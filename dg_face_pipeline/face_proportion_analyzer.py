# File: C:\AI\apps\Makehuman\dg_face_pipeline\face_proportion_analyzer.py
# Repo: makehumancommunity/makehuman  Branch: feat/face-proportion-pipeline
# Orchestrator: DG_Brain (inputs/outputs flow through C:\AI\apps\DG_Brain\)
"""
Photo -> MakeHuman Face Adaptor : Step A + B
=============================================
Extracts dimensionless Loomis/O'Reilly facial proportions from a portrait
photograph using MediaPipe FaceMesh (468 landmarks, optional iris refinement
adds 10 more), compares them to canonical Loomis baselines, and emits a
JSON delta payload mapped to MakeHuman measurement / modeling modifier names.

Pipeline:
  Step A : Landmark extraction + face-aligned 2D projection
  Step B : Ratio computation (eye-width W_eye = 1.0 base unit)
  Step C : Delta -> MakeHuman modifier suggestion table

Author: DG_Brain Pipeline
Target: Python 3.11+ on Windows 11 (PowerShell 7)

Install:
    py -3.11 -m pip install --upgrade pip
    py -3.11 -m pip install mediapipe==0.10.18 opencv-python==4.10.0.84 numpy==1.26.4

Run:
    py -3.11 C:\\AI\\apps\\Makehuman\\dg_face_pipeline\\face_proportion_analyzer.py ^
        --image C:\\AI\\apps\\DG_Brain\\assets\\portrait.jpg ^
        --out C:\\AI\\apps\\DG_Brain\\data\\face_proportions.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("face_proportion_analyzer")

# -----------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices (well-known constants from the 468 set;
# +iris when refine_landmarks=True adds 468..477 for the two iris rings).
# -----------------------------------------------------------------------------
LM = {
    "eye_l_outer": 33,
    "eye_l_inner": 133,
    "eye_r_inner": 362,
    "eye_r_outer": 263,
    "iris_l_center": 468,   # refine_landmarks=True
    "iris_r_center": 473,
    "nose_tip": 1,
    "nose_bottom": 2,
    "nose_wing_l": 98,
    "nose_wing_r": 327,
    "mouth_corner_l": 61,
    "mouth_corner_r": 291,
    "mouth_upper": 13,
    "mouth_lower": 14,
    "chin": 152,
    "forehead_top": 10,     # MediaPipe upper-forehead vertex; proxy for hairline
    "brow_center": 9,
    "brow_l_peak": 105,
    "brow_r_peak": 334,
}

# -----------------------------------------------------------------------------
# Canonical Loomis / O'Reilly baseline ratios.
# Unit: 1.0 == one eye-width (W_eye). Sourced from Loomis "Drawing the Head
# and Hands" and Andrew Reid (O'Reilly) revisions:
#   - Face width  = 5  W_eye   (eye-eye-eye-eye-eye across)
#   - Inter-eye   = 1  W_eye   (one eye fits between inner canthi)
#   - Nose width  = 1  W_eye   (~aligned with inner canthi)
#   - Mouth width = 1.5 W_eye  (corners under iris inner edge)
#   - Hairline -> Brow   = Brow -> Nose-base = Nose-base -> Chin
#         each segment = ~2.0 W_eye, total face-height ~7 W_eye
#   - Eyes on horizontal midline of the head (head_height ~ 7 W_eye)
# -----------------------------------------------------------------------------
LOOMIS_BASELINE: Dict[str, float] = {
    "face_width_per_eye": 5.0,
    "intereye_per_eye": 1.0,
    "nose_width_per_eye": 1.0,
    "mouth_width_per_eye": 1.5,
    "face_height_per_eye": 7.0,
    "hairline_to_brow_per_eye": 2.0,
    "brow_to_nose_per_eye": 2.0,
    "nose_to_chin_per_eye": 2.0,
    "eye_to_mouth_per_eye": 2.5,
    "ipd_per_eye": 2.0,  # inter-pupillary distance ~ 2 eye-widths
}

# -----------------------------------------------------------------------------
# Map abstract proportion delta -> concrete MakeHuman modifier path.
# Modifier paths follow the convention seen in
# data/modifiers/modeling_modifiers.json and measurement_modifiers.json,
# rendered as "<group>/<modifier>" so the downstream applier can look them up.
# Value sign convention: positive ratio_delta (subject larger than Loomis) ->
# positive MH modifier value in [-1.0, 1.0]; clamp + scale via SENSITIVITY.
# -----------------------------------------------------------------------------
MODIFIER_MAP: Dict[str, str] = {
    "face_width_per_eye":        "macrodetails/universal-face-width",
    "nose_width_per_eye":        "nose/nose-trans-scale-decr|incr",
    "mouth_width_per_eye":       "mouth/mouth-trans-scale-decr|incr",
    "face_height_per_eye":       "head/head-scale-vert-decr|incr",
    "hairline_to_brow_per_eye":  "forehead/forehead-trans-down|up",
    "brow_to_nose_per_eye":      "nose/nose-trans-up|down",
    "nose_to_chin_per_eye":      "chin/chin-trans-down|up",
    "eye_to_mouth_per_eye":      "mouth/mouth-trans-up|down",
    "intereye_per_eye":          "eyes/l-eye-trans-out|in",
    "ipd_per_eye":               "eyes/r-eye-trans-out|in",
}

SENSITIVITY = 0.8  # global scaler when converting ratio delta -> MH value
MH_VALUE_CLAMP = (-1.0, 1.0)


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass
class FaceLandmarks:
    """Aligned 2D landmarks in face-local coordinates (eye-line == x-axis)."""

    points: Dict[str, np.ndarray] = field(default_factory=dict)
    image_size: Tuple[int, int] = (0, 0)
    roll_rad: float = 0.0

    def dist(self, a: str, b: str) -> float:
        return float(np.linalg.norm(self.points[a] - self.points[b]))


@dataclass
class ProportionReport:
    image: str
    subject_ratios: Dict[str, float]
    baseline_ratios: Dict[str, float]
    delta_ratios: Dict[str, float]
    modifier_targets: List[Dict[str, float]]
    eye_width_px: float
    notes: str


# -----------------------------------------------------------------------------
# Step A : Landmark extraction
# -----------------------------------------------------------------------------
def extract_landmarks(image_path: Path) -> FaceLandmarks:
    """Run MediaPipe FaceMesh, return face-aligned landmarks in pixel space."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise IOError(f"OpenCV could not decode: {image_path}")
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_fm = mp.solutions.face_mesh
    with mp_fm.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(img_rgb)

    if not result.multi_face_landmarks:
        raise RuntimeError("FaceMesh found no face in image")

    raw = result.multi_face_landmarks[0].landmark
    pts_px: Dict[str, np.ndarray] = {}
    for name, idx in LM.items():
        if idx >= len(raw):
            log.warning("Landmark idx %d (%s) absent; enable refine_landmarks", idx, name)
            continue
        lm = raw[idx]
        pts_px[name] = np.array([lm.x * w, lm.y * h], dtype=np.float64)

    # Roll-align: rotate so eye line is horizontal.
    el = pts_px["eye_l_outer"]
    er = pts_px["eye_r_outer"]
    delta = er - el
    roll = float(np.arctan2(delta[1], delta[0]))
    cos_r, sin_r = np.cos(-roll), np.sin(-roll)
    rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    centroid = (el + er) / 2.0
    aligned = {k: (rot @ (p - centroid)) for k, p in pts_px.items()}

    return FaceLandmarks(points=aligned, image_size=(w, h), roll_rad=roll)


# -----------------------------------------------------------------------------
# Step B : Proportion computation (dimensionless, W_eye == 1.0)
# -----------------------------------------------------------------------------
def compute_ratios(face: FaceLandmarks) -> Tuple[Dict[str, float], float]:
    """Return dict of subject ratios + raw eye-width in pixels."""
    w_eye_l = abs(face.points["eye_l_inner"][0] - face.points["eye_l_outer"][0])
    w_eye_r = abs(face.points["eye_r_outer"][0] - face.points["eye_r_inner"][0])
    w_eye = (w_eye_l + w_eye_r) / 2.0
    if w_eye <= 0:
        raise ValueError("Degenerate eye width; landmarks invalid")

    def _abs_x(a: str, b: str) -> float:
        return abs(face.points[a][0] - face.points[b][0])

    def _abs_y(a: str, b: str) -> float:
        return abs(face.points[a][1] - face.points[b][1])

    has_iris = "iris_l_center" in face.points and "iris_r_center" in face.points
    if has_iris:
        ipd = float(np.linalg.norm(
            face.points["iris_l_center"] - face.points["iris_r_center"]
        ))
    else:
        # Fallback : midpoint of each eye's canthi
        eye_l_mid = (face.points["eye_l_outer"] + face.points["eye_l_inner"]) / 2.0
        eye_r_mid = (face.points["eye_r_outer"] + face.points["eye_r_inner"]) / 2.0
        ipd = float(np.linalg.norm(eye_l_mid - eye_r_mid))

    ratios: Dict[str, float] = {
        "face_width_per_eye":       _abs_x("eye_l_outer", "eye_r_outer") / w_eye,
        "intereye_per_eye":         _abs_x("eye_l_inner", "eye_r_inner") / w_eye,
        "nose_width_per_eye":       _abs_x("nose_wing_l", "nose_wing_r") / w_eye,
        "mouth_width_per_eye":      _abs_x("mouth_corner_l", "mouth_corner_r") / w_eye,
        "face_height_per_eye":      _abs_y("forehead_top", "chin") / w_eye,
        "hairline_to_brow_per_eye": _abs_y("forehead_top", "brow_center") / w_eye,
        "brow_to_nose_per_eye":     _abs_y("brow_center", "nose_bottom") / w_eye,
        "nose_to_chin_per_eye":     _abs_y("nose_bottom", "chin") / w_eye,
        "eye_to_mouth_per_eye":     _abs_y("eye_l_inner", "mouth_upper") / w_eye,
        "ipd_per_eye":              ipd / w_eye,
    }
    return ratios, w_eye


# -----------------------------------------------------------------------------
# Step C : Delta -> MakeHuman modifier suggestions
# -----------------------------------------------------------------------------
def build_modifier_targets(
    subject: Dict[str, float],
    baseline: Dict[str, float],
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    deltas: Dict[str, float] = {}
    targets: List[Dict[str, float]] = []
    for key, base_val in baseline.items():
        if key not in subject:
            continue
        delta = subject[key] - base_val
        normalised = delta / base_val if base_val != 0 else 0.0
        deltas[key] = round(delta, 5)
        mh_path = MODIFIER_MAP.get(key)
        if mh_path is None:
            continue
        value = float(np.clip(normalised * SENSITIVITY, *MH_VALUE_CLAMP))
        targets.append({
            "ratio_key": key,
            "modifier": mh_path,
            "value": round(value, 4),
            "subject_ratio": round(subject[key], 4),
            "baseline_ratio": round(base_val, 4),
            "ratio_delta": round(delta, 4),
        })
    return deltas, targets


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def analyze(image_path: Path, out_path: Path) -> ProportionReport:
    log.info("Step A : extracting landmarks from %s", image_path)
    face = extract_landmarks(image_path)
    log.info(
        "Step A : %d landmarks aligned (roll=%.2f deg)",
        len(face.points), np.degrees(face.roll_rad),
    )

    log.info("Step B : computing dimensionless ratios (W_eye = 1.0)")
    subject_ratios, eye_width_px = compute_ratios(face)
    for k, v in subject_ratios.items():
        log.info("  %-28s = %.3f W_eye  (baseline %.3f)", k, v, LOOMIS_BASELINE[k])

    log.info("Step C : building MakeHuman modifier delta payload")
    delta_ratios, mh_targets = build_modifier_targets(subject_ratios, LOOMIS_BASELINE)

    report = ProportionReport(
        image=str(image_path),
        subject_ratios={k: round(v, 4) for k, v in subject_ratios.items()},
        baseline_ratios=LOOMIS_BASELINE,
        delta_ratios=delta_ratios,
        modifier_targets=mh_targets,
        eye_width_px=round(eye_width_px, 3),
        notes=(
            "Ratios dimensionless; W_eye = 1.0 base unit (Loomis/O'Reilly). "
            "modifier_targets are suggestion values in [-1.0, 1.0] suitable for "
            "MakeHuman humanmodifier.HumanModifier.setValue() — apply via the "
            "modeling_modifiers.json registry at "
            "C:\\AI\\apps\\Makehuman\\makehuman\\data\\modifiers\\."
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    log.info("Wrote %s", out_path)
    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Photo -> MakeHuman face proportion analyzer (Loomis/O'Reilly).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(r"C:\AI\apps\DG_Brain\assets\portrait.jpg"),
        help="Path to source portrait image.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\AI\apps\DG_Brain\data\face_proportions.json"),
        help="Path to write JSON report.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        analyze(args.image, args.out)
    except FileNotFoundError as exc:
        log.error("Input missing: %s", exc)
        return 2
    except (RuntimeError, ValueError, IOError) as exc:
        log.error("Analysis failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
