# File: C:\AI\apps\Makehuman\dg_face_pipeline\render_canonical_report.py
# Repo: makehumancommunity/makehuman  Branch: feat/face-proportion-pipeline
"""
Render the canonical-proportion report produced by face_proportion_analyzer.py.

This is not a beauty score. It is a visual ruler: classical fifths/thirds are
drawn over the source image and the measured deltas are charted so we can turn
them into MakeHuman sculpt targets later.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("render_canonical_report")

PIPELINE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = PIPELINE_DIR / "examples"
OUTPUTS_DIR = PIPELINE_DIR / "outputs"


def _crop_image(
    img: np.ndarray,
    bbox: List[float] | None,
    pad_frac: float = 0.22,
) -> Tuple[np.ndarray, float, float]:
    if not bbox or len(bbox) != 4:
        return img, 0.0, 0.0
    h, w = img.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    pad = pad_frac * max(xmax - xmin, ymax - ymin)
    x0 = max(0, int(xmin - pad))
    y0 = max(0, int(ymin - pad))
    x1 = min(w, int(xmax + pad))
    y1 = min(h, int(ymax + pad))
    return img[y0:y1, x0:x1], float(x0), float(y0)


def _pt(landmarks: Dict[str, List[float]], name: str, x0: float, y0: float) -> Tuple[float, float]:
    x, y = landmarks[name]
    return float(x) - x0, float(y) - y0


def draw_grid(
    ax: plt.Axes,
    photo: Path,
    report: Dict[str, object],
) -> None:
    img = mpimg.imread(str(photo))
    crop, x0, y0 = _crop_image(img, report.get("face_bbox"))
    ax.imshow(crop)
    ax.set_title("Canon ruler overlay", color="white", fontsize=12, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#202024")

    landmarks = report.get("landmarks_px", {})
    if not isinstance(landmarks, dict):
        return

    required = {
        "face_l_edge", "face_r_edge", "eye_l_outer", "eye_l_inner",
        "eye_r_inner", "eye_r_outer", "forehead_top", "brow_center",
        "nose_bottom", "mouth_upper", "chin", "nose_wing_l", "nose_wing_r",
        "mouth_corner_l", "mouth_corner_r",
    }
    if not required.issubset(landmarks):
        return

    face_l = _pt(landmarks, "face_l_edge", x0, y0)
    face_r = _pt(landmarks, "face_r_edge", x0, y0)
    x_left, x_right = sorted([face_l[0], face_r[0]])

    eye_points = [
        _pt(landmarks, "eye_l_outer", x0, y0),
        _pt(landmarks, "eye_l_inner", x0, y0),
        _pt(landmarks, "eye_r_inner", x0, y0),
        _pt(landmarks, "eye_r_outer", x0, y0),
    ]
    eye_y = float(np.mean([p[1] for p in eye_points]))

    # Ideal fifths across approximate cheek-to-cheek width.
    fifths = np.linspace(x_left, x_right, 6)
    ax.plot([x_left, x_right], [eye_y, eye_y], color="#ff3b30", lw=4, alpha=0.92)
    for i, x in enumerate(fifths):
        ax.plot([x, x], [eye_y - 22, eye_y + 22], color="#ff3b30", lw=4, alpha=0.92)
        if i < 5:
            ax.text(
                (fifths[i] + fifths[i + 1]) / 2.0,
                eye_y - 28,
                str(i + 1),
                color="#ff3b30",
                fontsize=18,
                ha="center",
                va="bottom",
            )

    # Actual eye/nose/mouth widths.
    for a, b, color, yoff in [
        ("eye_l_outer", "eye_l_inner", "#37a2ff", 30),
        ("eye_r_inner", "eye_r_outer", "#37a2ff", 30),
        ("nose_wing_l", "nose_wing_r", "#42ff4f", 72),
        ("mouth_corner_l", "mouth_corner_r", "#ff4fb3", 108),
    ]:
        pa = _pt(landmarks, a, x0, y0)
        pb = _pt(landmarks, b, x0, y0)
        yy = max(pa[1], pb[1]) + yoff
        ax.plot([pa[0], pb[0]], [yy, yy], color=color, lw=4, solid_capstyle="round")
        ax.plot([pa[0], pa[0]], [yy - 14, yy + 14], color=color, lw=4, solid_capstyle="round")
        ax.plot([pb[0], pb[0]], [yy - 14, yy + 14], color=color, lw=4, solid_capstyle="round")

    # Actual thirds and ideal equal thirds.
    top = _pt(landmarks, "forehead_top", x0, y0)[1]
    brow = _pt(landmarks, "brow_center", x0, y0)[1]
    nose = _pt(landmarks, "nose_bottom", x0, y0)[1]
    chin = _pt(landmarks, "chin", x0, y0)[1]
    x_mid = (x_left + x_right) / 2.0
    actual_lines = [top, brow, nose, chin]
    ideal_lines = np.linspace(top, chin, 4)

    for y in ideal_lines:
        ax.plot([x_mid + 28, x_mid + 78], [y, y], color="#ffd60a", lw=3, alpha=0.95)
    ax.plot([x_mid + 53, x_mid + 53], [ideal_lines[0], ideal_lines[-1]], color="#ffd60a", lw=3, alpha=0.95)

    for y in actual_lines:
        ax.plot([x_mid - 78, x_mid - 28], [y, y], color="#37a2ff", lw=4, alpha=0.95)
    ax.plot([x_mid - 53, x_mid - 53], [actual_lines[0], actual_lines[-1]], color="#37a2ff", lw=4, alpha=0.95)

    for spine in ax.spines.values():
        spine.set_color("#404048")


def draw_delta_chart(ax: plt.Axes, report: Dict[str, object]) -> None:
    analysis = report.get("canonical_analysis", {})
    if not isinstance(analysis, dict):
        return
    deltas = analysis.get("identity_deltas", {})
    if not isinstance(deltas, dict):
        return

    preferred = [
        "face_width_per_eye",
        "eye_outer_span_per_eye",
        "intereye_per_eye",
        "nose_width_per_eye",
        "mouth_width_per_eye",
        "mouth_width_per_nose",
        "upper_third_share",
        "middle_third_share",
        "lower_third_share",
        "lower_third_upper_lip_share",
    ]
    rows = []
    for key in preferred:
        item = deltas.get(key)
        if isinstance(item, dict):
            rows.append((key, item))

    labels = [key.replace("_", " ") for key, _ in rows]
    values = [float(item.get("delta_pct", 0.0)) for _, item in rows]
    colors = ["#ff6b6b" if v < 0 else "#4dabf7" for v in values]
    y = np.arange(len(rows))

    ax.barh(y, values, color=colors, alpha=0.9)
    ax.axvline(0, color="#e8e8e8", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("delta from canon (%)", color="white")
    ax.set_title("Identity deltas vs classical scaffold", color="white", fontsize=12, pad=6)
    ax.tick_params(axis="x", colors="white")
    ax.set_facecolor("#202024")
    for spine in ax.spines.values():
        spine.set_color("#404048")

    limit = max(20.0, max(abs(v) for v in values) * 1.25 if values else 20.0)
    ax.set_xlim(-limit, limit)
    for i, _ in enumerate(rows):
        v = values[i]
        text_x = v + (1.0 if v >= 0 else -1.0)
        ha = "left" if v >= 0 else "right"
        ax.text(
            text_x,
            i,
            f"{v:+.1f}%",
            color="white",
            fontsize=8,
            va="center",
            ha=ha,
        )


def render(photo: Path, proportions: Path, out_png: Path) -> Path:
    report = json.loads(proportions.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#15151a")
    draw_grid(axes[0], photo, report)
    draw_delta_chart(axes[1], report)
    fig.suptitle(
        "Photo -> canon proportionality report",
        color="white",
        fontsize=14,
        y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Wrote canon report: %s", out_png)
    return out_png


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render canon proportionality overlay/report.")
    p.add_argument("--photo", type=Path, default=EXAMPLES_DIR / "winona_ref.png")
    p.add_argument("--proportions", type=Path, default=OUTPUTS_DIR / "data" / "face_proportions.json")
    p.add_argument("--out", type=Path, default=OUTPUTS_DIR / "renders" / "canon_report.png")
    args = p.parse_args(argv)
    try:
        render(args.photo, args.proportions, args.out)
    except FileNotFoundError as exc:
        log.error("Missing input: %s", exc)
        return 2
    except (RuntimeError, ValueError, IOError, KeyError) as exc:
        log.error("Render failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
