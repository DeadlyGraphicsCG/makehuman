# File: C:\AI\apps\Makehuman\dg_face_pipeline\learned_face_mesh.py
# Repo: makehumancommunity/makehuman  Branch: feat/face-proportion-pipeline
# Orchestrator: DG_Brain
"""
Photo -> face mesh via a learned 3DMM-style reconstructor
=========================================================
Uses MediaPipe FaceMesh's 478 3D landmarks (a learned face-shape network --
not a generic landmark detector, it predicts per-vertex 3D position from
image content) combined with the canonical face model's triangulation to
produce a Wavefront OBJ mesh of the subject's face directly from a photo.

This bypasses the MakeHuman modifier pipeline entirely. The output mesh
has 478 verts (face-only, no scalp/neck) which is sparser than FLAME's
~5000 or MICA's full head, but it ships zero install friction (we already
have MediaPipe + OpenCV from the analyzer step), no FLAME license required,
and no GPU dependency.

A subsequent step can fit MakeHuman modifier values to this mesh by
optimization (non-rigid ICP) when MH-topology output is needed downstream.

Run:
    python C:\\AI\\apps\\Makehuman\\dg_face_pipeline\\learned_face_mesh.py ^
        --image C:\\AI\\apps\\DG_Brain\\assets\\refs\\winona_ref.png ^
        --out   C:\\AI\\apps\\DG_Brain\\data\\winona_face_mesh.obj
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List

import cv2
import mediapipe as mp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("learned_face_mesh")

CANONICAL_OBJ = Path(__file__).parent / "canonical_face_model.obj"


# -----------------------------------------------------------------------------
# Canonical triangulation loader
# -----------------------------------------------------------------------------
def load_canonical_triangles(obj_path: Path) -> np.ndarray:
    """Parse triangle indices from MediaPipe's canonical_face_model.obj.

    OBJ faces may appear as "v", "v/vt", or "v/vt/vn"; we only need the v
    component. Quads (rare here) are fan-triangulated. Indices are converted
    from 1-based (OBJ convention) to 0-based.
    """
    face_re = re.compile(r"^f\s+")
    tris: List[List[int]] = []
    with obj_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not face_re.match(line):
                continue
            parts = line.split()[1:]
            idx = [int(p.split("/")[0]) - 1 for p in parts]
            if len(idx) == 3:
                tris.append(idx)
            elif len(idx) == 4:
                tris.append([idx[0], idx[1], idx[2]])
                tris.append([idx[0], idx[2], idx[3]])
            elif len(idx) > 4:
                for i in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(tris, dtype=np.int64)


# -----------------------------------------------------------------------------
# Midpoint subdivision (densification)
# -----------------------------------------------------------------------------
def subdivide_once(
    verts: np.ndarray, uvs: np.ndarray, tris: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One pass of midpoint subdivision -- every triangle becomes 4.

    Vertex positions interpolate linearly between the parent triangle's
    corners; same for UVs (which is the right thing for projective texture).
    Edges shared between triangles get a single shared midpoint so the mesh
    stays watertight.
    """
    new_verts: List[List[float]] = [list(v) for v in verts]
    new_uvs: List[List[float]] = [list(u) for u in uvs] if len(uvs) else []
    midpoint_cache: dict[tuple[int, int], int] = {}

    def get_midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached
        mid_v = ((verts[i] + verts[j]) / 2.0).tolist()
        new_verts.append(mid_v)
        if len(uvs):
            new_uvs.append(((uvs[i] + uvs[j]) / 2.0).tolist())
        idx = len(new_verts) - 1
        midpoint_cache[key] = idx
        return idx

    new_tris: List[List[int]] = []
    for tri in tris:
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)
        new_tris.append([v0, m01, m20])
        new_tris.append([m01, v1, m12])
        new_tris.append([m20, m12, v2])
        new_tris.append([m01, m12, m20])

    return (
        np.asarray(new_verts, dtype=np.float64),
        np.asarray(new_uvs, dtype=np.float64) if new_uvs else np.empty((0, 2)),
        np.asarray(new_tris, dtype=np.int64),
    )


def subdivide_n(
    verts: np.ndarray, uvs: np.ndarray, tris: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for level in range(n):
        verts, uvs, tris = subdivide_once(verts, uvs, tris)
        log.info(
            "Subdivision pass %d/%d -> %d verts, %d triangles",
            level + 1, n, len(verts), len(tris),
        )
    return verts, uvs, tris


# -----------------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------------
def reconstruct(image_path: Path, out_obj: Path, subdivisions: int = 0) -> Path:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not CANONICAL_OBJ.exists():
        raise FileNotFoundError(f"Canonical face model missing: {CANONICAL_OBJ}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise IOError(f"OpenCV could not decode: {image_path}")
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as fm:
        result = fm.process(img_rgb)
    if not result.multi_face_landmarks:
        raise RuntimeError("FaceMesh found no face in image")

    lms = result.multi_face_landmarks[0].landmark
    # MediaPipe lm.x, lm.y are in [0,1] normalized image coords (y-down).
    # lm.z is roughly in the same scale as lm.x (relative depth, negative
    # toward camera for the canonical model orientation).
    #
    # Flip y so the OBJ is y-up (matches our renderer's MakeHuman convention)
    # and scale by image width so x/y/z are in commensurate units.
    verts = np.array(
        [[lm.x * w, (1.0 - lm.y) * h, -lm.z * w] for lm in lms],
        dtype=np.float64,
    )
    log.info("Reconstructed %d landmark verts from FaceMesh", len(verts))

    tris = load_canonical_triangles(CANONICAL_OBJ)
    log.info("Loaded %d canonical triangles", len(tris))

    max_tri_idx = int(tris.max())
    if max_tri_idx >= len(verts):
        raise RuntimeError(
            f"Triangulation references vertex {max_tri_idx} but only "
            f"{len(verts)} predicted from FaceMesh"
        )

    # UV coordinates come for free: each landmark's normalized image (x, y) is
    # its UV. We flip v so OBJ convention (origin bottom-left) lines up with
    # MediaPipe convention (origin top-left). Vertex index == UV index, so the
    # face references will be of the form "f v/v v/v v/v".
    uvs = np.array([[lm.x, 1.0 - lm.y] for lm in lms], dtype=np.float64)

    if subdivisions > 0:
        verts, uvs, tris = subdivide_n(verts, uvs, tris, subdivisions)

    out_obj.parent.mkdir(parents=True, exist_ok=True)
    with out_obj.open("w", encoding="utf-8") as fh:
        fh.write("# Face mesh reconstructed via MediaPipe FaceMesh\n")
        fh.write(f"# Source image: {image_path}\n")
        fh.write(f"# Image size: {w} x {h}\n")
        fh.write(f"# Verts: {len(verts)}  UVs: {len(uvs)}  Triangles: {len(tris)}\n")
        fh.write("g face_mesh\n")
        for v in verts:
            fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for uv in uvs:
            fh.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for t in tris:
            # Vertex index == UV index for this mesh (1-indexed in OBJ).
            fh.write(
                f"f {t[0] + 1}/{t[0] + 1} "
                f"{t[1] + 1}/{t[1] + 1} "
                f"{t[2] + 1}/{t[2] + 1}\n"
            )
    log.info("Wrote face mesh OBJ: %s (verts+UVs)", out_obj)
    return out_obj


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a face OBJ mesh from a photo via MediaPipe FaceMesh.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(r"C:\AI\apps\DG_Brain\assets\refs\winona_ref.png"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\AI\apps\DG_Brain\data\subject_face_mesh.obj"),
    )
    parser.add_argument(
        "--subdivisions",
        type=int,
        default=0,
        help=(
            "Number of midpoint-subdivision passes. Each pass quadruples the "
            "triangle count and reduces texture-sampling facets. 0 (default) "
            "keeps the raw 478-vert canonical mesh; 2 -> ~7600 verts, 14336 "
            "triangles -- visibly smoother texture and still <1s to render."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reconstruct(args.image, args.out, subdivisions=args.subdivisions)
    except FileNotFoundError as exc:
        log.error("Missing input: %s", exc)
        return 2
    except (RuntimeError, ValueError, IOError) as exc:
        log.error("Reconstruction failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
