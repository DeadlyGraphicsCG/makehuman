# File: C:\AI\apps\Makehuman\dg_face_pipeline\render_learned_mesh.py
# Repo: makehumancommunity/makehuman  Branch: feat/face-proportion-pipeline
# Orchestrator: DG_Brain
"""
Render a learned face-mesh OBJ next to its source photograph.

Used to visually verify the output of `learned_face_mesh.py` (MediaPipe
FaceMesh -> 478-vert OBJ). Renders three panels:

  1. Source photo (cropped to the face bbox if a proportions JSON is given)
  2. Reconstructed mesh, frontal Lambert-shaded
  3. Reconstructed mesh, 3/4 view (yaw 25 deg) so depth detail is visible

Run:
    python C:\\AI\\apps\\Makehuman\\dg_face_pipeline\\render_learned_mesh.py ^
        --photo C:\\AI\\apps\\DG_Brain\\assets\\refs\\winona_ref.png ^
        --mesh  C:\\AI\\apps\\DG_Brain\\data\\winona_face_mesh.obj ^
        --out   C:\\AI\\apps\\DG_Brain\\data\\renders\\winona_learned_mesh.png
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("render_learned_mesh")

VERT_RE = re.compile(r"^v\s+")
FACE_RE = re.compile(r"^f\s+")

LIGHT_DIR = np.array([0.25, 0.35, 1.0])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
AMBIENT = 0.30


def load_obj_simple(obj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    verts: List[List[float]] = []
    tris: List[List[int]] = []
    with obj_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if VERT_RE.match(line):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif FACE_RE.match(line):
                parts = line.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                if len(idx) == 3:
                    tris.append(idx)
                elif len(idx) == 4:
                    tris.append([idx[0], idx[1], idx[2]])
                    tris.append([idx[0], idx[2], idx[3]])
    v = np.asarray(verts, dtype=np.float64)
    t = np.asarray(tris, dtype=np.int64)
    log.info("Loaded %s: %d verts, %d triangles", obj_path.name, len(v), len(t))
    return v, t


def rotate_y(verts: np.ndarray, yaw_deg: float) -> np.ndarray:
    centroid = verts.mean(axis=0)
    v = verts - centroid
    y = math.radians(yaw_deg)
    R = np.array([[math.cos(y), 0.0, math.sin(y)],
                  [0.0, 1.0, 0.0],
                  [-math.sin(y), 0.0, math.cos(y)]])
    return (v @ R.T) + centroid


def shade(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    nm = np.linalg.norm(n, axis=1, keepdims=True)
    nm = np.where(nm == 0, 1.0, nm)
    n = n / nm
    i = np.clip(n @ LIGHT_DIR, 0.0, 1.0)
    return AMBIENT + (1.0 - AMBIENT) * i


def draw_mesh(
    ax: plt.Axes,
    verts: np.ndarray,
    tris: np.ndarray,
    title: str,
    skin: Tuple[float, float, float] = (0.86, 0.71, 0.62),
) -> None:
    intensity = shade(verts, tris)
    polys = verts[tris][:, :, [0, 1]]
    colors = np.tile(skin, (len(tris), 1)) * intensity[:, None]
    order = np.argsort(verts[tris][:, :, 2].mean(axis=1))
    pc = PolyCollection(polys[order], facecolors=colors[order],
                        edgecolors="none", linewidths=0)
    ax.add_collection(pc)
    pad_x = 0.04 * (verts[:, 0].max() - verts[:, 0].min())
    pad_y = 0.06 * (verts[:, 1].max() - verts[:, 1].min())
    ax.set_xlim(verts[:, 0].min() - pad_x, verts[:, 0].max() + pad_x)
    ax.set_ylim(verts[:, 1].min() - pad_y, verts[:, 1].max() + pad_y)
    ax.set_aspect("equal")
    ax.set_facecolor("#202024")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color="white", fontsize=11, pad=6)
    for s in ax.spines.values():
        s.set_color("#404048")


def draw_photo(
    ax: plt.Axes, photo: Path, proportions: Path | None, pad_frac: float = 0.18,
) -> None:
    img = mpimg.imread(str(photo))
    h, w = img.shape[:2]
    crop = img
    if proportions and proportions.exists():
        meta = json.loads(proportions.read_text(encoding="utf-8"))
        bbox = meta.get("face_bbox")
        if bbox and len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            pad = pad_frac * max(xmax - xmin, ymax - ymin)
            x0 = max(0, int(xmin - pad))
            y0 = max(0, int(ymin - pad))
            x1 = min(w, int(xmax + pad))
            y1 = min(h, int(ymax + pad))
            crop = img[y0:y1, x0:x1]
    ax.imshow(crop)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#202024")
    ax.set_title("Source photograph", color="white", fontsize=11, pad=6)
    for s in ax.spines.values():
        s.set_color("#404048")


def render(
    photo: Path, mesh: Path, out_png: Path, proportions: Path | None,
) -> Path:
    verts, tris = load_obj_simple(mesh)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="#15151a")
    draw_photo(axes[0], photo, proportions)
    draw_mesh(axes[1], verts, tris, "Reconstructed mesh -- frontal")
    draw_mesh(axes[2], rotate_y(verts, 25.0), tris,
              "Reconstructed mesh -- 3/4 view (yaw +25 deg)")
    fig.suptitle(
        "Photo -> learned face mesh (MediaPipe FaceMesh, 478 verts)",
        color="white", fontsize=13, y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Wrote render: %s", out_png)
    return out_png


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--photo", type=Path,
                   default=Path(r"C:\AI\apps\DG_Brain\assets\refs\winona_ref.png"))
    p.add_argument("--mesh", type=Path,
                   default=Path(r"C:\AI\apps\DG_Brain\data\subject_face_mesh.obj"))
    p.add_argument("--proportions", type=Path,
                   default=Path(r"C:\AI\apps\DG_Brain\data\face_proportions.json"))
    p.add_argument("--out", type=Path,
                   default=Path(r"C:\AI\apps\DG_Brain\data\renders\learned_mesh.png"))
    args = p.parse_args(argv)
    try:
        render(args.photo, args.mesh, args.out, args.proportions)
    except FileNotFoundError as exc:
        log.error("Missing: %s", exc); return 2
    except (RuntimeError, ValueError, IOError) as exc:
        log.error("Render failed: %s", exc); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
