# File: C:\AI\apps\3D\Makehuman\dg_face_pipeline\render_learned_mesh.py
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
        --photo C:\\AI\\apps\\Core\\DG_Brain\\assets\\refs\\winona_ref.png ^
        --mesh  C:\\AI\\apps\\Core\\DG_Brain\\data\\winona_face_mesh.obj ^
        --out   C:\\AI\\apps\\Core\\DG_Brain\\data\\renders\\winona_learned_mesh.png
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

PIPELINE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = PIPELINE_DIR / "examples"
OUTPUTS_DIR = PIPELINE_DIR / "outputs"

VERT_RE = re.compile(r"^v\s+")
FACE_RE = re.compile(r"^f\s+")

LIGHT_DIR = np.array([0.25, 0.35, 1.0])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
AMBIENT = 0.30


VT_RE = re.compile(r"^vt\s+")
MTLLIB_RE = re.compile(r"^mtllib\s+")


def load_obj_with_uvs(
    obj_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (verts Nx3, uvs Mx2, tri_v Tx3, tri_uv Tx3).

    tri_v[i] = vertex indices for triangle i (0-indexed).
    tri_uv[i] = UV indices for triangle i (0-indexed). When the OBJ has no
    vt lines, tri_uv is identical to tri_v and `uvs` is empty -- callers
    should check `uvs.size > 0` before attempting texture sampling.
    """
    verts: List[List[float]] = []
    uvs: List[List[float]] = []
    tri_v: List[List[int]] = []
    tri_uv: List[List[int]] = []
    with obj_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if VERT_RE.match(line):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif VT_RE.match(line):
                parts = line.split()
                uvs.append([float(parts[1]), float(parts[2])])
            elif FACE_RE.match(line):
                parts = line.split()[1:]
                v_idx: List[int] = []
                vt_idx: List[int] = []
                for p in parts:
                    bits = p.split("/")
                    v_idx.append(int(bits[0]) - 1)
                    if len(bits) >= 2 and bits[1]:
                        vt_idx.append(int(bits[1]) - 1)
                    else:
                        vt_idx.append(int(bits[0]) - 1)
                if len(v_idx) == 3:
                    tri_v.append(v_idx)
                    tri_uv.append(vt_idx)
                elif len(v_idx) == 4:
                    tri_v.append([v_idx[0], v_idx[1], v_idx[2]])
                    tri_v.append([v_idx[0], v_idx[2], v_idx[3]])
                    tri_uv.append([vt_idx[0], vt_idx[1], vt_idx[2]])
                    tri_uv.append([vt_idx[0], vt_idx[2], vt_idx[3]])
    v = np.asarray(verts, dtype=np.float64)
    uv = np.asarray(uvs, dtype=np.float64) if uvs else np.empty((0, 2))
    tv = np.asarray(tri_v, dtype=np.int64)
    tu = np.asarray(tri_uv, dtype=np.int64)
    log.info(
        "Loaded %s: %d verts, %d UVs, %d triangles",
        obj_path.name, len(v), len(uv), len(tv),
    )
    return v, uv, tv, tu


def resolve_obj_texture(obj_path: Path) -> Path | None:
    """Return the first map_Kd texture referenced by the OBJ's MTL, if any."""
    mtllibs: list[str] = []
    with obj_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if MTLLIB_RE.match(line):
                mtllibs.append(line.split(maxsplit=1)[1])
    for mtl_name in mtllibs:
        mtl_path = obj_path.parent / mtl_name
        if not mtl_path.exists():
            continue
        with mtl_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if line.lower().startswith("map_kd "):
                    tex = line.split(maxsplit=1)[1]
                    texture = (mtl_path.parent / tex).resolve()
                    if texture.exists():
                        return texture
    return None


def load_obj_simple(obj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Back-compat shim: discard UV data, return only verts + tri vertex indices."""
    v, _uv, tv, _tu = load_obj_with_uvs(obj_path)
    return v, tv


def rotate_y(verts: np.ndarray, yaw_deg: float) -> np.ndarray:
    centroid = verts.mean(axis=0)
    v = verts - centroid
    y = math.radians(yaw_deg)
    R = np.array([[math.cos(y), 0.0, math.sin(y)],
                  [0.0, 1.0, 0.0],
                  [-math.sin(y), 0.0, math.cos(y)]])
    return (v @ R.T) + centroid


def sample_photo_per_triangle(
    photo_rgb: np.ndarray, uvs_per_tri: np.ndarray,
) -> np.ndarray:
    """Sample the source image at each triangle's centroid UV.

    photo_rgb : (H, W, 3 or 4) image, 0-1 float OR 0-255 uint8
    uvs_per_tri : (T, 3, 2) UV coords (u right, v up, OBJ convention)
    Returns : (T, 3) RGB in 0-1 float.
    """
    centroids = uvs_per_tri.mean(axis=1)  # (T, 2)
    h, w = photo_rgb.shape[:2]
    us = np.clip((centroids[:, 0] * (w - 1)).astype(int), 0, w - 1)
    # OBJ convention has v growing up; image arrays have y growing down -> flip
    vs = np.clip(((1.0 - centroids[:, 1]) * (h - 1)).astype(int), 0, h - 1)
    rgb = photo_rgb[vs, us]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = rgb.astype(np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return rgb


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
    uvs: np.ndarray | None = None,
    tri_uv: np.ndarray | None = None,
    photo_rgb: np.ndarray | None = None,
    shade_strength: float = 0.55,
) -> None:
    intensity = shade(verts, tris)
    polys = verts[tris][:, :, [0, 1]]

    if (uvs is not None and tri_uv is not None and photo_rgb is not None
            and len(uvs) > 0):
        uvs_per_tri = uvs[tri_uv]  # (T, 3, 2)
        base = sample_photo_per_triangle(photo_rgb, uvs_per_tri)
        # Dampen shading so the photo dominates; pure Lambert (intensity in
        # [AMBIENT, 1]) would over-darken textured surfaces because the photo
        # already contains baked-in lighting from the source.
        shade_factor = 1.0 - shade_strength + shade_strength * intensity
        colors = base * shade_factor[:, None]
    else:
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
    verts, uvs, tri_v, tri_uv = load_obj_with_uvs(mesh)
    photo_rgb: np.ndarray | None = None
    if photo and photo.exists():
        photo_rgb = mpimg.imread(str(photo))
        log.info(
            "Loaded photo for texture projection: %s (%dx%d)",
            photo.name, photo_rgb.shape[1], photo_rgb.shape[0],
        )

    texture_path = resolve_obj_texture(mesh)
    texture_rgb = None
    if texture_path is not None:
        texture_rgb = mpimg.imread(str(texture_path))
        log.info(
            "Loaded mesh texture from MTL: %s (%dx%d)",
            texture_path.name, texture_rgb.shape[1], texture_rgb.shape[0],
        )
    elif photo_rgb is not None:
        texture_rgb = photo_rgb
        log.info("No MTL texture found; using source photo as projective texture")

    has_texture = (texture_rgb is not None and len(uvs) > 0)
    suffix = " (photo-textured)" if has_texture else " (flat-shaded)"

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor="#15151a")
    draw_photo(axes[0], photo, proportions)
    draw_mesh(
        axes[1], verts, tri_v, "Reconstructed mesh -- frontal" + suffix,
        uvs=uvs, tri_uv=tri_uv, photo_rgb=texture_rgb,
    )
    draw_mesh(
        axes[2], rotate_y(verts, 25.0), tri_v,
        "Reconstructed mesh -- 3/4 view (yaw +25 deg)" + suffix,
        uvs=uvs, tri_uv=tri_uv, photo_rgb=texture_rgb,
    )
    fig.suptitle(
        "Photo -> learned face mesh (MediaPipe FaceMesh, 478 verts)"
        + (" with projective texture" if has_texture else ""),
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
                   default=EXAMPLES_DIR / "winona_ref.png")
    p.add_argument("--mesh", type=Path,
                   default=OUTPUTS_DIR / "data" / "subject_face_mesh.obj")
    p.add_argument("--proportions", type=Path,
                   default=OUTPUTS_DIR / "data" / "face_proportions.json")
    p.add_argument("--out", type=Path,
                   default=OUTPUTS_DIR / "renders" / "learned_mesh.png")
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
