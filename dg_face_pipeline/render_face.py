# File: C:\AI\apps\Makehuman\dg_face_pipeline\render_face.py
# Repo: makehumancommunity/makehuman  Branch: feat/face-proportion-pipeline
# Orchestrator: DG_Brain
"""
Photo -> MakeHuman Face Adaptor : Step D (visualization)
=========================================================
Renders the head region of a MakeHuman OBJ mesh as a flat-shaded frontal
portrait using only numpy + matplotlib (no GPU, no pyrender, no pyglet).
Designed for a two-up "baseline vs morphed" comparison panel so you can
eyeball the morph delta without launching the MakeHuman GUI.

Lighting model: Lambert with a single forward-up directional light. Face
normals derived from the cross product of triangle edges. Quad faces in
the base mesh are split into two triangles. Vertices are filtered to the
top portion of the mesh (head + neck) for legible portrait framing.

Run:
    python C:\\AI\\apps\\Makehuman\\dg_face_pipeline\\render_face.py ^
        --baseline C:\\AI\\apps\\Makehuman\\makehuman\\data\\3dobjs\\base.obj ^
        --morphed  C:\\AI\\apps\\DG_Brain\\data\\morphed.obj ^
        --out      C:\\AI\\apps\\DG_Brain\\data\\renders\\winona_mesh_compare.png
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

matplotlib.use("Agg")  # headless, no display required
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("render_face")

VERT_RE = re.compile(r"^v\s+")
FACE_RE = re.compile(r"^f\s+")
GROUP_RE = re.compile(r"^g\s+(\S+)")

# Head-region filter: keep verts whose y is within the top FRACTION of the
# mesh's y-extent. 0.12 captures only the head and a thin neck stub --
# anything more pulls in shoulders which look broken when the head is rotated.
HEAD_TOP_FRACTION = 0.12

# MakeHuman base.obj is segmented by `g <group>` directives. We keep only the
# real skin mesh; helper-* groups are hair/clothing/eyelash proxies and
# joint-* groups are tiny rigging markers -- both would render as visual
# noise on top of the face. The body skin group is named "body".
KEEP_GROUPS = {"body"}

# Light direction (camera-facing, slight elevation) for Lambert shading.
LIGHT_DIR = np.array([0.25, 0.35, 1.0])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
AMBIENT = 0.25


# -----------------------------------------------------------------------------
# OBJ loader
# -----------------------------------------------------------------------------
def load_obj(
    obj_path: Path,
    keep_groups: set[str] | None = None,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Return (vertices Nx3, faces list-of-list-of-int (0-indexed)).

    When `keep_groups` is provided, faces from any other `g <name>` block are
    dropped. Vertices are always loaded in full (they may be shared across
    groups); unreferenced vertices simply never appear in any face.
    """
    keep = keep_groups if keep_groups is not None else KEEP_GROUPS
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    current_group = ""
    keep_current = False
    group_face_counts: dict[str, int] = {}
    with obj_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            gm = GROUP_RE.match(line)
            if gm:
                current_group = gm.group(1)
                keep_current = current_group in keep
                continue
            if VERT_RE.match(line):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif FACE_RE.match(line):
                group_face_counts[current_group] = group_face_counts.get(current_group, 0) + 1
                if not keep_current:
                    continue
                parts = line.split()[1:]
                idx = []
                for p in parts:
                    # OBJ face entry may be "v", "v/vt", or "v/vt/vn"
                    v_str = p.split("/")[0]
                    idx.append(int(v_str) - 1)
                faces.append(idx)
    v = np.asarray(verts, dtype=np.float64)
    log.info(
        "Loaded %s: %d verts, %d faces kept (groups=%s, dropped %d helper/joint faces)",
        obj_path.name,
        len(v),
        len(faces),
        sorted(keep),
        sum(c for g, c in group_face_counts.items() if g not in keep),
    )
    return v, faces


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def triangulate(faces: List[List[int]]) -> np.ndarray:
    """Convert mixed tri/quad face list into a flat (M,3) triangle array."""
    tris: List[List[int]] = []
    for f in faces:
        if len(f) == 3:
            tris.append(f)
        elif len(f) == 4:
            tris.append([f[0], f[1], f[2]])
            tris.append([f[0], f[2], f[3]])
        elif len(f) > 4:
            # Fan-triangulate
            for i in range(1, len(f) - 1):
                tris.append([f[0], f[i], f[i + 1]])
    return np.asarray(tris, dtype=np.int64)


def filter_head_region(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Keep only triangles whose centroid lies in the top HEAD_TOP_FRACTION of y.

    Bounds are computed from referenced vertices only (the body group), so
    leftover unreferenced helper/joint verts can't skew the cutoff.
    """
    referenced = np.unique(tris.flatten())
    y_min = verts[referenced, 1].min()
    y_max = verts[referenced, 1].max()
    cutoff = y_max - HEAD_TOP_FRACTION * (y_max - y_min)
    centroids_y = verts[tris][:, :, 1].mean(axis=1)
    keep = centroids_y > cutoff
    return tris[keep]


def rotate_to_pose(
    verts: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    head_only_filter: bool = True,
) -> np.ndarray:
    """Apply yaw-pitch-roll rotation around the head centroid.

    yaw   = rotation around Y (up) axis      -- subject looking left/right
    pitch = rotation around X (side) axis    -- subject looking up/down
    roll  = rotation around Z (forward) axis -- head tilt

    The MakeHuman base mesh is centred near the origin with the head at +Y;
    we rotate around the head centroid (top portion of the mesh) so the body
    swings naturally rather than the whole figure pivoting at the feet.
    """
    if head_only_filter:
        y = verts[:, 1]
        head_mask = y > (y.max() - 0.18 * (y.max() - y.min()))
        centroid = verts[head_mask].mean(axis=0)
    else:
        centroid = verts.mean(axis=0)
    v = verts - centroid
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    Rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    R = Ry @ Rx @ Rz
    return (v @ R.T) + centroid


def compute_shading(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Per-triangle Lambert intensity in [0, 1]."""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normals = normals / norms
    intensity = normals @ LIGHT_DIR
    intensity = np.clip(intensity, 0.0, 1.0)
    return AMBIENT + (1.0 - AMBIENT) * intensity


# -----------------------------------------------------------------------------
# Single-panel render
# -----------------------------------------------------------------------------
def render_panel(
    ax: plt.Axes,
    verts: np.ndarray,
    tris: np.ndarray,
    title: str,
    skin_rgb: Tuple[float, float, float] = (0.82, 0.69, 0.60),
    drift_per_vert: np.ndarray | None = None,
    shared_xlim: Tuple[float, float] | None = None,
    shared_ylim: Tuple[float, float] | None = None,
) -> None:
    """Render `tris` shaded with Lambert OR colored by per-vertex drift.

    When `drift_per_vert` is provided, each triangle is colored by the mean
    drift of its 3 vertices using a hot colormap. Lambert shading is then
    multiplied in so the result still looks like a 3D head, not a flat mask.
    """
    tris_head = filter_head_region(verts, tris)
    if tris_head.size == 0:
        log.warning("Head filter removed all triangles; falling back to full mesh")
        tris_head = tris

    intensity = compute_shading(verts, tris_head)
    polys = verts[tris_head][:, :, [0, 1]]  # frontal projection (x, y)

    if drift_per_vert is not None:
        face_drift = drift_per_vert[tris_head].mean(axis=1)
        d_max = max(face_drift.max(), 1e-6)
        d_norm = np.clip(face_drift / d_max, 0.0, 1.0)
        cmap = plt.get_cmap("magma")
        base_colors = cmap(d_norm)[:, :3]
        face_colors = base_colors * intensity[:, None]
    else:
        face_colors = np.zeros((len(tris_head), 3))
        face_colors[:, 0] = intensity * skin_rgb[0]
        face_colors[:, 1] = intensity * skin_rgb[1]
        face_colors[:, 2] = intensity * skin_rgb[2]

    # Back-to-front painter's algorithm for crude depth occlusion
    centroids_z = verts[tris_head][:, :, 2].mean(axis=1)
    order = np.argsort(centroids_z)
    polys = polys[order]
    face_colors = face_colors[order]

    pc = PolyCollection(polys, facecolors=face_colors, edgecolors="none", linewidths=0)
    ax.add_collection(pc)

    # Frame to head bounding box (or to a shared bbox supplied by the caller
    # so the baseline / morphed / heatmap panels all render at identical scale
    # and the size difference of the morphed head is faithful, not an artefact
    # of matplotlib auto-framing).
    if shared_xlim is not None and shared_ylim is not None:
        ax.set_xlim(*shared_xlim)
        ax.set_ylim(*shared_ylim)
    else:
        head_verts = np.unique(tris_head.flatten())
        hv = verts[head_verts]
        pad_x = 0.02 * (hv[:, 0].max() - hv[:, 0].min() + 1e-6)
        pad_y = 0.04 * (hv[:, 1].max() - hv[:, 1].min() + 1e-6)
        ax.set_xlim(hv[:, 0].min() - pad_x, hv[:, 0].max() + pad_x)
        ax.set_ylim(hv[:, 1].min() - pad_y, hv[:, 1].max() + pad_y)
    ax.set_aspect("equal")
    ax.set_facecolor("#202024")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color="white", fontsize=12, pad=6)
    for spine in ax.spines.values():
        spine.set_color("#404048")


# -----------------------------------------------------------------------------
# Photo panel
# -----------------------------------------------------------------------------
def render_photo_panel(
    ax: plt.Axes,
    photo_path: Path,
    proportions_json: Path | None,
    title: str = "Source photograph",
    pad_frac: float = 0.18,
) -> None:
    img = mpimg.imread(str(photo_path))
    h, w = img.shape[:2]
    crop = img
    if proportions_json is not None and proportions_json.exists():
        meta = json.loads(proportions_json.read_text(encoding="utf-8"))
        bbox = meta.get("face_bbox")
        if bbox and len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            bw = xmax - xmin
            bh = ymax - ymin
            pad = pad_frac * max(bw, bh)
            x0 = max(0, int(xmin - pad))
            y0 = max(0, int(ymin - pad))
            x1 = min(w, int(xmax + pad))
            y1 = min(h, int(ymax + pad))
            crop = img[y0:y1, x0:x1]
            log.info(
                "Cropped photo to analyzed face bbox: %dx%d -> %dx%d",
                w, h, crop.shape[1], crop.shape[0],
            )
    ax.imshow(crop)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#202024")
    ax.set_title(title, color="white", fontsize=12, pad=6)
    for spine in ax.spines.values():
        spine.set_color("#404048")


# -----------------------------------------------------------------------------
# Three-up render: photo | baseline | morphed
# -----------------------------------------------------------------------------
def render_compare(
    baseline_obj: Path,
    morphed_obj: Path,
    out_png: Path,
    photo_path: Path | None = None,
    proportions_json: Path | None = None,
) -> Path:
    base_v, base_f = load_obj(baseline_obj)
    base_tris = triangulate(base_f)
    morph_v, morph_f = load_obj(morphed_obj)
    morph_tris = triangulate(morph_f)

    # Pull head pose from analyzer JSON (if available) and rotate both meshes
    # to match the source photograph's viewing angle.
    #
    # PnP head-pose absolute values are unreliable without true camera
    # intrinsics (we pass focal=image_width as a placeholder). Pitch in
    # particular tends to overshoot. We trust the direction of each axis
    # but clamp magnitudes to keep the rendered head from facing the floor.
    yaw = pitch = roll = 0.0
    raw_pose = (0.0, 0.0, 0.0)
    if proportions_json is not None and proportions_json.exists():
        meta = json.loads(proportions_json.read_text(encoding="utf-8"))
        pose = meta.get("head_pose_deg")
        if pose and len(pose) == 3:
            raw_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
            yaw = float(np.clip(raw_pose[0], -30.0, 30.0))
            pitch = float(np.clip(raw_pose[1], -10.0, 10.0))
            roll = float(np.clip(raw_pose[2], -30.0, 30.0))
            log.info(
                "Photo head pose raw=(%.1f, %.1f, %.1f)  clamped=(%.1f, %.1f, %.1f)",
                *raw_pose, yaw, pitch, roll,
            )
    if any(abs(a) > 0.01 for a in (yaw, pitch, roll)):
        # MediaPipe/OpenCV camera frame uses y-down, z-forward; MakeHuman uses
        # y-up, z-forward. Flip the yaw + pitch signs so a head turned subject-
        # left in the photo also turns subject-left in the rendered mesh.
        base_v = rotate_to_pose(base_v, -yaw, -pitch, roll)
        morph_v = rotate_to_pose(morph_v, -yaw, -pitch, roll)

    # Per-vertex drift = displacement magnitude from base to morphed (after the
    # same rotation has been applied to both, so rotation does not show up as
    # drift).
    drift = np.linalg.norm(morph_v - base_v, axis=1)
    log.info(
        "Per-vertex drift after pose rotation: mean=%.4f max=%.4f",
        float(drift.mean()), float(drift.max()),
    )

    # Compute a single shared frame from the UNION of both meshes' head verts
    # so baseline / morphed / heatmap panels render at identical scale.
    base_head_tris = filter_head_region(base_v, base_tris)
    morph_head_tris = filter_head_region(morph_v, morph_tris)
    head_verts = np.concatenate([
        base_v[np.unique(base_head_tris.flatten())],
        morph_v[np.unique(morph_head_tris.flatten())],
    ])
    pad_x = 0.04 * (head_verts[:, 0].max() - head_verts[:, 0].min() + 1e-6)
    pad_y = 0.06 * (head_verts[:, 1].max() - head_verts[:, 1].min() + 1e-6)
    sx = (float(head_verts[:, 0].min() - pad_x), float(head_verts[:, 0].max() + pad_x))
    sy = (float(head_verts[:, 1].min() - pad_y), float(head_verts[:, 1].max() + pad_y))

    n_panels = 4 if photo_path is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 7), facecolor="#15151a")
    pose_str = f"  yaw {yaw:+.0f}°  pitch {pitch:+.0f}°  roll {roll:+.0f}°"
    if photo_path is not None:
        render_photo_panel(axes[0], photo_path, proportions_json,
                           title="Source photograph")
        render_panel(axes[1], base_v, base_tris, "MH baseline" + pose_str,
                     shared_xlim=sx, shared_ylim=sy)
        render_panel(axes[2], morph_v, morph_tris, "Morphed (Loomis-deltas)" + pose_str,
                     skin_rgb=(0.88, 0.72, 0.62),
                     shared_xlim=sx, shared_ylim=sy)
        render_panel(axes[3], morph_v, morph_tris,
                     "Morph drift heatmap  (magma; bright = displaced)",
                     drift_per_vert=drift,
                     shared_xlim=sx, shared_ylim=sy)
    else:
        render_panel(axes[0], base_v, base_tris, "MH baseline" + pose_str,
                     shared_xlim=sx, shared_ylim=sy)
        render_panel(axes[1], morph_v, morph_tris, "Morphed (Loomis-deltas)" + pose_str,
                     skin_rgb=(0.88, 0.72, 0.62),
                     shared_xlim=sx, shared_ylim=sy)
        render_panel(axes[2], morph_v, morph_tris,
                     "Morph drift heatmap",
                     drift_per_vert=drift,
                     shared_xlim=sx, shared_ylim=sy)

    fig.suptitle(
        "Photo -> MakeHuman Face Adaptor : capture comparison",
        color="white",
        fontsize=14,
        y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Wrote render: %s", out_png)
    return out_png


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Side-by-side frontal mesh render: photo | baseline | morphed.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(r"C:\AI\apps\Makehuman\makehuman\data\3dobjs\base.obj"),
    )
    parser.add_argument(
        "--morphed",
        type=Path,
        default=Path(r"C:\AI\apps\DG_Brain\data\morphed.obj"),
    )
    parser.add_argument(
        "--photo",
        type=Path,
        default=None,
        help="Optional source photo. When set, adds a leftmost photo panel.",
    )
    parser.add_argument(
        "--proportions",
        type=Path,
        default=Path(r"C:\AI\apps\DG_Brain\data\face_proportions.json"),
        help="Analyzer JSON; used for face_bbox to crop the photo panel.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\AI\apps\DG_Brain\data\renders\mesh_compare.png"),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        render_compare(
            args.baseline,
            args.morphed,
            args.out,
            photo_path=args.photo,
            proportions_json=args.proportions,
        )
    except FileNotFoundError as exc:
        log.error("Missing input file: %s", exc)
        return 2
    except (RuntimeError, ValueError, IOError) as exc:
        log.error("Render failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
