"""
Project a source portrait into the canonical MediaPipe UV atlas.

Inputs are a source photo, runtime landmark UVs for that photo, and a canonical
OBJ whose `vt` records define the stable atlas. The output albedo is a PNG in
canonical UV space plus a diagnostic occupancy mask showing which atlas pixels
were covered by canonical faces.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from canonical_uv import canonical_vertex_uvs, fan_triangulate_face, faces_as_vertex_indices, load_canonical_obj

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("project_canonical_texture")

PIPELINE_DIR = Path(__file__).resolve().parent
CANONICAL_OBJ = PIPELINE_DIR / "canonical_face_model.obj"
EXAMPLES_DIR = PIPELINE_DIR / "examples"
OUTPUTS_DIR = PIPELINE_DIR / "outputs"


def extract_runtime_uvs(photo: Path) -> np.ndarray:
    """Run MediaPipe FaceMesh and return landmark UVs in OBJ convention."""
    import mediapipe as mp

    img_bgr = cv2.imread(str(photo))
    if img_bgr is None:
        raise IOError(f"OpenCV could not decode: {photo}")
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
    return np.asarray([[lm.x, 1.0 - lm.y] for lm in lms], dtype=np.float64)


def _coerce_uv_array(data: object) -> np.ndarray:
    if isinstance(data, dict):
        for key in ("uvs", "landmark_uvs", "landmarks_uv", "runtime_uvs"):
            if key in data:
                return _coerce_uv_array(data[key])
        rows = []
        for key in sorted(data, key=lambda x: int(x) if str(x).isdigit() else str(x)):
            value = data[key]
            if isinstance(value, dict):
                rows.append([value["u"], value["v"]])
            else:
                rows.append(value[:2])
        return np.asarray(rows, dtype=np.float64)
    return np.asarray(data, dtype=np.float64)


def load_runtime_uvs(path: Path, convention: str = "obj") -> np.ndarray:
    """Load runtime UVs from JSON, CSV, NPY, or NPZ."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        arr = _coerce_uv_array(json.loads(path.read_text(encoding="utf-8")))
    elif suffix == ".npy":
        arr = np.load(str(path))
    elif suffix == ".npz":
        payload = np.load(str(path))
        key = "uvs" if "uvs" in payload.files else payload.files[0]
        arr = payload[key]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            sample = fh.read(2048)
            fh.seek(0)
            has_header = csv.Sniffer().has_header(sample)
            if has_header:
                reader = csv.DictReader(fh)
                rows = [[float(row["u"]), float(row["v"])] for row in reader]
                arr = np.asarray(rows, dtype=np.float64)
            else:
                arr = np.loadtxt(fh, delimiter=",", dtype=np.float64)
    else:
        raise ValueError(f"Unsupported runtime UV file type: {path}")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Runtime UVs must be an Nx2 array: {path}")
    arr = arr[:, :2].copy()
    if convention == "image":
        arr[:, 1] = 1.0 - arr[:, 1]
    elif convention != "obj":
        raise ValueError(f"Unknown UV convention: {convention}")
    return arr


def _uv_to_pixels(uvs: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.empty_like(uvs, dtype=np.float64)
    out[:, 0] = uvs[:, 0] * (width - 1)
    out[:, 1] = (1.0 - uvs[:, 1]) * (height - 1)
    return out


def _sample_bilinear_bgr(image: np.ndarray, src_uvs: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.clip(src_uvs[:, 0] * (w - 1), 0.0, float(w - 1))
    y = np.clip((1.0 - src_uvs[:, 1]) * (h - 1), 0.0, float(h - 1))

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]

    c00 = image[y0, x0].astype(np.float64)
    c10 = image[y0, x1].astype(np.float64)
    c01 = image[y1, x0].astype(np.float64)
    c11 = image[y1, x1].astype(np.float64)
    top = c00 * (1.0 - wx) + c10 * wx
    bottom = c01 * (1.0 - wx) + c11 * wx
    return top * (1.0 - wy) + bottom * wy


def _raster_project_triangle(
    photo_bgr: np.ndarray,
    out_bgr: np.ndarray,
    occupancy: np.ndarray,
    dst_uv: np.ndarray,
    src_uv: np.ndarray,
) -> None:
    height, width = occupancy.shape
    tri_px = _uv_to_pixels(dst_uv, width, height)
    min_x = max(int(np.floor(np.min(tri_px[:, 0]))), 0)
    max_x = min(int(np.ceil(np.max(tri_px[:, 0]))), width - 1)
    min_y = max(int(np.floor(np.min(tri_px[:, 1]))), 0)
    max_y = min(int(np.ceil(np.max(tri_px[:, 1]))), height - 1)
    if min_x > max_x or min_y > max_y:
        return

    x0, y0 = tri_px[0]
    x1, y1 = tri_px[1]
    x2, y2 = tri_px[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denom)) < 1e-10:
        return

    xs, ys = np.meshgrid(
        np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5,
        np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5,
    )
    w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
    w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    if not np.any(inside):
        return

    bary = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)
    sample_uvs = bary @ src_uv
    sampled = _sample_bilinear_bgr(photo_bgr, sample_uvs)
    yy, xx = np.nonzero(inside)
    target_y = yy + min_y
    target_x = xx + min_x
    out_bgr[target_y, target_x] = np.clip(sampled, 0, 255).astype(np.uint8)
    occupancy[target_y, target_x] = 255


def _bleed_empty_pixels(out_bgr: np.ndarray, occupancy: np.ndarray, passes: int) -> None:
    if passes <= 0:
        return
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(passes):
        grown = cv2.dilate(occupancy, kernel, iterations=1)
        fill = (occupancy == 0) & (grown > 0)
        if not np.any(fill):
            break
        dilated_color = cv2.dilate(out_bgr, kernel, iterations=1)
        out_bgr[fill] = dilated_color[fill]
        occupancy[fill] = grown[fill]


def project_texture_from_arrays(
    photo: Path,
    runtime_uvs: np.ndarray,
    canonical_obj: Path,
    out_albedo: Path,
    out_mask: Path,
    size: int = 1024,
    bleed: int = 4,
) -> tuple[Path, Path]:
    """Bake photo pixels from runtime landmark UVs into canonical UV space."""
    if size < 16:
        raise ValueError("Texture size must be at least 16 pixels")

    photo_bgr = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    if photo_bgr is None:
        raise IOError(f"OpenCV could not decode: {photo}")

    mesh = load_canonical_obj(canonical_obj)
    canonical_uvs = canonical_vertex_uvs(mesh)
    runtime_uvs = np.asarray(runtime_uvs, dtype=np.float64)
    if runtime_uvs.ndim != 2 or runtime_uvs.shape[1] < 2:
        raise ValueError("Runtime UVs must be an Nx2 array")

    vertex_count = len(mesh.vertices)
    if len(runtime_uvs) < vertex_count:
        raise ValueError(
            f"Runtime UVs have {len(runtime_uvs)} rows but canonical OBJ needs {vertex_count}"
        )
    runtime_uvs = runtime_uvs[:vertex_count, :2]

    out_bgr = np.zeros((size, size, 3), dtype=np.uint8)
    occupancy = np.zeros((size, size), dtype=np.uint8)

    tri_count = 0
    for face in faces_as_vertex_indices(mesh):
        for tri in fan_triangulate_face(face):
            tri_idx = np.asarray(tri, dtype=np.int64)
            _raster_project_triangle(
                photo_bgr=photo_bgr,
                out_bgr=out_bgr,
                occupancy=occupancy,
                dst_uv=canonical_uvs[tri_idx],
                src_uv=runtime_uvs[tri_idx],
            )
            tri_count += 1

    raw_occupancy = occupancy.copy()
    _bleed_empty_pixels(out_bgr, occupancy, bleed)

    out_albedo.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_albedo), out_bgr):
        raise IOError(f"OpenCV could not write: {out_albedo}")
    if not cv2.imwrite(str(out_mask), raw_occupancy):
        raise IOError(f"OpenCV could not write: {out_mask}")

    covered = int(np.count_nonzero(raw_occupancy))
    log.info(
        "Projected %d triangles into %dx%d canonical atlas (%d occupied pixels, %.2f%%)",
        tri_count, size, size, covered, covered * 100.0 / float(size * size),
    )
    return out_albedo, out_mask


def project_texture(
    photo: Path,
    canonical_obj: Path,
    out_albedo: Path,
    out_mask: Path,
    runtime_uvs_path: Path | None = None,
    runtime_uv_convention: str = "obj",
    size: int = 1024,
    bleed: int = 4,
) -> tuple[Path, Path]:
    """Load or extract runtime UVs, then bake a canonical atlas texture."""
    runtime_uvs = (
        load_runtime_uvs(runtime_uvs_path, runtime_uv_convention)
        if runtime_uvs_path is not None
        else extract_runtime_uvs(photo)
    )
    return project_texture_from_arrays(photo, runtime_uvs, canonical_obj, out_albedo, out_mask, size, bleed)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bake a source photo into the canonical MediaPipe UV atlas.")
    parser.add_argument("--photo", type=Path, default=EXAMPLES_DIR / "winona_ref.png")
    parser.add_argument("--canonical", type=Path, default=CANONICAL_OBJ)
    parser.add_argument("--runtime-uvs", type=Path, default=None)
    parser.add_argument(
        "--runtime-uv-convention",
        choices=["obj", "image"],
        default="obj",
        help="'obj' means v=bottom-up. 'image' means v/y=top-down and will be flipped.",
    )
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "textures" / "canonical_albedo.png")
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--bleed", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    mask = args.mask or args.out.with_name(f"{args.out.stem}_occupancy.png")
    try:
        albedo, occupancy = project_texture(
            photo=args.photo,
            canonical_obj=args.canonical,
            out_albedo=args.out,
            out_mask=mask,
            runtime_uvs_path=args.runtime_uvs,
            runtime_uv_convention=args.runtime_uv_convention,
            size=args.size,
            bleed=args.bleed,
        )
    except FileNotFoundError as exc:
        log.error("Missing input: %s", exc)
        return 2
    except (RuntimeError, ValueError, IOError) as exc:
        log.error("Projection failed: %s", exc)
        return 1
    log.info("Wrote canonical albedo: %s", albedo)
    log.info("Wrote UV occupancy mask: %s", occupancy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
