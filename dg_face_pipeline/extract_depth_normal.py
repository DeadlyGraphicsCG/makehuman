# File: C:\AI\apps\Makehuman\dg_face_pipeline\extract_depth_normal.py
# Repo: makehumancommunity/makehuman  Branch: feat/face-proportion-pipeline
"""
Photo -> depth + normal extraction via Depth Anything v2
=========================================================
Single-photo monocular depth + derived normal-map extraction. Outputs the
raw scientific depth array (.npy), a viewable 8-bit depth PNG, a Sobel-
gradient normal map PNG suitable for use in a tangent-space shader, AND
per-vertex depth samples at each FaceMesh landmark's UV coordinate -- the
last is the key input for multi-source photogrammetry-style blending later.

Model: Depth Anything v2 (Apache-2.0), Base by default. Hugging Face
checkpoints:
    depth-anything/Depth-Anything-V2-Small-hf  (~25 MB params, fastest)
    depth-anything/Depth-Anything-V2-Base-hf   (~98 MB params, default)
    depth-anything/Depth-Anything-V2-Large-hf  (~335 MB params, best)

First run downloads the chosen checkpoint to your HF cache (typically
%USERPROFILE%\.cache\huggingface\hub on Windows).

Run:
    python dg_face_pipeline\\extract_depth_normal.py --subject carolyn_lilipaly_v002

Or override inputs explicitly:
    python dg_face_pipeline\\extract_depth_normal.py ^
        --photo C:\\path\\to\\image.png ^
        --out-dir outputs\\characters\\name\\textures ^
        --slug name ^
        --model-size large

Per-vertex depth output (CSV) is what you'd average across multiple photos
of the same subject to recover true geometric depth -- the "photogrammetry
for moving skin" use case. Each photo's depth is in arbitrary units (Depth
Anything is relative, not metric), so the blending step needs to align
scales across photos first (e.g. via the eye-IPD reference distance).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("extract_depth_normal")

PIPELINE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = PIPELINE_DIR / "examples"
OUTPUTS_DIR = PIPELINE_DIR / "outputs"

MODEL_NAMES = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


# -----------------------------------------------------------------------------
# Depth inference
# -----------------------------------------------------------------------------
def predict_depth(
    photo_path: Path,
    model_size: str = "base",
    device: str | None = None,
) -> Tuple[np.ndarray, Image.Image]:
    """Run Depth Anything v2 on a photo. Returns (depth_hxw_float, source_pil).

    Output depth is RELATIVE (closer = higher value), arbitrary scale. To
    get metric depth you'd need a different model family; for multi-source
    blending the relative output is sufficient because the blending step
    rescales each photo's depth against a known anchor (e.g. IPD).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = MODEL_NAMES[model_size]
    log.info("Loading Depth Anything v2 (%s) on %s ...", model_id, device)

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    img = Image.open(photo_path).convert("RGB")
    log.info("Photo: %s  (%dx%d)", photo_path.name, img.size[0], img.size[1])

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    predicted = outputs.predicted_depth  # (1, h_pred, w_pred), or (1, 1, h, w)
    if predicted.ndim == 3:
        predicted = predicted.unsqueeze(1)
    # Resize to the source image resolution.
    resized = F.interpolate(
        predicted, size=(img.size[1], img.size[0]),
        mode="bicubic", align_corners=False,
    ).squeeze().cpu().numpy()
    log.info(
        "Depth predicted: shape=%s  range=[%.3f, %.3f]  device=%s",
        resized.shape, float(resized.min()), float(resized.max()), device,
    )
    return resized.astype(np.float32), img


# -----------------------------------------------------------------------------
# Normal from depth
# -----------------------------------------------------------------------------
def normals_from_depth(
    depth: np.ndarray,
    z_scale: float = 200.0,
) -> np.ndarray:
    """Sobel-gradient tangent-space normal map from a depth array.

    z_scale controls bump strength; lower = stronger relief. 200 is a
    reasonable starting point for facial detail at 1k-2k photo resolution
    -- adjust per shot if normals look too soft or too aggressive.

    Returns uint8 RGB in OpenGL tangent-space convention:
        R = (Nx + 1) / 2 * 255
        G = (Ny + 1) / 2 * 255
        B = (Nz + 1) / 2 * 255   (Nz is mostly close to 1, so B near 255)
    """
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    nz = np.full_like(dx, z_scale)
    n = np.dstack([-dx, -dy, nz])
    length = np.linalg.norm(n, axis=2, keepdims=True)
    length = np.maximum(length, 1e-6)
    n = n / length
    rgb = ((n * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


# -----------------------------------------------------------------------------
# Depth visualisation
# -----------------------------------------------------------------------------
def depth_to_png(depth: np.ndarray) -> np.ndarray:
    """Normalise float depth to 8-bit greyscale for human-viewable PNG."""
    d = depth.astype(np.float64)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-9:
        return np.zeros_like(d, dtype=np.uint8)
    normalised = ((d - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return normalised


# -----------------------------------------------------------------------------
# Per-vertex depth sampling (the multi-source-blend input)
# -----------------------------------------------------------------------------
def sample_depth_at_uvs(
    depth: np.ndarray, uvs: np.ndarray,
) -> np.ndarray:
    """Sample the depth map at each UV coordinate.

    uvs : (N, 2) array in OBJ convention (u right, v up, origin bottom-left).
    Returns (N,) array of depth values (float, same scale as input depth).
    """
    h, w = depth.shape
    us = np.clip((uvs[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
    # Flip v (OBJ y-up -> image y-down)
    vs = np.clip(((1.0 - uvs[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)
    return depth[vs, us]


def load_obj_uvs(obj_path: Path) -> np.ndarray:
    """Pull the vt entries from an OBJ file. Returns (N, 2) array."""
    uvs = []
    for line in obj_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("vt "):
            parts = line.split()
            uvs.append([float(parts[1]), float(parts[2])])
    return np.array(uvs, dtype=np.float32)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def extract(
    photo_path: Path,
    out_dir: Path,
    slug: str,
    obj_path: Path | None = None,
    model_size: str = "base",
    z_scale: float = 200.0,
) -> dict:
    """Run inference + write outputs. Returns paths to written files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    depth, _img = predict_depth(photo_path, model_size=model_size)
    normal_rgb = normals_from_depth(depth, z_scale=z_scale)
    depth_8bit = depth_to_png(depth)

    written: dict = {}
    # Raw scientific depth for blending downstream.
    p = out_dir / f"{slug}_photo_depth_raw.npy"
    np.save(p, depth)
    written["depth_raw"] = p
    # Viewable depth PNG.
    p = out_dir / f"{slug}_photo_depth.png"
    cv2.imwrite(str(p), depth_8bit)
    written["depth_png"] = p
    # Tangent-space normal map.
    p = out_dir / f"{slug}_photo_normal.png"
    # cv2 writes BGR -> swap for correct OpenGL RGB convention
    cv2.imwrite(str(p), cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR))
    written["normal_png"] = p

    # Per-vertex depth sampling (the multi-source-blend input).
    if obj_path and obj_path.exists():
        uvs = load_obj_uvs(obj_path)
        if len(uvs) > 0:
            per_vert_depth = sample_depth_at_uvs(depth, uvs)
            csv_path = out_dir / f"{slug}_per_vertex_depth.csv"
            with csv_path.open("w", encoding="utf-8") as fh:
                fh.write("vertex_idx,u,v,depth\n")
                for i, ((u, v), d) in enumerate(zip(uvs, per_vert_depth)):
                    fh.write(f"{i},{u:.6f},{v:.6f},{d:.6f}\n")
            written["per_vertex_depth_csv"] = csv_path
            log.info(
                "Sampled depth at %d UV coords  range=[%.3f, %.3f]",
                len(uvs), float(per_vert_depth.min()), float(per_vert_depth.max()),
            )

    # Metadata for reproducibility.
    meta = {
        "source_photo": str(photo_path),
        "slug": slug,
        "model_size": model_size,
        "model_id": MODEL_NAMES[model_size],
        "z_scale": z_scale,
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "depth_mean": float(depth.mean()),
        "shape": list(depth.shape),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "obj_used_for_uv_sampling": str(obj_path) if obj_path else None,
    }
    p = out_dir / f"{slug}_photo_depth_meta.json"
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    written["meta"] = p

    for k, v in written.items():
        log.info("  wrote %-22s %s", k, v)
    return written


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract monocular depth + Sobel normal map from a photo (Depth Anything v2).",
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Character slug. When set, photo defaults to examples/<slug>/ref.png "
             "and outputs land in outputs/characters/<slug>/textures/. "
             "Also auto-discovers <slug>_face_mesh.obj for per-vertex sampling.",
    )
    parser.add_argument("--photo", type=Path, default=None)
    parser.add_argument("--obj", type=Path, default=None,
                        help="Face-mesh OBJ for per-vertex UV depth sampling.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--slug", type=str, default=None,
                        help="Output filename prefix; defaults to --subject value.")
    parser.add_argument(
        "--model-size", choices=("small", "base", "large"), default="base",
        help="Depth Anything v2 model size. base is the default; large is "
             "noticeably sharper on facial detail but ~3x slower and bigger.",
    )
    parser.add_argument(
        "--z-scale", type=float, default=200.0,
        help="Normal-map bump scale (lower = stronger relief). Default 200.",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, str]:
    if args.subject is None and args.photo is None:
        raise SystemExit("Provide --subject or --photo.")
    slug = args.slug or args.subject or args.photo.stem
    if args.photo is not None:
        photo = args.photo
    else:
        # Try examples/<subject>/ref.png; fall back to legacy examples/<subject>_ref.png
        c = EXAMPLES_DIR / args.subject / "ref.png"
        if c.exists():
            photo = c
        else:
            c2 = EXAMPLES_DIR / f"{args.subject}_ref.png"
            if c2.exists():
                photo = c2
            else:
                # Try the analyzer JSON for the original photo path
                pj = OUTPUTS_DIR / "characters" / args.subject / "data" / f"{args.subject}_face_proportions.json"
                if pj.exists():
                    meta = json.loads(pj.read_text(encoding="utf-8"))
                    photo = Path(meta.get("image", ""))
                else:
                    raise SystemExit(f"Could not auto-locate photo for subject '{args.subject}'.")
    if not photo.exists():
        raise SystemExit(f"Photo not found: {photo}")
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.subject is not None:
        out_dir = OUTPUTS_DIR / "characters" / args.subject / "textures"
    else:
        out_dir = OUTPUTS_DIR / "depth"
    obj = args.obj
    if obj is None and args.subject is not None:
        c = OUTPUTS_DIR / "characters" / args.subject / "data" / f"{args.subject}_face_mesh.obj"
        if c.exists():
            obj = c
    return photo, out_dir, obj, slug


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    photo, out_dir, obj, slug = resolve_paths(args)
    try:
        extract(photo, out_dir, slug, obj_path=obj,
                model_size=args.model_size, z_scale=args.z_scale)
    except (RuntimeError, ValueError, IOError) as exc:
        log.error("Depth extraction failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
