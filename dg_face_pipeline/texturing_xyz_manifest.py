"""
Inventory an external TexturingXYZ package without copying source assets.

The manifest is metadata only: absolute external paths, relative paths, sizes,
suffixes, and lightweight guesses that help downstream look-dev scripts decide
which maps are color, scalar, UDIM, reference, or geometry inputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("texturing_xyz_manifest")

PIPELINE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PIPELINE_DIR / "texturing_xyz_manifest_config.json"
UDIM_RE = re.compile(r"\.(1\d{3})(?=\.)")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def pipeline_relative(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PIPELINE_DIR / path


def normalized_parts(path: Path) -> tuple[str, ...]:
    return tuple(part.lower() for part in path.parts)


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def should_skip(path: Path, root: Path, config: dict[str, Any]) -> bool:
    rel_parts = path.relative_to(root).parts
    skip_dirs = {name.lower() for name in config.get("skip_dir_names", [])}
    skip_files = {name.lower() for name in config.get("skip_file_names", [])}
    return (
        any(part.lower() in skip_dirs for part in rel_parts[:-1])
        or path.name.lower() in skip_files
    )


def category_prefixes(config: dict[str, Any]) -> dict[str, tuple[tuple[str, ...], ...]]:
    prefixes: dict[str, tuple[tuple[str, ...], ...]] = {}
    for category, paths in config.get("categories", {}).items():
        prefixes[category] = tuple(tuple(Path(path).parts) for path in paths)
    return prefixes


def rel_startswith(rel_path: Path, prefix: tuple[str, ...]) -> bool:
    rel_parts = normalized_parts(rel_path)
    prefix_parts = tuple(part.lower() for part in prefix)
    return rel_parts[: len(prefix_parts)] == prefix_parts


def classify(path: Path, root: Path, config: dict[str, Any]) -> str:
    rel = path.relative_to(root)
    prefixes = category_prefixes(config)

    # More specific buckets first so ID masks and Mudbox support files do not
    # get absorbed by their broader parent folders.
    for category in (
        "id_masks",
        "mudbox_support_files",
        "eye_geometry",
        "eye_maps",
        "head_geometry",
        "head_maps",
        "calibration_reference_images",
        "groom_assets",
        "zbrush_tools",
    ):
        if any(rel_startswith(rel, prefix) for prefix in prefixes.get(category, ())):
            return category
    return "other"


def guess_udim(path: Path) -> str | None:
    match = UDIM_RE.search(path.name)
    return match.group(1) if match else None


def compact_name(path: Path) -> str:
    return path.name.lower().replace("-", "_").replace(" ", "_")


def guess_map_type(path: Path, category: str) -> str | None:
    name = compact_name(path)
    if path.suffix.lower() in {".obj", ".mud", ".ztl"}:
        return None
    if "albedo" in name:
        return "albedo"
    if "diffuse" in name:
        return "diffuse"
    if "dispcalibrated" in name:
        return "displacement_calibrated"
    if "dispmultichannel" in name:
        return "displacement_multichannel"
    if "normal" in name:
        return "normal"
    if "cavity" in name:
        return "cavity"
    if "utility" in name:
        return "utility"
    if "id_mask" in name or name.startswith("xyz_id_") or "_id_" in name:
        return "id_mask"
    if "mask" in name:
        return "mask"
    if "hdri" in name:
        return "hdri"
    if "lighting" in name:
        return "lighting_reference"
    if "calibration" in name or "macbeth" in name:
        return "calibration_reference"
    if category.endswith("reference_images"):
        return "reference_image"
    return "texture" if path.suffix.lower() in {".exr", ".tif", ".tiff", ".png", ".tx"} else None


def guess_color_space(path: Path, map_type: str | None) -> str:
    name = compact_name(path)
    if "raw" in name:
        return "Raw"
    if "lin_srgb" in name:
        return "linear_srgb"
    if "acescg" in name:
        return "ACEScg"
    if "srgb" in name:
        return "sRGB"
    if map_type in {
        "cavity",
        "displacement_calibrated",
        "displacement_multichannel",
        "id_mask",
        "mask",
        "normal",
        "utility",
    }:
        return "Raw"
    if path.suffix.lower() == ".exr":
        return "linear"
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        return "sRGB"
    return "unknown"


def guess_side(path: Path) -> str | None:
    name = compact_name(path)
    if "_left_" in name or name.endswith("_left_geo.obj"):
        return "left"
    if "_right_" in name or name.endswith("_right_geo.obj"):
        return "right"
    if "frontview" in name:
        return "front"
    if "sideview" in name:
        return "side"
    return None


def iter_source_files(root: Path, config: dict[str, Any]) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_file() and not should_skip(path, root, config):
            yield path


def file_entry(path: Path, root: Path, category: str) -> dict[str, Any]:
    stat = path.stat()
    map_type = guess_map_type(path, category)
    return {
        "path": str(path.resolve()),
        "relative_path": relative_posix(path, root),
        "name": path.name,
        "size_bytes": stat.st_size,
        "size_mib": round(stat.st_size / (1024 * 1024), 3),
        "suffixes": list(path.suffixes),
        "extension": path.suffix.lower(),
        "map_type_guess": map_type,
        "udim_tile_guess": guess_udim(path),
        "color_space_hint": guess_color_space(path, map_type),
        "side_hint": guess_side(path),
    }


def build_manifest(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"TexturingXYZ root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"TexturingXYZ root is not a directory: {root}")

    inventory: dict[str, list[dict[str, Any]]] = {
        category: [] for category in config.get("categories", {})
    }
    inventory.setdefault("other", [])

    for path in iter_source_files(root, config):
        category = classify(path, root, config)
        inventory.setdefault(category, []).append(file_entry(path, root, category))

    counts = {category: len(entries) for category, entries in inventory.items()}
    return {
        "schema_version": config.get("schema_version", 1),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "root": str(root),
            "package_name": root.name,
        },
        "policy": {
            "external_only": True,
            "note": (
                "This manifest records metadata and external file paths only. "
                "Do not copy TexturingXYZ source assets into this repository."
            ),
        },
        "counts": counts,
        "inventory": inventory,
    }


def write_manifest(manifest: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Write a metadata-only manifest for an external TexturingXYZ package."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(config["default_root"]),
        help="External TexturingXYZ package root to scan.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=pipeline_relative(config["default_out"]),
        help="Manifest JSON output path. Defaults under dg_face_pipeline/outputs/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    args = parse_args(argv)
    try:
        manifest = build_manifest(args.root, config)
        out = write_manifest(manifest, args.out)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        log.error("TexturingXYZ manifest failed: %s", exc)
        return 1

    total = sum(manifest["counts"].values())
    log.info("Wrote TexturingXYZ manifest: %s", out)
    log.info("Inventoried %d files from external root: %s", total, manifest["source"]["root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
