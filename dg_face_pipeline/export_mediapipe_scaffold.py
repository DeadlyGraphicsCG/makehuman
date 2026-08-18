"""
Export a non-triangulated MediaPipe scaffold from an OBJ.

The learned mesh OBJ is useful for quick rendering, but its triangle faces are
not production topology. This helper keeps the exact vertex and UV order while
dropping all faces, producing a point scaffold that can be used as a retopo or
refit reference without implying the triangle mesh is final.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("export_mediapipe_scaffold")

PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PIPELINE_DIR / "outputs"


def load_obj_points(obj_path: Path) -> tuple[list[list[float]], list[list[float]]]:
    verts: list[list[float]] = []
    uvs: list[list[float]] = []
    with obj_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vt":
                uvs.append([float(parts[1]), float(parts[2])])
    if not verts:
        raise ValueError(f"OBJ has no vertices: {obj_path}")
    return verts, uvs


def write_point_obj(
    src_obj: Path,
    out_obj: Path,
    verts: list[list[float]],
    uvs: list[list[float]],
    point_elements: bool,
) -> Path:
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    with out_obj.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# MediaPipe scaffold exported by DG Face Pipeline\n")
        fh.write(f"# Source OBJ: {src_obj}\n")
        fh.write("# Faces intentionally omitted: this is not triangulated topology.\n")
        fh.write(f"# Verts: {len(verts)}  UVs: {len(uvs)}\n")
        fh.write("g mediapipe_points\n")
        for x, y, z in verts:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            fh.write(f"vt {u:.6f} {v:.6f}\n")
        if point_elements:
            for idx in range(1, len(verts) + 1):
                fh.write(f"p {idx}\n")
    return out_obj


def write_csv(out_csv: Path, verts: list[list[float]], uvs: list[list[float]]) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "obj_index", "x", "y", "z", "u", "v"])
        for idx, xyz in enumerate(verts):
            uv = uvs[idx] if idx < len(uvs) else ["", ""]
            writer.writerow([idx, idx + 1, *xyz, *uv])
    return out_csv


def default_paths(subject: str, raw: bool) -> tuple[Path, Path, Path]:
    data_root = OUTPUTS_DIR / "characters" / subject / "data"
    suffix = "mediapipe_raw" if raw else "face_mesh"
    in_obj = data_root / f"{subject}_{suffix}.obj"
    out_stem = "mediapipe_raw_points" if raw else "face_mesh_points"
    return (
        in_obj,
        data_root / f"{subject}_{out_stem}.obj",
        data_root / f"{subject}_{out_stem}.csv",
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip triangle faces from a MediaPipe OBJ while preserving verts/UVs.",
    )
    parser.add_argument("--subject", default="winona")
    parser.add_argument("--in-obj", type=Path, default=None)
    parser.add_argument("--out-obj", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use <subject>_mediapipe_raw.obj defaults instead of the dense face mesh.",
    )
    parser.add_argument(
        "--no-point-elements",
        action="store_true",
        help="Write v/vt records only. By default OBJ p records are emitted too.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    default_in, default_out_obj, default_out_csv = default_paths(args.subject, args.raw)
    in_obj = args.in_obj or default_in
    out_obj = args.out_obj or default_out_obj
    out_csv = args.out_csv or default_out_csv

    try:
        verts, uvs = load_obj_points(in_obj)
        write_point_obj(in_obj, out_obj, verts, uvs, point_elements=not args.no_point_elements)
        write_csv(out_csv, verts, uvs)
    except (FileNotFoundError, ValueError, IOError) as exc:
        log.error("Scaffold export failed: %s", exc)
        return 1

    log.info("Wrote point OBJ: %s (%d verts, %d UVs, no faces)", out_obj, len(verts), len(uvs))
    log.info("Wrote point CSV: %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
