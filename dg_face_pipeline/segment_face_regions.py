# File: C:\AI\apps\3D\Makehuman\dg_face_pipeline\segment_face_regions.py
# Repo: makehumancommunity/makehuman  Branch: feat/face-proportion-pipeline
"""
Face-region segmentation for MediaPipe canonical meshes
========================================================
Post-processor that takes a learned face-mesh OBJ (the kind produced by
`learned_face_mesh.py`) and splits its faces into shading regions using
MediaPipe's well-known landmark groups:

  - LIPS    : 40 vert set (FACEMESH_LIPS).      Wet, specular, redder pigment.
  - EYES    : 32 verts (LEFT + RIGHT eyelid).   Sclera-adjacent, distinct
                                                shading from skin.
  - BROWS   : 20 verts (LEFT + RIGHT eyebrow).  Darker pigment, often glossy.
  - SKIN    : everything else.                  Default subsurface skin.

A face (triangle/quad/n-gon) is assigned to a region if a MAJORITY of its
verts belong to that region's landmark set. Ties resolve toward the more
specific region (lips > eye > brow > skin).

Output: a new OBJ with `g <region>` group blocks + `usemtl <region>` directives
between them, plus a companion MTL declaring four materials with sensible
default colours. Downstream DCC apps that honour usemtl (Maya/Blender/UE) will
then bind separate shaders per region; ones that ignore it still load the
geometry correctly.

Run:
    python dg_face_pipeline\\segment_face_regions.py --subject winona_v002
    # or with explicit paths:
    python dg_face_pipeline\\segment_face_regions.py ^
        --in-obj  outputs\\characters\\winona_v002\\data\\winona_v002_face_mesh.obj ^
        --out-obj outputs\\characters\\winona_v002\\data\\winona_v002_face_mesh_seg.obj

When --subject is given, writes alongside the input as
`<slug>_face_mesh_seg.obj` and `<slug>_face_mesh_seg.mtl`. Pass --in-place to
overwrite the original OBJ (the texture-referencing .mtl will be replaced).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import mediapipe as mp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("segment_face_regions")

PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PIPELINE_DIR / "outputs"


# -----------------------------------------------------------------------------
# Region definitions
# -----------------------------------------------------------------------------
def build_region_vert_sets() -> Dict[str, Set[int]]:
    """Resolve MediaPipe edge-sets into vertex-index sets per region.

    Order of return matters: regions are checked in this insertion order when
    assigning faces, so more-specific regions (lips) win over more-generic
    ones (skin).
    """
    con = mp.solutions.face_mesh_connections

    def verts(edge_set) -> Set[int]:
        s: Set[int] = set()
        for a, b in edge_set:
            s.add(a)
            s.add(b)
        return s

    return {
        "lips": verts(con.FACEMESH_LIPS),
        "eye":  verts(con.FACEMESH_LEFT_EYE) | verts(con.FACEMESH_RIGHT_EYE),
        "brow": verts(con.FACEMESH_LEFT_EYEBROW) | verts(con.FACEMESH_RIGHT_EYEBROW),
        # skin is implicit -- everything else.
    }


# Region material parameters. Tuned for an Arnold aiStandardSurface-like
# baseline; downstream Maya/Blender shader code can override.
REGION_MATERIALS: Dict[str, Dict[str, str]] = {
    "skin": {
        "Kd": "0.78 0.62 0.55",   # warm flesh tone
        "Ks": "0.15 0.15 0.15",
        "Ns": "10",                # low specular power (matte skin)
        "Ni": "1.40",              # mid IOR
        "illum": "2",
        "comment": "Default skin -- driver for SSS in DCC apps",
    },
    "lips": {
        "Kd": "0.62 0.32 0.32",   # redder, slightly desaturated
        "Ks": "0.35 0.35 0.35",   # more specular than skin
        "Ns": "40",                # tighter highlight (wet)
        "Ni": "1.45",
        "illum": "2",
        "comment": "Lips -- wetter, redder, glossier than skin",
    },
    "eye": {
        "Kd": "0.85 0.83 0.80",   # near-white sclera
        "Ks": "0.50 0.50 0.50",
        "Ns": "80",                # very tight specular
        "Ni": "1.38",              # close to aqueous humor
        "illum": "2",
        "comment": "Eyelid + sclera region (canonical mesh has no eyeball)",
    },
    "brow": {
        "Kd": "0.22 0.16 0.12",   # dark hair-pigment baseline
        "Ks": "0.18 0.18 0.18",
        "Ns": "20",
        "Ni": "1.55",              # melanin
        "illum": "2",
        "comment": "Eyebrow hair -- override per character (auburn/grey/etc.)",
    },
}


# -----------------------------------------------------------------------------
# OBJ I/O
# -----------------------------------------------------------------------------
VERT_RE = re.compile(r"^v\s+")
VT_RE = re.compile(r"^vt\s+")
VN_RE = re.compile(r"^vn\s+")
FACE_RE = re.compile(r"^f\s+")
MTLLIB_RE = re.compile(r"^mtllib\s+")
USEMTL_RE = re.compile(r"^usemtl\s+")
GROUP_RE = re.compile(r"^g\s+")


def parse_obj(obj_path: Path) -> Dict[str, List]:
    """Parse OBJ keeping verts, UVs, normals, faces, and header lines."""
    header: List[str] = []
    v_lines: List[str] = []
    vt_lines: List[str] = []
    vn_lines: List[str] = []
    faces: List[Tuple[str, List[int]]] = []  # (raw_line, [vert_indices_0based])

    in_geom = False
    with obj_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if FACE_RE.match(stripped):
                in_geom = True
                parts = stripped.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append((stripped, idx))
            elif VERT_RE.match(stripped):
                in_geom = True
                v_lines.append(stripped)
            elif VT_RE.match(stripped):
                in_geom = True
                vt_lines.append(stripped)
            elif VN_RE.match(stripped):
                in_geom = True
                vn_lines.append(stripped)
            elif not in_geom:
                # Preserve every line up to first geometry (mtllib, comments,
                # group declarations, usemtl, etc.) so we can rewrite a sane
                # header.
                header.append(stripped)
    return {
        "header": header,
        "v": v_lines,
        "vt": vt_lines,
        "vn": vn_lines,
        "faces": faces,
    }


# -----------------------------------------------------------------------------
# Face-to-region assignment
# -----------------------------------------------------------------------------
def assign_region(face_verts: List[int], region_sets: Dict[str, Set[int]]) -> str:
    """Majority-vote region for one face.

    Counts how many of the face's verts belong to each region's index set;
    picks the region with the highest count, breaking ties in favour of more
    specific regions (lips > eye > brow > skin).
    """
    counts: Dict[str, int] = {}
    for name, vset in region_sets.items():
        counts[name] = sum(1 for v in face_verts if v in vset)
    # Find max non-zero count. Regions are iterated in spec-precedence order.
    best_name = "skin"
    best_count = 0
    for name in region_sets.keys():  # lips, eye, brow
        if counts[name] > best_count:
            best_name = name
            best_count = counts[name]
    # Require majority (more than half of face's verts) to claim a non-skin
    # region. Otherwise the face is mostly skin.
    if best_count <= len(face_verts) // 2:
        return "skin"
    return best_name


# -----------------------------------------------------------------------------
# MTL writer
# -----------------------------------------------------------------------------
def write_segmented_mtl(
    mtl_path: Path,
    texture_filename: str | None,
) -> None:
    """Emit a Wavefront MTL with one material per region.

    All four materials reference the same base-colour texture (the photo);
    region-specific look is achieved via the diffuse tint + specular
    parameters. Downstream Arnold/Maya code can swap in per-region textures
    later.
    """
    with mtl_path.open("w", encoding="utf-8") as fh:
        fh.write("# Segmented face-region MTL\n")
        fh.write(f"# Generated by {Path(__file__).name}\n\n")
        for name, params in REGION_MATERIALS.items():
            fh.write(f"# {params['comment']}\n")
            fh.write(f"newmtl {name}\n")
            fh.write(f"Kd {params['Kd']}\n")
            fh.write(f"Ks {params['Ks']}\n")
            fh.write(f"Ns {params['Ns']}\n")
            fh.write(f"Ni {params['Ni']}\n")
            fh.write(f"illum {params['illum']}\n")
            if texture_filename:
                fh.write(f"map_Kd {texture_filename}\n")
            fh.write("\n")
    log.info("Wrote segmented MTL: %s", mtl_path)


# -----------------------------------------------------------------------------
# Segmented OBJ writer
# -----------------------------------------------------------------------------
def write_segmented_obj(
    out_obj: Path,
    parsed: Dict[str, List],
    mtl_filename: str,
    region_assignments: List[str],
) -> Dict[str, int]:
    """Write OBJ grouped by region. Returns per-region face counts."""
    faces = parsed["faces"]
    # Bucket face indices by region.
    by_region: Dict[str, List[int]] = {name: [] for name in REGION_MATERIALS.keys()}
    for i, region in enumerate(region_assignments):
        by_region[region].append(i)

    out_obj.parent.mkdir(parents=True, exist_ok=True)
    with out_obj.open("w", encoding="utf-8") as fh:
        fh.write("# Segmented face mesh -- per-region groups + materials\n")
        fh.write(f"# Generated by {Path(__file__).name}\n")
        fh.write(f"# Regions: {', '.join(REGION_MATERIALS.keys())}\n\n")
        fh.write(f"mtllib {mtl_filename}\n\n")

        # Geometry (verts, UVs, normals) is identical to source.
        for v in parsed["v"]:
            fh.write(v + "\n")
        for vt in parsed["vt"]:
            fh.write(vt + "\n")
        for vn in parsed["vn"]:
            fh.write(vn + "\n")

        # Faces emitted in region-group blocks.
        for region in REGION_MATERIALS.keys():
            idxs = by_region[region]
            if not idxs:
                continue
            fh.write(f"\ng {region}\n")
            fh.write(f"usemtl {region}\n")
            for i in idxs:
                raw_face = faces[i][0]
                fh.write(raw_face + "\n")
    log.info("Wrote segmented OBJ: %s", out_obj)
    return {name: len(by_region[name]) for name in REGION_MATERIALS.keys()}


# -----------------------------------------------------------------------------
# Texture discovery
# -----------------------------------------------------------------------------
def find_existing_texture(in_obj: Path) -> str | None:
    """Find the existing texture file referenced by the input OBJ's MTL.

    Reads `mtllib` and `map_Kd` directives to recover the same texture the
    learned-mesh pipeline already wrote, so the segmented OBJ keeps using it.
    """
    for line in in_obj.read_text(encoding="utf-8").splitlines():
        if MTLLIB_RE.match(line):
            mtl_name = line.split(maxsplit=1)[1].strip()
            mtl_path = in_obj.parent / mtl_name
            if mtl_path.exists():
                for ml in mtl_path.read_text(encoding="utf-8").splitlines():
                    if ml.strip().lower().startswith("map_kd"):
                        return ml.split(maxsplit=1)[1].strip()
            break
    return None


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def segment(in_obj: Path, out_obj: Path) -> Dict[str, int]:
    parsed = parse_obj(in_obj)
    log.info(
        "Parsed %s: %d v, %d vt, %d vn, %d faces",
        in_obj.name, len(parsed["v"]), len(parsed["vt"]),
        len(parsed["vn"]), len(parsed["faces"]),
    )
    region_sets = build_region_vert_sets()
    log.info(
        "Region vertex counts: %s",
        {n: len(s) for n, s in region_sets.items()},
    )
    region_assignments = [
        assign_region(face_verts, region_sets) for _, face_verts in parsed["faces"]
    ]
    texture = find_existing_texture(in_obj)
    if texture:
        log.info("Reusing existing texture: %s", texture)
    mtl_path = out_obj.with_suffix(".mtl")
    write_segmented_mtl(mtl_path, texture)
    counts = write_segmented_obj(out_obj, parsed, mtl_path.name, region_assignments)
    return counts


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a MediaPipe face OBJ into lip/eye/brow/skin regions.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Character slug; resolves --in-obj and --out-obj under outputs/characters/<slug>/data/.",
    )
    parser.add_argument(
        "--in-obj", type=Path, default=None,
        help="Source face-mesh OBJ (overrides --subject discovery).",
    )
    parser.add_argument(
        "--out-obj", type=Path, default=None,
        help="Where to write the segmented OBJ. Defaults to <in_stem>_seg.obj.",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the input OBJ + its MTL. Use with care.",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.in_obj is None and args.subject is None:
        raise SystemExit("Either --subject or --in-obj is required.")
    if args.in_obj is None:
        slug = args.subject
        in_obj = OUTPUTS_DIR / "characters" / slug / "data" / f"{slug}_face_mesh.obj"
    else:
        in_obj = args.in_obj
    if not in_obj.exists():
        raise SystemExit(f"Input OBJ not found: {in_obj}")
    if args.in_place:
        out_obj = in_obj
    elif args.out_obj is not None:
        out_obj = args.out_obj
    else:
        out_obj = in_obj.with_name(in_obj.stem + "_seg.obj")
    return in_obj, out_obj


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    in_obj, out_obj = resolve_paths(args)
    try:
        counts = segment(in_obj, out_obj)
    except (RuntimeError, ValueError, IOError) as exc:
        log.error("Segmentation failed: %s", exc)
        return 1
    log.info(
        "Region face counts -- %s",
        "  ".join(f"{name}={n}" for name, n in counts.items()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
