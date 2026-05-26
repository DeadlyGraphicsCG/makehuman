# File: C:\AI\apps\Makehuman\dg_face_pipeline\step_c_apply_modifiers.py
# Repo: makehumancommunity/makehuman  Branch: feat/face-proportion-pipeline
# Orchestrator: DG_Brain
"""
Photo -> MakeHuman Face Adaptor : Step C
=========================================
Consumes the JSON proportion delta payload produced by
`face_proportion_analyzer.py` (Step A+B) and applies the suggested
modifier values to the MakeHuman base mesh by linearly blending the
corresponding `.target` vertex-displacement files.

Output: a morphed OBJ at `--out_obj`, with the base mesh's face indices
preserved so any downstream renderer/exporter can use it unchanged.

Algorithm:
  1. Parse data/modifiers/modeling_modifiers.json -> modifier index
     keyed by "<group>/<target>" -> (min_keyword, max_keyword)
  2. Parse base.obj vertex list (19158 verts on stock MakeHuman master).
  3. For each modifier_targets entry in the analyzer JSON:
       - decode "<group>/<target>-<min_kw>|<max_kw>" + signed value v
       - pick <target>-<max_kw>.target if v > 0, else <target>-<min_kw>.target
       - load delta rows (vertex_idx dx dy dz) and add weight*delta
         (weight = abs(v)) to the working vertex array
  4. Stream-write a new OBJ that copies every non-vertex line verbatim
     from base.obj and replaces vertex lines with the morphed positions.

Run:
    python C:\\AI\\apps\\Makehuman\\dg_face_pipeline\\step_c_apply_modifiers.py ^
        --proportions C:\\AI\\apps\\DG_Brain\\data\\face_proportions.json ^
        --out_obj     C:\\AI\\apps\\DG_Brain\\data\\morphed.obj
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("step_c_apply_modifiers")

PIPELINE_DIR = Path(__file__).resolve().parent
MH_ROOT = PIPELINE_DIR.parent / "makehuman"
BASE_OBJ_DEFAULT = MH_ROOT / "data" / "3dobjs" / "base.obj"
TARGETS_ROOT = MH_ROOT / "data" / "targets"
MODIFIERS_JSON = MH_ROOT / "data" / "modifiers" / "modeling_modifiers.json"
OUTPUTS_DIR = PIPELINE_DIR / "outputs"

VERT_LINE_RE = re.compile(r"^v\s+")


# -----------------------------------------------------------------------------
# Modifier index
# -----------------------------------------------------------------------------
def build_modifier_index(modifiers_json: Path) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Map (group, target_base) -> (min_keyword, max_keyword)."""
    raw = json.loads(modifiers_json.read_text(encoding="utf-8"))
    idx: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for group_entry in raw:
        group = group_entry["group"]
        for mod in group_entry.get("modifiers", []):
            tgt = mod.get("target")
            mn = mod.get("min")
            mx = mod.get("max")
            if tgt and mn and mx:
                # Multiple entries can share the same target name with
                # different min/max axes (e.g. nose-trans down/up vs
                # in/out vs backward/forward). Index by the (min, max)
                # pair as well so the analyzer's modifier path resolves
                # to exactly one axis.
                idx[(group, f"{tgt}-{mn}|{mx}")] = (mn, mx)
    return idx


# -----------------------------------------------------------------------------
# OBJ I/O
# -----------------------------------------------------------------------------
def load_obj_vertices(obj_path: Path) -> np.ndarray:
    """Read v lines from an OBJ into an (N,3) float64 array."""
    verts: List[List[float]] = []
    with obj_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if VERT_LINE_RE.match(line):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    arr = np.asarray(verts, dtype=np.float64)
    log.info("Loaded %d vertices from %s", len(arr), obj_path.name)
    return arr


def write_obj_with_morphed_vertices(
    src_obj: Path,
    morphed: np.ndarray,
    out_path: Path,
) -> None:
    """Copy every non-`v` line verbatim, replace `v` lines with morphed coords."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vi = 0
    written_v = 0
    with src_obj.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if VERT_LINE_RE.match(line):
                x, y, z = morphed[vi]
                dst.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                vi += 1
                written_v += 1
            else:
                dst.write(line)
    if written_v != len(morphed):
        log.warning("Vertex line count mismatch: wrote %d, expected %d", written_v, len(morphed))
    log.info("Wrote morphed OBJ: %s (%d verts)", out_path, written_v)


# -----------------------------------------------------------------------------
# Target file parser + applier
# -----------------------------------------------------------------------------
def load_target_deltas(target_path: Path) -> np.ndarray:
    """Return (N,4) array: [vertex_idx, dx, dy, dz]."""
    rows: List[List[float]] = []
    with target_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 4), dtype=np.float64)


def resolve_target_path(
    modifier_path: str,
    value: float,
    mod_index: Dict[Tuple[str, str], Tuple[str, str]],
) -> Tuple[Path | None, float]:
    """
    Decode "<group>/<target>-<min_kw>|<max_kw>" + signed value into a concrete
    .target file path and a positive weight. Returns (None, 0.0) if the
    modifier cannot be resolved.
    """
    if "/" not in modifier_path or "|" not in modifier_path:
        return None, 0.0
    group, full = modifier_path.split("/", 1)
    key = (group, full)
    if key not in mod_index:
        return None, 0.0
    min_kw, max_kw = mod_index[key]
    # Strip the trailing "-<min>|<max>" off `full` to get the bare target base.
    suffix = f"-{min_kw}|{max_kw}"
    if not full.endswith(suffix):
        return None, 0.0
    target_base = full[: -len(suffix)]
    chosen_kw = max_kw if value > 0 else min_kw
    weight = abs(value)
    target_file = TARGETS_ROOT / group / f"{target_base}-{chosen_kw}.target"
    if not target_file.exists():
        log.warning("Target file missing on disk: %s", target_file)
        return None, 0.0
    return target_file, weight


def apply_one_target(verts: np.ndarray, deltas: np.ndarray, weight: float) -> int:
    """In-place: verts[idx] += weight * delta. Returns count of verts touched."""
    if deltas.size == 0:
        return 0
    idx = deltas[:, 0].astype(np.int64)
    valid = (idx >= 0) & (idx < len(verts))
    idx = idx[valid]
    disp = deltas[valid, 1:4]
    verts[idx] += weight * disp
    return int(len(idx))


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def apply_proportions(
    proportions_json: Path,
    base_obj: Path,
    out_obj: Path,
    amplify: float = 1.0,
) -> Path:
    payload = json.loads(proportions_json.read_text(encoding="utf-8"))
    mh_targets = payload.get("modifier_targets", [])
    log.info("Loaded %d modifier targets from %s", len(mh_targets), proportions_json.name)

    log.info("Building modifier index from %s", MODIFIERS_JSON.name)
    mod_index = build_modifier_index(MODIFIERS_JSON)
    log.info("Modifier index has %d axis entries", len(mod_index))

    log.info("Loading base mesh: %s", base_obj)
    base_verts = load_obj_vertices(base_obj)
    morphed = base_verts.copy()

    applied = 0
    skipped = 0
    for entry in mh_targets:
        modifier_path = entry.get("modifier", "")
        value = float(entry.get("value", 0.0)) * amplify
        if value == 0:
            skipped += 1
            continue
        tgt_file, weight = resolve_target_path(modifier_path, value, mod_index)
        if tgt_file is None:
            log.warning("Unresolved modifier (skipped): %s @ %.3f", modifier_path, value)
            skipped += 1
            continue
        deltas = load_target_deltas(tgt_file)
        touched = apply_one_target(morphed, deltas, weight)
        applied += 1
        log.info(
            "Applied %-55s weight=%.3f -> %d verts displaced",
            tgt_file.name, weight, touched,
        )

    drift = np.linalg.norm(morphed - base_verts, axis=1)
    log.info(
        "Applied=%d Skipped=%d  vertex drift: mean=%.5f  max=%.5f",
        applied, skipped, float(drift.mean()), float(drift.max()),
    )
    write_obj_with_morphed_vertices(base_obj, morphed, out_obj)
    return out_obj


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply face-proportion JSON deltas to MakeHuman base mesh.",
    )
    parser.add_argument(
        "--proportions",
        type=Path,
        default=OUTPUTS_DIR / "data" / "face_proportions.json",
    )
    parser.add_argument("--base_obj", type=Path, default=BASE_OBJ_DEFAULT)
    parser.add_argument(
        "--out_obj",
        type=Path,
        default=OUTPUTS_DIR / "data" / "morphed.obj",
    )
    parser.add_argument(
        "--amplify",
        type=float,
        default=1.0,
        help="Multiply all modifier values by this factor for stronger morph.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        apply_proportions(args.proportions, args.base_obj, args.out_obj, args.amplify)
    except FileNotFoundError as exc:
        log.error("Input missing: %s", exc)
        return 2
    except (RuntimeError, ValueError, IOError) as exc:
        log.error("Step C failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
