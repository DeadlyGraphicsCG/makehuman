"""
Canonical MediaPipe OBJ parsing and UV helpers.

The vendored canonical OBJ stores stable atlas coordinates as `vt` records,
but its face references are `v/vt` pairs where the texture index is not the
same number as the vertex index. The generated learned mesh preserves the
MediaPipe vertex index space, so callers usually want a per-vertex UV array
reordered through those face references.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class ObjFace:
    """A single OBJ polygon with 0-based vertex/UV/normal indices."""

    vertices: list[int]
    uvs: list[int]
    normals: list[int]


@dataclass(frozen=True)
class CanonicalObj:
    """Canonical OBJ payload needed by the face pipeline."""

    path: Path
    vertices: np.ndarray
    uvs: np.ndarray
    faces: list[ObjFace]


def _obj_index(raw: str, count: int, label: str) -> int:
    value = int(raw)
    if value > 0:
        return value - 1
    if value < 0:
        return count + value
    raise ValueError(f"OBJ {label} index 0 is invalid")


def _parse_face_ref(ref: str, vertex_count: int, uv_count: int, normal_count: int) -> tuple[int, int | None, int | None]:
    parts = ref.split("/")
    vertex = _obj_index(parts[0], vertex_count, "vertex")
    uv = None
    normal = None
    if len(parts) > 1 and parts[1]:
        uv = _obj_index(parts[1], uv_count, "texture")
    if len(parts) > 2 and parts[2]:
        normal = _obj_index(parts[2], normal_count, "normal")
    return vertex, uv, normal


def load_canonical_obj(obj_path: Path) -> CanonicalObj:
    """Load canonical OBJ vertices, texture vertices, and polygon refs.

    Faces are preserved as authored, including quads. Indices are converted to
    0-based arrays/lists for Python callers.
    """
    vertices: list[list[float]] = []
    uvs: list[list[float]] = []
    normal_count = 0
    faces: list[ObjFace] = []

    with obj_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "vt":
                uvs.append([float(parts[1]), float(parts[2])])
            elif tag == "vn":
                normal_count += 1
            elif tag == "f":
                face_vertices: list[int] = []
                face_uvs: list[int] = []
                face_normals: list[int] = []
                for ref in parts[1:]:
                    vertex, uv, normal = _parse_face_ref(ref, len(vertices), len(uvs), normal_count)
                    face_vertices.append(vertex)
                    if uv is not None:
                        face_uvs.append(uv)
                    if normal is not None:
                        face_normals.append(normal)
                if len(face_vertices) >= 3:
                    faces.append(ObjFace(face_vertices, face_uvs, face_normals))

    if not vertices:
        raise ValueError(f"OBJ has no vertices: {obj_path}")
    if not faces:
        raise ValueError(f"OBJ has no polygon faces: {obj_path}")

    return CanonicalObj(
        path=obj_path,
        vertices=np.asarray(vertices, dtype=np.float64),
        uvs=np.asarray(uvs, dtype=np.float64) if uvs else np.empty((0, 2), dtype=np.float64),
        faces=faces,
    )


def faces_as_vertex_indices(mesh: CanonicalObj) -> list[list[int]]:
    """Return polygon faces as vertex-index lists, preserving triangle/quads."""
    return [list(face.vertices) for face in mesh.faces]


def face_size_counts(faces: Iterable[Iterable[int]]) -> dict[int, int]:
    """Count face arities for logging and diagnostics."""
    counts: dict[int, int] = {}
    for face in faces:
        size = len(list(face))
        counts[size] = counts.get(size, 0) + 1
    return counts


def canonical_vertex_uvs(mesh: CanonicalObj) -> np.ndarray:
    """Return canonical `vt` coordinates reordered into vertex index space.

    MediaPipe's canonical OBJ uses face refs such as `174/43`; the stable atlas
    UV for vertex 173 is therefore `vt[42]`, not `vt[173]`. This helper follows
    every face corner, verifies that each vertex has a single canonical UV, and
    returns an `(N, 2)` array where row `i` is the base atlas UV for vertex `i`.
    """
    if len(mesh.uvs) == 0:
        raise ValueError(f"Canonical OBJ has no vt records: {mesh.path}")

    per_vertex = np.full((len(mesh.vertices), 2), np.nan, dtype=np.float64)
    assigned_uv: dict[int, int] = {}
    conflicts: list[tuple[int, int, int]] = []

    for face in mesh.faces:
        if not face.uvs:
            continue
        if len(face.uvs) != len(face.vertices):
            raise ValueError(f"Face has partial UV refs in {mesh.path}: {face}")
        for vertex_idx, uv_idx in zip(face.vertices, face.uvs):
            previous = assigned_uv.get(vertex_idx)
            if previous is not None and previous != uv_idx:
                conflicts.append((vertex_idx, previous, uv_idx))
                continue
            assigned_uv[vertex_idx] = uv_idx
            per_vertex[vertex_idx] = mesh.uvs[uv_idx]

    if conflicts:
        sample = ", ".join(f"v{v + 1}: vt{a + 1}/vt{b + 1}" for v, a, b in conflicts[:5])
        raise ValueError(f"Canonical OBJ has split UVs per vertex; cannot preserve vertex index space ({sample})")

    missing = np.flatnonzero(np.isnan(per_vertex[:, 0]))
    if len(missing) and len(mesh.uvs) == len(mesh.vertices):
        per_vertex[missing] = mesh.uvs[missing]
        missing = np.flatnonzero(np.isnan(per_vertex[:, 0]))
    if len(missing):
        sample = ", ".join(str(int(idx) + 1) for idx in missing[:10])
        raise ValueError(f"Canonical OBJ has vertices without UV refs: {sample}")

    return per_vertex


def triangulate_faces(faces: Iterable[Iterable[int]]) -> np.ndarray:
    """Fan-triangulate arbitrary polygon faces without reindexing vertices."""
    tris: list[list[int]] = []
    for raw_face in faces:
        face = list(raw_face)
        if len(face) == 3:
            tris.append(face)
        elif len(face) > 3:
            for i in range(1, len(face) - 1):
                tris.append([face[0], face[i], face[i + 1]])
    return np.asarray(tris, dtype=np.int64)


def fan_triangulate_face(face: Iterable[int]) -> list[list[int]]:
    """Return triangle vertex-index lists for one polygon face."""
    idx = list(face)
    if len(idx) < 3:
        return []
    if len(idx) == 3:
        return [idx]
    return [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
