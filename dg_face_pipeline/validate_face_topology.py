# File: C:\AI\apps\Makehuman\dg_face_pipeline\validate_face_topology.py
# Repo: makehumancommunity/makehuman  Branch: feat/face-proportion-pipeline
"""
OBJ topology validator for DG face-pipeline meshes.

The parser is intentionally small and dependency-free so it can run in Maya
handoff scripts, CI checks, and quick artist-side batch checks. It reports
validation issues without failing the process unless --strict is requested.

Run:
    python dg_face_pipeline\\validate_face_topology.py ^
        dg_face_pipeline\\canonical_face_model_v002.obj ^
        --json-out topology_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class ObjParseError(ValueError):
    """Raised when an OBJ file contains malformed syntax."""


@dataclass(frozen=True)
class FaceRef:
    """One parsed OBJ face corner.

    Indices are stored as OBJ's original signed, 1-based values and as resolved
    zero-based values when the reference is valid for the records seen so far.
    """

    vertex: int
    uv: int | None = None
    normal: int | None = None
    vertex_index: int | None = None
    uv_index: int | None = None
    normal_index: int | None = None
    token: str = ""


@dataclass(frozen=True)
class Face:
    line_number: int
    refs: tuple[FaceRef, ...]


@dataclass
class ObjMesh:
    path: Path
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    uvs: list[tuple[float, ...]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)


@dataclass
class InvalidReference:
    line: int
    face: int
    corner: int
    token: str
    kind: str
    value: int
    available: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "face": self.face,
            "corner": self.corner,
            "token": self.token,
            "kind": self.kind,
            "value": self.value,
            "available": self.available,
        }


@dataclass
class DegenerateFace:
    line: int
    face: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"line": self.line, "face": self.face, "reason": self.reason}


@dataclass
class TopologyReport:
    path: str
    vertex_count: int
    uv_count: int
    normal_count: int
    face_count: int
    face_arity_distribution: dict[str, int]
    invalid_vertex_refs: list[InvalidReference]
    invalid_uv_refs: list[InvalidReference]
    invalid_normal_refs: list[InvalidReference]
    degenerate_faces: list[DegenerateFace]
    boundary_edge_count: int
    nonmanifold_edge_count: int
    connected_component_count: int
    bbox_min: list[float] | None
    bbox_max: list[float] | None

    @property
    def issue_count(self) -> int:
        # Boundary edges are diagnostic rather than inherently invalid: the
        # source MediaPipe face masks are expected to be open before the
        # optional closure step. Strict mode should catch broken topology, not
        # reject a deliberately open canonical mesh.
        return (
            len(self.invalid_vertex_refs)
            + len(self.invalid_uv_refs)
            + len(self.invalid_normal_refs)
            + len(self.degenerate_faces)
            + self.nonmanifold_edge_count
        )

    @property
    def has_issues(self) -> bool:
        return self.issue_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "vertex_count": self.vertex_count,
            "uv_count": self.uv_count,
            "normal_count": self.normal_count,
            "face_count": self.face_count,
            "face_arity_distribution": self.face_arity_distribution,
            "invalid_vertex_refs": [ref.to_dict() for ref in self.invalid_vertex_refs],
            "invalid_uv_refs": [ref.to_dict() for ref in self.invalid_uv_refs],
            "invalid_normal_refs": [ref.to_dict() for ref in self.invalid_normal_refs],
            "degenerate_faces": [face.to_dict() for face in self.degenerate_faces],
            "boundary_edge_count": self.boundary_edge_count,
            "nonmanifold_edge_count": self.nonmanifold_edge_count,
            "connected_component_count": self.connected_component_count,
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "issue_count": self.issue_count,
            "has_issues": self.has_issues,
        }


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_float_tuple(
    parts: list[str],
    line_number: int,
    record_name: str,
    minimum: int,
) -> tuple[float, ...]:
    if len(parts) < minimum:
        raise ObjParseError(
            f"line {line_number}: {record_name} requires at least {minimum} numeric values"
        )
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ObjParseError(f"line {line_number}: invalid {record_name} numeric value") from exc


def _resolve_obj_index(value: int, available: int) -> int | None:
    if value > 0:
        index = value - 1
    elif value < 0:
        index = available + value
    else:
        return None
    if 0 <= index < available:
        return index
    return None


def _parse_face_token(
    token: str,
    line_number: int,
    vertex_count: int,
    uv_count: int,
    normal_count: int,
) -> FaceRef:
    parts = token.split("/")
    if len(parts) > 3 or not parts[0]:
        raise ObjParseError(f"line {line_number}: malformed face token '{token}'")

    try:
        vertex = int(parts[0])
        uv = int(parts[1]) if len(parts) >= 2 and parts[1] else None
        normal = int(parts[2]) if len(parts) == 3 and parts[2] else None
    except ValueError as exc:
        raise ObjParseError(f"line {line_number}: non-integer face index in '{token}'") from exc

    return FaceRef(
        vertex=vertex,
        uv=uv,
        normal=normal,
        vertex_index=_resolve_obj_index(vertex, vertex_count),
        uv_index=_resolve_obj_index(uv, uv_count) if uv is not None else None,
        normal_index=_resolve_obj_index(normal, normal_count) if normal is not None else None,
        token=token,
    )


def parse_obj(path: Path | str) -> ObjMesh:
    """Parse v/vt/vn/f records from an OBJ file.

    Unknown records are ignored. Malformed supported records raise
    ObjParseError; out-of-range face references are kept for validation.
    """
    obj_path = Path(path)
    mesh = ObjMesh(path=obj_path)
    try:
        with obj_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = _strip_comment(raw_line)
                if not line:
                    continue
                parts = line.split()
                record, values = parts[0], parts[1:]
                if record == "v":
                    coords = _parse_float_tuple(values, line_number, "v", 3)
                    mesh.vertices.append((coords[0], coords[1], coords[2]))
                elif record == "vt":
                    coords = _parse_float_tuple(values, line_number, "vt", 1)
                    mesh.uvs.append(coords)
                elif record == "vn":
                    coords = _parse_float_tuple(values, line_number, "vn", 3)
                    mesh.normals.append((coords[0], coords[1], coords[2]))
                elif record == "f":
                    refs = tuple(
                        _parse_face_token(
                            token,
                            line_number,
                            len(mesh.vertices),
                            len(mesh.uvs),
                            len(mesh.normals),
                        )
                        for token in values
                    )
                    mesh.faces.append(Face(line_number=line_number, refs=refs))
    except OSError as exc:
        raise ObjParseError(f"{obj_path}: could not read OBJ ({exc})") from exc
    return mesh


def bbox_for_vertices(vertices: Iterable[tuple[float, float, float]]) -> tuple[list[float], list[float]] | tuple[None, None]:
    vertices = list(vertices)
    if not vertices:
        return None, None
    mins = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    maxs = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    return mins, maxs


def polygon_area(vertices: list[tuple[float, float, float]]) -> float:
    """Approximate 3D polygon area by triangulating from the first vertex."""
    if len(vertices) < 3:
        return 0.0
    origin = vertices[0]
    total = 0.0
    for index in range(1, len(vertices) - 1):
        left = (
            vertices[index][0] - origin[0],
            vertices[index][1] - origin[1],
            vertices[index][2] - origin[2],
        )
        right = (
            vertices[index + 1][0] - origin[0],
            vertices[index + 1][1] - origin[1],
            vertices[index + 1][2] - origin[2],
        )
        cross = (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )
        total += math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) * 0.5
    return total


def _invalid_reference(
    face: Face,
    face_index: int,
    corner_index: int,
    ref: FaceRef,
    kind: str,
    value: int,
    available: int,
) -> InvalidReference:
    return InvalidReference(
        line=face.line_number,
        face=face_index,
        corner=corner_index,
        token=ref.token,
        kind=kind,
        value=value,
        available=available,
    )


def validate_topology(mesh: ObjMesh) -> TopologyReport:
    """Build a topology report from a parsed OBJ mesh."""
    face_arity = Counter(str(len(face.refs)) for face in mesh.faces)
    invalid_vertices: list[InvalidReference] = []
    invalid_uvs: list[InvalidReference] = []
    invalid_normals: list[InvalidReference] = []
    degenerate_faces: list[DegenerateFace] = []
    edges: defaultdict[tuple[int, int], int] = defaultdict(int)
    components = UnionFind(len(mesh.vertices))

    for face_index, face in enumerate(mesh.faces, start=1):
        resolved_vertex_indices: list[int | None] = []
        for corner_index, ref in enumerate(face.refs, start=1):
            if ref.vertex_index is None:
                invalid_vertices.append(
                    _invalid_reference(
                        face,
                        face_index,
                        corner_index,
                        ref,
                        "vertex",
                        ref.vertex,
                        len(mesh.vertices),
                    )
                )
                resolved_vertex_indices.append(None)
            else:
                resolved_vertex_indices.append(ref.vertex_index)

            if ref.uv is not None and ref.uv_index is None:
                invalid_uvs.append(
                    _invalid_reference(
                        face, face_index, corner_index, ref, "uv", ref.uv, len(mesh.uvs)
                    )
                )
            if ref.normal is not None and ref.normal_index is None:
                invalid_normals.append(
                    _invalid_reference(
                        face,
                        face_index,
                        corner_index,
                        ref,
                        "normal",
                        ref.normal,
                        len(mesh.normals),
                    )
                )

        valid_vertex_indices = [index for index in resolved_vertex_indices if index is not None]
        unique_valid = set(valid_vertex_indices)
        if len(face.refs) < 3:
            degenerate_faces.append(DegenerateFace(face.line_number, face_index, "fewer than 3 corners"))
        elif len(unique_valid) < 3:
            degenerate_faces.append(DegenerateFace(face.line_number, face_index, "fewer than 3 unique valid vertices"))
        elif len(unique_valid) != len(valid_vertex_indices):
            degenerate_faces.append(DegenerateFace(face.line_number, face_index, "repeated vertex reference"))
        else:
            face_vertices = [mesh.vertices[index] for index in valid_vertex_indices]
            if polygon_area(face_vertices) <= 1e-12:
                degenerate_faces.append(DegenerateFace(face.line_number, face_index, "zero area"))

        if len(valid_vertex_indices) == len(face.refs) and len(valid_vertex_indices) >= 2:
            for left, right in zip(valid_vertex_indices, valid_vertex_indices[1:] + valid_vertex_indices[:1]):
                if left == right:
                    continue
                edge = (left, right) if left < right else (right, left)
                edges[edge] += 1
                components.union(left, right)

    boundary_edge_count = sum(1 for uses in edges.values() if uses == 1)
    nonmanifold_edge_count = sum(1 for uses in edges.values() if uses > 2)
    if mesh.vertices:
        connected_component_count = len({components.find(index) for index in range(len(mesh.vertices))})
    else:
        connected_component_count = 0
    bbox_min, bbox_max = bbox_for_vertices(mesh.vertices)

    return TopologyReport(
        path=str(mesh.path),
        vertex_count=len(mesh.vertices),
        uv_count=len(mesh.uvs),
        normal_count=len(mesh.normals),
        face_count=len(mesh.faces),
        face_arity_distribution=dict(sorted(face_arity.items(), key=lambda item: int(item[0]))),
        invalid_vertex_refs=invalid_vertices,
        invalid_uv_refs=invalid_uvs,
        invalid_normal_refs=invalid_normals,
        degenerate_faces=degenerate_faces,
        boundary_edge_count=boundary_edge_count,
        nonmanifold_edge_count=nonmanifold_edge_count,
        connected_component_count=connected_component_count,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def validate_obj_path(path: Path | str) -> TopologyReport:
    """Parse and validate one OBJ path."""
    return validate_topology(parse_obj(path))


def write_json_report(reports: list[TopologyReport], out_path: Path) -> None:
    payload = {
        "reports": [report.to_dict() for report in reports],
        "summary": {
            "file_count": len(reports),
            "files_with_issues": sum(1 for report in reports if report.has_issues),
            "total_issue_count": sum(report.issue_count for report in reports),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_text_report(report: TopologyReport) -> str:
    lines = [
        f"OBJ topology: {report.path}",
        f"  vertices: {report.vertex_count}",
        f"  uvs: {report.uv_count}",
        f"  normals: {report.normal_count}",
        f"  faces: {report.face_count}",
        f"  face arity: {report.face_arity_distribution}",
        f"  invalid refs: vertex={len(report.invalid_vertex_refs)}, uv={len(report.invalid_uv_refs)}, normal={len(report.invalid_normal_refs)}",
        f"  degenerate faces: {len(report.degenerate_faces)}",
        f"  boundary edges: {report.boundary_edge_count}",
        f"  nonmanifold edges: {report.nonmanifold_edge_count}",
        f"  connected components: {report.connected_component_count}",
        f"  bbox min: {report.bbox_min}",
        f"  bbox max: {report.bbox_max}",
        f"  issues: {report.issue_count}",
    ]
    if report.invalid_vertex_refs or report.invalid_uv_refs or report.invalid_normal_refs:
        lines.append("  invalid ref samples:")
        for ref in (report.invalid_vertex_refs + report.invalid_uv_refs + report.invalid_normal_refs)[:8]:
            lines.append(
                f"    line {ref.line}, face {ref.face}, corner {ref.corner}: "
                f"{ref.kind} {ref.value} with {ref.available} available ({ref.token})"
            )
    if report.degenerate_faces:
        lines.append("  degenerate samples:")
        for face in report.degenerate_faces[:8]:
            lines.append(f"    line {face.line}, face {face.face}: {face.reason}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate OBJ topology for DG face-pipeline meshes.",
    )
    parser.add_argument("objs", nargs="+", type=Path, help="One or more OBJ files to validate.")
    parser.add_argument("--json-out", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when validation issues are present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    missing = [path for path in args.objs if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing OBJ: {path}", file=sys.stderr)
        return 2

    reports: list[TopologyReport] = []
    for path in args.objs:
        try:
            reports.append(validate_obj_path(path))
        except ObjParseError as exc:
            print(f"Parse error: {exc}", file=sys.stderr)
            return 1

    for index, report in enumerate(reports):
        if index:
            print()
        print(format_text_report(report))

    if args.json_out:
        write_json_report(reports, args.json_out)
        print(f"\nWrote JSON report: {args.json_out}")

    if args.strict and any(report.has_issues for report in reports):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
