"""
Abstract texture-baking adapter contract for the DG face pipeline.

Version 1 intentionally implements only the ``manifest`` backend. It writes a
dry-run JSON report that downstream Maya, Blender, or farm adapters can consume
without requiring those tools to be installed on the current machine.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("bake_maps")

DEFAULT_MAPS = ("normal", "ambient_occlusion", "curvature")
BACKENDS = ("manifest", "maya", "blender")


class BackendUnavailableError(RuntimeError):
    """Raised when a backend adapter exists but cannot bake in this runtime."""


@dataclass(frozen=True)
class BakeRequest:
    low: Path
    high: Path
    out_dir: Path
    resolution: int
    cage_distance: float
    backend: str
    maps: tuple[str, ...]
    report_path: Path


class BakingAdapter(ABC):
    """Common contract all texture-baking backends must implement."""

    backend: str
    executable_names: tuple[str, ...] = ()

    def __init__(self, request: BakeRequest):
        self.request = request

    @abstractmethod
    def bake(self) -> dict[str, Any]:
        """Run or describe the bake and return a serialisable report."""

    @property
    @abstractmethod
    def command_contract(self) -> dict[str, Any]:
        """Document the external command expected by this adapter."""

    def availability_notes(self) -> dict[str, Any]:
        found = {
            name: shutil.which(name)
            for name in self.executable_names
        }
        return {
            "backend": self.backend,
            "implemented": isinstance(self, ManifestAdapter),
            "required_executables": list(self.executable_names),
            "discovered_executables": {name: path for name, path in found.items() if path},
            "available": isinstance(self, ManifestAdapter),
            "notes": self._availability_text(found),
        }

    def _availability_text(self, found: dict[str, str | None]) -> list[str]:
        if isinstance(self, ManifestAdapter):
            return ["Manifest backend is always available; no DCC application is required."]
        missing = [name for name, path in found.items() if path is None]
        if missing:
            return [
                f"Missing executable(s): {', '.join(missing)}.",
                "Install the DCC application or pass the full command to a future adapter wrapper.",
                "This v1 adapter is a contract stub and does not launch the backend yet.",
            ]
        return [
            "Executable was found, but this v1 adapter is still a contract stub.",
            "Use backend=manifest to generate bake requests until execution support is wired.",
        ]

    def build_base_report(self, status: str) -> dict[str, Any]:
        output_paths = {
            map_name: (self.request.out_dir / f"{map_name}_{self.request.resolution}.png").as_posix()
            for map_name in self.request.maps
        }
        return {
            "schema": "dg_face_pipeline.bake_manifest.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "backend": self.backend,
            "inputs": {
                "low": describe_path(self.request.low),
                "high": describe_path(self.request.high),
            },
            "requested_maps": list(self.request.maps),
            "resolution": {
                "width": self.request.resolution,
                "height": self.request.resolution,
                "unit": "pixels",
            },
            "cage": {
                "distance": self.request.cage_distance,
                "unit": "scene_units",
                "mode": "uniform_distance",
            },
            "output_paths": output_paths,
            "report_path": self.request.report_path.as_posix(),
            "backend_availability": self.availability_notes(),
            "command_contract": self.command_contract,
        }


class ManifestAdapter(BakingAdapter):
    backend = "manifest"

    @property
    def command_contract(self) -> dict[str, Any]:
        return {
            "purpose": "Write a dry-run bake manifest without launching a DCC backend.",
            "invocation": (
                "python dg_face_pipeline/bake_maps.py --backend manifest "
                "--low <retopo.obj> --high <scan.obj> --out-dir <textures>"
            ),
            "outputs": [
                "bake_manifest.json",
                "declared map paths only; no image maps are rendered by this backend",
            ],
        }

    def bake(self) -> dict[str, Any]:
        report = self.build_base_report(status="dry_run")
        report["notes"] = [
            "No bake was executed.",
            "Use this manifest to hand off identical inputs to Maya, Blender, or a render-farm adapter.",
        ]
        return report


class MayaAdapter(BakingAdapter):
    backend = "maya"
    executable_names = ("mayapy", "maya")

    @property
    def command_contract(self) -> dict[str, Any]:
        return {
            "purpose": "Bake selected maps from a high-resolution mesh to a low-resolution mesh in Maya.",
            "expected_entrypoint": "mayapy <future_maya_bake_script.py>",
            "required_arguments": [
                "--low <low_retopo_mesh>",
                "--high <high_source_mesh>",
                "--out-dir <texture_output_directory>",
                "--resolution <square_texture_size>",
                "--cage-distance <uniform_projection_distance>",
                "--maps <comma-separated map list>",
            ],
            "expected_outputs": [
                "<map>_<resolution>.png for each requested map",
                "bake_manifest.json with backend result metadata",
            ],
        }

    def bake(self) -> dict[str, Any]:
        report = self.build_base_report(status="adapter_stub")
        report["error"] = (
            "The Maya baking backend is a v1 contract stub. Generate a manifest now, "
            "or wire this contract to a mayapy script that imports the low/high meshes, "
            "sets transfer-map cage distance, and writes the requested maps."
        )
        raise BackendUnavailableError(report["error"])


class BlenderAdapter(BakingAdapter):
    backend = "blender"
    executable_names = ("blender",)

    @property
    def command_contract(self) -> dict[str, Any]:
        return {
            "purpose": "Bake selected maps from a high-resolution mesh to a low-resolution mesh in Blender.",
            "expected_entrypoint": "blender --background --python <future_blender_bake_script.py> --",
            "required_arguments": [
                "--low <low_retopo_mesh>",
                "--high <high_source_mesh>",
                "--out-dir <texture_output_directory>",
                "--resolution <square_texture_size>",
                "--cage-distance <uniform_projection_distance>",
                "--maps <comma-separated map list>",
            ],
            "expected_outputs": [
                "<map>_<resolution>.png for each requested map",
                "bake_manifest.json with backend result metadata",
            ],
        }

    def bake(self) -> dict[str, Any]:
        report = self.build_base_report(status="adapter_stub")
        report["error"] = (
            "The Blender baking backend is a v1 contract stub. Generate a manifest now, "
            "or wire this contract to a blender --background script that loads the meshes, "
            "configures cage extrusion/distance, and writes the requested maps."
        )
        raise BackendUnavailableError(report["error"])


def describe_path(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "resolved": path.resolve().as_posix(),
        "exists": path.exists(),
    }


def parse_maps(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_MAPS
    maps: list[str] = []
    for value in values:
        for item in value.split(","):
            cleaned = item.strip().lower().replace("-", "_")
            if cleaned:
                maps.append(cleaned)
    deduped = tuple(dict.fromkeys(maps))
    if not deduped:
        raise ValueError("At least one requested map is required.")
    return deduped


def make_adapter(request: BakeRequest) -> BakingAdapter:
    if request.backend == "manifest":
        return ManifestAdapter(request)
    if request.backend == "maya":
        return MayaAdapter(request)
    if request.backend == "blender":
        return BlenderAdapter(request)
    raise ValueError(f"Unsupported bake backend: {request.backend}")


def write_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or execute DG face-pipeline texture bake requests.",
    )
    parser.add_argument("--low", type=Path, required=True, help="Low-resolution retopo mesh receiving baked maps.")
    parser.add_argument("--high", type=Path, required=True, help="High-resolution source mesh used for projection.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the bake report and map outputs.")
    parser.add_argument("--resolution", type=int, default=2048, help="Square output texture size in pixels.")
    parser.add_argument("--cage-distance", type=float, default=0.02, help="Uniform cage/projection distance in scene units.")
    parser.add_argument("--backend", choices=BACKENDS, default="manifest", help="Bake backend adapter to use.")
    parser.add_argument(
        "--maps",
        action="append",
        default=None,
        help=(
            "Requested map names. May be repeated or comma-separated. "
            f"Default: {', '.join(DEFAULT_MAPS)}."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Report JSON path. Defaults to <out-dir>/bake_manifest.json.",
    )
    return parser.parse_args(argv)


def request_from_args(args: argparse.Namespace) -> BakeRequest:
    if args.resolution <= 0:
        raise ValueError("--resolution must be greater than zero.")
    if args.cage_distance < 0:
        raise ValueError("--cage-distance must be zero or greater.")
    report_path = args.report or args.out_dir / "bake_manifest.json"
    return BakeRequest(
        low=args.low,
        high=args.high,
        out_dir=args.out_dir,
        resolution=args.resolution,
        cage_distance=args.cage_distance,
        backend=args.backend,
        maps=parse_maps(args.maps),
        report_path=report_path,
    )


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    report: dict[str, Any] | None = None
    try:
        request = request_from_args(args)
        adapter = make_adapter(request)
        if isinstance(adapter, (MayaAdapter, BlenderAdapter)):
            report = adapter.build_base_report(status="adapter_stub")
            report["error"] = (
                f"The {adapter.backend} backend is documented but not executable in v1. "
                "Run with --backend manifest to write a dry-run request, or implement the "
                "documented command contract."
            )
            write_report(report, request.report_path)
            raise BackendUnavailableError(report["error"])
        report = adapter.bake()
        write_report(report, request.report_path)
    except ValueError as exc:
        log.error("Invalid bake request: %s", exc)
        return 2
    except BackendUnavailableError as exc:
        log.error("%s", exc)
        if report:
            log.error("Wrote adapter report: %s", report["report_path"])
        return 1
    except OSError as exc:
        log.error("Could not write bake report: %s", exc)
        return 1

    log.info("Wrote bake report: %s", report["report_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
