"""
Validate that a generated Maya scene uses the intended Arnold look-dev graph.

Run with Maya's Python, for example:

    "C:\\Program Files\\Autodesk\\Maya2027\\bin\\mayapy.exe" ^
        dg_face_pipeline\\validate_maya_arnold_scene.py ^
        dg_face_pipeline\\outputs\\characters\\winona_v003\\maya\\winona_v003_arnold_skin_origin.mb ^
        --subject winona_v003
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def maya_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def initialize_maya():
    import maya.standalone  # type: ignore

    try:
        maya.standalone.initialize(name="python")
    except RuntimeError:
        pass
    from maya import cmds  # type: ignore

    return cmds


def plug_exists(cmds, node: str, attrs: tuple[str, ...]) -> str | None:
    for attr in attrs:
        plug = f"{node}.{attr}"
        if cmds.objExists(plug):
            return plug
    return None


def connected_sources(cmds, node: str, attrs: tuple[str, ...]) -> list[str]:
    plug = plug_exists(cmds, node, attrs)
    if plug is None:
        return []
    return cmds.listConnections(plug, plugs=True, source=True, destination=False) or []


def add_issue(issues: list[str], message: str) -> None:
    issues.append(message)


def validate_scene(scene: Path, subject: str, require_three: bool, require_physical_sky: bool) -> dict[str, Any]:
    cmds = initialize_maya()
    issues: list[str] = []
    try:
        if not cmds.pluginInfo("mtoa", query=True, loaded=True):
            cmds.loadPlugin("mtoa", quiet=True)
    except Exception as exc:
        add_issue(issues, f"Could not load MtoA: {exc}")

    cmds.file(maya_path(scene), open=True, force=True)
    report: dict[str, Any] = {
        "scene": str(scene.resolve()),
        "subject": subject,
        "issues": issues,
    }

    try:
        renderer = cmds.getAttr("defaultRenderGlobals.ren")
    except Exception:
        renderer = None
    report["renderer"] = renderer
    if renderer != "arnold":
        add_issue(issues, f"defaultRenderGlobals.ren is {renderer!r}, expected 'arnold'")

    shader = f"{subject}_skin_aiStandardSurface"
    sg = f"{subject}_skin_SG"
    report["shader"] = shader
    report["shader_type"] = cmds.nodeType(shader) if cmds.objExists(shader) else None
    if not cmds.objExists(shader):
        add_issue(issues, f"Missing shader: {shader}")
    elif cmds.nodeType(shader) != "aiStandardSurface":
        add_issue(issues, f"{shader} is {cmds.nodeType(shader)}, expected aiStandardSurface")

    report["shading_engine"] = sg
    if not cmds.objExists(sg):
        add_issue(issues, f"Missing shadingEngine: {sg}")
    else:
        surface_sources = cmds.listConnections(f"{sg}.surfaceShader", plugs=True, source=True, destination=False) or []
        report["surface_shader_sources"] = surface_sources
        if not any(src.startswith(f"{shader}.") for src in surface_sources):
            add_issue(issues, f"{sg}.surfaceShader is not driven by {shader}")

    expected_meshes = [f"{subject}_GEO_center"]
    if require_three:
        expected_meshes.extend([f"{subject}_GEO_left", f"{subject}_GEO_right"])
    transforms: dict[str, dict[str, Any]] = {}
    for transform in expected_meshes:
        if not cmds.objExists(transform):
            add_issue(issues, f"Missing mesh transform: {transform}")
            continue
        shapes = cmds.listRelatives(transform, shapes=True, type="mesh", fullPath=False) or []
        tx = tuple(round(v, 6) for v in cmds.getAttr(f"{transform}.translate")[0])
        ry = round(float(cmds.getAttr(f"{transform}.rotateY")), 6)
        transforms[transform] = {"translate": tx, "rotateY": ry, "shapes": shapes}
        for shape in shapes:
            shading_groups = cmds.listConnections(shape, type="shadingEngine") or []
            if sg not in shading_groups:
                add_issue(issues, f"{shape} is not assigned to {sg}; got {shading_groups}")
    report["transforms"] = transforms
    if require_three and f"{subject}_GEO_left" in transforms and transforms[f"{subject}_GEO_left"]["rotateY"] != -45.0:
        add_issue(issues, f"{subject}_GEO_left.rotateY is not -45")
    if require_three and f"{subject}_GEO_right" in transforms and transforms[f"{subject}_GEO_right"]["rotateY"] != 45.0:
        add_issue(issues, f"{subject}_GEO_right.rotateY is not 45")
    if f"{subject}_GEO_center" in transforms and transforms[f"{subject}_GEO_center"]["rotateY"] != 0.0:
        add_issue(issues, f"{subject}_GEO_center.rotateY is not 0")

    file_nodes = {
        "albedo": f"{subject}_albedo_file",
        "roughness": f"{subject}_roughness_file",
        "normal": f"{subject}_normal_file",
    }
    files: dict[str, dict[str, Any]] = {}
    for label, node in file_nodes.items():
        if not cmds.objExists(node):
            add_issue(issues, f"Missing file node: {node}")
            continue
        files[label] = {
            "node": node,
            "path": cmds.getAttr(f"{node}.fileTextureName"),
            "colorSpace": cmds.getAttr(f"{node}.colorSpace"),
        }
    report["files"] = files
    if files.get("albedo", {}).get("colorSpace") != "sRGB":
        add_issue(issues, "Albedo file node is not sRGB")
    for label in ("roughness", "normal"):
        if files.get(label, {}).get("colorSpace") != "Raw":
            add_issue(issues, f"{label} file node is not Raw")

    if cmds.objExists(shader):
        report["base_color_sources"] = connected_sources(cmds, shader, ("baseColor", "base_color"))
        report["roughness_sources"] = connected_sources(cmds, shader, ("specularRoughness", "specular_roughness"))
        report["normal_sources"] = connected_sources(cmds, shader, ("normalCamera", "normal", "n"))
        if not report["base_color_sources"]:
            add_issue(issues, "Shader base color has no texture/color-correct source")
        if not report["roughness_sources"]:
            add_issue(issues, "Shader roughness has no roughness source")
        if not report["normal_sources"]:
            add_issue(issues, "Shader normal has no bump/normal source")

    bump = f"{subject}_normal_bump2d"
    report["bump_node"] = bump if cmds.objExists(bump) else None
    if not cmds.objExists(bump):
        add_issue(issues, f"Missing tangent normal bump2d node: {bump}")
    else:
        report["bumpInterp"] = cmds.getAttr(f"{bump}.bumpInterp") if cmds.objExists(f"{bump}.bumpInterp") else None
        if report["bumpInterp"] != 1:
            add_issue(issues, f"{bump}.bumpInterp is {report['bumpInterp']}, expected 1 for tangent-space normals")

    report["aiPhysicalSky"] = cmds.ls(type="aiPhysicalSky") or []
    report["aiSkyDomeLight"] = cmds.ls(type="aiSkyDomeLight") or []
    report["aiAreaLight"] = cmds.ls(type="aiAreaLight") or []
    if require_physical_sky:
        if not report["aiPhysicalSky"]:
            add_issue(issues, "Missing aiPhysicalSky")
        if not report["aiSkyDomeLight"]:
            add_issue(issues, "Missing aiSkyDomeLight")
        if report["aiAreaLight"]:
            add_issue(issues, f"Unexpected aiAreaLight nodes: {report['aiAreaLight']}")

    report["ok"] = not issues
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated Maya Arnold shader/look-dev scene.")
    parser.add_argument("scene", type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--no-three", dest="require_three", action="store_false", default=True)
    parser.add_argument("--no-physical-sky", dest="require_physical_sky", action="store_false", default=True)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_scene(args.scene, args.subject, args.require_three, args.require_physical_sky)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
