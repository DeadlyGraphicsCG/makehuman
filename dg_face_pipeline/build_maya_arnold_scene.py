"""
Build a clean Maya/Arnold look-dev scene for a DG Face Pipeline character.

Run this with Maya's Python, not normal CPython, for example:

    "C:\\Program Files\\Autodesk\\Maya2027\\bin\\mayapy.exe" ^
        dg_face_pipeline\\build_maya_arnold_scene.py --subject winona

The script opens the lighting template when present, removes old generated
face meshes, imports the normalized OBJ, creates an Arnold skin shader with
explicit texture color spaces, lays out three comparison meshes, and saves a
clean Maya scene.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PIPELINE_DIR / "outputs"

ARNOLD_SUBDIV_TYPES = {
    "none": 0,
    "catclark": 1,
    "linear": 2,
}

log = logging.getLogger("build_maya_arnold_scene")


def maya_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def default_paths(subject: str) -> dict[str, Path | None]:
    root = OUTPUTS_DIR / "characters" / subject
    maya_root = root / "maya"
    textures_root = root / "textures"
    subject_template = maya_root / "lightingscene_v001.mb"
    versioned_template = PIPELINE_DIR / "maya_templates" / "face_lighting_v001.mb"
    template = subject_template if subject_template.exists() else versioned_template
    return {
        "obj": maya_root / f"{subject}_face_mesh_maya_cm.obj",
        "albedo": textures_root / f"{subject}_albedo.jpg",
        "roughness": textures_root / f"{subject}_roughness.jpg",
        "normal": textures_root / f"{subject}_normal.jpg",
        "template": template if template.exists() else None,
        "out": maya_root / f"{subject}_arnold_skin_clean.ma",
    }


def initialize_maya():
    import maya.standalone  # type: ignore

    try:
        maya.standalone.initialize(name="python")
    except RuntimeError:
        pass
    from maya import cmds  # type: ignore

    return cmds


def load_plugins(cmds) -> None:
    for plugin in ("mtoa", "objExport"):
        try:
            if not cmds.pluginInfo(plugin, query=True, loaded=True):
                cmds.loadPlugin(plugin, quiet=True)
                log.info("Loaded Maya plugin: %s", plugin)
        except Exception as exc:  # Maya can still proceed with fallback nodes.
            log.warning("Could not load Maya plugin %s: %s", plugin, exc)


def set_if_exists(cmds, node: str, attr: str, value, attr_type: str | None = None) -> None:
    plug = f"{node}.{attr}"
    try:
        if not cmds.objExists(plug):
            return
        if attr_type:
            cmds.setAttr(plug, value, type=attr_type)
        else:
            cmds.setAttr(plug, value)
    except Exception as exc:
        log.debug("Could not set %s: %s", plug, exc)


def connect_if_possible(cmds, src: str, dst: str, force: bool = True) -> None:
    try:
        if cmds.objExists(src) and cmds.objExists(dst):
            cmds.connectAttr(src, dst, force=force)
    except Exception as exc:
        log.debug("Could not connect %s -> %s: %s", src, dst, exc)


def open_template_or_new(cmds, template: Path | None) -> None:
    if template and template.exists():
        cmds.file(maya_path(template), open=True, force=True)
        log.info("Opened lighting template: %s", template)
    else:
        cmds.file(new=True, force=True)
        log.info("Created a fresh Maya scene; no lighting template found")
    cmds.currentUnit(linear="cm", angle="deg", time="film")


def mesh_parent_transforms(cmds, nodes: Iterable[str] | None = None) -> list[str]:
    if nodes is None:
        shapes = cmds.ls(type="mesh", long=True) or []
    else:
        shapes = []
        for node in nodes:
            shapes.extend(cmds.listRelatives(node, allDescendents=True, type="mesh", fullPath=True) or [])
            if cmds.nodeType(node) == "mesh":
                shapes.append(node)
    parents: list[str] = []
    seen: set[str] = set()
    for shape in shapes:
        parent = (cmds.listRelatives(shape, parent=True, fullPath=True) or [None])[0]
        if parent and parent not in seen:
            seen.add(parent)
            parents.append(parent)
    return parents


def clear_template_meshes(cmds) -> None:
    for group in ("DG_FACE_PIPELINE_GRP", "DG_FACE_LOOKDEV_GRP"):
        if cmds.objExists(group):
            cmds.delete(group)
    transforms = mesh_parent_transforms(cmds)
    if transforms:
        cmds.delete(transforms)
        log.info("Removed %d existing mesh transforms from template", len(transforms))


def remove_unknown_scene_data(cmds) -> None:
    """Remove stale unknown nodes/plugins that stop binary templates saving as .ma."""
    unknown_nodes = cmds.ls(type="unknown") or []
    removed_nodes = 0
    for node in unknown_nodes:
        try:
            cmds.lockNode(node, lock=False)
            cmds.delete(node)
            removed_nodes += 1
        except Exception as exc:
            log.debug("Could not delete unknown node %s: %s", node, exc)
    unknown_plugins = cmds.unknownPlugin(query=True, list=True) or []
    removed_plugins = 0
    for plugin in unknown_plugins:
        try:
            cmds.unknownPlugin(plugin, remove=True)
            removed_plugins += 1
        except Exception as exc:
            log.debug("Could not remove unknown plugin %s: %s", plugin, exc)
    if removed_nodes or removed_plugins:
        log.info("Removed %d unknown nodes and %d unknown plugin records", removed_nodes, removed_plugins)


def delete_nodes(cmds, nodes: Iterable[str]) -> int:
    removed = 0
    for node in sorted(set(nodes)):
        if not cmds.objExists(node):
            continue
        try:
            cmds.lockNode(node, lock=False)
            cmds.delete(node)
            removed += 1
        except Exception as exc:
            log.debug("Could not delete node %s: %s", node, exc)
    return removed


def purge_generated_shader_clutter(cmds, subject: str) -> None:
    """Delete stale DG-generated material nodes while leaving cameras/lights alone."""
    patterns = [
        "*_skin_aiStandardSurface*",
        "*_skin_standardSurface*",
        "*_skin_SG*",
        "*_albedo_file*",
        "*_albedo_place2d*",
        "*_roughness_file*",
        "*_roughness_place2d*",
        "*_normal_file*",
        "*_normal_place2d*",
        "*_normal_bump2d*",
        "*_photo_file*",
        "*_photo_place2d*",
        "face_photo_mat*",
        "Char_skin",
        "aiStandardSurface1SG",
    ]
    candidates: list[str] = []
    protected = {"initialShadingGroup", "initialParticleSE", "lambert1", "standardSurface1", "particleCloud1"}
    for pattern in patterns:
        candidates.extend(cmds.ls(pattern) or [])
    candidates.extend(cmds.ls(f"dg_{subject}_import:*") or [])
    candidates = [node for node in candidates if node.split("|")[-1] not in protected]
    removed = delete_nodes(cmds, candidates)
    if removed:
        log.info("Removed %d stale/generated shader nodes", removed)


def create_fallback_camera_and_lights(cmds) -> None:
    cameras = [c for c in (cmds.ls(type="camera") or []) if not c.startswith(("persp", "top", "front", "side"))]
    if not cameras:
        cam_transform, cam_shape = cmds.camera(name="render_cam")
        cmds.setAttr(f"{cam_transform}.translate", 0, 2, 120, type="double3")
        cmds.setAttr(f"{cam_shape}.focalLength", 50)
        cmds.setAttr(f"{cam_shape}.nearClipPlane", 0.1)
        cmds.setAttr(f"{cam_shape}.farClipPlane", 1000)
        cmds.setAttr(f"{cam_shape}.renderable", True)
    if not cmds.ls(type="aiAreaLight"):
        key = cmds.createNode("transform", name="key_area_light")
        key_shape = cmds.createNode("aiAreaLight", name="key_area_lightShape", parent=key)
        cmds.setAttr(f"{key}.translate", -45, 55, 80, type="double3")
        cmds.setAttr(f"{key}.rotate", -30, -20, 0, type="double3")
        set_if_exists(cmds, key_shape, "intensity", 650)
        set_if_exists(cmds, key_shape, "aiSamples", 3)
        fill = cmds.createNode("transform", name="fill_area_light")
        fill_shape = cmds.createNode("aiAreaLight", name="fill_area_lightShape", parent=fill)
        cmds.setAttr(f"{fill}.translate", 55, 35, 75, type="double3")
        cmds.setAttr(f"{fill}.rotate", -25, 25, 0, type="double3")
        set_if_exists(cmds, fill_shape, "intensity", 130)
        set_if_exists(cmds, fill_shape, "aiSamples", 2)


def camera_transforms(cmds) -> list[str]:
    default_cameras = {"persp", "top", "front", "side"}
    camera_shapes = cmds.ls(type="camera", long=True) or []
    renderable: list[str] = []
    candidates: list[str] = []
    for shape in camera_shapes:
        parent = (cmds.listRelatives(shape, parent=True, fullPath=True) or [None])[0]
        if not parent:
            continue
        short = parent.split("|")[-1]
        if short in default_cameras:
            continue
        candidates.append(parent)
        try:
            if cmds.getAttr(f"{shape}.renderable"):
                renderable.append(parent)
        except Exception:
            pass
    return renderable or candidates


def aim_render_camera_at_meshes(cmds, transforms: Iterable[str]) -> None:
    cameras = camera_transforms(cmds)
    if not cameras:
        return
    bbox_nodes = [node for node in transforms if cmds.objExists(node)]
    if not bbox_nodes:
        return
    bbox = cmds.exactWorldBoundingBox(bbox_nodes)
    target = (
        (bbox[0] + bbox[3]) * 0.5,
        (bbox[1] + bbox[4]) * 0.5,
        (bbox[2] + bbox[5]) * 0.5,
    )
    locator = cmds.spaceLocator(name="DG_FACE_CAMERA_AIM_TMP")[0]
    cmds.setAttr(f"{locator}.translate", *target, type="double3")
    for camera in cameras:
        try:
            constraint = cmds.aimConstraint(
                locator,
                camera,
                aimVector=(0, 0, -1),
                upVector=(0, 1, 0),
                worldUpType="scene",
            )[0]
            cmds.delete(constraint)
            shape = (cmds.listRelatives(camera, shapes=True, type="camera", fullPath=True) or [None])[0]
            if shape:
                set_if_exists(cmds, shape, "renderable", True)
                set_if_exists(cmds, shape, "focalLength", 70)
                set_if_exists(cmds, shape, "nearClipPlane", 0.1)
                set_if_exists(cmds, shape, "farClipPlane", 1000)
            log.info("Aimed camera %s at generated face layout", camera)
        except Exception as exc:
            log.warning("Could not aim camera %s: %s", camera, exc)
    cmds.delete(locator)


def create_file_node(cmds, subject: str, label: str, texture: Path, color_space: str) -> str:
    file_node = cmds.shadingNode("file", asTexture=True, name=f"{subject}_{label}_file")
    place_node = cmds.shadingNode("place2dTexture", asUtility=True, name=f"{subject}_{label}_place2d")
    set_if_exists(cmds, file_node, "fileTextureName", maya_path(texture), "string")
    set_if_exists(cmds, file_node, "colorSpace", color_space, "string")
    if color_space == "Raw":
        set_if_exists(cmds, file_node, "alphaIsLuminance", True)
    for src, dst in (
        ("coverage", "coverage"),
        ("translateFrame", "translateFrame"),
        ("rotateFrame", "rotateFrame"),
        ("mirrorU", "mirrorU"),
        ("mirrorV", "mirrorV"),
        ("stagger", "stagger"),
        ("wrapU", "wrapU"),
        ("wrapV", "wrapV"),
        ("repeatUV", "repeatUV"),
        ("offset", "offset"),
        ("rotateUV", "rotateUV"),
        ("noiseUV", "noiseUV"),
        ("vertexUvOne", "vertexUvOne"),
        ("vertexUvTwo", "vertexUvTwo"),
        ("vertexUvThree", "vertexUvThree"),
        ("vertexCameraOne", "vertexCameraOne"),
        ("outUV", "uvCoord"),
        ("outUvFilterSize", "uvFilterSize"),
    ):
        connect_if_possible(cmds, f"{place_node}.{src}", f"{file_node}.{dst}")
    return file_node


def create_skin_shader(
    cmds,
    subject: str,
    albedo: Path,
    roughness: Path | None,
    normal: Path | None,
) -> tuple[str, str]:
    shader_type = "aiStandardSurface"
    try:
        shader = cmds.shadingNode(shader_type, asShader=True, name=f"{subject}_skin_aiStandardSurface")
    except Exception:
        shader_type = "standardSurface"
        shader = cmds.shadingNode(shader_type, asShader=True, name=f"{subject}_skin_standardSurface")
        log.warning("Fell back to Maya standardSurface because aiStandardSurface was unavailable")
    sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=f"{subject}_skin_SG")
    connect_if_possible(cmds, f"{shader}.outColor", f"{sg}.surfaceShader")

    set_if_exists(cmds, shader, "base", 0.92)
    set_if_exists(cmds, shader, "specular", 0.42)
    set_if_exists(cmds, shader, "specularRoughness", 0.5)
    set_if_exists(cmds, shader, "metalness", 0.0)
    set_if_exists(cmds, shader, "transmission", 0.0)
    set_if_exists(cmds, shader, "subsurface", 0.22)
    for attr, values in (
        ("baseColor", (0.78, 0.48, 0.38)),
        ("subsurfaceColor", (1.0, 0.58, 0.42)),
        ("subsurfaceRadius", (1.0, 0.35, 0.18)),
    ):
        try:
            if cmds.objExists(f"{shader}.{attr}"):
                cmds.setAttr(f"{shader}.{attr}", *values, type="double3")
        except Exception:
            pass

    albedo_node = create_file_node(cmds, subject, "albedo", albedo, "sRGB")
    connect_if_possible(cmds, f"{albedo_node}.outColor", f"{shader}.baseColor")
    if roughness and roughness.exists():
        rough_node = create_file_node(cmds, subject, "roughness", roughness, "Raw")
        connect_if_possible(cmds, f"{rough_node}.outColorR", f"{shader}.specularRoughness")
    if normal and normal.exists():
        normal_node = create_file_node(cmds, subject, "normal", normal, "Raw")
        bump = cmds.shadingNode("bump2d", asUtility=True, name=f"{subject}_normal_bump2d")
        set_if_exists(cmds, bump, "bumpInterp", 1)
        set_if_exists(cmds, bump, "bumpDepth", 0.08)
        set_if_exists(cmds, bump, "bumpFilter", 1.25)
        connect_if_possible(cmds, f"{normal_node}.outAlpha", f"{bump}.bumpValue")
        connect_if_possible(cmds, f"{bump}.outNormal", f"{shader}.normalCamera")
    return shader, sg


def import_obj(cmds, obj: Path, subject: str) -> list[str]:
    namespace = f"dg_{subject}_import"
    if cmds.namespace(exists=namespace):
        cmds.namespace(removeNamespace=namespace, mergeNamespaceWithRoot=True)
    new_nodes = cmds.file(
        maya_path(obj),
        i=True,
        type="OBJ",
        ignoreVersion=True,
        ra=True,
        mergeNamespacesOnClash=False,
        namespace=namespace,
        returnNewNodes=True,
    ) or []
    transforms = mesh_parent_transforms(cmds, new_nodes)
    if not transforms:
        transforms = mesh_parent_transforms(cmds, cmds.ls(f"{namespace}:*", long=True) or [])
    if not transforms:
        raise RuntimeError(f"No mesh transforms imported from {obj}")
    main = transforms[0]
    short = main.split("|")[-1]
    if short != f"{subject}_GEO":
        main = cmds.rename(main, f"{subject}_GEO")
    return [main]


def apply_smoothing(cmds, transforms: Iterable[str], subdiv_type: str, subdiv_iterations: int) -> None:
    subdiv_type_id = ARNOLD_SUBDIV_TYPES[subdiv_type]
    for transform in transforms:
        shapes = cmds.listRelatives(transform, allDescendents=True, type="mesh", fullPath=True) or []
        try:
            cmds.select(transform, replace=True)
            cmds.polySoftEdge(angle=180, constructionHistory=False)
        except Exception as exc:
            log.debug("Could not soften %s: %s", transform, exc)
        for shape in shapes:
            set_if_exists(cmds, shape, "displaySmoothMesh", 2)
            set_if_exists(cmds, shape, "smoothLevel", 1)
            set_if_exists(cmds, shape, "renderSmoothLevel", max(0, subdiv_iterations))
            set_if_exists(cmds, shape, "aiSubdivType", subdiv_type_id)
            set_if_exists(cmds, shape, "aiSubdivIterations", max(0, subdiv_iterations))
            set_if_exists(cmds, shape, "aiSubdivSmoothDerivs", True)


def assign_shader(cmds, transforms: Iterable[str], shading_group: str) -> None:
    shapes: list[str] = []
    for transform in transforms:
        shapes.extend(cmds.listRelatives(transform, allDescendents=True, type="mesh", fullPath=True) or [])
    if shapes:
        cmds.sets(shapes, edit=True, forceElement=shading_group)


def delete_unused_shader_nodes(cmds, keep_prefix: str) -> None:
    """Remove material nodes created by OBJ import after our shader is assigned."""
    used_nodes: set[str] = set()
    for node in cmds.ls(f"{keep_prefix}_*", materials=True) or []:
        used_nodes.add(node)
    for node in cmds.ls(f"{keep_prefix}_*", type="file") or []:
        used_nodes.add(node)
    for node in cmds.ls(f"{keep_prefix}_*", type="place2dTexture") or []:
        used_nodes.add(node)
    for node in cmds.ls(f"{keep_prefix}_*", type="bump2d") or []:
        used_nodes.add(node)
    unused: list[str] = []
    for node_type in ("shadingEngine", "lambert", "phong", "blinn", "file", "place2dTexture", "materialInfo"):
        for node in cmds.ls(type=node_type) or []:
            if node in used_nodes or node in {"initialShadingGroup", "initialParticleSE", "lambert1"}:
                continue
            if node.startswith(keep_prefix):
                continue
            history = cmds.listConnections(node, source=True, destination=True) or []
            connected_to_mesh = False
            for linked in history:
                try:
                    if cmds.nodeType(linked) == "mesh":
                        connected_to_mesh = True
                        break
                except Exception:
                    pass
            if not connected_to_mesh and (
                node.startswith("dg_")
                or "face_photo_mat" in node
                or node.startswith("aiStandardSurface")
                or node == "Char_skin"
                or node.endswith("_photo_file")
                or node.endswith("_photo_place2d")
            ):
                unused.append(node)
    removed = delete_nodes(cmds, unused)
    if removed:
        log.info("Removed %d unused OBJ/template shader nodes", removed)


def layout_meshes(cmds, subject: str, main: str, layout: str, front_rotate_y: float) -> list[str]:
    group = cmds.group(empty=True, name="DG_FACE_PIPELINE_GRP")
    main = cmds.parent(main, group)[0]
    main = cmds.rename(main, f"{subject}_GEO_center")
    cmds.setAttr(f"{main}.translate", 0, 0, 0, type="double3")
    cmds.setAttr(f"{main}.rotateY", front_rotate_y)
    transforms = [main]
    if layout == "three":
        left = cmds.duplicate(main, rr=True, name=f"{subject}_GEO_left")[0]
        right = cmds.duplicate(main, rr=True, name=f"{subject}_GEO_right")[0]
        cmds.setAttr(f"{left}.translateX", -24)
        cmds.setAttr(f"{left}.rotateY", front_rotate_y + 24.0)
        cmds.setAttr(f"{right}.translateX", 24)
        cmds.setAttr(f"{right}.rotateY", front_rotate_y - 24.0)
        transforms.extend([left, right])
    return transforms


def build_scene(
    subject: str,
    obj: Path,
    albedo: Path,
    roughness: Path | None,
    normal: Path | None,
    out: Path,
    template: Path | None,
    layout: str,
    front_rotate_y: float,
    arnold_subdiv_type: str,
    arnold_subdiv_iterations: int,
    clear_meshes: bool,
) -> Path:
    cmds = initialize_maya()
    load_plugins(cmds)
    open_template_or_new(cmds, template)
    if clear_meshes:
        clear_template_meshes(cmds)
    create_fallback_camera_and_lights(cmds)
    purge_generated_shader_clutter(cmds, subject)
    try:
        cmds.setAttr("defaultRenderGlobals.ren", "arnold", type="string")
    except Exception:
        pass
    _, sg = create_skin_shader(cmds, subject, albedo, roughness, normal)
    imported = import_obj(cmds, obj, subject)
    transforms = layout_meshes(cmds, subject, imported[0], layout, front_rotate_y)
    assign_shader(cmds, transforms, sg)
    delete_unused_shader_nodes(cmds, subject)
    apply_smoothing(cmds, transforms, arnold_subdiv_type, arnold_subdiv_iterations)
    aim_render_camera_at_meshes(cmds, transforms)
    remove_unknown_scene_data(cmds)
    out.parent.mkdir(parents=True, exist_ok=True)
    file_type = "mayaBinary" if out.suffix.lower() == ".mb" else "mayaAscii"
    cmds.file(rename=maya_path(out))
    cmds.file(save=True, type=file_type, force=True)
    log.info("Saved clean Maya scene: %s", out)
    return out


def existing_path(path: Path | None) -> Path | None:
    if path and path.exists():
        return path
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a clean Maya Arnold face look-dev scene.")
    p.add_argument("--subject", default="winona")
    p.add_argument("--obj", type=Path)
    p.add_argument("--albedo", type=Path)
    p.add_argument("--roughness", type=Path)
    p.add_argument("--normal", type=Path)
    p.add_argument("--template", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--layout", choices=["single", "three"], default="three")
    p.add_argument("--front-rotate-y", type=float, default=-24.0)
    p.add_argument("--arnold-subdiv-type", choices=sorted(ARNOLD_SUBDIV_TYPES), default="catclark")
    p.add_argument("--arnold-subdiv-iterations", type=int, default=2)
    p.add_argument("--keep-template-meshes", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args(argv)
    paths = default_paths(args.subject)
    obj = args.obj or paths["obj"]
    albedo = args.albedo or paths["albedo"]
    roughness = args.roughness or existing_path(paths["roughness"])
    normal = args.normal or existing_path(paths["normal"])
    template = args.template if args.template is not None else paths["template"]
    out = args.out or paths["out"]
    for label, path in (("obj", obj), ("albedo", albedo)):
        if path is None or not path.exists():
            log.error("Missing %s path: %s", label, path)
            return 1
    try:
        build_scene(
            args.subject,
            obj,
            albedo,
            roughness if roughness and roughness.exists() else None,
            normal if normal and normal.exists() else None,
            out,
            template if template and template.exists() else None,
            args.layout,
            args.front_rotate_y,
            args.arnold_subdiv_type,
            args.arnold_subdiv_iterations,
            not args.keep_template_meshes,
        )
    except Exception as exc:
        log.exception("Maya scene build failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
