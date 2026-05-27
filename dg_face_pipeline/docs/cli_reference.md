# CLI Reference

All paths default to pipeline-relative locations (`examples/`, `outputs/`,
`../makehuman/` for the MH data files). Override any of them with explicit
flags. Run any script with `--help` for the live argparse-generated help.

## run_full_pipeline.py
The one-shot orchestrator. Runs analyzer → step C → learned mesh → canon
report → MH render → learned render → final stacked frame. It does not export
the Maya handoff.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--subject` | str | *required* | Short slug (e.g. `winona`); used to name all outputs. |
| `--image` | Path | `examples/<subject>/ref.png` | Optional source portrait override. Falls back to legacy `examples/<subject>_ref.png`. |
| `--subdivisions` | int | `2` | Mesh-densification passes for the learned mesh. With the default/v002 468-landmark topology: `0` = 468 verts, `1` = 1833, `2` = 7257, `3` = 28881. |
| `--amplify` | float | `1.5` | Multiplier on MH modifier values. Above ~2.5 begins distorting MH geometry. |
| `--canonical` | Path | `canonical_face_model.obj` | Canonical topology OBJ for the learned mesh stage. Use v002 with `--subdivisions 0` to preserve quads. |
| `--uv-mode` | `photo` / `canonical` | `photo` | Passed to `learned_face_mesh.py`; canonical mode writes stable base UVs and bakes an atlas albedo. |
| `--texture-size` | int | `1024` | Canonical atlas size when `--uv-mode canonical` is used. |
| `--texture-bleed` | int | `4` | Seam bleed passes for canonical atlas baking. |

```powershell
python dg_face_pipeline\run_full_pipeline.py --subject winona
```

Canonical UV one-shot:

```powershell
python dg_face_pipeline\run_full_pipeline.py --subject winona --canonical dg_face_pipeline\canonical_face_model_v002.obj --subdivisions 0 --uv-mode canonical --texture-size 1024
```

Outputs go under `outputs\characters\<subject>\data\` and
`outputs\characters\<subject>\renders\`.

## face_proportion_analyzer.py
Step A + B. Photo → JSON with Loomis ratios, face bbox, head pose.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--image` | Path | `examples/winona_ref.png` | Source portrait. |
| `--out`   | Path | `outputs/data/face_proportions.json` | Where to write the JSON. |

## step_c_apply_modifiers.py
Step C. JSON → morphed MH base OBJ.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--proportions` | Path | `outputs/data/face_proportions.json` | Analyzer output JSON. |
| `--base_obj`    | Path | `../makehuman/data/3dobjs/base.obj` | MH baseline mesh. |
| `--out_obj`     | Path | `outputs/data/morphed.obj` | Morphed output. |
| `--amplify`     | float | `1.0` | Multiplier on modifier values (clamped to ±1 per modifier inside the analyzer). |

## learned_face_mesh.py
Photo → MediaPipe-FaceMesh OBJ with UVs, true geometric normals, optional pose-neutralization and boundary closure.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--image` | Path | `examples/winona_ref.png` | Source portrait. |
| `--out`   | Path | `outputs/data/subject_face_mesh.obj` | Output OBJ. Vertex idx == UV idx == normal idx. |
| `--subdivisions` | int | `0` | Midpoint-subdivision passes (triangle path only). |
| `--canonical` | Path | `canonical_face_model.obj` | Canonical topology OBJ. Use `canonical_face_model_v002.obj` for the local quad-dominant experiment. |
| `--neutralize-pose` / `--no-neutralize-pose` | flag | on | Undo the photo's yaw + roll via `cv2.solvePnP` so the mesh is Z-forward neutral (production rig orientation). Pitch is **intentionally left alone** because PnP without real camera intrinsics overshoots pitch -- supply real intrinsics to safely include it. |
| `--write-normals` / `--no-write-normals` | flag | on | Compute true per-vertex geometric normals from the actual mesh topology and emit `vn` lines + `v/vt/vn` face refs. |
| `--front-upright` / `--no-front-upright` | flag | on | Final Maya-facing guard that rotates the mesh in front-view XY space from a weighted face-roll solve: outer eyes, inner eyes, nose-to-chin, and forehead-to-chin. |
| `--center-origin` / `--no-center-origin` | flag | on | Center the final closed mesh bounding box at world origin before writing the OBJ. |
| `--close-boundary` / `--no-close-boundary` | flag | on | Close the open face-mask boundary by extruding the `FACEMESH_FACE_OVAL` ring backward + capping with a triangle fan. **Required** for clean Catmull-Clark subdivision in Maya/Arnold -- open meshes collapse inward with each CC level. With closure: 468 v / 508 f → 505 v / 580 f. |
| `--uv-mode` | `photo` / `canonical` | `photo` | `photo` preserves the legacy per-photo landmark UVs. `canonical` reorders the canonical OBJ `vt` records into vertex-index space and bakes the portrait into that stable atlas. |
| `--canonical-texture` | Path | `<out_stem>_canonical_albedo.png` | Output PNG for the baked canonical-atlas albedo when `--uv-mode canonical` is used. |
| `--canonical-uv-mask` | Path | `<out_stem>_canonical_uv_occupancy.png` | Diagnostic UV occupancy mask for the canonical atlas bake. |
| `--texture-size` | int | `1024` | Square atlas size in pixels for the canonical bake. |
| `--texture-bleed` | int | `4` | One-pixel dilation passes to bleed albedo into empty seam pixels after occupancy is recorded. |

When using a quad-dominant canonical topology, keep `--subdivisions 0` so the
output OBJ preserves quads for Maya/Arnold Catmull-Clark subdivision.

Canonical UV mode keeps vertex, UV, and normal indices aligned in the generated
OBJ. The helper follows the source canonical OBJ's `f v/vt` records first,
because the canonical `vt` list is not authored in vertex-index order.

```powershell
python dg_face_pipeline\learned_face_mesh.py --image dg_face_pipeline\examples\winona\ref.png --out dg_face_pipeline\outputs\characters\winona\data\winona_face_mesh.obj --uv-mode canonical
```

## project_canonical_texture.py
Standalone canonical atlas baker. Input is a source photo, runtime landmark UVs
for that photo, and a canonical OBJ. If `--runtime-uvs` is omitted, the script
runs MediaPipe FaceMesh on `--photo` and uses those runtime UVs directly.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--photo` | Path | `examples/winona_ref.png` | Source portrait to sample. |
| `--canonical` | Path | `canonical_face_model.obj` | Canonical OBJ whose `vt` records define the destination atlas. Triangle and quad faces are supported. |
| `--runtime-uvs` | Path | *(MediaPipe from photo)* | Optional JSON, CSV, NPY, or NPZ array of runtime landmark UVs. |
| `--runtime-uv-convention` | `obj` / `image` | `obj` | `obj` means `v` is bottom-up; `image` means top-down and is flipped on load. |
| `--out` | Path | `outputs/textures/canonical_albedo.png` | Baked canonical-atlas albedo PNG. |
| `--mask` | Path | `<out_stem>_occupancy.png` | Diagnostic raw occupancy mask before seam bleed. |
| `--size` | int | `1024` | Square atlas size in pixels. |
| `--bleed` | int | `4` | One-pixel dilation passes to bleed color into empty seam pixels. |

```powershell
python dg_face_pipeline\project_canonical_texture.py --photo dg_face_pipeline\examples\winona\ref.png --canonical dg_face_pipeline\canonical_face_model.obj --out dg_face_pipeline\outputs\characters\winona\textures\winona_canonical_albedo.png --size 1024
```

## render_face.py
4-panel MH comparison render.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--baseline` | Path | `../makehuman/data/3dobjs/base.obj` | Unmorphed MH base. |
| `--morphed`  | Path | `outputs/data/morphed.obj` | Morphed MH mesh from step C. |
| `--photo`    | Path | `None` | Optional photo to add as leftmost panel. |
| `--proportions` | Path | `outputs/data/face_proportions.json` | Used for face bbox (photo cropping) and head pose. |
| `--out`      | Path | `outputs/renders/mesh_compare.png` | Output PNG. |

## render_learned_mesh.py
3-panel learned-mesh render with projective photo texture.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--photo`       | Path | `examples/winona_ref.png` | Source photo (used both as panel 1 and as projective texture). |
| `--mesh`        | Path | `outputs/data/subject_face_mesh.obj` | Learned-mesh OBJ. |
| `--proportions` | Path | `outputs/data/face_proportions.json` | Used for face bbox to crop the photo panel. |
| `--out`         | Path | `outputs/renders/learned_mesh.png` | Output PNG. |

## render_canonical_report.py
Canon thirds/fifths overlay and identity-delta chart.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--photo`       | Path | `examples/winona_ref.png` | Source photo. |
| `--proportions` | Path | `outputs/data/face_proportions.json` | Analyzer JSON with `canonical_analysis`. |
| `--out`         | Path | `outputs/renders/canon_report.png` | Output PNG. |

## export_maya_arnold_scene.py
Maya / Arnold handoff stage. Writes a Maya-centimeter OBJ and a Maya ASCII
loader scene with an Arnold skin shader. Rerun this when scale, normals,
smoothing, shader, lighting, or camera settings change.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--subject` | str | `winona` | Character slug. |
| `--obj` | Path | `outputs/characters/<subject>/data/<subject>_face_mesh.obj` | Source learned mesh OBJ override. |
| `--texture` | Path | `examples/<subject>/ref.png` | Source photo texture override. |
| `--albedo` | Path | generated texture if present | Albedo/base color JPG override. |
| `--roughness` | Path | generated texture if present | Roughness JPG override, connected as Raw. |
| `--normal` | Path | generated texture if present | Normal JPG override, connected as Raw through tangent-space `bump2d`. |
| `--out` | Path | `outputs/characters/<subject>/maya/<subject>_arnold_skin.ma` | Maya ASCII output path. |
| `--height-cm` | float | `22.0` | Normalize mesh Y height in Maya centimeters. |
| `--arnold-subdiv-type` | str | `catclark` | Arnold subdivision request: `catclark`, `linear`, or `none`. |
| `--arnold-subdiv-iterations` | int | `1` | Arnold render subdivision iterations / Maya render smooth level. |
| `--layout` | str | `three` | Import one centered mesh, or make a three-copy comparison layout. |
| `--front-rotate-y` | float | `-24.0` | Yaw correction so the center mesh faces the render camera. |

```powershell
python dg_face_pipeline\export_maya_arnold_scene.py --subject winona --height-cm 22 --arnold-subdiv-type catclark --layout three --front-rotate-y 0
```

See `docs\maya_handoff.md` for the lighting-template contract and shader map
policy.

## build_maya_arnold_scene.py
Maya 2027 scene builder. Run with Maya's Python (`mayapy.exe`). It opens the
lighting template when available, removes stale generated meshes/shader nodes,
imports the normalized Maya-centimeter OBJ, builds the Arnold skin shader,
assigns albedo/roughness/normal maps with explicit color spaces, lays out the
three meshes, enables Arnold Catmull-Clark subdivision, removes unknown nodes,
and saves a clean Maya ASCII scene.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--subject` | str | `winona` | Character slug. |
| `--obj` | Path | `outputs/characters/<subject>/maya/<subject>_face_mesh_maya_cm.obj` | Maya-normalized OBJ override. |
| `--albedo` | Path | generated albedo JPG | Albedo/base-color texture. |
| `--roughness` | Path | generated roughness JPG if present | Roughness texture, connected as Raw. |
| `--normal` | Path | generated normal JPG if present | Normal texture, connected as Raw through tangent-space `bump2d`. |
| `--template` | Path | subject `lightingscene_v001.mb`, then versioned template | Lighting/camera Maya scene to open before mesh import. |
| `--out` | Path | `outputs/characters/<subject>/maya/<subject>_arnold_skin_clean.ma` | Clean saved Maya scene. |
| `--layout` | str | `three` | Build one centered mesh or a three-copy comparison layout. |
| `--front-rotate-y` | float | `-24.0` | Yaw correction for the center mesh. |
| `--arnold-subdiv-type` | str | `catclark` | Arnold subdivision: `catclark`, `linear`, or `none`. |
| `--arnold-subdiv-iterations` | int | `2` | Arnold per-shape subdivision iterations. |
| `--bake-subdivision-levels` | int | `0` | Bake this many Catmull-Clark levels into real geometry before saving. Use `4` for the current high-res Winona look-dev scene. |
| `--keep-template-meshes` | bool | `false` | Keep meshes already in the lighting template. |

```powershell
mayapy dg_face_pipeline\build_maya_arnold_scene.py --subject winona --arnold-subdiv-type none --arnold-subdiv-iterations 0 --bake-subdivision-levels 4
```

If `mayapy` is not visible in a freshly opened terminal, use the full Maya
2027 path:

```powershell
& "C:\Program Files\Autodesk\Maya2027\bin\mayapy.exe" dg_face_pipeline\build_maya_arnold_scene.py --subject winona
```

## segment_face_regions.py
Post-processor that splits a learned-mesh OBJ into lip / eye / brow / skin
groups using MediaPipe's landmark sets. Writes a parallel `_seg.obj` +
`_seg.mtl` with four materials so downstream Maya / Blender / UE can bind
distinct shaders per region.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--subject` | str | *(one of --subject or --in-obj is required)* | Character slug; resolves OBJ path under `outputs/characters/<slug>/data/`. |
| `--in-obj` | Path | *(see --subject)* | Source face-mesh OBJ (overrides --subject discovery). |
| `--out-obj` | Path | `<in_stem>_seg.obj` next to input | Where to write the segmented OBJ. |
| `--in-place` | flag | off | Overwrite the input OBJ + its MTL. Use with care. |

Typical region face counts on a v002 mesh: `skin=468, lips=16, eye=16, brow=8` (sum 508 matches the v002 face count exactly). Vertex / UV / normal counts are unchanged from input.

```powershell
python dg_face_pipeline\segment_face_regions.py --subject winona_v002
```

## validate_face_topology.py
Dependency-free OBJ topology validator for retopo, bake, and Maya handoff
checks. It accepts one or more OBJ paths, prints a human-readable report, and
optionally writes a machine-readable JSON report. Validation issues are reported
but do not fail the command unless `--strict` is passed.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `objs` | Path[] | *required* | One or more OBJ files to validate. |
| `--json-out` | Path | none | Optional JSON report path. |
| `--strict` | bool | `false` | Exit nonzero when topology issues are present. Parse errors and missing files always exit nonzero. |

Report fields include vertex / UV / normal / face counts, face arity
distribution, invalid vertex / UV / normal references, degenerate faces,
boundary and nonmanifold edge counts, connected component count, and bounding
box min/max.

```powershell
python dg_face_pipeline\validate_face_topology.py dg_face_pipeline\canonical_face_model_v002.obj --json-out outputs\topology_report.json
```

## extract_depth_normal.py
Monocular depth + Sobel-gradient normal extraction via **Depth Anything v2**
(Apache-2.0, Hugging Face). Writes raw depth (`.npy`), viewable depth PNG,
tangent-space normal map PNG, and per-vertex depth sampled at each FaceMesh
landmark's UV -- the last is the key input for multi-source blending.

First run downloads the chosen Hugging Face checkpoint to
`%USERPROFILE%\.cache\huggingface\hub` (Small ~25 MB, Base ~98 MB, Large ~335 MB params).

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--subject` | str | *(one of --subject or --photo required)* | Auto-discovers `examples/<slug>/ref.png` + `outputs/characters/<slug>/data/<slug>_face_mesh.obj` for UV sampling. |
| `--photo` | Path | *(see --subject)* | Explicit photo path. |
| `--obj` | Path | auto from --subject | Face-mesh OBJ for per-vertex UV depth sampling. |
| `--out-dir` | Path | `outputs/characters/<subject>/textures/` | Output folder. |
| `--slug` | str | from --subject | Output filename prefix. |
| `--model-size` | str | `base` | One of `small` / `base` / `large`. Large is sharpest on facial detail but ~3x slower. |
| `--z-scale` | float | `200.0` | Normal-map bump strength. **Lower = stronger relief**. `2.0` produces visible facial features; `0.5` is aggressive. |

Outputs (per subject):
- `<slug>_photo_depth_raw.npy` -- float depth (HxW), arbitrary scale (Depth Anything is relative, not metric)
- `<slug>_photo_depth.png` -- 8-bit normalised depth visualisation
- `<slug>_photo_normal.png` -- tangent-space RGB normal map (OpenGL convention)
- `<slug>_per_vertex_depth.csv` -- `(vertex_idx, u, v, depth)` for each vert (the multi-source-blend input)
- `<slug>_photo_depth_meta.json` -- model id, depth range, device for reproducibility

```powershell
python dg_face_pipeline\extract_depth_normal.py --subject carolyn_lilipaly_v002 --model-size base --z-scale 2.0
```

## face_pipeline.bat
Drag-and-drop entry point. Drop any portrait image onto this `.bat` in
Explorer; it derives a clean slug from the filename (lowercased, non-
alphanumerics → underscores), runs the full pipeline with the v002 canonical
+ `--subdivisions 0`, and writes everything under
`outputs/characters/<slug>/`. No arguments needed -- everything is inferred
from the dropped file.

```text
"Carolyn Lilipaly.jpg"   -> slug "carolyn_lilipaly"
"hugh-griffith.webp"     -> slug "hugh_griffith"
```

## export_mediapipe_scaffold.py
Face/topology stripper for retopo or refit work. Keeps the exact vertex and UV
order from a MediaPipe OBJ, omits all `f` face records, and writes an index
CSV.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--subject` | str | `winona` | Character slug. |
| `--in-obj` | Path | subject data OBJ | Source OBJ override. |
| `--out-obj` | Path | subject data point OBJ | Point-only OBJ output. |
| `--out-csv` | Path | subject data CSV | Vertex/UV index table. |
| `--raw` | bool | `false` | Use `<subject>_mediapipe_raw.obj` defaults instead of dense face mesh. |
| `--no-point-elements` | bool | `false` | Write only `v`/`vt`; default also writes OBJ `p` point records. |

```powershell
python dg_face_pipeline\export_mediapipe_scaffold.py --subject jim_varney --raw
```

## generate_texture_maps.py
Single-photo JPG texture helper. Writes albedo, roughness, and normal maps into
`outputs/characters/<subject>/textures/`, then updates the subject OBJ `.mtl`
files to reference them.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--subject` | str | `winona` | Character slug. |
| `--photo` | Path | `examples/<subject>/ref.png` | Source photo override. |

```powershell
python dg_face_pipeline\generate_texture_maps.py --subject jim_varney
```

## bake_maps.py
Abstract texture-baking adapter. In v1, `--backend manifest` is fully
implemented as a dry-run/report writer; `maya` and `blender` are contract stubs
that write an adapter report and fail with actionable setup/implementation
messages.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--low` | Path | *required* | Low-resolution retopo mesh that receives baked maps. |
| `--high` | Path | *required* | High-resolution source mesh used for projection. |
| `--out-dir` | Path | *required* | Directory for the report and declared map outputs. |
| `--resolution` | int | `2048` | Square output texture size in pixels. |
| `--cage-distance` | float | `0.02` | Uniform cage/projection distance in scene units. |
| `--backend` | str | `manifest` | One of `manifest`, `maya`, or `blender`. |
| `--maps` | str | `normal, ambient_occlusion, curvature` | Optional repeated or comma-separated requested map names. |
| `--report` | Path | `<out-dir>/bake_manifest.json` | Optional report JSON override. |

```powershell
python dg_face_pipeline\bake_maps.py --backend manifest --low outputs\characters\winona\data\winona_retopo.obj --high outputs\characters\winona\data\winona_face_mesh.obj --out-dir outputs\characters\winona\textures --resolution 2048 --cage-distance 0.02
```

See `docs\baking_adapter_contract.md` for the adapter contract and the planned
Maya/Blender command shapes.

## texturing_xyz_manifest.py
External-only TexturingXYZ package inventory. Scans an external source root and
writes metadata-only JSON under generated outputs by default. It records file
paths, sizes, suffixes, map type guesses, UDIM tile guesses, color-space hints,
and `generated_at`; it does not copy source assets.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--root` | Path | `B:\RESOURCES\TexturingXYZ\vFace_030` | External TexturingXYZ package root to scan. |
| `--out` | Path | `outputs/texturing_xyz/vFace_030_manifest.json` | Manifest JSON output path. |

```powershell
python dg_face_pipeline\texturing_xyz_manifest.py --root B:\RESOURCES\TexturingXYZ\vFace_030 --out dg_face_pipeline\outputs\texturing_xyz\vFace_030_manifest.json
```

## Exit codes
All scripts use the same convention:

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Runtime error during processing (bad OBJ, no face detected, etc.). |
| `2` | Missing input file. |
| `3` | Topology validation issues found when `validate_face_topology.py --strict` is used. |
