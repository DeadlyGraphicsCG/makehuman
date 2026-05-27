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

```powershell
python dg_face_pipeline\run_full_pipeline.py --subject winona
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
Photo → MediaPipe-FaceMesh OBJ with UVs.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--image` | Path | `examples/winona_ref.png` | Source portrait. |
| `--out`   | Path | `outputs/data/subject_face_mesh.obj` | Output OBJ. Vertex idx == UV idx. |
| `--subdivisions` | int | `0` | Midpoint-subdivision passes. |
| `--canonical` | Path | `canonical_face_model.obj` | Canonical topology OBJ. Use `canonical_face_model_v002.obj` for the local quad-dominant experiment. |

When using a quad-dominant canonical topology, keep `--subdivisions 0` so the
output OBJ preserves quads for Maya/Arnold Catmull-Clark subdivision.

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

## Exit codes
All scripts use the same convention:

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Runtime error during processing (bad OBJ, no face detected, etc.). |
| `2` | Missing input file. |
