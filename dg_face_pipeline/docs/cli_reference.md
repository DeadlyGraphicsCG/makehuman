# CLI Reference

All paths default to pipeline-relative locations (`examples/`, `outputs/`,
`../makehuman/` for the MH data files). Override any of them with explicit
flags. Run any script with `--help` for the live argparse-generated help.

## run_full_pipeline.py
The one-shot orchestrator. Runs analyzer → step C → learned mesh → both
renders → final stacked frame.

| Flag | Type | Default | Purpose |
|------|------|---------|---------|
| `--image` | Path | *required* | Source portrait PNG / JPG. |
| `--subject` | str | *required* | Short slug (e.g. `winona`); used to name all outputs. |
| `--subdivisions` | int | `2` | Mesh-densification passes. `0` = 478 verts, `1` = 1843, `2` = 7267, `3` = 28867. |
| `--amplify` | float | `1.5` | Multiplier on MH modifier values. Above ~2.5 begins distorting MH geometry. |

```powershell
python run_full_pipeline.py --image examples\winona_ref.png --subject winona
```

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

## Exit codes
All scripts use the same convention:

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Runtime error during processing (bad OBJ, no face detected, etc.). |
| `2` | Missing input file. |
