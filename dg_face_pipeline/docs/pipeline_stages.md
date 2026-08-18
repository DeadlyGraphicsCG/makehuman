# Pipeline Stages

The face pipeline is staged. Do not rerun every step when only a downstream
asset changes.

## Stage Map

| Stage | Command | Main outputs | Rerun when |
|-------|---------|--------------|------------|
| Analyze | `python dg_face_pipeline\face_proportion_analyzer.py --image <ref> --out <json>` | `<subject>_face_proportions.json` | Landmark choice, canon rules, ratios, face bbox, or head-pose logic changes. |
| Mesh | `python dg_face_pipeline\learned_face_mesh.py --image <ref> --out <obj> --subdivisions 2` | `<subject>_face_mesh.obj` | MediaPipe mesh generation, canonical topology, subdivisions, UV mode, or source photo changes. |
| Canonical UV atlas | `python dg_face_pipeline\project_canonical_texture.py --photo <ref> --canonical <canonical.obj> --out <albedo.png>` | canonical albedo PNG + occupancy mask | Reproject the portrait when the source photo, runtime landmark UVs, canonical OBJ `vt` records, or atlas resolution changes. |
| MediaPipe scaffold | `python dg_face_pipeline\export_mediapipe_scaffold.py --subject <subject>` | point-only OBJ + CSV | Retopo/refit scaffold needed without face records. |
| MakeHuman morph | `python dg_face_pipeline\step_c_apply_modifiers.py --proportions <json> --out_obj <obj>` | `<subject>_morphed.obj` | Modifier mapping, MakeHuman target application, amplify value, or analysis JSON changes. |
| Debug renders | `python dg_face_pipeline\run_full_pipeline.py --subject <subject>` | canon/MH/learned/full PNGs | Render style changes or any upstream analyze/mesh/MH output changes. |
| Maya OBJ handoff | `python dg_face_pipeline\export_maya_arnold_scene.py --subject <subject> --layout three --front-rotate-y 0` | Maya cm-scale OBJ + loader `.ma` scene | Maya scale, OBJ normals, basic shader settings, or loader-scene settings change. |
| Clean Maya scene | `mayapy dg_face_pipeline\build_maya_arnold_scene.py --subject <subject>` | `<subject>_arnold_skin_clean.ma` | Lighting template, clean Arnold shader network, mesh layout, baked/render subdivision, or final Maya scene cleanup changes. |

`run_full_pipeline.py` is the convenient all-up command. It runs analyze,
MakeHuman morph, learned mesh, canon report, MH render, learned render, and
the stacked PNG. It does **not** need to run when only the Maya handoff is
being tuned.

## Character Layout

Each character should live under the same shape:

```text
dg_face_pipeline/
├── examples/
│   └── <subject>/
│       └── ref.png
└── outputs/
    └── characters/
        └── <subject>/
            ├── data/
            │   ├── <subject>_face_proportions.json
            │   ├── <subject>_face_mesh.obj
            │   └── <subject>_morphed.obj
            ├── renders/
            │   ├── <subject>_canon_report.png
            │   ├── <subject>_mh_compare.png
            │   ├── <subject>_learned_mesh.png
            │   └── <subject>_full_pipeline.png
            └── maya/
                ├── lightingscene_v001.mb
                ├── <subject>_face_mesh_maya_cm.obj
                ├── <subject>_arnold_skin.ma
                └── <subject>_arnold_skin_clean.ma
```

The `outputs/` tree is local/generated and gitignored. If a lighting template
becomes canonical, copy it into a versioned template location before relying on
it across machines.

## Practical Regeneration Rules

- If the photo changes, rerun the full pipeline and then the Maya handoff.
- If canon rules or landmarks change, rerun the full pipeline and then the
  Maya handoff.
- If OBJ scale or exported vertex normals change, rerun
  `export_maya_arnold_scene.py`, then `build_maya_arnold_scene.py`.
- If only Arnold shader settings, lighting-template cleanup, Catmull-Clark
  settings, or the three-copy Maya layout changes, rerun only
  `build_maya_arnold_scene.py`.
- If only the report PNG layout changes, rerun the render/report stage, not
  MediaPipe.
- If only the MakeHuman modifier mapping changes, rerun analysis if ratio
  names changed; otherwise rerun MakeHuman morph and renders.

## Current Gap

The Maya handoff is now two-stage. `export_maya_arnold_scene.py` prepares the
centimeter-scale OBJ and a fallback loader `.ma`. `build_maya_arnold_scene.py`
is the preferred look-dev push: it runs inside Maya 2027, opens the lighting
template, removes stale generated mesh/shader clutter, imports the subject OBJ,
creates one clean Arnold skin shader, assigns the three generated JPG maps with
explicit color spaces, creates the three comparison meshes, applies soft edges,
and either requests Arnold Catmull-Clark subdivision or bakes Catmull-Clark
into real geometry with `--bake-subdivision-levels`, removes unknown nodes, and saves
`<subject>_arnold_skin_clean.ma`.

For the local quad-dominant MediaPipe test, generate the learned mesh with:

```powershell
python dg_face_pipeline\learned_face_mesh.py --image <ref> --out <obj> --canonical dg_face_pipeline\canonical_face_model_v002.obj --subdivisions 0
```

Keeping `--subdivisions 0` preserves the v002 face sizes so Maya can do the
Catmull-Clark smoothing.

## Retopo / UV / Bake Contract

The branch-level contract for quad topology work, canonical UV reprojection,
adapter-based baking, and external-only TexturingXYZ integration lives in
`retopo_texture_bake_contract.md`. Follow that contract when adding new mesh
or texture stages so parallel agent work stays compatible.
