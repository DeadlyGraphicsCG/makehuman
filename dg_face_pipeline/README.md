# DG Face Pipeline

End-to-end Python pipeline for `portrait photograph → 3D face mesh`, with two
parallel back-ends:

1. **Learned mesh** (recommended for likeness) — MediaPipe FaceMesh predicts
   478 3D landmarks from the photo, combined with the canonical face model's
   triangulation and optional midpoint subdivision. Output: a self-contained
   OBJ + UV-keyed projective texture from the source photo. ~3 seconds end-to-end
   on a CPU. No GPU, no FLAME/BFM license, no extra installs beyond
   `mediapipe + opencv-python + numpy + matplotlib`.

2. **MakeHuman-modifier path** (recommended for downstream MH rigging /
   animation) — extracts dimensionless Loomis / O'Reilly proportions from the
   photo, maps them onto nine MakeHuman shape modifiers, applies the weighted
   `.target` displacements to MH's 19,158-vertex base mesh, and emits a
   morphed body OBJ. The morph is honest but limited by how few modifiers we
   drive; the learned mesh is the one that looks like the subject.

The two back-ends run side-by-side in `run_full_pipeline.py` and produce a
stacked comparison frame so you can see them together.

```
photo --> [analyzer] --> proportions JSON --> [step C] --> morphed MH OBJ
                              |
                              +----------> [learned_face_mesh] --> dense OBJ + UVs
                              |
                              +----------> [render_face]        --> MH 4-panel PNG
                              |
                              +----------> [render_learned_mesh] --> learned 3-panel PNG
                                                                       |
                                                                       v
                                                              [stack_renders]
                                                                       |
                                                                       v
                                                          <subject>_full_pipeline.png
```

## Example output

The combined frame for `examples/winona_ref.png`:

![winona pipeline output](docs/images/winona_full_pipeline.png)

Top half is the MakeHuman-modifier path (photo / baseline / Loomis-morphed /
drift heatmap). Bottom half is the learned-mesh path (photo / frontal
textured / 3/4 textured). Same data; two completely different mesh sources.

## Quickstart

### Prerequisites
- Windows 11 (only platform tested; nothing platform-specific in the code
  but `cv2.solvePnP` and `mediapipe` install cleanly on Win via pip)
- Python 3.10 or 3.11 (MediaPipe 0.10.x ships wheels for both)
- A local MakeHuman checkout at `..\makehuman\` (sibling to this directory)
  for the MH-modifier back-end. The learned-mesh back-end has no MH dependency.

### Install
```powershell
python -m pip install --upgrade pip
python -m pip install mediapipe==0.10.18 opencv-python==4.10.0.84 numpy==1.26.4 matplotlib==3.9.2
```

### Run the full pipeline
```powershell
python dg_face_pipeline\run_full_pipeline.py `
    --image dg_face_pipeline\examples\winona_ref.png `
    --subject winona
```

Outputs land at:
- `dg_face_pipeline\outputs\data\winona_face_proportions.json`
- `dg_face_pipeline\outputs\data\winona_morphed.obj`        (MH-modifier path)
- `dg_face_pipeline\outputs\data\winona_face_mesh.obj`      (learned, with UVs)
- `dg_face_pipeline\outputs\renders\winona_mh_compare.png`
- `dg_face_pipeline\outputs\renders\winona_learned_mesh.png`
- `dg_face_pipeline\outputs\renders\winona_full_pipeline.png`  ← combined

### Run only one back-end

Learned mesh + textured render (recommended for "show me the face"):
```powershell
python dg_face_pipeline\learned_face_mesh.py     --image examples\winona_ref.png --out outputs\data\winona_face_mesh.obj --subdivisions 2
python dg_face_pipeline\render_learned_mesh.py   --photo examples\winona_ref.png --mesh outputs\data\winona_face_mesh.obj --out outputs\renders\learned.png
```

MakeHuman-modifier path only:
```powershell
python dg_face_pipeline\face_proportion_analyzer.py --image examples\winona_ref.png --out outputs\data\proportions.json
python dg_face_pipeline\step_c_apply_modifiers.py   --proportions outputs\data\proportions.json --out_obj outputs\data\morphed.obj
python dg_face_pipeline\render_face.py              --photo examples\winona_ref.png --proportions outputs\data\proportions.json --morphed outputs\data\morphed.obj --out outputs\renders\mh.png
```

## Directory layout

```
dg_face_pipeline/
├── README.md                         <- you are here
├── face_proportion_analyzer.py        <- Step A+B: photo → ratios + pose JSON
├── step_c_apply_modifiers.py          <- Step C: ratios → morphed MH OBJ
├── learned_face_mesh.py               <- photo → learned dense OBJ + UVs
├── render_face.py                     <- 4-panel MH-modifier comparison render
├── render_learned_mesh.py             <- 3-panel learned-mesh + projective tex
├── run_full_pipeline.py               <- orchestrator (runs all five)
├── canonical_face_model.obj           <- vendored from google/mediapipe (Apache 2.0)
├── examples/                          <- committed input portraits
│   ├── winona_ref.png
│   └── subject_b_ref.png
├── outputs/                           <- generated artifacts (gitignored)
│   ├── data/                          <-   JSON + OBJ
│   └── renders/                       <-   PNG comparison frames
└── docs/
    ├── architecture.md                <- data flow + design decisions
    ├── cli_reference.md               <- every script flag
    ├── extending.md                   <- swap in MICA / add modifiers / new exporters
    ├── modifier_mapping.md            <- Loomis ratio ↔ MH modifier reference
    └── images/                        <- doc-embedded example frames
```

## Repo context

This pipeline lives **inside** a fork of MakeHuman Community
(<https://github.com/makehumancommunity/makehuman>) at
`makehumancommunity/makehuman` → `DeadlyGraphicsCG/makehuman`. The fork's
`feat/face-proportion-pipeline` branch is the active development branch.

The pipeline is **orchestrated by the DG_Brain hub**
(`C:\AI\apps\DG_Brain\`) — that's where production artifacts and per-subject
configurations live in the wider DG production stack. This `dg_face_pipeline/`
folder is the self-contained execution surface; DG_Brain is the conductor.

## Licensing

- Pipeline code (this folder, except `canonical_face_model.obj`):
  inherits the parent repository's AGPL-3.0 from MakeHuman Community.
- `canonical_face_model.obj`: Apache-2.0, vendored verbatim from
  `google/mediapipe`, source at
  `mediapipe/modules/face_geometry/data/canonical_face_model.obj`.
- Example portraits (`examples/`): third-party images, included for
  pipeline-demonstration purposes only. Replace with your own portraits for
  production use.
- MediaPipe (Python package) is Apache-2.0; OpenCV is Apache-2.0; matplotlib
  is PSF-style.

If commercial use is a constraint, the MakeHuman-modifier path is
AGPL-3.0-bound (because the MH `.target` files it reads are AGPL); the
learned-mesh path is fully Apache-2.0 in its dependency chain. See
`docs/extending.md` for how to detach the learned-mesh half from the MH fork
if needed.

## See also

- [docs/architecture.md](docs/architecture.md) — full data-flow walk-through
- [docs/cli_reference.md](docs/cli_reference.md) — every flag, every script
- [docs/extending.md](docs/extending.md) — adding modifiers, swapping
  reconstructor (MICA / PIXEL3DMM / FLAME), wiring back into MakeHuman
- [docs/modifier_mapping.md](docs/modifier_mapping.md) — which MH modifier
  each Loomis ratio drives, with rationale
