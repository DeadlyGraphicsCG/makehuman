# Architecture — DG Face Pipeline

## Data flow

```
                          ┌─────────────────────┐
                          │  source portrait    │   examples/<name>.png
                          │  (any aspect ratio) │
                          └──────────┬──────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
                ▼                    ▼                    ▼
    ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
    │ face_proportion_   │ │ learned_face_mesh  │ │ (the photo itself, │
    │ analyzer.py        │ │ .py                │ │  passed through for│
    │                    │ │                    │ │  texture sampling) │
    │ - MediaPipe Face   │ │ - MediaPipe Face   │ │                    │
    │   Mesh, 478 lms    │ │   Mesh, 478 lms    │ │                    │
    │ - cv2.solvePnP →   │ │ - canonical tri-   │ │                    │
    │   yaw/pitch/roll   │ │   angulation       │ │                    │
    │ - Loomis ratios    │ │ - subdivide N×     │ │                    │
    │   (W_eye = 1.0)    │ │ - UVs from lm.x,y  │ │                    │
    │ - face bbox (px)   │ │ - flip-y, scale    │ │                    │
    └─────────┬──────────┘ └─────────┬──────────┘ └────────────────────┘
              │                      │
              ▼                      ▼
    outputs/data/                outputs/data/
    <subj>_face_proportions.json <subj>_face_mesh.obj
              │                      │
              ▼                      ▼
    ┌────────────────────┐  ┌──────────────────────┐
    │ step_c_apply_      │  │ render_learned_mesh  │  ← uses original photo
    │ modifiers.py       │  │ .py                  │    as projective texture
    │                    │  │                      │
    │ - load MH base.obj │  │ - per-tri photo      │
    │   (19158 verts)    │  │   sampling at UV     │
    │ - resolve modifier │  │   centroid           │
    │   path → .target   │  │ - dampened Lambert   │
    │ - weight × delta   │  │ - 3-panel: photo /   │
    │ - apply to verts   │  │   frontal / 3/4 yaw  │
    └─────────┬──────────┘  └─────────┬────────────┘
              │                       │
              ▼                       ▼
    outputs/data/                 outputs/renders/
    <subj>_morphed.obj            <subj>_learned_mesh.png
              │
              ▼
    ┌────────────────────┐
    │ render_face.py     │  ← uses photo + bbox + pose from analyzer JSON
    │                    │
    │ - 4-panel:         │
    │   photo / base /   │
    │   morphed / drift  │
    │   heatmap          │
    │ - body-only filter │
    │ - clamped head     │
    │   pose rotation    │
    │ - shared bbox      │
    └─────────┬──────────┘
              │
              ▼
    outputs/renders/<subj>_mh_compare.png
              │
              │       ┌────────────────────────────────────────┐
              └──────►│ run_full_pipeline.py :: stack_renders()│
                      └────────────────────┬───────────────────┘
                                           │
                                           ▼
                            outputs/renders/<subj>_full_pipeline.png
```

## Per-script responsibilities

### face_proportion_analyzer.py
Reads a portrait, runs MediaPipe FaceMesh (`refine_landmarks=True` for iris),
and produces a JSON containing:

- `subject_ratios` — 10 dimensionless ratios in W_eye units (Loomis canon).
- `baseline_ratios` — the canonical Loomis values for each ratio.
- `delta_ratios` — `subject - baseline`.
- `modifier_targets` — suggested MH modifier value list, mapped via
  `MODIFIER_MAP`, each entry includes ratio key, modifier path
  (`<group>/<target>-<min_kw>|<max_kw>`), proposed value, and ratio context.
- `face_bbox` — `[xmin, ymin, xmax, ymax]` of all 478 landmarks in
  original-image pixel coords (used downstream for photo cropping).
- `head_pose_deg` — `[yaw, pitch, roll]` in degrees from a 6-point
  `cv2.solvePnP` solve using a canonical 3D face model in image-coord
  conventions (chin +y, eyes -y, depth +z behind the nose).
- `eye_width_px` — the W_eye scalar in pixels, for forensic reproducibility.
- `notes` — human-readable context line.

### step_c_apply_modifiers.py
Reads the analyzer JSON and the MakeHuman modifier registry
(`makehuman/data/modifiers/modeling_modifiers.json`), resolves each modifier
path to one or two concrete `.target` files in `makehuman/data/targets/`,
loads the per-vertex `(idx, dx, dy, dz)` rows, and applies
`weight = abs(value) * amplify` to MH base.obj's 19,158 vertices. Writes a
new OBJ that copies every non-vertex line from the base verbatim so the face
indices, groups, and topology survive intact.

### learned_face_mesh.py
Runs MediaPipe FaceMesh on the same image, but instead of computing
proportions it builds an OBJ:

1. Vertices are MediaPipe's 478 landmark positions, scaled to image
   pixel space (x · w, (1 − y) · h, −z · w). y is flipped so the OBJ is
   y-up.
2. UVs are `(lm.x, 1 − lm.y)` — origin bottom-left, OBJ convention.
3. Triangulation comes from `canonical_face_model.obj` (898 triangles,
   898 if `subdivisions=0`).
4. With `--subdivisions N`, each pass midpoint-subdivides every triangle
   into 4. Edge midpoints are shared via a cache so the mesh stays
   watertight. UVs interpolate linearly (correct for projective texture).

### render_face.py
The MH-modifier-path renderer. Four panels (photo / baseline / morphed /
drift heatmap), all four rotated to the analyzer-estimated head pose so the
visual comparison is at the same viewing angle. Pose magnitudes are clamped
(`pitch ±10°`, `yaw/roll ±30°`) because `solvePnP` with our placeholder
focal-length intrinsics overestimates pitch.

Body-only filter (`g body` group, ~13,378 faces) drops MH's helper-* and
joint-* groups (hair-fit proxies, eyelashes, teeth, joint markers) so the
face is clean.

### render_learned_mesh.py
The learned-mesh renderer. Three panels (photo / frontal textured /
3/4 yaw +25° textured). Texture sampling: each triangle picks the source-photo
RGB at its centroid UV. Lambert shading is dampened (`shade_strength=0.55`)
because the photo already contains baked-in lighting from the original
exposure.

### run_full_pipeline.py
Imports every other script as a Python module (no `subprocess`), runs them
in sequence with per-subject output naming, then concatenates the MH and
learned PNGs vertically into a single combined frame.

## Design decisions

### Why MediaPipe and not MICA / PIXEL3DMM / DECA / EMOCA?
The first-pass goal was likeness fidelity quickly. MICA and PIXEL3DMM both
require a FLAME license (manual registration at
`flame.is.tue.mpg.de`, email-gated), and PIXEL3DMM additionally pulls
`pytorch3d` which is painful on Windows. MediaPipe FaceMesh is a learned
3D face shape predictor that happens to ship as part of a library we
already had installed for the landmark-extraction step. Its mesh is
sparser (478 vs ~5000) but completely unencumbered and zero-install.

The trade-off: no scalp, no ears, no back of head, no neck. Just the face
mask. For the immediate goal ("show me a 3D Winona from a photo") that is
fine; for production-quality character generation, swap in MICA when you
have the FLAME license sorted. See `extending.md` for the swap pattern.

### Why also keep the MakeHuman-modifier path?
Two reasons.

1. The MH path produces output in MH topology (19,158 verts, the full
   body, rigged-ready), which is what downstream tools in the DG pipeline
   actually consume. The learned mesh is great for visualisation and for
   "is this the right person" verification, but isn't MH-compatible.
2. The Loomis ratios computed by the analyzer are useful artefacts in
   their own right — even if you never apply them to MH, they're a
   compact, dimensionless description of a face's proportions, and the
   delta-from-canonical signal is interpretable.

### Why subdivision instead of per-vertex barycentric texture sampling?
Both produce a similar visual result. Subdivision was simpler to wire in
because it leaves the existing per-triangle flat-sampling renderer
unchanged — each triangle shrinks until it covers ~1 pixel and the
flat-fill becomes effectively per-pixel. Barycentric in matplotlib
needs custom triangle rasterisation, which is more code for the same
output.

### Why is `head_pose_deg.pitch` clamped to ±10°?
`cv2.solvePnP` with no real camera intrinsics (we pass `focal = image_w`
as a placeholder) overestimates pitch substantially — on a true frontal
shot it can produce `pitch ≈ −23°`. The other two angles are more stable.
The clamp keeps the rendered head from tipping unrealistically forward
while still letting the roll match the photo's head tilt. To remove the
clamp, supply real intrinsics (focal length, principal point) at the
analyzer level.

### Why 9 MH modifiers and not 50?
The current 9 cover broad scale axes (head/forehead/nose/mouth horiz +
vert, mouth y-translate, eye spacing). They demonstrate the path
end-to-end but cannot reach actual likeness. Adding more (chin-width,
cheek-bones, eye-height, head-oval, etc.) is a config-only change — see
`extending.md` for the recipe. The reason it's still 9 is that the
learned-mesh path already nails likeness, so investing in more MH
modifiers is only worthwhile if the downstream consumer specifically
needs MH topology.
