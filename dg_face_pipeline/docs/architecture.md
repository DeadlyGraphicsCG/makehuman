# Architecture — DG Face Pipeline

## Data flow

When orchestrated by `run_full_pipeline.py`, all generated artefacts land under
`outputs/characters/<subject>/{data,renders,maya,textures}/`. When the
individual scripts are run standalone they default to the flatter
`outputs/{data,renders}/` layout (see `cli_reference.md` for each script's
defaults).

```
                          ┌─────────────────────┐
                          │  source portrait    │   examples/<subject>/ref.png
                          │  (any aspect ratio) │   (legacy: examples/<name>_ref.png)
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
    │ - cv2.solvePnP →   │ │ - canonical OBJ    │ │                    │
    │   yaw/pitch/roll   │ │   topology (v1 tri │ │                    │
    │ - Loomis ratios    │ │   or v002 quads)   │ │                    │
    │   (W_eye = 1.0)    │ │ - subdivide N×     │ │                    │
    │ - canonical thirds │ │   (tri path only)  │ │                    │
    │   / fifths report  │ │ - UVs from lm.x,y  │ │                    │
    │ - face bbox (px)   │ │ - flip-y, scale    │ │                    │
    └─────────┬──────────┘ └─────────┬──────────┘ └────────────────────┘
              │                      │
              ▼                      ▼
    .../data/                    .../data/
    <subj>_face_proportions.json <subj>_face_mesh.obj
              │                      │
              ├──────────────────────┤
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
    .../data/                     .../renders/
    <subj>_morphed.obj            <subj>_learned_mesh.png
              │                       │
              ▼                       ▼
    ┌────────────────────┐  ┌──────────────────────┐
    │ render_face.py     │  │ render_canonical_    │  ← thirds/fifths
    │                    │  │ report.py            │    scaffold + delta chart
    │ - 4-panel:         │  │                      │
    │   photo / base /   │  │ - photo overlay      │
    │   morphed / drift  │  │ - subject vs canon   │
    │   heatmap          │  │   delta bars         │
    │ - body-only filter │  │                      │
    │ - clamped head     │  │                      │
    │   pose rotation    │  │                      │
    └─────────┬──────────┘  └─────────┬────────────┘
              │                       │
              ▼                       ▼
    .../renders/                  .../renders/
    <subj>_mh_compare.png         <subj>_canon_report.png
              │
              │       ┌────────────────────────────────────────┐
              └──────►│ run_full_pipeline.py :: stack_renders()│
                      └────────────────────┬───────────────────┘
                                           │
                                           ▼
                         .../renders/<subj>_full_pipeline.png

      ── separate Maya / Arnold handoff stage (manual, not in run_full_pipeline) ──

    .../data/<subj>_face_mesh.obj
              │
              ▼
    ┌────────────────────────────┐    ┌────────────────────────────┐
    │ export_maya_arnold_scene   │    │ generate_texture_maps.py   │
    │ .py                        │    │ - photo → albedo / rough / │
    │ - normalize Y to cm        │    │   normal JPGs              │
    │ - write Maya-cm OBJ        │    │ - rewrites OBJ .mtl refs   │
    │ - write fallback Arnold .ma│    └────────────────────────────┘
    └─────────┬──────────────────┘
              │
              ▼
    ┌────────────────────────────┐
    │ build_maya_arnold_scene.py │  ← run with mayapy.exe (Maya 2027)
    │                            │
    │ - open lighting template   │
    │ - import normalized OBJ    │
    │ - layout (one / three)     │
    │ - aiStandardSurface +      │
    │   aiColorCorrect skin grade│
    │ - Arnold subdiv (catclark) │
    │ - bake-subdivision-levels  │
    │   optional (live x4 etc.)  │
    └─────────┬──────────────────┘
              │
              ▼
    .../maya/<subj>_arnold_skin_clean.ma
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
3. Topology comes from `--canonical <obj>`, defaulting to the vendored
   `canonical_face_model.obj` (Google MediaPipe, 468 verts, 898
   triangles). The pipeline also ships
   `canonical_face_model_v002.obj`, a local quad-dominant retopology
   (468 verts, 508 faces — 390 quads + 118 triangles) that preserves
   MediaPipe's vertex/UV index space but gives Catmull-Clark a much
   better starting point. When the canonical mesh has 468 verts but
   MediaPipe returns 478 (refine_landmarks=True), the trailing 10
   iris landmarks are dropped before writing.
4. The OBJ writer preserves whatever face arity the canonical uses
   (triangles, quads, or n-gons); it does not force triangulation.
5. With `--subdivisions N` (triangle path only), each pass midpoint-
   subdivides every triangle into 4. Edge midpoints are shared via a
   cache so the mesh stays watertight; UVs interpolate linearly. Use
   `--subdivisions 0` with the v002 canonical to keep authored quads
   intact for Maya/Arnold Catmull-Clark.

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

### render_canonical_report.py
Renders the canonical proportionality report — a thirds-and-fifths
scaffold overlaid on the source photograph, plus a bar chart of subject-
ratio deltas against the Loomis baselines. Reads the
`canonical_analysis` block from the analyzer JSON; writes a single PNG.

### export_mediapipe_scaffold.py
Stripped-down OBJ + CSV exporter used for retopo / refit work. Keeps the
exact vertex and UV order from a MediaPipe OBJ, drops the `f` records
(point-only by default; `--raw` switches to the raw MediaPipe predicted
mesh as input). Useful when the next step is a hand- or tool-assisted
retopology that needs the original index space.

### generate_texture_maps.py
Single-photo texture helper. Reads `examples/<subject>/ref.png` and
writes `albedo.jpg`, `roughness.jpg`, and `normal.jpg` into
`outputs/characters/<subject>/textures/`, then rewrites the subject's
OBJ `.mtl` files to reference them. Pragmatic, not photogrammetry — but
enough to drive the Arnold skin shader for look-dev.

Note: the `normal.jpg` here is a Sobel gradient of photo luminance — a
fake bump derived from pixel brightness, not real surface geometry. For
true geometric normals see `extract_depth_normal.py`.

### segment_face_regions.py
Post-processor that splits a learned-mesh OBJ into lip / eye / brow / skin
groups using MediaPipe's `FACEMESH_LIPS`, `FACEMESH_LEFT_EYE`,
`FACEMESH_RIGHT_EYE`, and `FACEMESH_*_EYEBROW` landmark sets. A face is
assigned to a region if a majority of its verts belong to that region's
index set; ties resolve toward the more specific region (lips > eye >
brow > skin). Output is a parallel `_seg.obj` + `_seg.mtl` with four
materials so Maya / Blender / UE can bind distinct shaders per region.
Geometry is unchanged from input.

### extract_depth_normal.py
Monocular depth extraction via **Depth Anything v2** (Apache-2.0, Hugging
Face). Loads the chosen checkpoint (Small / Base / Large), runs inference
on the photo, then writes:

1. `<slug>_photo_depth_raw.npy` — float depth array (HxW, arbitrary scale)
2. `<slug>_photo_depth.png` — 8-bit normalised depth visualisation
3. `<slug>_photo_normal.png` — tangent-space normal map from Sobel of depth
4. `<slug>_per_vertex_depth.csv` — depth sampled at each FaceMesh
   landmark's UV coordinate (505 rows for a closed v002 mesh)
5. `<slug>_photo_depth_meta.json` — model id + depth range + device

The per-vertex CSV is the input format for the (future) multi-source
blending step: collect N CSVs from N photos of the same subject, align
their depth scales via the IPD anchor, average per vertex, then apply as
Z-displacement on the v002 mesh. That's "photogrammetry for moving skin"
in practical form.

### export_maya_arnold_scene.py
Maya / Arnold handoff stage, pure-Python (no Maya install needed). Reads
the subject's `<subject>_face_mesh.obj`, normalises Y-height to Maya
centimetres (default 22 cm, head-height for a centimetre-scale Maya
scene), and writes both a Maya-cm OBJ and a fallback Maya ASCII (`.ma`)
loader that builds an `aiStandardSurface` skin shader, wires
albedo/roughness/normal maps, and lays out one or three comparison
meshes. The `.ma` is usable as-is; the matching `build_maya_arnold_scene`
upgrade is preferred when Maya 2027 + Arnold is available.

### build_maya_arnold_scene.py
Maya 2027 scene builder; requires `mayapy.exe`. Opens a lighting template
(per-subject `lightingscene_v001.mb` if present, else the versioned
default), removes stale generated meshes / shader nodes from prior runs,
imports the normalised Maya-cm OBJ, builds an Arnold skin shader (albedo
piped through `aiColorCorrect` for darker base tones, raw roughness /
normal inputs, smaller SSS scale tuned for cm-scale geometry), positions
the three comparison meshes at `rotateY ∈ {-45, 0, 45}` with the centre
at `[0, 100, 0]`, enables Arnold Catmull-Clark subdivision, and
optionally bakes subdivision (`--bake-subdivision-levels N`) into the
stored mesh. Saves a clean Maya ASCII at
`<subject>_arnold_skin_clean.ma`. This stage is **decoupled** from
`run_full_pipeline.py` — rerun only when shader / lighting / camera /
smoothing / scale change. See `docs/maya_handoff.md` for the lighting-
template contract and `docs/pipeline_stages.md` for the regen rules.

### run_full_pipeline.py
Imports the analysis + render scripts as Python modules (no
`subprocess`), runs them in sequence with per-subject output naming
under `outputs/characters/<subject>/`, then concatenates the MH and
learned PNGs vertically into a single combined frame. Accepts
`--canonical <obj>` and passes it through to the learned-mesh stage so
the v002 quad topology is selectable. Does **not** run the Maya /
Arnold export — that stage is intentionally separate.

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
mask. The v002 canonical improves Catmull-Clark behavior for Maya look-dev,
but it is still a MediaPipe face mask, not a production rig topology. For
production-quality character generation, either swap in a full-head model
such as MICA/FLAME or transfer the likeness onto MakeHuman topology. See
`extending.md` for the swap pattern.

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

### Why is pitch *intentionally* left in the mesh after pose-neutralization?
The pose-neutralization step (`learned_face_mesh.py --neutralize-pose`)
applies `cv2.solvePnP` to recover the photo's yaw / pitch / roll, then
inverts only **yaw + roll** before writing the mesh. Pitch is deliberately
left at the photo's natural value.

Reason: `solvePnP` with our placeholder camera intrinsics (`focal = image_w`,
principal point at image centre) systematically **overshoots pitch**. On a
true frontal shot it can report ~+10° pitch where the real value is closer
to zero. If we inverted that bogus pitch we'd over-rotate the mesh upward
visibly (chin lifts toward camera). Yaw and roll are robust because they
are essentially landmark-derived (yaw from horizontal asymmetry, roll from
eye-line angle), so we invert those fully.

Workarounds: pass `--neutralize-pitch` to apply the full PnP pitch (use only
when you have real intrinsics), or `--pitch-scale 0.5` to apply a fraction.

### Why is the face-mask boundary closed (extrude + fan-cap) before Maya?
The v002 canonical is an **open-boundary face mask** (no scalp, no closed
top of head). Maya/Arnold Catmull-Clark subdivision on an open mesh collapses
the boundary inward with each level -- after 4 levels of CC, the boundary
detaches visibly from the interior surface, creating a "lip" with a gap
between it and the inner face. This was visible in any non-frontal camera
angle.

`learned_face_mesh.py --close-boundary` walks MediaPipe's
`FACEMESH_FACE_OVAL` ring (36 verts), extrudes it backward by 50% of the
face depth, stitches the boundary to the back-ring with 36 wall quads, and
caps the back with a 36-triangle fan to a central pole vertex. Result: a
closed shell (505 verts / 580 faces) that CC subdivides cleanly. The
high-valence pole at the back is acceptable because no production camera
angle in this pipeline ever looks at the back of the head.

This is a v002-specific patch. When we swap to a full-head model
(MICA/FLAME) the boundary is already closed and this step becomes a no-op.

### Why does depth-from-photo (Depth Anything v2) belong in the pipeline?
The mesh from MediaPipe is **single-photo** -- it's a learned 3DMM-style
prediction of "what's likely 3D given this 2D image", not a measurement. For
production likeness we need real depth, ideally fused from multiple photos
of the same subject (like photogrammetry, but tolerating skin
deformation/expression changes).

`extract_depth_normal.py` is the first half of that workflow: it runs
**Depth Anything v2** (Apache-2.0, Hugging Face) on the photo to produce a
relative-depth map, derives a Sobel-gradient normal map from it, and
**samples depth at each of the 505 mesh-vertex UV coordinates**. That CSV
output is the input the (future) multi-source blending step will consume:
N CSV files from N photos of the same subject, scale-aligned via an IPD
anchor, averaged per vertex, then applied as Z-displacement on the v002 mesh.

The single-photo case already gives improved albedo/roughness/normal maps
for the Arnold skin shader -- the depth-derived normal map is real surface
relief (pore-scale features show up) rather than the luminance-derived bump
that `generate_texture_maps.py` writes from a Sobel of pixel brightness.

### Why 9 MH modifiers and not 50?
The current 9 cover broad scale axes (head/forehead/nose/mouth horiz +
vert, mouth y-translate, eye spacing). They demonstrate the path
end-to-end but cannot reach actual likeness. Adding more (chin-width,
cheek-bones, eye-height, head-oval, etc.) is a config-only change — see
`extending.md` for the recipe. The reason it's still 9 is that the
learned-mesh path already nails likeness, so investing in more MH
modifiers is only worthwhile if the downstream consumer specifically
needs MH topology.
