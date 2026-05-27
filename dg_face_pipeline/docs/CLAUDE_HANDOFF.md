# Claude handoff: v002 quad-dominant Maya face workflow

## Current goal

The working direction is to turn the MediaPipe/photo face output into something that can sit inside a proper DCC character pipeline: Maya scale, Arnold skin shading, correct camera framing, and a topology path that can eventually feed a quad rig such as MakeHuman.

The user specifically wants to compare:

- the default MediaPipe triangle mesh,
- the new quad-dominant v002 canonical mesh,
- and a Maya/Arnold scene where the meshes are lit, shaded, scaled, rotated, and framed consistently.

## What changed

### Quad-dominant canonical source

New canonical files live at:

- `dg_face_pipeline/canonical_face_model_v002.ma`
- `dg_face_pipeline/canonical_face_model_v002.mb`
- `dg_face_pipeline/canonical_face_model_v002.obj`

The OBJ is the pipeline input. It was exported from the v002 Maya scene and keeps the MediaPipe vertex/UV index space intact:

- 468 vertices
- 468 UVs
- 508 faces
- 390 quads
- 118 triangles

This is not a perfect production face topology. It is still a MediaPipe frontal mask, but it gives Catmull-Clark a much better starting point than the stock 898-triangle canonical face.

### Learned mesh pipeline

`dg_face_pipeline/learned_face_mesh.py` now accepts:

```powershell
python dg_face_pipeline\learned_face_mesh.py `
  --image dg_face_pipeline\examples\winona\ref.png `
  --out dg_face_pipeline\outputs\characters\winona\data\winona_face_mesh.obj `
  --canonical dg_face_pipeline\canonical_face_model_v002.obj `
  --subdivisions 0
```

Important behavior:

- `--canonical` can point at the v002 OBJ.
- The OBJ writer preserves quads/ngons instead of forcing triangles.
- If MediaPipe returns 478 landmarks but the canonical mesh only uses 468, the pipeline trims the trailing iris landmarks.
- Use `--subdivisions 0` when using v002 and you want to preserve the authored quad-dominant topology. The old midpoint subdivision path still triangulates because it works on triangles.

`dg_face_pipeline/run_full_pipeline.py` also passes the new `--canonical` option through.

### Maya/Arnold scene builder

`dg_face_pipeline/build_maya_arnold_scene.py` now does more of the final Maya packaging:

- Preserves the template scene layout height and spacing where possible.
- Uses the three-mesh layout the user asked for:
  - left mesh: `rotateY = -45`
  - centre mesh: `rotateY = 0`
  - right mesh: `rotateY = 45`
- Supports baked Catmull-Clark subdivision via `--bake-subdivision-levels`.
- Sets Arnold-style viewport/render subdivision attributes when requested.
- Adds a skin-oriented `aiStandardSurface` setup.
- Pipes albedo through `aiColorCorrect` before `baseColor` so blacks can be pushed darker.
- Uses a smaller SSS scale suitable for Maya centimetre scenes.

The latest clean Winona scene was generated locally at:

```text
dg_face_pipeline/outputs/characters/winona/maya/winona_arnold_skin_clean.ma
```

That output is intentionally gitignored because `outputs/` is generated data.

## Latest verified Winona scene

The current local scene was built with baked Catmull-Clark subdivision:

```powershell
& "C:\Program Files\Autodesk\Maya2027\bin\mayapy.exe" `
  dg_face_pipeline\build_maya_arnold_scene.py `
  --subject winona `
  --template dg_face_pipeline\outputs\characters\winona\maya\lightingscene_v001.mb `
  --arnold-subdiv-type none `
  --arnold-subdiv-iterations 0 `
  --bake-subdivision-levels 4 `
  --layout three `
  --front-rotate-y 0
```

Verification from Maya:

- `winona_GEO_left`: `rotateY -45`, position `[-25, 100, 0]`
- `winona_GEO_center`: `rotateY 0`, position `[0, 100, 0]`
- `winona_GEO_right`: `rotateY 45`, position `[25, 100, 0]`
- each baked mesh: `122785` vertices, `122496` faces
- camera: `render_cam`, aimed at the template/model height around Y `100`
- albedo is connected through `winona_albedo_grade_aiColorCorrect.outColor`
- grade values: gamma `1.12`, contrast `1.28`, contrast pivot `0.32`, exposure `-0.12`
- SSS scale: `0.12`

## What is not done yet

This is the key point: the saved Maya scene now has real baked subdivision, but it does not yet have true high-frequency sculpt/detail reprojection.

Current state:

- photo -> MediaPipe landmarks -> v002 quad-dominant mesh positions
- v002 mesh -> Maya centimetre scale and template placement
- Maya mesh -> baked Catmull-Clark x4
- Arnold-ish skin shader with albedo, roughness, normal, and graded base color

Still missing:

- dense surface/detail source,
- projection of high-res wrinkles/planes/pores/likeness detail onto the subdivided v002 mesh,
- displacement or normal baking from that projected detail,
- transfer of this result onto a MakeHuman head/base topology.

Do not describe the current scene as "reprojected detail" yet. It is subdivided and shaded, not sculpt-projected.

## Recommended next plan

1. Render-check the latest `winona_arnold_skin_clean.ma` in Maya 2027/Arnold.
2. Confirm the left/centre/right rotations read correctly from camera.
3. Tune skin color grade and SSS in real Arnold renders, not just scene inspection.
4. Add a reproducible one-command Winona v002 build script so the user can rerun the exact current state.
5. Implement actual detail reprojection:
   - generate or load a dense high-res reference surface,
   - align it to the v002 face,
   - project deltas onto the subdivided v002 mesh,
   - bake normal/displacement maps against the v002 UVs.
6. Keep MediaPipe v002 as a reference/capture bridge, not the final rig mesh.
7. Start the MakeHuman transfer as a separate step:
   - isolate the MakeHuman head region,
   - add a compatible UV/detail set if needed,
   - use landmarks/anchors to fit the MakeHuman head to the v002/reference face,
   - project likeness deltas back to MakeHuman while preserving its riggable quad topology.

## Maya shader notes

The intended Maya texture handling is:

- albedo/base color: sRGB/color-managed input
- roughness: Raw/non-color
- normal: Raw/non-color, connected through `bump2d`/normalCamera style network
- displacement later: Raw/non-color, applied at the shading group/displacement slot

The current script builds a pragmatic Arnold skin material. It is good enough for iteration, but the final target should probably split:

- clean albedo,
- pore/detail normal,
- micro roughness,
- larger displacement or sculpt deltas,
- and tuned SSS per Maya centimetre scale.

## Files to inspect first

- `dg_face_pipeline/learned_face_mesh.py`
- `dg_face_pipeline/run_full_pipeline.py`
- `dg_face_pipeline/build_maya_arnold_scene.py`
- `dg_face_pipeline/docs/maya_handoff.md`
- `dg_face_pipeline/docs/cli_reference.md`
- `dg_face_pipeline/canonical_face_model_v002.obj`

