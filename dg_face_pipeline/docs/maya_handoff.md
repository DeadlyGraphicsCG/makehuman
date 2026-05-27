# Maya / Arnold Handoff

The MediaPipe face mesh is a likeness reference and quick look-dev mesh. It is
not the final MakeHuman topology. The Maya handoff exists so we can inspect the
photo-projected face under real Arnold lighting while we build the MakeHuman
refit path.

## Local Lighting Template

For Winona there is a local Maya lighting scene at:

```text
dg_face_pipeline\outputs\characters\winona\maya\lightingscene_v001.mb
```

This scene is the current visual target for Maya pushes. It contains:

- a lighting setup
- a set camera
- shaders
- three repositioned versions of the model for quick comparison renders

Because it lives under `outputs/`, it is currently treated as local/generated
data. If this scene becomes the canonical render setup, move or copy a clean
template into a versioned path such as:

```text
dg_face_pipeline\maya_templates\face_lighting_v001.mb
```

Then the exporter can reliably use that template for every character and every
machine.

## Desired Export Contract

Each Maya push should produce a lit and shaded scene with three mesh variants:

```text
<subject>_face_mesh_maya_cm.obj
<subject>_arnold_skin.ma
<subject>_arnold_skin_clean.ma
```

The scene should:

- load or reference the lighting template
- import the normalized subject mesh at Maya centimeter scale
- keep the main face mesh centered at world origin before placement
- create three positioned copies or update three template placeholders
- assign Arnold skin shading consistently to all copies
- keep the render camera and lights from the template
- leave generated assets under `outputs\characters\<subject>\maya\`

The current exporter already handles the safe geometry basics:

- centers the mesh at world origin
- scales it to `--height-cm` (default `22.0`)
- writes vertex normals (`vn`) for smoother shading
- writes UVs tied to the source photo
- creates a Maya ASCII loader scene with Arnold skin shader wiring
- applies soft-edge and Arnold subdivision hints on import

The preferred Maya push is now `build_maya_arnold_scene.py`, which runs inside
Maya 2027. It opens the lighting template, deletes stale generated mesh/shader
clutter from the template, imports the normalized OBJ, builds one subject
Arnold skin network, assigns it only to the generated face meshes, applies
Catmull-Clark subdivision on the mesh shapes, removes unknown nodes that block
ASCII saves, and writes `<subject>_arnold_skin_clean.ma`.

## Current Command

```powershell
python dg_face_pipeline\export_maya_arnold_scene.py --subject winona --height-cm 22 --layout three --front-rotate-y -24
mayapy dg_face_pipeline\build_maya_arnold_scene.py --subject winona --arnold-subdiv-type catclark --arnold-subdiv-iterations 2
```

Only rerun these stages when the Maya handoff changes. Do not rerun MediaPipe
or the MakeHuman morph just because scale, smoothing, lighting, camera, or
shader values changed. If only the lighting template or shader network changes,
rerun just `build_maya_arnold_scene.py`.

Maya 2027 is expected at:

```text
C:\Program Files\Autodesk\Maya2027\bin
```

That directory should be on the user PATH, so a fresh terminal can call
`maya`, `mayapy`, or `mayabatch` directly. If the current app process has not
refreshed its inherited environment, call the full path:

```powershell
& "C:\Program Files\Autodesk\Maya2027\bin\mayapy.exe" dg_face_pipeline\build_maya_arnold_scene.py --subject winona
```

## Shading Notes

For the current single-photo mesh, treat the photo as color/albedo-ish input,
not as physically clean scan data. It contains baked lighting, shadows,
specular highlights, makeup, camera response, and compression. Use it for fast
likeness inspection, then replace it later with proper albedo/normal/
displacement bakes.

Initial Arnold skin shader guidance:

- Use `aiStandardSurface`.
- Connect the generated albedo JPG to `baseColor`.
- For 8/16-bit JPG, PNG, or TIFF photos, set the file/aiImage color space to
  `sRGB` / `srgb_texture`.
- If a texture is already linear EXR or ACEScg, tag it explicitly as that.
  Arnold's working space is commonly ACEScg; wrong input tags make faces look
  washed out, too dark, or over-saturated.
- Keep scalar maps such as roughness, masks, displacement, and bump/normal
  data in `Raw`.
- Use moderate specular and roughness; avoid glossy skin until maps are clean.
- Use light subsurface only for this prototype because the projected photo
  already contains lighting and skin scattering.
- Keep shader weights/colors at or below `1.0` unless there is a deliberate
  reason to overdrive them.
- Prefer mesh vertex normals and soft edges for this MediaPipe mesh.
- Treat Arnold Catmull-Clark subdivision as preview/look-dev only here; the
  final production solution should be a fitted quad MakeHuman mesh.

Single-photo starting point:

| Attribute | Starting value |
|-----------|----------------|
| Base Weight | `1.0` |
| Base Color | generated `<subject>_albedo.jpg` |
| Specular Weight | `0.35`-`0.55` |
| Specular Roughness | generated `<subject>_roughness.jpg`, Raw |
| Metalness | `0.0` |
| Transmission | `0.0` |
| Subsurface Weight | `0.15`-`0.35` |
| Subsurface Radius | about `R 1.0, G 0.35, B 0.2` |

For a more physically oriented skin setup later, move a cleaned skin albedo into
Subsurface Color, raise Subsurface Weight, and tune Subsurface Scale to the
scene's centimeter units. Bad outward normals can break SSS, so normal checks
are mandatory before trusting skin renders.

## Normal, Bump, and Displacement Policy

Current state:

- A shallow single-photo normal JPG can be generated for look-dev.
- No real displacement map is generated.
- The exporter writes averaged vertex normals for smoother triangle shading.

Future texture-bake state:

- Albedo/base color: `sRGB`
- Normal maps: `Raw`, connected through `file`/`aiImage -> aiNormalMap ->
  aiStandardSurface.normalCamera` or through Maya `bump2d` set to tangent-space
  normals. The current clean-scene builder uses Maya `bump2d` with
  `bumpInterp=1`, `bumpDepth=0.08`, and `bumpFilter=1.25`.
- Bump maps: `Raw`
- Displacement maps: `Raw`, connected through the shading group displacement
  slot with controlled subdivision/displacement bounds

Do not invent high-frequency detail from the MediaPipe mesh itself. Its shape
is useful as a likeness guide, but it is too sparse and triangulated to be the
source of production pore detail.

Normal-map caveats:

- Do not expect Maya `bump2d` Bump Depth to behave like a normal-map strength
  control in all Arnold paths.
- If green-channel lighting is inverted, flip G.
- If tangent axes look wrong, use Arnold controls such as FlipR, FlipG, or Swap
  Tangents.
- MtoA 5.6.1 adds `normal_map.tangent_space_type` with MIKKTSpace support. Use
  `mikk` only when the normal map was baked/exported with MikkTSpace tangents;
  otherwise keep the default tangent space.

Displacement caveats:

- Connect displacement through the shading group's displacement slot, not the
  surface shader.
- Use high-quality maps and Arnold `.tx` / auto-TX workflows for serious
  renders.
- Set displacement bounds/padding high enough to prevent clipping.
- If the height map is mid-gray centered, document the zero/midpoint, commonly
  `0.5`. If black is zero height, use `0`.
- Prefer bump/normal for single-photo pore detail. Use displacement only for
  real silhouette or geometry relief.

## Smoothing and Subdivision

For the current OBJ face mesh:

- Verify normals face outward.
- Export averaged vertex normals.
- Unlock/soften skin normals if the OBJ imports faceted.
- Use soft normals for shading smoothness.
- For quick Maya inspection, export the learned mesh at `--subdivisions 3`
  before the Maya stage, then use `--arnold-subdiv-type catclark
  --arnold-subdiv-iterations 2`.
- Use Arnold Catmull-Clark subdivision carefully. It changes silhouette and can
  shift projected texture detail around eyes, lips, nostrils, and UV seams.
- Per-object Arnold subdivision iterations live on the mesh shape; global
  Arnold subdivision caps can limit them.
- In saved Maya ASCII, MtoA serializes these attrs as `.ai_subdiv_type`,
  `.ai_subdiv_iterations`, and `.ai_subdiv_smooth_derivs`. For the current
  clean scenes, `.ai_subdiv_type 1` means Catmull-Clark and
  `.ai_subdiv_iterations 2` is the look-dev smoothing level.

The deeper production answer remains a fitted quad MakeHuman head with proper
loops and Catmull-Clark subdivision. The MediaPipe triangle mesh is the
likeness guide.

## Maya 2027 / MtoA Notes

- Maya 2027 ships with MtoA 5.6.0 / Arnold 7.5.0.0.
- MtoA 5.6.1 adds MIKKTSpace normal mapping support and unit auto-scaling
  improvements for linked shader parameters.
- Maya 2027 uses newer OCIO color management. Do not rely on automatic color
  space detection for photo, normal, bump, roughness, or displacement maps.
- Arnold for Maya is the `mtoa.mll` plugin.
- Interactive Arnold rendering is included; batch rendering may watermark
  without the correct Arnold licensing.
- If the team uses Arnold GPU/OptiX, validate CPU vs GPU lookdev parity and
  driver requirements.

## References

- [Arnold Color Management](https://help.autodesk.com/cloudhelp/ENU/AR-Core/files/arnold_user_guide_ac_color_management_html.html)
- [Arnold for Maya Standard Surface / Normal Mapping](https://help.autodesk.com/cloudhelp/ENU/AR-Maya/files/am-Arnold_for_Maya_User_Guide/shaders/surface/arnold_for_maya_surface_am_Standard_Surface_html.html)
- [Arnold Standard Surface](https://help.autodesk.com/cloudhelp/ENU/AR-Core/files/ac-shading/ac-surface-shaders/arnold_user_guide_ac_surface_shaders_ac_standard_surface_html.html)
- [Arnold Subsurface](https://help.autodesk.com/view/ARNOL/ENU/?guid=arnold_user_guide_ac_standard_surface_ac_standard_subsurface_html)
- [Arnold Displacement in MtoA](https://help.autodesk.com/cloudhelp/2024/ENU/AR-Maya/files/am-Arnold_for_Maya_User_Guide/shaders/arnold_for_maya_shaders_displacement_html.html)
- [Maya 2027 Arnold for Maya 5.6.0](https://help.autodesk.com/cloudhelp/2027/ENU/Maya-WhatsNew/files/GUID-82D13118-E987-4B5E-B1DA-C2A01F51058C.htm)
- [MtoA 5.6.1 Release Notes](https://help.autodesk.com/view/ARNOL/ENU/?guid=arnold_for_maya_561_html)
