# Extending — DG Face Pipeline

How to add a new ratio, swap in a different face reconstructor (MICA /
FLAME / 3DDFA), add more MH modifiers, or wire the output into Blender,
Three.js, or back into a live MakeHuman session.

## Add a new Loomis-style ratio

Each ratio is a dimensionless `W_eye = 1.0` proportion computed from a
small subset of MediaPipe landmarks and mapped onto one MakeHuman modifier
path.

1. Pick the landmark indices. Reference: see the `LM` dict at the top of
   `face_proportion_analyzer.py`. Add new entries if needed; MediaPipe's
   478-landmark layout is documented in
   `mediapipe/modules/face_geometry/data/canonical_face_model.fbx` (or its
   `.obj` sibling).
2. Add a `compute_ratios` line. Inside `face_proportion_analyzer.py`,
   extend the `ratios` dict with the new key:
   ```python
   ratios["new_ratio_name"] = _abs_x("landmark_a", "landmark_b") / w_eye
   ```
3. Add a baseline. Set the canonical Loomis value in `LOOMIS_BASELINE`:
   ```python
   LOOMIS_BASELINE["new_ratio_name"] = 1.5
   ```
4. Map to an MH modifier. Append to `MODIFIER_MAP`:
   ```python
   MODIFIER_MAP["new_ratio_name"] = "group/target-min_kw|max_kw"
   ```
   The format is `<group>/<target>-<min_keyword>|<max_keyword>`. The group
   and target names must match a `modifiers` entry in
   `makehuman/data/modifiers/modeling_modifiers.json`, and the
   `min_keyword` / `max_keyword` must be the actual `min` / `max` strings
   from that entry. `step_c_apply_modifiers.py` will refuse to apply a
   modifier path it can't resolve against the registry.

That's it — re-run the pipeline and the new ratio shows up in the JSON, in
`modifier_targets`, and in the morphed MH OBJ.

## Swap in MICA (full-head, denser, FLAME topology)

MICA gives a ~5,000-vertex FLAME-topology mesh including scalp, ears, neck
— a real upgrade over MediaPipe's 478-vert face mask.

**Prerequisites the human has to handle:**
1. Register at <https://flame.is.tue.mpg.de/> (MPI gates the FLAME model
   weights with email approval, usually <1 day). Download
   `FLAME2020.zip` and `flame_model.pkl`.
2. Clone <https://github.com/Zielon/MICA>. Follow their install — on
   Windows, the `pytorch3d` dependency is the only painful one; pre-built
   wheels on conda-forge are the easiest path.
3. Place FLAME weights where MICA expects them (its README is explicit).
4. Download MICA's pretrained checkpoint from the Google Drive link in
   the repo.

**Pipeline integration:**

Add a new module `mica_face_mesh.py` next to `learned_face_mesh.py`
following the same contract — `reconstruct(image, out_obj)` writes an
OBJ. Then in `run_full_pipeline.py` add a CLI flag and route:
```python
if args.backend == "mica":
    mica.reconstruct(image, learned_obj)
else:
    learned.reconstruct(image, learned_obj, subdivisions=args.subdivisions)
```
`render_learned_mesh.py` doesn't need to change — it reads any OBJ that
has `v` / `vt` / `f` lines. If MICA's output lacks UVs, fall back to flat
shading; the renderer handles that automatically when `uvs` is empty.

The downstream MH-fitting step (currently unimplemented) becomes much more
tractable with FLAME topology because FLAME ↔ MH landmark correspondences
have been published.

## Swap in 3DDFA_V2 (lighter, BFM topology, Windows-friendly)

3DDFA_V2 is faster than MICA, builds on Windows with `pip install -e .`
in the cloned repo, and outputs a ~35,000-vert BFM-style mesh. The catch
is BFM also has an academic-only license (free for research, paid
otherwise).

Same integration shape — write `tddfa_face_mesh.py` exposing
`reconstruct(image, out_obj)`, add a CLI route in
`run_full_pipeline.py`.

## Add more MakeHuman modifiers

The current `MODIFIER_MAP` drives only 9 modifiers (broad scale axes).
Push toward likeness by adding region-specific modifiers — chin width,
cheek bones, eye height/depth, head shape macros, etc.

The fastest path:
1. Open `makehuman/data/modifiers/modeling_modifiers.json` and copy the
   `{"target": ..., "min": ..., "max": ...}` entries you want.
2. For each, pick or compute a ratio that should drive it. Some examples:
   - `head/head-oval-decr|incr` → could be driven by the ratio of
     `eye_to_mouth` over `intereye` (longer face → higher).
   - `chin/chin-width-decr|incr` → ratio of chin-tip to jaw-corner width
     over W_eye.
   - `eyes/l-eye-height1-decr|incr` → ratio of eyelid aperture to W_eye.
3. Add to `MODIFIER_MAP` keyed by a ratio name, and to `LOOMIS_BASELINE`
   with the canonical value.

Note that `head-oval`, `head-triangular`, `head-diamond`, etc. are
**no-min/max** macros (single-target, value 0..1). They don't fit the
`-min|-max` convention. Step C currently skips them. To support them,
extend `resolve_target_path()` in `step_c_apply_modifiers.py` to handle
the no-min/max case (positive-only, target file is
`<group>/<target>.target`, no `-min` / `-max` suffix).

## Project the mesh into a different renderer

The OBJs are vanilla Wavefront — open them in Blender, Maya, Three.js,
Unreal Engine. The `learned` OBJ ships with UVs keyed to the source
photo dimensions, so you can:

1. Load `<subject>_face_mesh.obj` and `examples/<subject>_ref.png` into
   Blender.
2. Create a new Image Texture node in a Principled BSDF material,
   point it at the photo, link to Base Color via the UV that came in
   with the OBJ.
3. The mesh is now textured with the source photo, ready for relighting,
   compositing, animation, etc.

For Three.js / glTF: convert via `obj2gltf` or `pyassimp`. UVs survive
the conversion.

## Hook back into a live MakeHuman session

The current MH path produces a *static* morphed OBJ — it doesn't write a
`.mhm` file or set live modifier values in a running MakeHuman GUI. To
drive a live session:

1. Start MakeHuman with its socket-server plugin enabled (it ships as
   `makehuman/plugins/1_modeling_module.py`-style; configure via
   `Settings → Plugins`).
2. From the analyzer JSON, read the `modifier_targets` list.
3. For each entry, send `Set Modifier <path> <value>` over the socket.
4. MakeHuman applies the modifier values to its live human in real time,
   blending the same `.target` files our step C does directly.

Code skeleton:
```python
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 12345))
for entry in modifier_targets:
    cmd = f"Set Modifier {entry['modifier']} {entry['value']}\n"
    s.sendall(cmd.encode())
s.close()
```

The exact command grammar depends on which MH socket plugin is enabled;
check the upstream community plugins repo.

## Detach the learned-mesh half from the MH fork

If commercial/AGPL constraints push you to keep only the Apache-2.0 half
of this pipeline, extract:

- `learned_face_mesh.py`
- `render_learned_mesh.py`
- `canonical_face_model.obj`
- the `examples/` directory you care about

into a fresh repo. Delete:
- `face_proportion_analyzer.py` (no MH dependency itself, but Loomis
  ratios are most useful with the MH back-end)
- `step_c_apply_modifiers.py` (AGPL-bound by `.target` reads)
- `render_face.py` (renders MH output)
- `run_full_pipeline.py` (orchestrates the AGPL half)

The standalone learned-mesh subset has no AGPL dependency and is fully
Apache-2.0 in its dependency chain.
