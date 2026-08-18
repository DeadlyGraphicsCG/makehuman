# Manifest-Driven Character Build Order

This document separates the **current implemented Winona v003 build** from the
**intended full manifest-driven build**. The current Maya files are useful
look-dev artifacts, but they are not proof that every future pipeline stage has
run.

## Current Winona v003 Path

`outputs/characters/winona_v003/maya/winona_v003_arnold_skin_origin.ma` was
created by the existing scripts in this order:

1. Run face analysis from the source portrait.
2. Generate the MediaPipe/canonical face mesh.
3. Neutralize photo yaw/roll from the analyzer.
4. Apply the final Maya-facing front-upright solve.
5. Close the face boundary with a back cap.
6. Center the final closed mesh bounding box at world origin.
7. Bake the portrait into the canonical UV atlas.
8. Generate provisional canonical roughness and normal maps from that atlas.
9. Export a Maya-centimeter OBJ centered at origin.
10. Build an Arnold Maya scene from the Maya-centimeter OBJ.
11. Assign canonical albedo, roughness, and normal file nodes.
12. Build a physical-sky Arnold lighting rig.
13. Save clean `.ma` and `.mb` look-dev scenes.

The scene was also reopened with Maya Python to verify:

- the center mesh transform is at `translate 0 0 0`;
- the center mesh has `rotateY 0`;
- the three-up scene has center, left, and right meshes at `0`, `-45`, and
  `+45` degrees;
- physical sky and skydome Arnold nodes are present;
- no generated legacy area lights are present;
- the shader file nodes resolve;
- albedo uses `sRGB`;
- roughness and normal use `Raw`;
- both the source OBJ and Maya-centimeter OBJ pass topology validation.

## Not Yet Part Of That File

The current Maya scene does **not** yet include these future stages:

1. tri-to-quad or manual retopo completion;
2. retopo landmark correspondence validation;
3. high/low bake execution;
4. projection cage generation;
5. baked AO, curvature, displacement, or high-fidelity normal maps;
6. TexturingXYZ alignment and map blending;
7. segmentation-mask authoring beyond first-pass procedural outputs;
8. a single character manifest that records every stage, artifact, gate, and
   artist comment.

Those pieces exist as contracts, docs, helpers, or stubs. They still need to be
wired into a manifest-driven orchestrator before a Maya scene can honestly be
marked as a full finished build.

## Intended Full Build Order

Use this order for the manifest-led pipeline:

1. **Create / update the subject manifest.**
   The manifest is the build recipe and review ledger. It declares inputs,
   output paths, coordinate policy, UV policy, texture policy, validation gates,
   stage status, and artist comments.

2. **Validate source inputs.**
   Check portrait path, canonical topology, optional retopo mesh, optional
   high-resolution source, TexturingXYZ root, and Maya lighting template.

3. **Run face analysis.**
   Write the landmark/proportion JSON and record analyzer version, image size,
   detected pose, and confidence notes.

4. **Generate the base face mesh.**
   Reconstruct the MediaPipe/canonical mesh, apply pose neutralization, apply
   front-upright alignment, close the boundary, center at origin, and write
   normals.

5. **Validate topology and coordinates.**
   Gate on invalid references, degenerate faces, boundary edges, nonmanifold
   edges, component count, expected counts, orientation checks, and bbox center.

6. **Resolve UV policy.**
   Use canonical UVs as the stable base UV set. Store runtime/photo projection
   UVs separately so future reprojection can be repeated.

7. **Project the canonical albedo.**
   Bake the source portrait into the canonical atlas and write an occupancy mask
   plus projection metadata.

8. **Generate provisional map set.**
   Produce first-pass albedo, segmentation masks, roughness, and normal maps in
   canonical UV space. Mark these as provisional until high/low baking and
   TexturingXYZ merge are complete.

9. **Run retopo / quad stage.**
   Create or import the quad-oriented retopo mesh. Validate vertex order,
   landmark correspondence, UV transfer, face arity, and deformation fit.

10. **Set up high/low baking.**
    Define high mesh, low mesh, cage distance, cage asset, requested maps,
    resolution, color space, and backend.

11. **Execute bake backend.**
    Run Maya, Blender, or another adapter. Record bake command, map outputs,
    warnings, and validation thumbnails.

12. **Inventory TexturingXYZ package.**
    Scan the external source root and record metadata-only file references.
    Never copy TexturingXYZ source assets into the repository.

13. **Merge texture sources.**
    Blend photo projection, baked maps, segmentation masks, and TexturingXYZ
    detail according to the manifest's map policy.

14. **Export Maya-normalized OBJ.**
    Convert to centimeters, center at origin, preserve UV/material references,
    and validate again.

15. **Build Arnold shader.**
    Create the skin material and connect map nodes with explicit color spaces:
    albedo in `sRGB`, scalar/vector maps in `Raw`.

16. **Build Maya scenes.**
    Save a single-origin asset scene and a three-up look-dev scene in `.ma` and
    `.mb` forms.

17. **Verify Maya scenes.**
    Reopen with Maya Python and verify meshes, transforms, shader connections,
    plugin requirements, missing textures, unknown nodes, and camera framing.

18. **Finalize the manifest.**
    Record final artifact paths, validation stats, thumbnails, timestamps, git
    commit, and human review comments.

## Manifest Comment Pattern

Use JSONC or YAML if you want real inline comments. If the manifest must remain
plain JSON, store comments as data:

```json
{
  "schema": "dg_face_pipeline.character_manifest.v1",
  "subject": "winona_v003",
  "artist_comments": [
    {
      "stage": "orientation",
      "author": "dg",
      "status": "needs_review",
      "comment": "Check side tilt and X centering in Maya front/side views."
    }
  ],
  "stages": {
    "maya_scene": {
      "status": "built",
      "inputs": [
        "outputs/characters/winona_v003/maya/winona_v003_face_mesh_maya_cm.obj"
      ],
      "artifacts": [
        "outputs/characters/winona_v003/maya/winona_v003_arnold_skin_origin.ma",
        "outputs/characters/winona_v003/maya/winona_v003_arnold_skin_origin.mb"
      ],
      "validation": {
        "center_mesh_translate": [0.0, 0.0, 0.0],
        "center_mesh_rotate_y": 0.0,
        "shader_textures_resolved": true
      },
      "comments": []
    }
  }
}
```

## Honest Status Labels

Use these labels consistently:

- `not_started`: no artifact exists;
- `planned`: manifest entry exists, but no build artifact exists;
- `built`: artifact exists, but has not passed validation;
- `validated`: artifact passed automated gates;
- `artist_review`: artifact needs visual or manual review;
- `approved`: artist has signed off;
- `blocked`: stage cannot proceed without a missing input or decision;
- `provisional`: useful output, but expected to be replaced by a later stage.

For example, the current `winona_v003_arnold_skin_origin.ma` should be recorded
as `validated` for Maya scene construction, but `provisional` for final skin
look-dev because retopo, high/low baking, and TexturingXYZ detail are not yet in
that scene.
