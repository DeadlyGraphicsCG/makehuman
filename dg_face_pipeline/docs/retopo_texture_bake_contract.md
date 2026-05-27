# Retopo, UV, Bake, and TexturingXYZ Contract

This branch turns the current MediaPipe face pipeline into a more stable DCC
handoff path. The shared contract below is the boundary between parallel work
tracks.

## Branch Policy

- Work branch: `codex/face-pipeline-retopo-texture-bake`.
- Base branch: `feat/face-proportion-pipeline`.
- Generated outputs stay under `dg_face_pipeline/outputs/`.
- TexturingXYZ source assets stay external and must not be committed.

## Mesh Contract

- Canonical face meshes preserve MediaPipe vertex index space wherever
  possible.
- OBJ output remains Y-up.
- The intended production-facing convention is Z-forward, with any unavoidable
  source-photo pose stored in metadata instead of hidden in geometry.
- OBJ writers must preserve authored face arity unless a command explicitly
  requests triangulation.
- Saved face records must keep valid `v/vt/vn` references when UVs or normals
  exist.

## UV Contract

- The MediaPipe canonical OBJ `vt` records are the base UV set.
- Per-photo landmark `(lm.x, lm.y)` values are projection inputs, not the
  long-lived UV layout.
- UV-atlas outputs are written in canonical UV space:
  - albedo/base color
  - segmentation masks
  - roughness
  - normal/depth-derived normal when available
- Diagnostics should include a UV occupancy mask and enough metadata to
  reproduce the projection.

## Baking Contract

- Baking is driven by an adapter interface, not a one-off DCC script.
- Required inputs:
  - low mesh OBJ
  - high mesh OBJ
  - output directory
  - map list
  - resolution
  - cage distance or cage mesh
- Required output:
  - bake report JSON
  - declared map paths, even for dry-run manifests
- Backends may be implemented independently. The manifest backend is the
  portable baseline; Maya and Blender backends must fail clearly when their
  host applications are unavailable.

## TexturingXYZ Contract

- Default external root: `B:\RESOURCES\TexturingXYZ\vFace_030`.
- The repo may store code, documentation, and generated manifests.
- The repo must not store TexturingXYZ source geometry, textures, ZTools,
  Mudbox files, or converted maps derived directly from the package.
- Every generated merge output must record provenance:
  - source path
  - map type
  - color-space hint
  - UDIM tile
  - blend mode
  - generation timestamp

## Validation Gates

Before merging a worker track:

1. Run the new or affected script with `--help`.
2. Run a small local smoke command against `winona` or a canonical OBJ.
3. Check `git status --short` for unexpected generated assets.
4. Confirm no file under `B:\RESOURCES\TexturingXYZ\vFace_030` was copied into
   the repository.
