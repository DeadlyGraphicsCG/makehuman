# Baking Adapter Contract

`bake_maps.py` is the DG face-pipeline handoff layer between retopo meshes and
DCC-specific texture baking. The contract is deliberately small so it can be
implemented by Maya, Blender, or a farm wrapper without adding heavy Python
dependencies to this repository.

## Request

Every backend receives the same request fields:

| Field | Meaning |
|-------|---------|
| `low` | Low-resolution retopo mesh that receives baked maps. |
| `high` | High-resolution source mesh used for projection. |
| `out_dir` | Directory for map outputs and the report JSON. |
| `resolution` | Square output map size in pixels. |
| `cage_distance` | Uniform cage/projection distance in scene units. |
| `maps` | Requested map names, defaulting to `normal`, `ambient_occlusion`, and `curvature`. |
| `backend` | One of `manifest`, `maya`, or `blender`. |

The report JSON always includes `inputs`, `backend`, `requested_maps`,
`resolution`, `cage`, `output_paths`, `backend_availability`, and
`command_contract`.

## Backends

### `manifest`

Fully implemented in v1. It performs no bake and writes a dry-run
`bake_manifest.json` that declares exactly which inputs, maps, cage settings,
and output paths a real adapter should use.

### `maya`

Contract stub in v1. It writes an adapter report and exits with a clear error.
A future implementation should run under `mayapy`, import the low/high meshes,
configure transfer-map projection using `--cage-distance`, and write one image
per requested map to the declared output paths.

Expected external entrypoint:

```powershell
mayapy <future_maya_bake_script.py> --low <low.obj> --high <high.obj> --out-dir <textures> --resolution <size> --cage-distance <distance> --maps <maps>
```

### `blender`

Contract stub in v1. It writes an adapter report and exits with a clear error.
A future implementation should run Blender headlessly, load the meshes,
configure cage extrusion/distance, and write one image per requested map to the
declared output paths.

Expected external entrypoint:

```powershell
blender --background --python <future_blender_bake_script.py> -- --low <low.obj> --high <high.obj> --out-dir <textures> --resolution <size> --cage-distance <distance> --maps <maps>
```

## Example

```powershell
python dg_face_pipeline\bake_maps.py --backend manifest --low outputs\characters\winona\data\winona_retopo.obj --high outputs\characters\winona\data\winona_face_mesh.obj --out-dir outputs\characters\winona\textures --resolution 2048 --cage-distance 0.02
```
