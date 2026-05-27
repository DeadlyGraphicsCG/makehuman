# Multiagent Execution Log

Branch: `codex/face-pipeline-retopo-texture-bake`

## Tracks

| Track | Owner | Responsibility | Primary write scope |
|---|---|---|---|
| Topology validation | Worker | OBJ topology metrics and validation CLI | `validate_face_topology.py` |
| Canonical UV projection | Worker | Base MediaPipe UVs and atlas projection | UV/projection scripts, `learned_face_mesh.py` if needed |
| Baking adapter | Worker | Backend-neutral bake manifest and adapter stubs | `bake_maps.py`, baking docs |
| TexturingXYZ manifest | Worker | External-only inventory and provenance manifest | `texturing_xyz_manifest.py`, docs |
| Integration | Lead | Contracts, merge review, smoke tests, final docs | docs and final glue |

## Merge Order

1. Topology validation, because every later mesh candidate needs metrics.
2. Canonical UV projection, because map generation depends on UV space.
3. TexturingXYZ manifest, because it is external-only and low-conflict.
4. Baking adapter, because it consumes mesh/map paths from earlier stages.
5. Final docs and CLI reference updates.

## Smoke Matrix

Run these before final handoff:

```powershell
python dg_face_pipeline\validate_face_topology.py dg_face_pipeline\canonical_face_model.obj dg_face_pipeline\canonical_face_model_v002.obj
python dg_face_pipeline\texturing_xyz_manifest.py --root B:\RESOURCES\TexturingXYZ\vFace_030
python dg_face_pipeline\bake_maps.py --backend manifest --low dg_face_pipeline\canonical_face_model_v002.obj --high dg_face_pipeline\canonical_face_model_v002.obj --out-dir dg_face_pipeline\outputs\bake_smoke
python dg_face_pipeline\run_full_pipeline.py --subject winona --canonical dg_face_pipeline\canonical_face_model_v002.obj --subdivisions 0
```

The full pipeline command is the expensive smoke test. Run it after script
level checks pass.
