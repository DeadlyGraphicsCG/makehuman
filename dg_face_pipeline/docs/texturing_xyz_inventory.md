# TexturingXYZ External Inventory

TexturingXYZ source packages stay outside this repository. The DG face pipeline
only records a metadata manifest with external paths, sizes, suffixes, map type
guesses, UDIM tile guesses, and color-space hints. Do not copy TexturingXYZ OBJ,
EXR, TIFF, TX, ZTL, MUD, or reference-image files into `dg_face_pipeline/` or
commit them to git.

## Default vFace_030 Scan

The inventory script defaults to the local external package root:

```powershell
python dg_face_pipeline\texturing_xyz_manifest.py
```

That writes:

```text
dg_face_pipeline\outputs\texturing_xyz\vFace_030_manifest.json
```

`outputs/` is gitignored, so the generated manifest remains a local artifact.

## Explicit Paths

Use `--root` when the TexturingXYZ package lives somewhere else, and `--out` to
write the manifest to a different generated location:

```powershell
python dg_face_pipeline\texturing_xyz_manifest.py --root B:\RESOURCES\TexturingXYZ\vFace_030 --out dg_face_pipeline\outputs\texturing_xyz\vFace_030_manifest.json
```

The script reads `dg_face_pipeline/texturing_xyz_manifest_config.json` for the
known package layout. The current buckets are:

- `head_geometry`
- `head_maps`
- `id_masks`
- `calibration_reference_images`
- `eye_geometry`
- `eye_maps`
- `groom_assets`
- `zbrush_tools`
- `mudbox_support_files`
- `other`

Generated swatch/cache files and OS thumbnails are skipped. All retained
entries include both an absolute external path and a root-relative path so
downstream tools can validate availability without vendoring the source data.
