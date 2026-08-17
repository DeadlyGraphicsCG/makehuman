# Sapiens + MediaPipe Character Modeller Plan

## Purpose

Build a new DG character-modelling path that keeps MediaPipe as the stable
face-topology spine and adds Sapiens2 as a higher-fidelity human perception
layer for masks, normals, pointmaps, and body context.

The intended destination workspace is:

```text
C:\AI\apps\3D\DG_CharacterModeller
```

That directory did not exist when this plan was written. Start the next Codex
conversation from that folder after creating it, or start in `C:\AI\apps` and
create the workspace as the first action.

## Core Decision

Do not replace MediaPipe outright.

MediaPipe Face Landmarker / FaceMesh V2 provides the stable face landmark
contract this pipeline already depends on:

- 478 3D face landmarks.
- A known landmark index space.
- Compatibility with the current canonical face OBJ/UV workflow.
- Optional blendshape scores and facial transform matrices.

Sapiens2 should be added as an auxiliary perception layer:

- Human matting.
- Body-part segmentation.
- Surface normals.
- Pointmaps.
- Pose/body context.
- Cleaner masks for texture projection and DCC handoff.

The hybrid model is:

```text
source photo
  -> MediaPipe Face Landmarker V2
       -> canonical face landmarks
       -> canonical face mesh / UV spine
       -> facial transform / blendshape metadata
  -> Sapiens2 tasks
       -> human matte
       -> semantic segmentation
       -> surface normals
       -> pointmap
       -> pose/body context
  -> fusion layer
       -> canonical albedo
       -> face/body masks
       -> normal/depth/pointmap assets
       -> MakeHuman and Maya handoff package
```

## DG_Brain Governance Check

Before implementing in `DG_CharacterModeller`, check back with DG_Brain and
follow its hub/spoke rules.

Read these files first:

```text
C:\AI\apps\Core\DG_Brain\AGENT.md
C:\AI\apps\Core\DG_Brain\.agent\rules\rules.md
C:\AI\apps\Core\DG_Brain\DG_RULES.md
C:\AI\apps\Core\DG_Brain\ARCHITECTURE.md
C:\AI\apps\Core\DG_Brain\path_registry.py
```

Apply these DG_Brain rules to the new spoke:

- Verify before claim: inspect files and command output before reporting facts.
- Use absolute paths in docs, config, and handoff notes.
- PR-first workflow: branch, change, verify, commit, PR, review, merge.
- Never store secrets in the spoke; use the DG_Brain secrets pattern.
- Archive before delete; do not remove source/config/docs directly.
- Register the new spoke in DG_Brain governance if that registry expects it.
- Keep generated model outputs out of git unless deliberately versioned.
- Run Desloppify/code-health checks once the project has enough code surface.

## Proposed Branches

For the existing face-pipeline experiment:

```text
codex/sapiens-human-priors
```

For the new standalone app:

```text
codex/dg-character-modeller-oneshot
```

## One-Shot Target

The user-facing command should be simple:

```powershell
python -m dg_character_modeller oneshot --image C:\path\to\ref.png --subject winona
```

Expected output layout:

```text
C:\AI\apps\3D\DG_CharacterModeller\
├── inputs\
│   └── <subject>\ref.png
├── outputs\
│   └── characters\
│       └── <subject>\
│           ├── data\
│           │   ├── <subject>_landmarks_mediapipe.json
│           │   ├── <subject>_face_mesh.obj
│           │   ├── <subject>_fusion_manifest.json
│           │   └── <subject>_handoff_manifest.json
│           ├── textures\
│           │   ├── <subject>_canonical_albedo.png
│           │   ├── <subject>_human_matte.png
│           │   ├── <subject>_face_mask.png
│           │   ├── <subject>_normal_sapiens.png
│           │   └── <subject>_pointmap_sapiens.exr
│           ├── renders\
│           │   └── <subject>_character_modeller_report.png
│           └── maya\
│               └── <subject>_maya_handoff\
└── docs\
```

## Implementation Phases

### Phase 0: Workspace Bootstrap

- Create `C:\AI\apps\3D\DG_CharacterModeller`.
- Add `AGENTS.md` that points back to DG_Brain governance.
- Add `README.md`, `pyproject.toml`, and `src\dg_character_modeller\`.
- Add `.gitignore` for generated outputs, model caches, checkpoints, and local
  secrets.
- Decide whether this is a new git repo or a spoke inside an existing DG repo.

### Phase 1: MediaPipe Spine

- Port or wrap the current `dg_face_pipeline` MediaPipe logic.
- Prefer MediaPipe Face Landmarker V2 over legacy `mp.solutions.face_mesh`.
- Emit landmark JSON with:
  - image path and dimensions
  - 478 landmarks
  - canonical mesh mapping metadata
  - facial transform matrix if available
  - blendshape coefficients if available
- Preserve the current canonical OBJ contract:
  - Y-up OBJ output
  - stable `v/vt/f` references
  - canonical UV mode as the long-lived atlas path

### Phase 2: Sapiens2 Adapter

- Keep Sapiens2 optional and external.
- Do not vendor Sapiens2 weights into the repo.
- Add environment discovery:
  - `SAPIENS_ROOT`
  - `SAPIENS_CHECKPOINT_ROOT`
  - explicit model checkpoint paths
- Start with these tasks:
  - matting
  - segmentation
  - surface normal
  - pointmap
- Write every Sapiens output with metadata:
  - model id
  - checkpoint path/hash if available
  - task type
  - input image
  - output resolution
  - device
  - timestamp

### Phase 3: Fusion Layer

- Use MediaPipe UVs and canonical mesh for identity-stable face projection.
- Use Sapiens matte/segmentation to suppress background, hair spill, clothing,
  and non-face regions during texture projection.
- Use Sapiens normals/pointmap as auxiliary maps, not as topology authority.
- Keep all coordinate-space conversions explicit in metadata.
- Produce a `fusion_manifest.json` that records every input and output.

### Phase 4: DCC Handoff

- Reuse the existing Maya/Arnold handoff ideas from
  `C:\AI\apps\3D\Makehuman\dg_face_pipeline`.
- Keep adapter contracts small and explicit:
  - low mesh
  - high mesh
  - canonical texture maps
  - Sapiens maps
  - bake or render settings
- Make the Maya stage rerunnable without rerunning perception.

### Phase 5: Verification

Minimum smoke tests:

```powershell
python -m dg_character_modeller --help
python -m dg_character_modeller oneshot --image C:\AI\apps\3D\Makehuman\dg_face_pipeline\examples\winona\ref.png --subject winona --dry-run
python -m dg_character_modeller inspect --subject winona
```

Visual verification:

- Produce a report PNG showing source, MediaPipe landmarks, Sapiens matte,
  Sapiens normal, canonical texture preview, and fused mesh preview.
- Confirm generated files stay under `outputs\`.
- Confirm no Sapiens checkpoints or external licensed assets are committed.

## Licensing Notes

MediaPipe is the safer default for the core face topology contract.

Sapiens2 uses a custom Meta license. Treat it as optional, external, and
documented. Do not silently make it mandatory. Add clear usage notes for
consent, biometric restrictions, model provenance, and redistribution.

## Initial File Plan

```text
C:\AI\apps\3D\DG_CharacterModeller\
├── AGENTS.md
├── README.md
├── pyproject.toml
├── src\
│   └── dg_character_modeller\
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── paths.py
│       ├── manifest.py
│       ├── mediapipe_spine.py
│       ├── sapiens_adapter.py
│       ├── fusion.py
│       └── report.py
├── docs\
│   ├── architecture.md
│   ├── dg_brain_governance.md
│   ├── sapiens_adapter_contract.md
│   └── oneshot_pipeline.md
└── tests\
```

## Starter Prompt For Next Conversation

Use this prompt in the new Codex thread:

```text
We are starting C:\AI\apps\3D\DG_CharacterModeller as a DG_Brain-governed spoke.
First read C:\AI\apps\Core\DG_Brain\AGENT.md and
C:\AI\apps\Core\DG_Brain\.agent\rules\rules.md, then create the project scaffold.

Goal: build a one-shot hybrid character modeller where MediaPipe Face
Landmarker V2 provides the stable 478-landmark face spine and Sapiens2 is an
optional external adapter for matting, segmentation, normals, pointmaps, and
body context. Do not vendor Sapiens2 weights. Keep generated outputs out of
git. Add docs and dry-run CLI first, then implement the MediaPipe spine.

Reference plan:
C:\AI\apps\3D\Makehuman\dg_face_pipeline\docs\sapiens_mediapipe_character_modeller_plan.md
```

