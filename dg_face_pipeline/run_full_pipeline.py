# File: C:\AI\apps\Makehuman\dg_face_pipeline\run_full_pipeline.py
# Repo: makehumancommunity/makehuman  Branch: feat/face-proportion-pipeline
# Orchestrator: DG_Brain
"""
Photo -> MakeHuman Face Adaptor : one-shot orchestrator
=======================================================
Runs the full pipeline for a single portrait photo:

  1. face_proportion_analyzer    -- landmarks -> Loomis ratios + bbox + pose
  2. step_c_apply_modifiers      -- ratios -> morphed MH base.obj
  3. learned_face_mesh           -- photo -> dense subdivided face OBJ + UVs
  4. render_canonical_report      -- thirds/fifths canon overlay + deltas
  5. render_face                  -- 4-panel MH comparison render
  6. render_learned_mesh          -- 3-panel learned-mesh + photo-texture
  7. assemble final combined PNG  -- stacks all renders vertically

Run:
    python C:\\AI\\apps\\Makehuman\\dg_face_pipeline\\run_full_pipeline.py ^
        --subject winona

By default the image is resolved from examples\\<subject>\\ref.png, falling
back to examples\\<subject>_ref.png for older layouts. Outputs go under
outputs\\characters\\<subject>\\.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Allow `python run_full_pipeline.py ...` invocation from anywhere by importing
# sibling modules via their absolute paths.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import face_proportion_analyzer as analyzer  # noqa: E402
import learned_face_mesh as learned          # noqa: E402
import render_canonical_report as render_canon  # noqa: E402
import render_face as render_mh              # noqa: E402
import render_learned_mesh as render_learned  # noqa: E402
import step_c_apply_modifiers as step_c       # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("run_full_pipeline")

PIPELINE_DIR = SCRIPT_DIR  # alias for readability
EXAMPLES_DIR = PIPELINE_DIR / "examples"
OUTPUTS_DIR = PIPELINE_DIR / "outputs"
CHARACTERS_ROOT = OUTPUTS_DIR / "characters"
MH_ROOT = PIPELINE_DIR.parent / "makehuman"


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def resolve_character_image(subject: str, image: Path | None = None) -> Path:
    if image is not None:
        return image
    candidates = [
        EXAMPLES_DIR / subject / "ref.png",
        EXAMPLES_DIR / subject / f"{subject}_ref.png",
        EXAMPLES_DIR / f"{subject}_ref.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def character_paths(subject: str) -> tuple[Path, Path]:
    root = CHARACTERS_ROOT / subject
    return root / "data", root / "renders"


# -----------------------------------------------------------------------------
# Final compositor: stacks all comparison renders vertically.
# -----------------------------------------------------------------------------
def stack_renders(render_pngs: List[Path], out_png: Path) -> Path:
    images = [mpimg.imread(str(p)) for p in render_pngs]
    # Match widths by padding the narrower one with the background colour.
    bg = (0.082, 0.082, 0.102)  # #15151a, the panel facecolor used elsewhere
    target_w = max(img.shape[1] for img in images)

    def pad_to(img, w):
        if img.shape[1] == w:
            if img.shape[-1] == 4:
                return img[..., :3]
            return img
        if img.shape[-1] == 4:
            img = img[..., :3]
        pad = ((0, 0), (0, w - img.shape[1]), (0, 0))
        import numpy as np
        out = np.pad(img, pad, mode="constant", constant_values=0.0)
        # Recolour the pad columns to bg
        out[:, img.shape[1]:, 0] = bg[0]
        out[:, img.shape[1]:, 1] = bg[1]
        out[:, img.shape[1]:, 2] = bg[2]
        return out

    import numpy as np
    combined = np.concatenate([pad_to(img, target_w) for img in images], axis=0)

    fig = plt.figure(figsize=(combined.shape[1] / 140.0, combined.shape[0] / 140.0),
                     facecolor="#15151a")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(combined)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#15151a")
    for s in ax.spines.values():
        s.set_visible(False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Wrote combined render: %s", out_png)
    return out_png


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def run(
    subject: str,
    image: Path | None = None,
    subdivisions: int = 2,
    amplify: float = 1.5,
    canonical_obj: Path = learned.CANONICAL_OBJ,
) -> Path:
    image = resolve_character_image(subject, image)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    data_root, renders_root = character_paths(subject)
    proportions_json = data_root / f"{subject}_face_proportions.json"
    morphed_obj = data_root / f"{subject}_morphed.obj"
    learned_obj = data_root / f"{subject}_face_mesh.obj"
    canon_render_png = renders_root / f"{subject}_canon_report.png"
    mh_render_png = renders_root / f"{subject}_mh_compare.png"
    learned_render_png = renders_root / f"{subject}_learned_mesh.png"
    final_png = renders_root / f"{subject}_full_pipeline.png"

    log.info("=" * 72)
    log.info("STEP 1/6 : analyzer -- %s", image)
    log.info("=" * 72)
    analyzer.analyze(image, proportions_json)

    log.info("=" * 72)
    log.info("STEP 2/6 : step_c MH modifier apply (amplify=%.2f)", amplify)
    log.info("=" * 72)
    step_c.apply_proportions(
        proportions_json=proportions_json,
        base_obj=step_c.BASE_OBJ_DEFAULT,
        out_obj=morphed_obj,
        amplify=amplify,
    )

    log.info("=" * 72)
    log.info("STEP 3/6 : learned face mesh (subdivisions=%d)", subdivisions)
    log.info("=" * 72)
    learned.reconstruct(
        image,
        learned_obj,
        subdivisions=subdivisions,
        canonical_obj=canonical_obj,
    )

    log.info("=" * 72)
    log.info("STEP 4/6 : render canon proportionality report")
    log.info("=" * 72)
    render_canon.render(
        photo=image,
        proportions=proportions_json,
        out_png=canon_render_png,
    )

    log.info("=" * 72)
    log.info("STEP 5/6 : render MH 4-panel comparison")
    log.info("=" * 72)
    render_mh.render_compare(
        baseline_obj=MH_ROOT / "data" / "3dobjs" / "base.obj",
        morphed_obj=morphed_obj,
        out_png=mh_render_png,
        photo_path=image,
        proportions_json=proportions_json,
    )

    log.info("=" * 72)
    log.info("STEP 6/6 : render learned-mesh 3-panel")
    log.info("=" * 72)
    render_learned.render(
        photo=image,
        mesh=learned_obj,
        out_png=learned_render_png,
        proportions=proportions_json,
    )

    log.info("=" * 72)
    log.info("FINAL    : stacking combined frame")
    log.info("=" * 72)
    stack_renders([canon_render_png, mh_render_png, learned_render_png], final_png)
    return final_png


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full Photo -> face pipeline.")
    p.add_argument(
        "--subject", type=str, required=True,
        help="Short name used to derive output file names (e.g. 'winona').",
    )
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image override. Defaults to examples/<subject>/ref.png.",
    )
    p.add_argument("--subdivisions", type=int, default=2)
    p.add_argument("--amplify", type=float, default=1.5)
    p.add_argument(
        "--canonical",
        type=Path,
        default=learned.CANONICAL_OBJ,
        help=(
            "Canonical topology OBJ for the learned mesh stage. Use "
            "dg_face_pipeline/canonical_face_model_v002.obj to test the "
            "local quad-dominant topology."
        ),
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        out = run(args.subject, args.image, args.subdivisions, args.amplify, args.canonical)
        log.info("DONE -- final frame: %s", out)
    except FileNotFoundError as exc:
        log.error("Missing input: %s", exc); return 2
    except (RuntimeError, ValueError, IOError) as exc:
        log.error("Pipeline failed: %s", exc); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
