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
  4. render_face                  -- 4-panel MH comparison render
  5. render_learned_mesh          -- 3-panel learned-mesh + photo-texture
  6. assemble final combined PNG  -- stacks both renders vertically

Run:
    python C:\\AI\\apps\\Makehuman\\dg_face_pipeline\\run_full_pipeline.py ^
        --image C:\\AI\\apps\\DG_Brain\\assets\\refs\\winona_ref.png ^
        --subject winona

Outputs go under C:\\AI\\apps\\DG_Brain\\data\\ as <subject>_*.{json,obj,png}
and a final combined frame at data\\renders\\<subject>_full_pipeline.png.
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
import render_face as render_mh              # noqa: E402
import render_learned_mesh as render_learned  # noqa: E402
import step_c_apply_modifiers as step_c       # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("run_full_pipeline")

DATA_ROOT = Path(r"C:\AI\apps\DG_Brain\data")
RENDERS_ROOT = DATA_ROOT / "renders"


# -----------------------------------------------------------------------------
# Final compositor: stacks the MH 4-panel above the learned-mesh 3-panel.
# -----------------------------------------------------------------------------
def stack_renders(mh_png: Path, learned_png: Path, out_png: Path) -> Path:
    a = mpimg.imread(str(mh_png))
    b = mpimg.imread(str(learned_png))
    # Match widths by padding the narrower one with the background colour.
    bg = (0.082, 0.082, 0.102)  # #15151a, the panel facecolor used elsewhere
    target_w = max(a.shape[1], b.shape[1])

    def pad_to(img, w):
        if img.shape[1] == w:
            return img
        pad = ((0, 0), (0, w - img.shape[1]), (0, 0))
        # alpha-strip if present
        if img.shape[-1] == 4:
            img = img[..., :3]
        import numpy as np
        out = np.pad(img, pad, mode="constant", constant_values=0.0)
        # Recolour the pad columns to bg
        out[:, img.shape[1]:, 0] = bg[0]
        out[:, img.shape[1]:, 1] = bg[1]
        out[:, img.shape[1]:, 2] = bg[2]
        return out

    import numpy as np
    a3 = pad_to(a if a.shape[-1] != 4 else a[..., :3], target_w)
    b3 = pad_to(b if b.shape[-1] != 4 else b[..., :3], target_w)
    combined = np.concatenate([a3, b3], axis=0)

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
    image: Path,
    subject: str,
    subdivisions: int = 2,
    amplify: float = 1.5,
) -> Path:
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    proportions_json = DATA_ROOT / f"{subject}_face_proportions.json"
    morphed_obj = DATA_ROOT / f"{subject}_morphed.obj"
    learned_obj = DATA_ROOT / f"{subject}_face_mesh.obj"
    mh_render_png = RENDERS_ROOT / f"{subject}_mh_compare.png"
    learned_render_png = RENDERS_ROOT / f"{subject}_learned_mesh.png"
    final_png = RENDERS_ROOT / f"{subject}_full_pipeline.png"

    log.info("=" * 72)
    log.info("STEP 1/5 : analyzer -- %s", image.name)
    log.info("=" * 72)
    analyzer.analyze(image, proportions_json)

    log.info("=" * 72)
    log.info("STEP 2/5 : step_c MH modifier apply (amplify=%.2f)", amplify)
    log.info("=" * 72)
    step_c.apply_proportions(
        proportions_json=proportions_json,
        base_obj=step_c.BASE_OBJ_DEFAULT,
        out_obj=morphed_obj,
        amplify=amplify,
    )

    log.info("=" * 72)
    log.info("STEP 3/5 : learned face mesh (subdivisions=%d)", subdivisions)
    log.info("=" * 72)
    learned.reconstruct(image, learned_obj, subdivisions=subdivisions)

    log.info("=" * 72)
    log.info("STEP 4/5 : render MH 4-panel comparison")
    log.info("=" * 72)
    render_mh.render_compare(
        baseline_obj=Path(r"C:\AI\apps\Makehuman\makehuman\data\3dobjs\base.obj"),
        morphed_obj=morphed_obj,
        out_png=mh_render_png,
        photo_path=image,
        proportions_json=proportions_json,
    )

    log.info("=" * 72)
    log.info("STEP 5/5 : render learned-mesh 3-panel")
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
    stack_renders(mh_render_png, learned_render_png, final_png)
    return final_png


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full Photo -> face pipeline.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument(
        "--subject", type=str, required=True,
        help="Short name used to derive output file names (e.g. 'winona').",
    )
    p.add_argument("--subdivisions", type=int, default=2)
    p.add_argument("--amplify", type=float, default=1.5)
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        out = run(args.image, args.subject, args.subdivisions, args.amplify)
        log.info("DONE -- final frame: %s", out)
    except FileNotFoundError as exc:
        log.error("Missing input: %s", exc); return 2
    except (RuntimeError, ValueError, IOError) as exc:
        log.error("Pipeline failed: %s", exc); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
