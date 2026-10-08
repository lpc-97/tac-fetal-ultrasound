# Test-time Anatomical Canonicalization (TAC)

Reference implementation and analysis code for *Cross-domain fetal ultrasound structure detection
through anatomy-guided nuisance inference and uncertainty-aware evidence fusion*.

TAC adapts a frozen anatomical structure detector online, one unlabeled image at a time, without
access to source images. It infers the apparent anatomical scale of each test image and corrects it
before any prediction is fused. Three modules act in a closed loop:

* **ANI**, anatomical nuisance inference: every first-pass detection is treated either as a true
  structure whose log-size has been displaced by one unknown per-image factor, or as an uninformative
  false positive. These votes are fused with a source-domain prior over structure sizes by three
  expectation-maximization alternations of closed-form updates, which return the factor and the spread
  of a Gaussian surrogate for its posterior.
* **UVA**, uncertainty-guided view acquisition: further canonicalized views are placed at the
  Gauss-Hermite nodes of that surrogate, and acquired only where the surrogate spread is wide.
* **SDF**, structure-aware decision fusion: the boxes of all views are merged by weighted boxes fusion
  and resolved into one detection per structure under a pairwise anatomical layout prior, by iterated
  conditional modes. The result returns to ANI for a single refinement round.

Nothing is trained, the detector is never updated, and the only information that leaves the source site
is a few hundred Gaussian statistics over structure sizes and relative positions.

## Interactive walkthrough

<https://lpc-97.github.io/tac-fetal-ultrasound/> explains the method as four stepped animations,
each one driven by the measurements behind the paper rather than by a sketch.

1. **The problem.** Objects slide across the band edges of the detector's own level-assignment rule,
   `level = clip(floor(4 + log2(sqrt(area)/224)), 2, 5)`, and the level histogram moves off the one
   the detector was trained on. The three histograms are measured on the four-chamber view.
2. **ANI.** The nine first-pass boxes of a real image vote for one factor. The three alternations
   replay the intermediate values the deployed estimator wrote, responsibilities included.
3. **UVA.** The objective, the parabola fitted at its minimum, the width that comes out of the
   curvature, and the quadrature nodes the extra views are placed at.
4. **SDF.** Weighted fusion over views rather than over cluster size, then resolution of a duplicate
   under the pairwise layout prior.

A fifth panel runs the same four steps on four real test frames. Colored boxes are what the frozen
detector actually produced, and a structure counts as recovered when a box of its class overlaps the
annotation by at least half:

| frame | shift | structures recovered |
| --- | --- | --- |
| Heart, four-chamber, center 1 model on a center 3 image | x2.85 | 0 of 9 -> 7 of 9 |
| Heart, four-chamber, center 3 model on a center 2 image | x0.33 | 0 of 8 -> 8 of 8 |
| Abdomen, transverse, GE model on a Philips image | x1.57 | 4 of 5 -> 4 of 5, one unmatched box dropped |
| Spine, sagittal, GE model on a Samsung image | x2.05 | 4 of 4 -> 4 of 4, one unmatched box dropped |

The page is a single self-contained file under `docs/`, with no external scripts. The four frames in
`docs/assets/` are de-identified examples that also appear in the qualitative figure of the paper;
they are the only images in this repository, and the datasets themselves are not redistributed.

## What this repository contains

This is **not a standalone package**. TAC is implemented as an addition to the
[Adaptive Teacher](https://github.com/facebookresearch/adaptive_teacher) detection codebase, and the
files below are meant to be dropped into a clone of it. See `NOTICE` for attribution and licensing.

```
adapteacher/engine/tta_canon.py       the TAC test-time loop: ANI, UVA, view construction, fusion
adapteacher/engine/canon_input.py     resampling of the input to the canonical scale
adapteacher/modeling/anat_fusion.py   structure resolution under the pairwise layout prior (ICM)
adapteacher/modeling/fusion_fast.py   the Gaussian heat-map fusion used as a reference method
tools/anat_prior.py                   estimation of the source size and layout priors from annotations
configs/canon_config_keys.py.in       the SEMISUPNET.CANON keys to add to adapteacher/config.py
configs/od_r50fpn_retinanet_FZ.yaml   the one-stage detector configuration
runners/                              evaluation and experiment launchers
analysis/                             every analysis behind a number in the paper
paper_figures/                        the table and figure generators
```

## Installation

Build the Adaptive Teacher environment as its own README describes (Python 3.7, PyTorch 1.9.1+cu111,
Detectron2 0.5), then copy the files of this repository over the clone, keeping the paths, and append
`configs/canon_config_keys.py.in` to the `add_*_config` function of `adapteacher/config.py`.

```bash
pip install ensemble-boxes        # weighted boxes fusion
```

Two environment variables locate the project at run time. `TAC_ROOT` is the project root, the
directory holding `datasets/`, `output/` and `results/`; it defaults to the current directory.
`TAC_SCRATCH` is a writable scratch prefix for the caches the runners export, and defaults to
`/tmp`.

## Reproducing the experiments

1. **Train the source detectors.** `runners/eval_fz_seeds.sh` and `runners/eval_rt_seeds.sh` train and
   evaluate the frozen-BatchNorm Faster R-CNN and RetinaNet source models, for seeds 42, 43 and 44.
2. **Estimate the source priors.** `tools/anat_prior.py` writes one JSON per source domain, holding the
   per-structure log-size Gaussians, the pairwise layout Gaussians and the calibrated reference scale.
3. **Run TAC and the ablations.** `runners/run_canon.sh` dispatches every configuration reported in the
   paper by name, for example

   ```bash
   VARIANTS="tac tac_v7 tacr2" SPLIT=test SEED=42 GPU=0 bash runners/run_canon.sh
   ```

   `runners/launch_utest.sh` queues the uncertainty-aware variants over seeds and directions.
4. **Baselines.** `runners/run_baselines.sh` runs the test-time adaptation baselines with the
   configuration search described in the paper; `runners/run_compose.sh` runs them on the canonicalized
   input.
5. **Analysis and tables.** The scripts in `analysis/` each write a JSON under `output/analysis/`, and
   the generators in `paper_figures/` turn those into the table bodies and figures. Point `TAC_ROOT` at
   the project root first:

   ```bash
   TAC_ROOT=/path/to/project python paper_figures/gen_tables_reviewer.py
   ```

   Every number quoted in the paper is read from one of those JSON files; none is typed by hand.

## Data

The fetal ultrasound datasets are not redistributed here. Availability is stated in the paper's data
availability section; requests go to the data owners, under the ethics constraints described there.
The code expects COCO-format annotations under `datasets/<plane>/<domain>/{train,val,test}.json`.

## Citation

```bibtex
@article{tac2026,
  title   = {Cross-domain fetal ultrasound structure detection through anatomy-guided nuisance
             inference and uncertainty-aware evidence fusion},
  journal = {Information Fusion},
  year    = {2026}
}
```

## License

The upstream Adaptive Teacher code this work extends is released under Attribution-NonCommercial 4.0
International (CC BY-NC 4.0), whose text is reproduced in `LICENSE`. The additions in this repository
are released on the same terms, for non-commercial research use, with attribution. See `NOTICE`.
