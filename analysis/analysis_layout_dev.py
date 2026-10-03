# -*- coding: utf-8 -*-
"""Stratification of the Heart-4CC test images by how far their ground-truth anatomy deviates from the source layout prior.

For every test image we compute the mean pairwise layout log-density of its GROUND-TRUTH boxes under the source
pairwise prior (the same quantity the structure-aware re-scoring uses, via AnatPrior), squashed exactly as in
AnatRescorer. A low value means the spatial arrangement of that fetus deviates from what the source centre saw, which is
the closest label-free proxy for unusual anatomy that these datasets allow. Images are binned by that value, and for
each bin we report COCO AP restricted to the bin for the source model, TAC-3 with and without the layout re-scoring,
and TAC-7 (pooled over the six directions and three seeds, averaged over the (direction, seed) cells with >= 5 images).

The two questions it answers: (a) does the canonicalisation still help on the images whose anatomy deviates most, and
(b) does the layout re-scoring, which is estimated on source anatomy, harm them?

Writes output/analysis/layout_deviation.{json,md}.  usage: python analysis_layout_dev.py
"""
import contextlib
import io
import json
import os
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.chdir(os.environ.get("TAC_ROOT", "."))
import sys
sys.path.insert(0, ".")
from adapteacher.engine.tta_distill import AnatPrior

DOMS = ["c1", "c2", "c3"]; SEEDS = ["42", "43", "44"]
ROWS = {"source": "output/so_fz_s{s}_{a}_to_{b}",
        "tac3": "output/canon/tac_s{s}_{a}_to_{b}_test",
        "tac3_noar": "output/canon/tac_noar_s{s}_{a}_to_{b}_test",
        "tac7": "output/canon/tac_v7_s{s}_{a}_to_{b}_test"}
BINS = [("most deviant (lowest 10%)", 0, 10), ("10-25%", 10, 25), ("25-50%", 25, 50), ("typical (top 50%)", 50, 101)]


def preds_file(d):
    f = os.path.join(d, "inference", "coco_instances_results.json")
    return f if os.path.exists(f) else os.path.join(d, "coco_instances_results.json")


def ap_on(gt, dt, img_ids):
    if len(img_ids) < 5:
        return float("nan")
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, dt, "bbox"); E.params.imgIds = sorted(img_ids); E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[0] * 100)


def image_plausibility(prior, anns, W, H):
    """mean pairwise layout log-density of the GT boxes (prior.box_scores, the same routine the re-scoring uses),
    squashed to [0,1] exactly as in AnatRescorer; None if no pair of the image is covered by the prior."""
    if not anns:
        return None
    xyxy = [(a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]) for a in anns]
    cls = [a["category_id"] - 1 for a in anns]
    per_box = prior.box_scores(xyxy, cls, W, H)
    if np.all(np.isnan(per_box)):
        return None
    lo = prior.thr; hi = prior.thr + 3.0
    return float(np.clip((np.nanmean(per_box) - lo) / (hi - lo), 0.0, 1.0))


# ---------------------------------------------------------------- per-image deviation, pooled over directions
dev = {}          # (dir, img_id) -> squashed plausibility of the GT layout
gts = {}
for a in DOMS:
    prior = AnatPrior(f"output/anat_prior_{a}.json")
    for b in DOMS:
        if a == b:
            continue
        gt = gts.setdefault(b, COCO(f"datasets/Heart-4cc/{b}/test.json"))
        for iid in gt.getImgIds():
            im = gt.loadImgs(iid)[0]
            anns = gt.loadAnns(gt.getAnnIds(imgIds=iid))
            v = image_plausibility(prior, anns, im["width"], im["height"])
            if v is not None:
                dev[(f"{a}_to_{b}", iid)] = v
print("images with a layout score:", len(dev), flush=True)
vals = np.array(list(dev.values()))
cuts = {p: float(np.percentile(vals, p)) for p in (10, 25, 50)}
print("percentiles of the GT layout plausibility:", {k: round(v, 3) for k, v in cuts.items()}, flush=True)

acc = defaultdict(list); counts = defaultdict(int)
for s in SEEDS:
    for a in DOMS:
        for b in DOMS:
            if a == b:
                continue
            d = f"{a}_to_{b}"; gt = gts[b]
            dts = {}
            ok = True
            for r, tpl in ROWS.items():
                f = preds_file(tpl.format(s=s, a=a, b=b))
                if not os.path.exists(f):
                    ok = False; break
                with contextlib.redirect_stdout(io.StringIO()):
                    dts[r] = gt.loadRes(f)
            if not ok:
                print("missing", d, s, flush=True); continue
            groups = defaultdict(list)
            for iid in gt.getImgIds():
                v = dev.get((d, iid))
                if v is None:
                    continue
                pct = 100.0 * float((vals < v).mean())
                for lab, lo, hi in BINS:
                    if lo <= pct < hi:
                        groups[lab].append(iid); break
            for lab, ids in groups.items():
                counts[lab] += len(ids)
                acc[lab].append({r: ap_on(gt, dts[r], ids) for r in ROWS})
    print("seed", s, "done", flush=True)

out = {"n_images_scored": len(dev), "percentiles": cuts, "bins": {}}
md = ["## Heart-4CC test images stratified by the deviation of their ground-truth layout from the source prior",
      "(Faster R-CNN, six directions x three seeds; AP restricted to the bin, averaged over the (direction, seed) cells with >= 5 images)\n",
      "| bin | images | source | TAC-3 | TAC-3 w/o layout re-scoring | TAC-7 | TAC-3 - source | re-scoring effect |",
      "|---|---|---|---|---|---|---|---|"]
for lab, _, _ in BINS:
    cells = [c for c in acc[lab] if not np.isnan(c["source"])]
    if not cells:
        continue
    m = {r: float(np.mean([c[r] for c in cells])) for r in ROWS}
    rs = m["tac3"] - m["tac3_noar"]
    out["bins"][lab] = {"n_images": counts[lab], "n_cells": len(cells), **{k: round(v, 2) for k, v in m.items()},
                        "tac3_minus_source": round(m["tac3"] - m["source"], 2), "rescoring_effect": round(rs, 2)}
    md.append(f"| {lab} | {counts[lab]} | {m['source']:.2f} | {m['tac3']:.2f} | {m['tac3_noar']:.2f} | {m['tac7']:.2f} | {m['tac3'] - m['source']:+.2f} | {rs:+.2f} |")
json.dump(out, open("output/analysis/layout_deviation.json", "w"), indent=1)
open("output/analysis/layout_deviation.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("LAYOUT_DEV_DONE")
