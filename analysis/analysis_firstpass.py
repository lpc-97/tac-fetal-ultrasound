# -*- coding: utf-8 -*-
"""Reviewer item 4: first-pass failure / scale-estimate reliability analysis (Heart-4CC, Faster R-CNN, test, seeds 42/43/44).
Groups the target test images by (a) the number of first-pass candidates (p >= 0.3) that TAC used, (b) the magnitude of the
oracle (GT-derived) scale shift |log f_oracle|, (c) the maximum confidence of the source detector at the default scale (proxy for a
low-confidence first pass). For every group: COCO AP of source / TAC-3 / TAC-3 (2 rounds) / TAC-7 restricted to the group's images
(COCOeval with params.imgIds), pooled over the 6 directions x 3 seeds, and the scale-estimate error |log(f_TAC / f_oracle)|.
Writes output/analysis/firstpass_analysis.json and prints a markdown table."""
import contextlib
import io
import json
import os
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.chdir(os.environ.get("TAC_ROOT", "."))
SEEDS = [42, 43, 44]; DOMS = ["c1", "c2", "c3"]; DIRS = [(a, b) for a in DOMS for b in DOMS if a != b]
ROWS = {"source": "output/so_fz_s{s}_{a}_to_{b}", "tac3": "output/canon/tac_s{s}_{a}_to_{b}_test", "tac3_2r": "output/canon/tac_rounds2_s{s}_{a}_to_{b}_test", "tac7": "output/canon/tac_v7_s{s}_{a}_to_{b}_test",
        "est_median": "output/canon/est_median_s{s}_{a}_to_{b}_test", "est_confmean": "output/canon/est_confmean_s{s}_{a}_to_{b}_test", "est_noclass": "output/canon/est_noclass_s{s}_{a}_to_{b}_test",
        "oracle3": "output/canon/fac_oracle_s{s}_{a}_to_{b}_test", "global3": "output/canon/fac_global_s{s}_{a}_to_{b}_test"}
EST = {"irls": "output/canon/tac_v7_s{s}_{a}_to_{b}_test", "median": "output/canon/est_median_s{s}_{a}_to_{b}_test", "confmean": "output/canon/est_confmean_s{s}_{a}_to_{b}_test", "noclass": "output/canon/est_noclass_s{s}_{a}_to_{b}_test"}
NDET_BINS = [("0", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5), (">5", 6, 10 ** 9)]
SHIFT_BINS = [("|log f*| <= log 1.25", 0.0, np.log(1.25)), ("log 1.25 - log 1.6", np.log(1.25), np.log(1.6)), ("log 1.6 - log 2", np.log(1.6), np.log(2.0)), ("> log 2", np.log(2.0), 99.0)]
CONF_BINS = [("max p < 0.5", 0.0, 0.5), ("0.5 - 0.8", 0.5, 0.8), ("0.8 - 0.95", 0.8, 0.95), (">= 0.95", 0.95, 1.01)]


def preds_file(d):
    f = os.path.join(d, "inference", "coco_instances_results.json")
    return f if os.path.exists(f) else os.path.join(d, "coco_instances_results.json")


def ap_on(gt, dt, img_ids):
    if not img_ids:
        return float("nan")
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, dt, "bbox"); E.params.imgIds = sorted(img_ids); E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[0] * 100)


gts, dts = {}, {}
acc = defaultdict(lambda: defaultdict(list))   # grouping -> bin label -> list of (row -> AP) dicts per (dir, seed)
err = defaultdict(lambda: defaultdict(list))   # grouping -> bin label -> list of |log f_tac/f_oracle|
counts = defaultdict(lambda: defaultdict(int))
within15 = defaultdict(lambda: defaultdict(list))
est_err = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # grouping -> bin -> estimator -> |log f_est/f_oracle|
for a, b in DIRS:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(f"datasets/Heart-4cc/{b}/test.json")
    name2id = {os.path.basename(im["file_name"]): im["id"] for im in gt.loadImgs(gt.getImgIds())}
    oracle = json.load(open(f"output/analysis/oracle_factors_{a}_to_{b}_test.json"))["factors"]
    for s in SEEDS:
        # [f, v, ndet] per image from the TAC-7 run: its round-1 estimate and pass-1 candidate count are identical to TAC-3's
        fac = json.load(open(f"output/canon/tac_v7_s{s}_{a}_to_{b}_test/canon_factors.json"))["factors"]
        src_preds = json.load(open(preds_file(ROWS["source"].format(s=s, a=a, b=b))))
        maxp = defaultdict(float)
        for p in src_preds:
            maxp[p["image_id"]] = max(maxp[p["image_id"]], p["score"])
        with contextlib.redirect_stdout(io.StringIO()):
            dt = {r: gt.loadRes(preds_file(ROWS[r].format(s=s, a=a, b=b))) for r in ROWS}
        est_fac = {k: json.load(open(os.path.join(p.format(s=s, a=a, b=b), "canon_factors.json")))["factors"] for k, p in EST.items()}
        groups = {"ndet": defaultdict(list), "shift": defaultdict(list), "conf": defaultdict(list)}
        for name, (f, v, nd) in fac.items():
            iid = name2id[name]; fo = oracle[name][0]; e = abs(np.log(f / fo))
            ee = {k: abs(np.log(est_fac[k][name][0] / fo)) for k in EST}
            sh = abs(np.log(fo)); mp = maxp.get(iid, 0.0)
            for g, val, bins in [("ndet", nd, NDET_BINS), ("shift", sh, SHIFT_BINS), ("conf", mp, CONF_BINS)]:
                for lab, lo, hi in bins:
                    if (lo <= val <= hi) if g == "ndet" else (lo <= val < hi):
                        groups[g][lab].append(iid); err[g][lab].append(e); within15[g][lab].append(e <= np.log(1.15))
                        for k in EST:
                            est_err[g][lab][k].append(ee[k])
        for g, bins in groups.items():
            for lab, ids in bins.items():
                counts[g][lab] += len(ids)
                if len(ids) >= 5:   # AP on very small subsets is meaningless
                    acc[g][lab].append({r: ap_on(gt, dt[r], ids) for r in ROWS})
    print("done", a, "->", b, flush=True)

out = {}
md = []
for g, title in [("ndet", "first-pass candidates (p >= 0.3)"), ("shift", "oracle scale shift |log f*|"), ("conf", "max source confidence at the default scale")]:
    md.append(f"\n### grouped by {title}\n\n| group | images (6 dirs x 3 seeds) | (dir,seed) cells | source AP | TAC-3 | TAC-3 2r | TAC-7 | gain TAC-7 | TAC-3 w/ oracle f | TAC-3 w/ global f | est. median | est. conf-mean | est. no-class | median |log f/f*| (IRLS) | within +-15% | median err median / conf-mean / no-class |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    out[g] = {}
    labs = [l for l, _, _ in {"ndet": NDET_BINS, "shift": SHIFT_BINS, "conf": CONF_BINS}[g]]
    for lab in labs:
        cells = acc[g][lab]; n = counts[g][lab]
        if not cells:
            md.append(f"| {lab} | {n} | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - |"); out[g][lab] = {"n_images": n}; continue
        m = {r: float(np.nanmean([c[r] for c in cells])) for r in ROWS}
        e = float(np.median(err[g][lab])); w = float(np.mean(within15[g][lab]))
        ee = {k: float(np.median(v)) for k, v in est_err[g][lab].items()}
        out[g][lab] = {"n_images": n, "n_cells": len(cells), **m, "gain_tac7": m["tac7"] - m["source"], "median_abs_log_err": e, "within15": w, "est_median_abs_log_err": ee}
        md.append(f"| {lab} | {n} | {len(cells)} | {m['source']:.1f} | {m['tac3']:.1f} | {m['tac3_2r']:.1f} | {m['tac7']:.1f} | {m['tac7'] - m['source']:+.1f} | {m['oracle3']:.1f} | {m['global3']:.1f} | {m['est_median']:.1f} | {m['est_confmean']:.1f} | {m['est_noclass']:.1f} | {e:.3f} (x{np.exp(e):.2f}) | {100 * w:.0f}% | {ee['median']:.3f} / {ee['confmean']:.3f} / {ee['noclass']:.3f} |")
json.dump(out, open("output/analysis/firstpass_analysis.json", "w"), indent=1)
open("output/analysis/firstpass_analysis.md", "w").write("\n".join(md) + "\n")
print("\n".join(md)); print("FIRSTPASS_DONE")
