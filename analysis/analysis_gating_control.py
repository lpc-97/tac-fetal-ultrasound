# -*- coding: utf-8 -*-
"""Is the surrogate spread the right thing to gate on? Equal-budget control.

Gating only decides WHICH images receive the bracket. The per-image detections of \tac-3 (no bracket)
and \tac-7 (bracket on every image) are both on disk, so for any subset S of images the stream that
brackets exactly S can be assembled off-line: take the \tac-7 detections on S and the \tac-3
detections elsewhere. Every rule below brackets the same number of images, so the mean forward-pass
budget is identical by construction.

Rules compared at a budget fraction phi:
  top-s     : the phi fraction with the largest first-pass spread s   (what UVA does)
  random    : a uniformly random phi fraction, averaged over R draws
  bottom-s  : the phi fraction with the smallest s                    (the adversarial control)
  all / none: \tac-7 and \tac-3, for reference

CPU only. Writes output/analysis/gating_control.json and prints a markdown table.
adapted from analysis_uva_quartiles.py (same repository)
"""
import contextlib
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.chdir(os.environ.get("TAC_ROOT", "."))
SEEDS = [42, 43, 44]
DOMS = ["c1", "c2", "c3"]
DIRS = [(a, b) for a in DOMS for b in DOMS if a != b]
TAC3 = "output/canon/tac_s{s}_{a}_to_{b}_test"
TAC7 = "output/canon/tac_v7_s{s}_{a}_to_{b}_test"
SPREAD = "output/canon/u7_k17_s42_{a}_to_{b}_test/canon_factors.json"
PHIS = [0.24, 0.50]
R = 10
RNG = np.random.RandomState(0)


def preds_file(d):
    f = os.path.join(d, "inference", "coco_instances_results.json")
    return f if os.path.exists(f) else os.path.join(d, "coco_instances_results.json")


def ap_of(gt, dets):
    if not dets:
        return float("nan")
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, "bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
    return float(E.stats[0] * 100)


rows = defaultdict(dict)          # rule -> cell -> AP
for a, b in DIRS:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO("datasets/Heart-4cc/%s/test.json" % b)
    name2id = {os.path.basename(im["file_name"]): im["id"] for im in gt.loadImgs(gt.getImgIds())}
    fac = json.load(open(SPREAD.format(a=a, b=b)))["factors"]
    s_by_id = {name2id[fn]: v[3] for fn, v in fac.items() if fn in name2id and v[3] >= 0}
    ids = sorted(s_by_id)
    order = sorted(ids, key=lambda i: s_by_id[i])       # ascending s

    for s in SEEDS:
        cell = "%s_to_%s_s%d" % (a, b, s)
        d3, d7 = defaultdict(list), defaultdict(list)
        for r in json.load(open(preds_file(TAC3.format(s=s, a=a, b=b)))):
            d3[r["image_id"]].append(r)
        for r in json.load(open(preds_file(TAC7.format(s=s, a=a, b=b)))):
            d7[r["image_id"]].append(r)

        def mix(sel):
            sel = set(sel)
            out = []
            for i in ids:
                out += (d7 if i in sel else d3)[i]
            # images without a spread (no candidates) keep their TAC-3 detections
            for i in set(list(d3) + list(d7)) - set(ids):
                out += d3[i]
            return out

        rows["none"][cell] = ap_of(gt, mix([]))
        rows["all"][cell] = ap_of(gt, mix(ids))
        for phi in PHIS:
            k = int(round(phi * len(ids)))
            rows["top_%.2f" % phi][cell] = ap_of(gt, mix(order[-k:] if k else []))
            rows["bottom_%.2f" % phi][cell] = ap_of(gt, mix(order[:k]))
            rnd = [ap_of(gt, mix(RNG.choice(ids, k, replace=False))) for _ in range(R)]
            rows["random_%.2f" % phi][cell] = float(np.mean(rnd))
    print("  done %s -> %s" % (a, b))

CELLS = ["%s_to_%s_s%d" % (a, b, s) for a, b in DIRS for s in SEEDS]
out = {"phis": PHIS, "n_random_draws": R, "rules": {}}
for rule, d in rows.items():
    v = np.array([d[c] for c in CELLS], dtype=np.float64)
    out["rules"][rule] = {"mean": float(np.nanmean(v)), "per_cell": {c: d[c] for c in CELLS}}

base = np.array([rows["none"][c] for c in CELLS])
full = np.array([rows["all"][c] for c in CELLS])
print("\n| rule | bracketed | mean AP | gain over TAC-3 | share of the full gain |")
print("|---|---|---|---|---|")
print("| \\tac-3, no bracket | 0%% | %.2f | -- | -- |" % out["rules"]["none"]["mean"])
for phi in PHIS:
    for pre in ("top", "random", "bottom"):
        k = "%s_%.2f" % (pre, phi)
        v = np.array([rows[k][c] for c in CELLS])
        g = float(np.nanmean(v - base))
        print("| %s-$s$ | %.0f%% | %.2f | %+.2f | %.0f%% |"
              % (pre, 100 * phi, out["rules"][k]["mean"], g, 100 * g / float(np.nanmean(full - base))))
print("| \\tac-7, bracket on all | 100%% | %.2f | %+.2f | 100%% |"
      % (out["rules"]["all"]["mean"], float(np.nanmean(full - base))))

for phi in PHIS:
    t = np.array([rows["top_%.2f" % phi][c] for c in CELLS])
    r = np.array([rows["random_%.2f" % phi][c] for c in CELLS])
    bo = np.array([rows["bottom_%.2f" % phi][c] for c in CELLS])
    for lab, other in (("top_minus_random", r), ("top_minus_bottom", bo)):
        d = t - other
        half = 2.110 * float(np.std(d, ddof=1)) / np.sqrt(len(d))
        out["%s_%.2f" % (lab, phi)] = {"mean": float(np.mean(d)), "ci95_half": half,
                                       "n_cells": int(len(d)),
                                       "frac_positive": float(np.mean(d > 0))}
        print("phi=%.2f  %-18s %+.2f +- %.2f AP over %d cells, positive in %.0f%%"
              % (phi, lab, np.mean(d), half, len(d), 100 * np.mean(d > 0)))

json.dump(out, open("output/analysis/gating_control.json", "w"), indent=1)
print("wrote output/analysis/gating_control.json")
