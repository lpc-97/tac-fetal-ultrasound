# -*- coding: utf-8 -*-
"""Does the surrogate spread s identify the images on which extra views pay off?

For every Heart-4CC direction and seed (Faster R-CNN), group the target test images by the quartile of
the first-pass surrogate spread s of Eq. (4), and report inside each quartile

  AP(TAC-3), AP(TAC-7), Delta = AP(TAC-7) - AP(TAC-3)

computed with COCOeval restricted to the group's images (params.imgIds), exactly as
analysis_firstpass.py does, plus the mean true scale error |log(f_TAC / f_oracle)| and the mean number
of first-pass candidates.

The claim UVA needs is that Delta grows with s: spending the extra forward passes where s is large is
worth more than spending them where s is small. CPU only, from stored artifacts.

s is read from the u7_k17 runs, the only ones whose canon_factors.json carries the first-pass spread in
field 3. The first pass is identical across TAC configurations (same frozen detector, same s_0, same
MIN_SCORE, same prior), which analysis_uncert_exact.py verified by reconstructing s to 1e-9.

Writes output/analysis/uva_quartiles.json and prints a markdown table.
adapted from analysis_firstpass.py (same repository)
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
SEEDS = [42, 43, 44]
DOMS = ["c1", "c2", "c3"]
DIRS = [(a, b) for a in DOMS for b in DOMS if a != b]
ROWS = {"tac3": "output/canon/tac_s{s}_{a}_to_{b}_test",
        "tac7": "output/canon/tac_v7_s{s}_{a}_to_{b}_test",
        "source": "output/so_fz_s{s}_{a}_to_{b}"}
SPREAD = "output/canon/u7_k17_s{s}_{a}_to_{b}_test/canon_factors.json"
NQ = 4


def preds_file(d):
    f = os.path.join(d, "inference", "coco_instances_results.json")
    return f if os.path.exists(f) else os.path.join(d, "coco_instances_results.json")


def ap_on(gt, dt, img_ids):
    if len(img_ids) < 5:
        return float("nan")
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, dt, "bbox")
        E.params.imgIds = sorted(img_ids)
        E.evaluate()
        E.accumulate()
        E.summarize()
    return float(E.stats[0] * 100)


# The spread depends on the direction, so global quartile edges do not partition every cell and would
# confound the comparison with direction difficulty. Quartiles are therefore taken WITHIN each
# (direction, seed) cell, and every cell contributes one AP per quartile.
all_s = []
per_cell = {}
for a, b in DIRS:
    for s in SEEDS:
        fac = json.load(open(SPREAD.format(s=s, a=a, b=b)))["factors"]
        per_cell[(a, b, s)] = fac
        all_s += [v[3] for v in fac.values() if v[3] >= 0]
edges = [float(np.percentile(all_s, 100.0 * k / NQ)) for k in range(1, NQ)]
print("pooled spread quartile edges (reference only): %s  (n=%d image-cells)"
      % ([round(e, 4) for e in edges], len(all_s)))

LAB = ["Q1 (narrowest)", "Q2", "Q3", "Q4 (widest)"]


def cell_quartiles(fac, keep):
    """quartile index per file name, computed within this cell"""
    vals = sorted(fac[fn][3] for fn in keep)
    e = [vals[int(round(len(vals) * k / NQ)) - 1] for k in range(1, NQ)]
    out = {}
    for fn in keep:
        x = fac[fn][3]
        q = NQ - 1
        for k, lim in enumerate(e):
            if x <= lim:
                q = k
                break
        out[fn] = q
    return out, e


acc = defaultdict(lambda: defaultdict(dict))   # bin -> row -> {(dir, seed): AP}
err = defaultdict(list)
ndet = defaultdict(list)
sval = defaultdict(list)
cnt = defaultdict(int)

for a, b in DIRS:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO("datasets/Heart-4cc/%s/test.json" % b)
    name2id = {os.path.basename(im["file_name"]): im["id"] for im in gt.loadImgs(gt.getImgIds())}
    oracle = json.load(open("output/analysis/oracle_factors_%s_to_%s_test.json" % (a, b)))["factors"]
    for s in SEEDS:
        fac = per_cell[(a, b, s)]
        keep = [fn for fn, v in fac.items() if fn in name2id and v[3] >= 0]
        qmap, _ = cell_quartiles(fac, keep)
        groups = defaultdict(list)
        for fn in keep:
            v = fac[fn]
            q = qmap[fn]
            groups[q].append(name2id[fn])
            sval[q].append(v[3])
            ndet[q].append(v[2])
            if fn in oracle and oracle[fn][0] > 0:
                err[q].append(abs(np.log(max(v[0], 1e-9) / oracle[fn][0])))
        dt = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for k, tpl in ROWS.items():
                dt[k] = gt.loadRes(preds_file(tpl.format(s=s, a=a, b=b)))
        for q, ids in groups.items():
            cnt[q] += len(ids)
            for k in ROWS:
                acc[q][k][(a, b, s)] = ap_on(gt, dt[k], ids)
    print("  done %s -> %s" % (a, b))

out = {"edges": edges, "n_quartiles": NQ, "bins": {}}
print("\n| spread quartile | images | mean s | mean cand. | mean |log f/f*| | source | TAC-3 | TAC-7 | TAC-7 - TAC-3 |")
print("|---|---|---|---|---|---|---|---|---|")
CELLS = [(a, b, s) for a, b in DIRS for s in SEEDS]
for q in range(NQ):
    m = {k: float(np.nanmean([acc[q][k].get(c, np.nan) for c in CELLS])) for k in ROWS}
    d = m["tac7"] - m["tac3"]
    per_cell_delta = {"%s_to_%s_s%d" % c: (float(acc[q]["tac7"][c] - acc[q]["tac3"][c])
                                           if c in acc[q]["tac7"] and c in acc[q]["tac3"] else None)
                      for c in CELLS}
    rec = {"n_images": cnt[q], "mean_s": float(np.mean(sval[q])), "mean_ndet": float(np.mean(ndet[q])),
           "mean_abs_log_err": float(np.mean(err[q])) if err[q] else None,
           "source": m["source"], "tac3": m["tac3"], "tac7": m["tac7"], "delta_tac7_tac3": d,
           "delta_per_cell": per_cell_delta}
    out["bins"][LAB[q]] = rec
    print("| %s | %d | %.3f | %.1f | %.3f | %.2f | %.2f | %.2f | %+.2f |"
          % (LAB[q], cnt[q], rec["mean_s"], rec["mean_ndet"], rec["mean_abs_log_err"] or float("nan"),
             m["source"], m["tac3"], m["tac7"], d))

# paired across the 18 (direction, seed) cells: is Q4 - Q1 of the delta positive?
d1 = out["bins"][LAB[0]]["delta_per_cell"]
d4 = out["bins"][LAB[NQ - 1]]["delta_per_cell"]
keys = [k for k in d1 if d1[k] is not None and d4.get(k) is not None]
diff = np.array([d4[k] - d1[k] for k in keys])
half = 2.110 * float(np.std(diff, ddof=1)) / np.sqrt(len(diff))   # t_{.975, 17} = 2.110
out["q4_minus_q1"] = {"mean": float(np.mean(diff)), "std": float(np.std(diff, ddof=1)),
                      "ci95_half": half, "n_cells": int(len(diff)),
                      "frac_positive": float(np.mean(diff > 0))}
n = len(diff)
out["monotone"] = bool(all(out["bins"][LAB[k]]["delta_tac7_tac3"] <= out["bins"][LAB[k + 1]]["delta_tac7_tac3"]
                           for k in range(NQ - 1)))
print("\nQ4 - Q1 of the TAC-7 gain: %+.2f AP (std %.2f over %d cells, positive in %.0f%% of them)"
      % (out["q4_minus_q1"]["mean"], out["q4_minus_q1"]["std"], n, 100 * out["q4_minus_q1"]["frac_positive"]))
print("gain monotone in the spread quartile: %s" % out["monotone"])
json.dump(out, open("output/analysis/uva_quartiles.json", "w"), indent=1)
print("wrote output/analysis/uva_quartiles.json")
