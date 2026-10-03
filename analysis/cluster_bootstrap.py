# -*- coding: utf-8 -*-
"""Reviewer item 6: data independence and case-level uncertainty.
(1) Clusters of near-duplicate / same-case test frames per Heart-4CC centre from the file names (c1/c2 DICOM UIDs -> study-level
    prefix; '<id>_<frame>' names -> id; 'YYYY.MM.DD HH.MM.SS' capture names -> frames within 60 s are linked).
(2) De-duplicated evaluation: AP of the main rows on one frame per cluster (first frame), 3 seeds.
(3) Cluster-level paired bootstrap (B resamples of clusters with replacement, per direction, same resample for all methods and seeds):
    95% percentile CI of the 6-direction mean AP difference for the key comparisons. AP under image multiplicities is computed from
    COCOeval's per-image match records (validated against COCOeval on the unweighted set).
usage: python cluster_bootstrap.py [dsn: 4c|abdomen|spine] [B]"""
import contextlib
import io
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.chdir(os.environ.get("TAC_ROOT", "."))
DSN = sys.argv[1] if len(sys.argv) > 1 else "4c"; B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SEEDS = [42, 43, 44]
if DSN == "4c":
    ROOT, DOMS, DP = "datasets/Heart-4cc", ["c1", "c2", "c3"], ""
else:
    ROOT, DOMS, DP = ("datasets/abdomen" if DSN == "abdomen" else "datasets/FUSSD"), ["ge", "ph", "sa"], DSN + "_"
DIRS = [(a, b) for a in DOMS for b in DOMS if a != b]
best = json.load(open("output/baselines/best_val.json" if DSN == "4c" else f"output/baselines/best_val_{DSN}.json"))
ROWS = {
    "source": lambda s, a, b: f"output/so_fz_{DP}s{s}_{a}_to_{b}",
    "heatmap18": lambda s, a, b: f"output/tta_heatmap_{DP}s{s}_{a}_to_{b}",
    "wbf18_ar": lambda s, a, b: f"output/scalesel/{DP}ss_all_flip_wbf_ar_s{s}_{a}_to_{b}_test",
    "cotta": lambda s, a, b: f"output/baselines/{DP}{best['cotta'][f'{a}_to_{b}']}_s{s}_{a}_to_{b}_test",
    "tac3": lambda s, a, b: f"output/canon/{DP}tac_s{s}_{a}_to_{b}_test",
    "tac3_2r": lambda s, a, b: f"output/canon/{DP}{'tac_rounds2' if DSN == '4c' else 'tacr2'}_s{s}_{a}_to_{b}_test",
    "tac7": lambda s, a, b: f"output/canon/{DP}tac_v7_s{s}_{a}_to_{b}_test",
}
PAIRS = [("tac7", "source"), ("tac7", "heatmap18"), ("tac7", "wbf18_ar"), ("tac7", "cotta"), ("tac3_2r", "heatmap18"), ("tac3_2r", "wbf18_ar"), ("tac3_2r", "cotta"), ("tac3", "wbf18_ar"), ("tac7", "tac3_2r")]


def preds_file(d):
    f = os.path.join(d, "inference", "coco_instances_results.json")
    return f if os.path.exists(f) else os.path.join(d, "coco_instances_results.json")


# ---------------------------------------------------------------- clusters from file names
def cluster_keys(names):
    keys = {}
    ts = []
    for n in names:
        base = os.path.splitext(n)[0]
        m = re.match(r"^(\d{4})\.(\d{2})\.(\d{2}) (\d{2})\.(\d{2})\.(\d{2})$", base)
        if m:
            y, mo, d, h, mi, se = map(int, m.groups()); ts.append((n, ((y * 12 + mo) * 31 + d) * 86400 + h * 3600 + mi * 60 + se)); continue
        m = re.match(r"^(\d{8})-(\d{6})$", base)
        if m:
            ts.append((n, int(m.group(1)) * 86400 + int(m.group(2)[:2]) * 3600 + int(m.group(2)[2:4]) * 60 + int(m.group(2)[4:]))); continue
        m = re.match(r"^(\d+)_(\d+)$", base)          # '<patient id>_<frame>' (numeric id); anonymised 'te_00032' stays image-level
        if m:
            keys[n] = "id:" + m.group(1); continue
        if base.count(".") >= 6:                       # DICOM UID: drop the trailing frame counters
            keys[n] = "uid:" + ".".join(base.split(".")[:-2]); continue
        keys[n] = "img:" + base                       # anonymised names: each image its own cluster
    ts.sort(key=lambda t: t[1]); cur = None; last = None
    for n, t in ts:                                     # capture time stamps: link frames within 60 s
        if last is None or t - last > 60:
            cur = f"ts:{t}"
        keys[n] = cur; last = t
    return keys


results = {"dsn": DSN, "B": B, "clusters": {}, "dedup_ap": {}, "bootstrap": {}}
records = {}   # (dir, seed, row) -> per-image records
gt_imgs = {}
for a, b in DIRS:
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(f"{ROOT}/{b if DSN == '4c' else b.upper()}/test.json")
    imgs = gt.loadImgs(gt.getImgIds()); names = {im["id"]: os.path.basename(im["file_name"]) for im in imgs}
    keys = cluster_keys(list(names.values()))
    clusters = defaultdict(list)
    for iid, n in names.items():
        clusters[keys[n]].append(iid)
    cl = [sorted(v) for v in clusters.values()]
    if b not in results["clusters"]:
        sizes = [len(c) for c in cl]
        results["clusters"][b] = {"n_images": len(imgs), "n_clusters": len(cl), "max_cluster": max(sizes), "n_multi": sum(s > 1 for s in sizes), "kinds": dict(zip(*np.unique([k.split(':')[0] for k in clusters], return_counts=True)))}
        results["clusters"][b]["kinds"] = {k: int(v) for k, v in results["clusters"][b]["kinds"].items()}
    gt_imgs[(a, b)] = (gt, cl)
    for s in SEEDS:
        for r, pat in ROWS.items():
            f = preds_file(pat(s, a, b))
            with contextlib.redirect_stdout(io.StringIO()):
                dt = gt.loadRes(f); E = COCOeval(gt, dt, "bbox"); E.evaluate()   # evaluateImg uses maxDet = 100 for every (cat, area, img)
            recs = {}
            nI = len(E.params.imgIds); nA = len(E.params.areaRng)   # evalImgs ordered by (cat, areaRng, img); area index 0 = 'all'
            for k_i, cat in enumerate(E.params.catIds):
                for i_i, iid in enumerate(E.params.imgIds):
                    e = E.evalImgs[k_i * nA * nI + 0 * nI + i_i]
                    if e is None:
                        continue
                    dtS = np.asarray(e["dtScores"], dtype=np.float64); dtm = np.asarray(e["dtMatches"]); dtIg = np.asarray(e["dtIgnore"], dtype=bool); gtIg = np.asarray(e["gtIgnore"], dtype=bool)
                    recs[(cat, iid)] = (dtS, (dtm > 0) & (~dtIg), (dtm == 0) & (~dtIg), int((~gtIg).sum()))
            records[(a, b, s, r)] = (recs, list(E.params.catIds), list(E.params.iouThrs), list(E.params.recThrs))
            # validation: unweighted AP must equal COCOeval
            with contextlib.redirect_stdout(io.StringIO()):
                E.accumulate(); E.summarize()
            records[(a, b, s, r, "ref")] = float(E.stats[0] * 100)
    print("records", a, "->", b, "clusters", len(cl), "of", len(imgs), flush=True)


def weighted_ap(rec_tuple, weights):
    """COCO AP@[.5:.95] (area all, maxDet 100) for image multiplicities `weights` (dict img -> count, 0 = excluded)."""
    recs, cats, ious, recThrs = rec_tuple; T = len(ious); R = np.asarray(recThrs); aps = []
    for cat in cats:
        S, TP, FP, npig = [], [], [], 0
        for (c, iid), (dtS, tp, fp, ng) in recs.items():
            if c != cat:
                continue
            w = weights.get(iid, 0)
            if w == 0:
                continue
            npig += w * ng
            if len(dtS):
                S.append(dtS); TP.append(tp * w); FP.append(fp * w)
        if npig == 0:
            continue
        if not S:
            aps.append(0.0); continue
        S = np.concatenate(S); order = np.argsort(-S, kind="mergesort")
        tp = np.concatenate(TP, axis=1)[:, order].astype(np.float64); fp = np.concatenate(FP, axis=1)[:, order].astype(np.float64)
        tpc = np.cumsum(tp, axis=1); fpc = np.cumsum(fp, axis=1)
        rc = tpc / npig; pr = tpc / np.maximum(tpc + fpc, np.spacing(1))
        for t in range(T):
            p = pr[t].copy()
            for i in range(len(p) - 2, -1, -1):
                p[i] = max(p[i], p[i + 1])
            inds = np.searchsorted(rc[t], R, side="left"); q = np.zeros(len(R))
            ok = inds < len(p); q[ok] = p[inds[ok]]
            aps.append(float(q.mean()))
    return 100 * float(np.mean(aps)) if aps else float("nan")


# validation of the weighted AP against COCOeval
maxdiff = 0.0
for (a, b, s, r), rt in [(k, v) for k, v in records.items() if len(k) == 4]:
    gt, cl = gt_imgs[(a, b)]; w = {iid: 1 for c in cl for iid in c}
    d = abs(weighted_ap(rt, w) - records[(a, b, s, r, "ref")]); maxdiff = max(maxdiff, d)
results["validation_max_abs_diff_vs_cocoeval"] = maxdiff; print("weighted-AP validation: max |diff| vs COCOeval =", round(maxdiff, 6), flush=True)

# (2) de-duplicated evaluation: one frame per cluster
for r in ROWS:
    per_seed = []
    for s in SEEDS:
        vals = []
        for a, b in DIRS:
            gt, cl = gt_imgs[(a, b)]; w = {c[0]: 1 for c in cl}
            vals.append(weighted_ap(records[(a, b, s, r)], w))
        per_seed.append(float(np.mean(vals)))
    full = [float(np.mean([records[(a, b, s, r, "ref")] for a, b in DIRS])) for s in SEEDS]
    results["dedup_ap"][r] = {"dedup_mean": float(np.mean(per_seed)), "dedup_std": float(np.std(per_seed)), "full_mean": float(np.mean(full)), "full_std": float(np.std(full))}
    print(f"dedup {r:10s} full {np.mean(full):.2f} -> dedup {np.mean(per_seed):.2f} ± {np.std(per_seed):.2f}", flush=True)

# (3) cluster-level paired bootstrap
rng = np.random.RandomState(0); t0 = time.time()
boot = {f"{x}-{y}": [] for x, y in PAIRS}; boot_dir = {f"{x}-{y}": {f"{a}_to_{b}": [] for a, b in DIRS} for x, y in PAIRS}
for it in range(B):
    ws = {}
    for a, b in DIRS:
        gt, cl = gt_imgs[(a, b)]; idx = rng.randint(0, len(cl), len(cl)); w = defaultdict(int)
        for i in idx:
            for iid in cl[i]:
                w[iid] += 1
        ws[(a, b)] = w
    ap = {}
    for (a, b) in DIRS:
        for s in SEEDS:
            for r in ROWS:
                ap[(a, b, s, r)] = weighted_ap(records[(a, b, s, r)], ws[(a, b)])
    for x, y in PAIRS:
        diffs = []
        for a, b in DIRS:
            d = float(np.mean([ap[(a, b, s, x)] - ap[(a, b, s, y)] for s in SEEDS])); diffs.append(d); boot_dir[f"{x}-{y}"][f"{a}_to_{b}"].append(d)
        boot[f"{x}-{y}"].append(float(np.mean(diffs)))
    if (it + 1) % 100 == 0:
        print(f"bootstrap {it + 1}/{B} ({time.time() - t0:.0f}s)", flush=True)
for k, v in boot.items():
    v = np.asarray(v); x, y = k.split("-")
    point = float(np.mean([np.mean([records[(a, b, s, x, "ref")] - records[(a, b, s, y, "ref")] for s in SEEDS]) for a, b in DIRS]))
    results["bootstrap"][k] = {"point": point, "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "p_le_0": float(np.mean(v <= 0)),
                               "per_dir": {d: {"point": float(np.mean(vals)), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]} for d, vals in boot_dir[k].items()}}
    print(f"{k:22s} Δ = {point:+.2f}  95% CI [{results['bootstrap'][k]['ci95'][0]:+.2f}, {results['bootstrap'][k]['ci95'][1]:+.2f}]  P(Δ<=0)={results['bootstrap'][k]['p_le_0']:.3f}")
json.dump(results, open(f"output/analysis/cluster_bootstrap_{DSN}.json", "w"), indent=1)
print("CLUSTER_BOOTSTRAP_DONE", DSN)
