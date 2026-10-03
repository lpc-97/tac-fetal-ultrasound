# -*- coding: utf-8 -*-
"""De-duplicated in-domain evaluation (Heart-4CC, Faster R-CNN, source centre = target centre, three seeds).

The released splits are at image level and the centre-3 test split contains frames captured within one minute of
training frames, so the in-domain numbers of the control experiment could be inflated by near-duplicate frames. This
script groups the in-domain test images into case clusters with the same rule as cluster_bootstrap.py (DICOM study
prefix, capture time stamps within 60 s, otherwise one image per cluster), keeps the first frame of every cluster, and
recomputes COCO AP for the source model, TAC-3 and TAC-7 on that de-duplicated set.

Writes output/analysis/indomain_dedup.{json,md}.  usage: python analysis_indomain_dedup.py
"""
import contextlib
import io
import json
import os
import re
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

os.chdir(os.environ.get("TAC_ROOT", "."))
DOMS = ["c1", "c2", "c3"]; SEEDS = ["42", "43", "44"]
ROWS = {"source": "output/so_fz_s{s}_{a}_to_{a}",
        "tac3": "output/canon/tac_s{s}_{a}_to_{a}_test",
        "tac7": "output/canon/tac_v7_s{s}_{a}_to_{a}_test"}


def preds_file(d):
    f = os.path.join(d, "inference", "coco_instances_results.json")
    return f if os.path.exists(f) else os.path.join(d, "coco_instances_results.json")


def cluster_keys(names):
    keys = {}; ts = []
    for n in names:
        base = os.path.splitext(n)[0]
        m = re.match(r"^(\d{4})\.(\d{2})\.(\d{2}) (\d{2})\.(\d{2})\.(\d{2})$", base)
        if m:
            y, mo, d, h, mi, se = map(int, m.groups()); ts.append((n, ((y * 12 + mo) * 31 + d) * 86400 + h * 3600 + mi * 60 + se)); continue
        m = re.match(r"^(\d{8})-(\d{6})$", base)
        if m:
            ts.append((n, int(m.group(1)) * 86400 + int(m.group(2)[:2]) * 3600 + int(m.group(2)[2:4]) * 60 + int(m.group(2)[4:]))); continue
        m = re.match(r"^(\d+)_(\d+)$", base)
        if m:
            keys[n] = "id:" + m.group(1); continue
        if base.count(".") >= 6:
            keys[n] = "uid:" + ".".join(base.split(".")[:-2]); continue
        keys[n] = "img:" + base
    ts.sort(key=lambda t: t[1]); cur = None; last = None
    for n, t in ts:
        if last is None or t - last > 60:
            cur = f"ts:{t}"
        keys[n] = cur; last = t
    return keys


def ap_on(gt, dt, img_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, dt, "bbox"); E.params.imgIds = sorted(img_ids); E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[0] * 100)


full = defaultdict(list); dedup = defaultdict(list); indep = defaultdict(list); stats = {}
for a in DOMS:
    gt = COCO(f"datasets/Heart-4cc/{a}/test.json")
    names = {i: gt.loadImgs(i)[0]["file_name"] for i in gt.getImgIds()}
    tr = json.load(open(f"datasets/Heart-4cc/{a}/train.json"))
    tr_names = [im["file_name"] for im in tr["images"]]
    # clusters over TEST u TRAIN together, so that a test frame acquired in the same study / within 60 s of a
    # training frame lands in the same cluster as that training frame
    keys = cluster_keys([os.path.basename(n) for n in list(names.values()) + tr_names])
    tr_keys = {keys[os.path.basename(n)] for n in tr_names}
    by_cluster = defaultdict(list)
    for iid, n in sorted(names.items()):
        by_cluster[keys[os.path.basename(n)]].append(iid)
    keep = sorted(v[0] for v in by_cluster.values())                      # one frame per test cluster
    keep_indep = sorted(v[0] for k, v in by_cluster.items() if k not in tr_keys)   # and not sharing a cluster with training
    stats[a] = {"n_images": len(names), "n_clusters": len(by_cluster), "max_cluster": max(len(v) for v in by_cluster.values()),
                "n_clusters_shared_with_train": sum(1 for k in by_cluster if k in tr_keys),
                "n_images_shared_with_train": sum(len(v) for k, v in by_cluster.items() if k in tr_keys),
                "n_independent_clusters": len(keep_indep)}
    print(a, stats[a], flush=True)
    for s in SEEDS:
        for r, tpl in ROWS.items():
            f = preds_file(tpl.format(s=s, a=a))
            if not os.path.exists(f):
                print("missing", f, flush=True); continue
            with contextlib.redirect_stdout(io.StringIO()):
                dt = gt.loadRes(f)
            full[(r, s)].append(ap_on(gt, dt, list(names)))
            dedup[(r, s)].append(ap_on(gt, dt, keep))
            if len(keep_indep) >= 5:
                indep[(r, s)].append(ap_on(gt, dt, keep_indep))

out = {"clusters": stats, "rows": {}}
md = ["## In-domain control: full test splits, one frame per case cluster, and clusters independent of the training split",
      "(Heart-4CC, Faster R-CNN, three centres x three seeds)\n",
      "| row | full AP | one frame per cluster | clusters not shared with train |", "|---|---|---|---|"]
for r in ROWS:
    fm = [float(np.mean(full[(r, s)])) for s in SEEDS if (r, s) in full]
    dm = [float(np.mean(dedup[(r, s)])) for s in SEEDS if (r, s) in dedup]
    im = [float(np.mean(indep[(r, s)])) for s in SEEDS if (r, s) in indep]
    if not fm:
        continue
    out["rows"][r] = {"full_mean": round(float(np.mean(fm)), 2), "full_std": round(float(np.std(fm)), 2),
                      "dedup_mean": round(float(np.mean(dm)), 2), "dedup_std": round(float(np.std(dm)), 2),
                      "indep_mean": round(float(np.mean(im)), 2) if im else None,
                      "indep_std": round(float(np.std(im)), 2) if im else None,
                      "diff": round(float(np.mean(dm) - np.mean(fm)), 2)}
    v = out["rows"][r]
    ic = f"{v['indep_mean']:.2f} ± {v['indep_std']:.2f}" if v["indep_mean"] is not None else "n/a"
    md.append(f"| {r} | {v['full_mean']:.2f} ± {v['full_std']:.2f} | {v['dedup_mean']:.2f} ± {v['dedup_std']:.2f} | {ic} |")
if "tac3" in out["rows"] and "source" in out["rows"]:
    out["gain_full"] = round(out["rows"]["tac3"]["full_mean"] - out["rows"]["source"]["full_mean"], 2)
    out["gain_dedup"] = round(out["rows"]["tac3"]["dedup_mean"] - out["rows"]["source"]["dedup_mean"], 2)
    out["gain_full_tac7"] = round(out["rows"]["tac7"]["full_mean"] - out["rows"]["source"]["full_mean"], 2)
    out["gain_dedup_tac7"] = round(out["rows"]["tac7"]["dedup_mean"] - out["rows"]["source"]["dedup_mean"], 2)
    md.append(f"\nTAC-3 gain over the source model: {out['gain_full']:+.2f} on the full splits, {out['gain_dedup']:+.2f} de-duplicated.")
    md.append(f"TAC-7 gain over the source model: {out['gain_full_tac7']:+.2f} on the full splits, {out['gain_dedup_tac7']:+.2f} de-duplicated.")
    if out["rows"]["source"].get("indep_mean") is not None:
        out["gain_indep"] = round(out["rows"]["tac3"]["indep_mean"] - out["rows"]["source"]["indep_mean"], 2)
        out["gain_indep_tac7"] = round(out["rows"]["tac7"]["indep_mean"] - out["rows"]["source"]["indep_mean"], 2)
        md.append(f"On the clusters that share no case with the training split: {out['gain_indep']:+.2f} (TAC-3) and {out['gain_indep_tac7']:+.2f} (TAC-7).")
json.dump(out, open("output/analysis/indomain_dedup.json", "w"), indent=1)
open("output/analysis/indomain_dedup.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("INDOMAIN_DEDUP_DONE")
