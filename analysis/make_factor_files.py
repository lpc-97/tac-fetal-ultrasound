# -*- coding: utf-8 -*-
"""Externally supplied scale factors for the reviewer experiments (Heart-4CC by default; DSN=abdomen|spine, DET=fs|rt supported):
  oracle   : per-image factor from the GROUND-TRUTH boxes under the source prior (weighted LS of mu_c - z over GT boxes, no shrinkage,
             r_i = 1) -> output/analysis/oracle_factors_<DP><src>_to_<tgt>_<split>.json
  global   : one factor per stream = median of the TAC per-image factors of the corresponding tacr2 run (per seed)
             -> output/analysis/global_factors_<DP><src>_to_<tgt>_s<seed>.json
  valscale : one factor per direction = (AP-optimal single scale on the target VAL split, seed 42) / s0
             -> output/analysis/valscale_factors_<DP><src>_to_<tgt>.json   (Heart-4CC FRCNN only: output/scales/scale_analysis_val.json)
usage: python make_factor_files.py [DSN] [DET]"""
import glob
import json
import math
import os
import sys

import numpy as np

os.chdir(os.environ.get("TAC_ROOT", "."))
DSN = sys.argv[1] if len(sys.argv) > 1 else "4c"; DET = sys.argv[2] if len(sys.argv) > 2 else "fs"
DP = "" if DSN == "4c" else DSN + "_"; PT = "rt_" if DET == "rt" else ""; OP = "rt_" if DET == "rt" else ""
ROOT = {"4c": "datasets/Heart-4cc", "abdomen": "datasets/abdomen", "spine": "datasets/FUSSD"}[DSN]
DOMS = ["c1", "c2", "c3"] if DSN == "4c" else ["ge", "ph", "sa"]
DIRS = [(a, b) for a in DOMS for b in DOMS if a != b]
os.makedirs("output/analysis", exist_ok=True)


def prior_of(src):
    pj = json.load(open(f"output/anat_prior_{PT}{DP}{src}.json"))
    s0 = int(pj.get("indomain_best_scale", 800)); shift = math.log(s0 / 800.0)
    cat_ids = sorted(int(k) for k in pj["names"])  # contiguous ids 0..K-1
    mu = {k: v["logsize_mean"] + shift for k, v in ((int(k), v) for k, v in pj["canonical_at_800"].items())}
    sig = {k: v["logsize_std"] for k, v in ((int(k), v) for k, v in pj["canonical_at_800"].items())}
    return s0, mu, sig


def gt_json(dom, split):
    d = dom if DSN == "4c" else dom.upper()
    return f"{ROOT}/{d}/{split}.json"


n_written = 0
for src, tgt in DIRS:
    s0, mu, sig = prior_of(src)
    for split in ("test", "val"):
        g = json.load(open(gt_json(tgt, split)))
        cat_ids = sorted(c["id"] for c in g["categories"]); cid2idx = {c: i for i, c in enumerate(cat_ids)}
        per = {}
        for a in g["annotations"]:
            per.setdefault(a["image_id"], []).append(a)
        fac = {}
        for im in g["images"]:
            H, W = im["height"], im["width"]; k0 = min(s0 / min(H, W), 4000 / max(H, W))
            num = den = 0.0; n = 0
            for a in per.get(im["id"], []):
                w, h = a["bbox"][2], a["bbox"][3]
                if w <= 0 or h <= 0:
                    continue
                k = cid2idx[a["category_id"]]; z = math.log(math.sqrt(w * h) * k0); s2 = sig.get(k, 0.3) ** 2 + 0.1 ** 2
                num += (mu.get(k, 0.0) - z) / s2; den += 1.0 / s2; n += 1
            u = num / den if den > 0 else 0.0
            u = float(np.clip(u, math.log(0.3), math.log(3.0)))
            fac[os.path.basename(im["file_name"])] = [float(math.exp(u)), 0.0, n]
        out = f"output/analysis/oracle_factors_{DP}{src}_to_{tgt}_{split}.json"
        json.dump({"s0": s0, "kind": "oracle", "factors": fac}, open(out, "w")); n_written += 1
        if split == "test":
            f = np.array([v[0] for v in fac.values()]); print(f"oracle {src}->{tgt} test: n={len(f)} median f={np.median(f):.2f} p10-90=[{np.percentile(f,10):.2f},{np.percentile(f,90):.2f}]")
    # global (stream-level) factor from the TAC per-image factors of each seed's tacr2 test run
    for seed in (42, 43, 44):
        p = f"output/canon/{OP}{DP}tacr2_s{seed}_{src}_to_{tgt}_test/canon_factors.json"
        if not os.path.exists(p):
            print("missing", p); continue
        j = json.load(open(p)); f = np.array([v[0] for v in j["factors"].values()]); g_med = float(np.median(f))
        out = f"output/analysis/global_factors_{DP}{src}_to_{tgt}_s{seed}.json"
        json.dump({"s0": j["s0"], "kind": "global-median", "value": g_med, "factors": {k: [g_med, 0.0, 0] for k in j["factors"]}}, open(out, "w")); n_written += 1
    # val-selected fixed scale (Heart-4CC FRCNN; scale_analysis_val.json keys 'c1->c2' with per-scale AP)
    sa = "output/scales/scale_analysis_val.json"
    if DSN == "4c" and DET == "fs" and os.path.exists(sa):
        d = json.load(open(sa))[f"{src}->{tgt}"]; best = max(d, key=lambda k: d[k]["AP"]); fv = int(best) / s0
        g = json.load(open(gt_json(tgt, "test")))
        out = f"output/analysis/valscale_factors_{DP}{src}_to_{tgt}.json"
        json.dump({"s0": s0, "kind": "val-selected-scale", "scale": int(best), "value": fv, "factors": {os.path.basename(im["file_name"]): [fv, 0.0, 0] for im in g["images"]}}, open(out, "w")); n_written += 1
        print(f"valscale {src}->{tgt}: best val scale {best} -> factor {fv:.3f} (s0={s0})")
print("FACTOR_FILES_DONE", n_written)
