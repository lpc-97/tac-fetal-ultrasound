# -*- coding: utf-8 -*-
"""Calibration of the posterior scale uncertainty of TAC (Heart-4CC, Faster R-CNN, test, three seeds).
For every test image the pass-1 IRLS gives u* and the Laplace posterior std s (canon_factors.json of a v0.5 run: [f, v, ndet, s]);
the oracle factor f* comes from output/analysis/oracle_factors_<dir>_test.json. Reports, per bin of s: number of images, median
|log f - log f*|, the fraction of images whose oracle lies inside u* +- 1.96 s (nominal 95 %) and inside the acquired bracket
u* +- clip(K s, MIN, MAX), plus the same statistics per number of first-pass candidates, and the mean number of views of the
count-mode variant per candidate group. Usage: python analysis_uncert.py <bracket_variant> <count_variant>."""
import json
import math
import os
import sys

import numpy as np

os.chdir(os.environ.get("TAC_ROOT", "."))
VB = sys.argv[1] if len(sys.argv) > 1 else "u7_k17"; VC = sys.argv[2] if len(sys.argv) > 2 else "ua_t10"
K, MIN, MAX = 1.732, 0.15, 0.5
DIRS = ["c1_to_c2", "c1_to_c3", "c2_to_c1", "c2_to_c3", "c3_to_c1", "c3_to_c2"]
rows = []
for s in ("42", "43", "44"):
    for d in DIRS:
        fb = f"output/canon/{VB}_s{s}_{d}_test/canon_factors.json"; fo = f"output/analysis/oracle_factors_{d}_test.json"
        if not (os.path.exists(fb) and os.path.exists(fo)):
            continue
        F = json.load(open(fb))["factors"]; O = json.load(open(fo))["factors"]
        fc = f"output/canon/{VC}_s{s}_{d}_test/canon_stats.json"
        C = json.load(open(f"output/canon/{VC}_s{s}_{d}_test/canon_factors.json"))["factors"] if os.path.exists(fc) else {}
        for k, v in F.items():
            if k not in O or len(v) < 4 or v[3] is None or v[3] < 0:
                continue
            u = math.log(v[0]); uo = math.log(max(O[k][0], 1e-3)); sd = v[3]; nd = v[2]
            rows.append({"seed": s, "dir": d, "u": u, "uo": uo, "s": sd, "ndet": nd, "err": abs(u - uo), "in95": abs(u - uo) <= 1.96 * sd,
                         "inbr": abs(u - uo) <= float(np.clip(K * sd, MIN, MAX)), "hw": float(np.clip(K * sd, MIN, MAX))})
print("images:", len(rows))
S = np.array([r["s"] for r in rows]); E = np.array([r["err"] for r in rows]); ND = np.array([r["ndet"] for r in rows])
IN95 = np.array([r["in95"] for r in rows]); INB = np.array([r["inbr"] for r in rows]); HW = np.array([r["hw"] for r in rows])
out = {"variants": [VB, VC], "n": len(rows), "by_s": [], "by_ndet": [], "spearman_s_err": None}
from scipy.stats import spearmanr
rho = spearmanr(S, E); out["spearman_s_err"] = [float(rho[0]), float(rho[1])]
print(f"Spearman(s, |err|) = {rho[0]:.3f} (p={rho[1]:.1e})")
print(f"{'s bin':14s} {'n':>5s} {'med err':>8s} {'x':>6s} {'in95%':>6s} {'in brk':>7s} {'hw':>6s}")
for lo, hi in [(0, 0.06), (0.06, 0.08), (0.08, 0.10), (0.10, 0.15), (0.15, 0.25), (0.25, 10)]:
    m = (S >= lo) & (S < hi)
    if m.sum() == 0:
        continue
    rec = {"lo": lo, "hi": hi, "n": int(m.sum()), "med_err": float(np.median(E[m])), "in95": float(IN95[m].mean()), "in_bracket": float(INB[m].mean()), "hw_mean": float(HW[m].mean())}
    out["by_s"].append(rec); print(f"[{lo:.2f},{hi:.2f}) {rec['n']:5d} {rec['med_err']:8.3f} {math.exp(rec['med_err']):6.2f} {rec['in95']:6.2f} {rec['in_bracket']:7.2f} {rec['hw_mean']:6.3f}")
print(f"{'ndet group':14s} {'n':>5s} {'med s':>7s} {'med err':>8s} {'in95%':>6s} {'in brk':>7s}")
for name, m in [("1-2", (ND >= 1) & (ND <= 2)), ("3-5", (ND >= 3) & (ND <= 5)), (">5", ND > 5)]:
    if m.sum() == 0:
        continue
    rec = {"group": name, "n": int(m.sum()), "med_s": float(np.median(S[m])), "med_err": float(np.median(E[m])), "in95": float(IN95[m].mean()), "in_bracket": float(INB[m].mean())}
    out["by_ndet"].append(rec); print(f"{name:14s} {rec['n']:5d} {rec['med_s']:7.3f} {rec['med_err']:8.3f} {rec['in95']:6.2f} {rec['in_bracket']:7.2f}")
# views of the count-mode variant per candidate group (from its canon_stats: extra_frac) and overall
vc = []
for s in ("42", "43", "44"):
    for d in DIRS:
        p = f"output/canon/{VC}_s{s}_{d}_test/canon_stats.json"
        if os.path.exists(p):
            j = json.load(open(p)); vc.append((j.get("views_per_img"), j.get("extra_frac")))
if vc:
    out["count_variant"] = {"views_per_img": float(np.mean([x[0] for x in vc])), "extra_frac": float(np.mean([x[1] for x in vc if x[1] is not None]))}
    print("count variant:", out["count_variant"])
json.dump(out, open("output/analysis/uncert_calibration.json", "w"), indent=1); print("UNCERT_CALIB_DONE")
