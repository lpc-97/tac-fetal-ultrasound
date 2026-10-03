"""Fig. 10 - uncertainty-aware TAC (Heart-4CC, Faster R-CNN):
(a) calibration of the posterior std s of the log-scale factor: median |log f - log f*| per bin of s against the value expected for a
    calibrated Gaussian (0.674 s), with the coverage of the 95 % interval u* +- 1.96 s annotated (test, three seeds, oracle f* from the labels);
(b) accuracy against the number of views per image: TAC-3 / TAC-7 / TAC-11, the uncertainty-aware variants (bracket, + posterior weights,
    adaptive count) and the blind ensembles (test, three seeds);
(c) selection of the count threshold tau_s on the validation split (seed 42): mean val AP and views per image against tau_s.
Data: results/reviewer/uncert_calibration.json, results/reviewer/uncert_summary.json, results/reviewer/reviewer_summary.json."""
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import C, OURS, BASE, RES, panel_label, save

CAL = json.load(open(os.path.join(RES, "reviewer", "uncert_calibration.json"), encoding="utf-8"))
U = json.load(open(os.path.join(RES, "reviewer", "uncert_summary.json"), encoding="utf-8"))
R = json.load(open(os.path.join(RES, "reviewer", "reviewer_summary.json"), encoding="utf-8"))["rows"]
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), gridspec_kw={"width_ratios": [1.0, 1.15, 1.0], "wspace": 0.42})

# (a) calibration
ax = axes[0]
bins = CAL["by_s"]; x = np.arange(len(bins))
labs = [f"{b['lo']:.2f}–{b['hi']:.2f}" if b["hi"] < 5 else f"≥{b['lo']:.2f}" for b in bins]
med = [b["med_err"] for b in bins]; ref = [0.674 * (b["lo"] + min(b["hi"], 1.0)) / 2 for b in bins]
ax.bar(x, med, 0.6, color=OURS, label="median $|\\log f-\\log f^{\\star}|$")
ax.plot(x, ref, color=C["ink2"], marker="_", ms=14, mew=1.4, ls="none", label="$0.674\\,s$ (calibrated Gaussian)")
for i, b in enumerate(bins):
    ax.text(x[i], med[i] + 0.012, f"{100*b['in95']:.0f}%", ha="center", va="bottom", fontsize=5.6, color=C["ink"])
    ax.text(x[i], -0.006, f"n={b['n']}", ha="center", va="top", fontsize=5.0, color=C["ink2"])
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=5.6, rotation=30, ha="right", rotation_mode="anchor")
ax.set_xlabel("posterior std $s$ of the log-scale factor", fontsize=7); ax.set_ylabel("error of the estimate (log units)")
ax.set_ylim(-0.03, max(med) * 1.55); ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2, frameon=False)
ax.text(0.98, 0.97, "labels: coverage of $u^{\\star}\\pm1.96s$", transform=ax.transAxes, ha="right", va="top", fontsize=5.4, color=C["ink2"])
panel_label(ax, "(a)", x=-0.2, y=1.08)

# (b) accuracy vs views (test, three seeds)
ax = axes[1]
T = U["test"]
def pt(key, src, lab, col, mk, ms=4.5, mfc=None, zorder=3):
    r = src[key]
    v = r["views"] if "views" in r and r["views"] else None
    ax.errorbar(v, r["mean"], yerr=r["std"], color=col, marker=mk, ms=ms, ls="none", capsize=1.5, elinewidth=0.6, label=lab, mfc=mfc or col, zorder=zorder)
    return v, r["mean"]
line = []
for k, lab in [("tac", "TAC-3"), ("tac_v7", "TAC-7"), ("tac_v11", "TAC-11")]:
    line.append(pt(k, R, lab if k == "tac" else None, OURS, "o"))
line.sort(); ax.plot([p[0] for p in line], [p[1] for p in line], color=OURS, lw=0.9, alpha=0.6, zorder=2, label="TAC, fixed bracket (3 / 7 / 11 views)")
for k, lab, mk in [(U["test_keys"]["bracket"], "TAC-U, bracket $u^{\\star}\\pm h$", "s"), (U["test_keys"]["weights"], "TAC-U + posterior weights", "D"), (U["test_keys"]["count"], "TAC-U, adaptive count", "*")]:
    if k in T:
        pt(k, T, lab, C["teal"], mk, ms=6 if mk == "*" else 4.5, zorder=4)
for k, lab in [("ss_fix3_wbf_ar", "blind fixed scales + re-scoring"), ("ss_fix7_wbf_ar", None)]:
    pt(k, R, lab, C["gray"], "s")
for k, lab, col in [("ss_all_flip_wbf_ar", "18-view WBF + re-scoring", C["blue"]), ("heatmap18", "18-view heat-map", C["gold"])]:
    pt(k, R, lab, col, "H")
ax.set_xscale("log", basex=2); ax.set_xticks([3, 4, 5, 7, 11, 18]); ax.set_xticklabels(["3", "4", "5", "7", "11", "18"]); ax.minorticks_off()
ax.set_xlabel("views per image (mean)", fontsize=7); ax.set_ylabel("mean AP (%)")
ax.legend(loc="lower right", fontsize=5.3, handlelength=1.1, handletextpad=0.4, frameon=False)
panel_label(ax, "(b)", x=-0.2, y=1.08)

# (c) tau selection on val
ax = axes[2]
V = U["val"]; taus = sorted((float(k.split("_t")[1][:2]) / 100, k) for k in V if k.startswith("ua_t") and not k.endswith("_w") and "AP" in V[k])
xs = [t for t, _ in taus]; ys = [V[k]["AP"] for _, k in taus]; vs = [V[k]["views"] for _, k in taus]
ax.plot(xs, ys, color=C["teal"], marker="o", ms=4, lw=1.0, label="TAC-U, adaptive count")
for xv, yv, vv in zip(xs, ys, vs):
    ax.text(xv, yv + 0.05, f"{vv:.1f}", ha="center", va="bottom", fontsize=5.4, color=C["ink2"])
ax.axhline(V["tac_v7"]["AP"], color=OURS, lw=0.8, ls="--", label="TAC-7 (7 views)")
ax.axhline(V["tac"]["AP"], color=C["gray"], lw=0.8, ls=":", label="TAC-3 (3 views)")
ax.set_xlabel("count threshold $\\tau_s$ (val, seed 42)", fontsize=7); ax.set_ylabel("mean val AP (%)")
ax.legend(loc="lower left", fontsize=5.6, handlelength=1.2, frameon=False)
ax.text(0.98, 0.97, "labels: views per image", transform=ax.transAxes, ha="right", va="top", fontsize=5.4, color=C["ink2"])
panel_label(ax, "(c)", x=-0.22, y=1.08)
save(fig, "fig10_uncert")
