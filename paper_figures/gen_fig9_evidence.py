"""Fig. 9 - the three evidence chains requested by the reviewers (Heart-4CC, Faster R-CNN, test, 3 seeds):
(a) is scale the shift? AP of the source model with one view at the default scale, the val-selected fixed scale, a stream-level
    factor, the TAC per-image factor and the oracle (GT-derived) factor, and the same factors inside the TAC-3 pipeline;
(b) is per-image inference needed? gain of TAC-7 over the source model as a function of the oracle shift of the image
    (groups of |log f*|), with the scale-estimate error of TAC;
(c) first-pass reliability: AP by number of first-pass candidates and the estimate error of the four estimators.
Data: results/reviewer/reviewer_summary.json, results/reviewer/firstpass_analysis.json."""
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import C, OURS, BASE, RES, W_FULL, panel_label, layout, save

R = json.load(open(os.path.join(RES, "reviewer", "reviewer_summary.json"), encoding="utf-8"))["rows"]
FP = json.load(open(os.path.join(RES, "reviewer", "firstpass_analysis.json"), encoding="utf-8"))
from tabdata import tac_single_view  # noqa: E402  the TAC single view is the run of Table 6(a), as in Table S10 (2026-11-10)
R["tac_1v"] = tac_single_view()
fig, axes = plt.subplots(1, 4, figsize=(W_FULL, 2.5), gridspec_kw={"width_ratios": [1.22, 1.12, 1.0, 0.88]})

# (a) scale chain
ax = axes[0]
labs = ["default", "fixed", "stream", "TAC", "oracle"]  # spelled out in the caption
k1 = ["source", "fac_valscale_1v", "fac_global_1v", "tac_1v", "fac_oracle_1v"]; k3 = [None, "fac_valscale", "fac_global", "tac", "fac_oracle"]
x = np.arange(5); w = 0.16
# dot-and-error-bar plot (the axis starts at 45, so bars would exaggerate the differences): filled marker = single view,
# open marker = the same factor inside the TAC-3 pipeline; the values are given in the scale-chain table
cols = [C["gray"], C["gray"], C["gray"], OURS, C["teal"]]
for i, (a_, c) in enumerate(zip(k1, k3)):
    if c:
        ax.plot([x[i] - w, x[i] + w], [R[a_]["mean"], R[c]["mean"]], color=cols[i], lw=0.8, alpha=0.5, zorder=3)
    ax.errorbar(x[i] - w if c else x[i], R[a_]["mean"], yerr=R[a_]["std"], color=cols[i], marker="o", ms=4.8, mfc=cols[i], mec=cols[i], ls="none", elinewidth=0.6, capsize=1.5, zorder=4)
    if c:
        ax.errorbar(x[i] + w, R[c]["mean"], yerr=R[c]["std"], color=cols[i], marker="o", ms=4.8, mfc="white", mec=cols[i], mew=1.0, ls="none", elinewidth=0.6, capsize=1.5, zorder=4)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.5); ax.set_ylim(45, 57); ax.set_xlim(-0.6, 4.6); ax.set_ylabel("mean test AP (%)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", color=C["gray"], mfc=C["gray"], mec=C["gray"], ms=4.8, ls="none", label="single view"),
                   Line2D([], [], marker="o", color=C["gray"], mfc="white", mec=C["gray"], mew=1.0, ms=4.8, ls="none", label="TAC-3 pipeline")],
          loc="upper left", handlelength=1.0)
ax.set_xlabel("scale factor of the canonicalized view")
panel_label(ax, "(a)", "Fixed, per-image and oracle scale")

# (b) gain vs oracle shift
ax = axes[1]
keys = ["|log f*| <= log 1.25", "log 1.25 - log 1.6", "log 1.6 - log 2", "> log 2"]; xl = ["≤1.25", "1.25–1.6", "1.6–2", ">2"]
src = [FP["shift"][k]["source"] for k in keys]; t7 = [FP["shift"][k]["tac7"] for k in keys]; orc = [FP["shift"][k]["oracle3"] for k in keys]; n = [FP["shift"][k]["n_images"] for k in keys]
x = np.arange(4); w = 0.27
ax.bar(x - w, src, w, color=C["gray"], label="source"); ax.bar(x, t7, w, color=OURS, label="TAC-7"); ax.bar(x + w, orc, w, color=C["teal"], label="TAC-3, oracle $f$")
for i in range(4):
    ax.text(x[i], max(t7[i], orc[i]) + 1.0, f"+{t7[i] - src[i]:.1f}", ha="center", va="bottom", fontsize=7, color=OURS, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=6.5); ax.set_xlabel("oracle shift $\\max(f^{\\star}, 1/f^{\\star})$"); ax.tick_params(axis="x", length=0)
ax.set_ylabel("AP on the group (%)"); ax.set_ylim(0, 78)  # no legend: the legend of (c) lists every series drawn here (caption says so)
panel_label(ax, "(b)", "By oracle shift")

# (c) first-pass candidates: AP and estimator error
ax = axes[2]
keys = ["1-2", "3-5", ">5"]; xl = ["1–2", "3–5", ">5"]
x = np.arange(3); w = 0.2
for j, (k, lab, col) in enumerate([("source", "source", C["gray"]), ("tac3", "TAC-3", C["orange"]), ("tac7", "TAC-7, fixed", OURS), ("oracle3", "TAC-3, oracle $f$", C["teal"])]):
    ax.bar(x + (j - 1.5) * w, [FP["ndet"][g][k] for g in keys], w, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_xlabel("first-pass candidates ($p_i \\geq 0.3$)"); ax.set_ylabel("AP on the group (%)"); ax.set_ylim(0, 78)
ax.tick_params(axis="x", length=0)
ax.legend(loc="upper left", handlelength=1.0)
panel_label(ax, "(c)", "By first-pass evidence")

# (d) estimator error on the same groups, on its own axis
ax = axes[3]
for k, lab, col, mk in [("irls", "IRLS (TAC)", OURS, "o"), ("median", "median", C["ink2"], "s"), ("noclass", "class-agnostic", C["sky"], "^")]:
    ax.plot(x, [math.exp(FP["ndet"][g]["est_median_abs_log_err"][k]) for g in keys], color=col, marker=mk, ms=3.8, lw=1.2, label=lab)
ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_xlabel("first-pass candidates"); ax.set_xlim(-0.25, 2.25)
ax.set_ylim(1.0, 2.05); ax.set_ylabel("median $\\max(f/f^{\\star}, f^{\\star}/f)$")
ax.legend(loc="upper right", handlelength=1.5)
panel_label(ax, "(d)", "Scale error")
layout(fig)
save(fig, "fig9_evidence", tight=False)
