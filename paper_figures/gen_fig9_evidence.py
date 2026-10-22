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
from pubstyle import C, OURS, BASE, RES, panel_label, save

R = json.load(open(os.path.join(RES, "reviewer", "reviewer_summary.json"), encoding="utf-8"))["rows"]
FP = json.load(open(os.path.join(RES, "reviewer", "firstpass_analysis.json"), encoding="utf-8"))
fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.62), gridspec_kw={"width_ratios": [1.3, 1.05, 1.0, 0.75], "wspace": 0.52})

# (a) scale chain
ax = axes[0]
labs = ["default", "val-sel.", "stream", "TAC", "oracle"]  # spelled out in the caption
k1 = ["source", "fac_valscale_1v", "fac_global_1v", "tac_1v", "fac_oracle_1v"]; k3 = [None, "fac_valscale", "fac_global", "tac", "fac_oracle"]
x = np.arange(5); w = 0.36
v1 = [R[k]["mean"] for k in k1]; e1 = [R[k]["std"] for k in k1]
ax.bar(x - w / 2, v1, w, color=[BASE, BASE, BASE, OURS, C["teal"]], yerr=e1, error_kw=dict(lw=0.5, capsize=1.5, ecolor=C["ink2"]), label="single view")
v3 = [R[k]["mean"] if k else np.nan for k in k3]; e3 = [R[k]["std"] if k else 0 for k in k3]
ax.bar(x + w / 2, v3, w, color=[BASE, BASE, BASE, OURS, C["teal"]], yerr=e3, error_kw=dict(lw=0.5, capsize=1.5, ecolor=C["ink2"]), hatch="////", edgecolor="white", label="TAC-3 pipeline (3 views)")
# bar values are given in the scale-chain table, so the bars carry no labels
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7, rotation=30, ha="right", rotation_mode="anchor"); ax.set_ylim(45, 57); ax.set_ylabel("mean AP (%)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=C["gray"], label="single view"), Patch(facecolor=C["gray"], hatch="////", edgecolor="white", label="TAC-3 pipeline")], loc="upper left", fontsize=7, handlelength=1.4)
ax.set_xlabel("scale factor of the canonicalized view", fontsize=7)
panel_label(ax, "(a)", x=-0.14, y=1.08)

# (b) gain vs oracle shift
ax = axes[1]
keys = ["|log f*| <= log 1.25", "log 1.25 - log 1.6", "log 1.6 - log 2", "> log 2"]; xl = ["≤1.25", "1.25–1.6", "1.6–2", ">2"]
src = [FP["shift"][k]["source"] for k in keys]; t7 = [FP["shift"][k]["tac7"] for k in keys]; orc = [FP["shift"][k]["oracle3"] for k in keys]; n = [FP["shift"][k]["n_images"] for k in keys]
x = np.arange(4); w = 0.27
ax.bar(x - w, src, w, color=C["gray"], label="source"); ax.bar(x, t7, w, color=OURS, label="TAC-7"); ax.bar(x + w, orc, w, color=C["teal"], label="TAC-3, oracle $f$")
for i in range(4):
    ax.text(x[i], t7[i] + 0.6, f"+{t7[i] - src[i]:.1f}", ha="center", va="bottom", fontsize=7, color=OURS, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=7, rotation=28, ha="right", rotation_mode="anchor"); ax.set_xlabel("oracle scale shift $\\max(f^{\\star}, 1/f^{\\star})$", fontsize=7)
ax.set_ylabel("AP on the group (%)"); ax.set_ylim(0, 84); ax.legend(loc="upper right", ncol=1, fontsize=7, handlelength=1.0, handletextpad=0.4, frameon=False)
panel_label(ax, "(b)", x=-0.22, y=1.08)

# (c) first-pass candidates: AP and estimator error
ax = axes[2]
keys = ["1-2", "3-5", ">5"]; xl = ["1–2", "3–5", ">5"]
x = np.arange(3); w = 0.2
for j, (k, lab, col) in enumerate([("source", "source", C["gray"]), ("tac3", "TAC-3", "#F4A261"), ("tac7", "TAC-7", OURS), ("oracle3", "TAC-3, oracle $f$", C["teal"])]):
    ax.bar(x + (j - 1.5) * w, [FP["ndet"][g][k] for g in keys], w, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=7); ax.set_xlabel("first-pass candidates ($p_i \\geq 0.3$)", fontsize=7); ax.set_ylabel("AP on the group (%)"); ax.set_ylim(0, 84)
ax.legend(loc="upper left", fontsize=7, handlelength=1.0, handletextpad=0.4, frameon=False)
panel_label(ax, "(c)", x=-0.22, y=1.08)

# (d) estimator error on the same groups, on its own axis
ax = axes[3]
for k, lab, col, mk in [("irls", "IRLS (TAC)", OURS, "o"), ("median", "median", C["ink2"], "s"), ("noclass", "class-agnostic", C["sky"], "^")]:
    ax.plot(x, [math.exp(FP["ndet"][g]["est_median_abs_log_err"][k]) for g in keys], color=col, marker=mk, ms=3.5, lw=1.1, label=lab)
ax.set_xticks(x); ax.set_xticklabels(xl, fontsize=7); ax.set_xlabel("first-pass candidates", fontsize=7)
ax.set_ylim(1.0, 2.05); ax.set_ylabel("median $\\max(f/f^{\\star}, f^{\\star}/f)$", fontsize=7)
ax.legend(loc="upper right", fontsize=7, handlelength=1.2, handletextpad=0.4, frameon=False, title="estimator", title_fontsize=7)
panel_label(ax, "(d)", x=-0.3, y=1.08)
save(fig, "fig9_evidence")
