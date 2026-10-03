"""Fig. 1 - motivation: (a) per-structure object size differs systematically between centers; (b) the best test scale of a
source detector changes with the source->target pair (val sweep, seed 42); (c) the per-image scale factors inferred by TAC
spread widely even inside one target center. Data: results/paper_aux/dataset_size_stats.json,
results/scale_analysis_val_2026-09-11.json, results/paper_aux/tac_factors_s42.json."""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import C, DOM, load, load_aux, panel_label, save, MARK

stats = load_aux("dataset_size_stats.json")["4c"]
sweep = load("scale_analysis_val_2026-09-11.json")
fac = load_aux("tac_factors_s42.json")

CLASSES = ["LA", "RA", "LV", "RV", "VS", "CRO", "SP", "DAO", "RIB"]
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.15), gridspec_kw={"width_ratios": [1.25, 1.05, 1.0], "wspace": 0.32})

# (a) per-class median size (px, short edge 800) per center, train split
ax = axes[0]
x = np.arange(len(CLASSES)); w = 0.27
for j, dom in enumerate(["c1", "c2", "c3"]):
    cls = stats[f"{dom}_train"]["cls"]
    med = [cls[c]["median"] for c in CLASSES]
    lo = [cls[c]["median"] - cls[c]["p10"] for c in CLASSES]; hi = [cls[c]["p90"] - cls[c]["median"] for c in CLASSES]
    ax.bar(x + (j - 1) * w, med, w, color=DOM[dom], label=f"center {dom[-1]}", yerr=[lo, hi], error_kw=dict(lw=0.4, capsize=0, ecolor=C["ink2"], alpha=0.55))
ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=6.3, rotation=35, ha="right", rotation_mode="anchor")
ax.set_ylabel("object size $\\sqrt{wh}$ (px @ short edge 800)")
ax.legend(loc="upper left", ncol=1, handlelength=1.0, fontsize=6.5, bbox_to_anchor=(0.0, 1.0))
ax.set_ylim(0, 330)
ratio = stats["c3_train"]["all_median"] / stats["c1_train"]["all_median"]
panel_label(ax, "(a)")

# (b) val AP vs test scale per direction
ax = axes[1]
dirs = ["c1->c2", "c1->c3", "c2->c1", "c2->c3", "c3->c1", "c3->c2"]
cols = [C["blue"], C["sky"], C["cyan"], C["teal"], C["orange"], C["gold"]]
for i, d in enumerate(dirs):
    s = sorted(int(k) for k in sweep[d]); ap = [sweep[d][str(k)]["AP"] for k in s]
    ax.plot(s, ap, color=cols[i], marker=MARK[i], markersize=3, lw=1.1, label=d.replace("->", "→"))
    k = int(np.argmax(ap)); ax.plot(s[k], ap[k], marker=MARK[i], markersize=6.5, color=cols[i], markeredgecolor="black", markeredgewidth=0.6, zorder=5)
ax.axvline(800, color=C["ink2"], lw=0.8, ls="--"); ax.text(785, 8, "default\nscale 800", fontsize=6, color=C["ink2"], va="bottom", ha="right")
ax.set_xlabel("test-time short-edge scale (px)"); ax.set_ylabel("AP on target val (%)")
ax.set_xticks([300, 500, 700, 900, 1200]); ax.set_ylim(5, 72)
ax.legend(ncol=2, loc="upper left", fontsize=6, handlelength=1.3, columnspacing=0.7, bbox_to_anchor=(-0.02, 1.03))
panel_label(ax, "(b)")

# (c) per-image scale factors (TAC, seed 42, test): histograms of log2 f for three directions
ax = axes[2]
bins = np.linspace(-1.5, 2.0, 36)
for d, col, lab in [("c1_to_c2", C["blue"], "c1→c2"), ("c3_to_c2", C["gold"], "c3→c2"), ("c1_to_c3", C["sky"], "c1→c3")]:
    f = np.asarray(fac[f"tacr2_s42_{d}_test"]["f"]); lf = np.log2(np.clip(f, 0.25, 4))
    ax.hist(lf, bins=bins, color=col, alpha=0.75, lw=0, label=f"{lab}  (median f = {np.median(f):.2f})")
ax.axvline(0, color=C["ink2"], lw=0.8, ls="--")
ax.set_xticks([-1, 0, 1, 2]); ax.set_xticklabels(["½", "1", "2", "4"])
ax.set_xlabel("per-image scale factor $f$ (log axis)"); ax.set_ylabel("test images")
ax.legend(loc="upper right", fontsize=6, handlelength=1.0)
panel_label(ax, "(c)")

save(fig, "fig1_motivation")
