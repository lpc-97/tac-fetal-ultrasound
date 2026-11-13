"""Fig. 6 - per-structure AP improvement over the source detector (test, 6 directions x 3 seeds) on the three datasets for the
18-view heat-map fusion, 18-view WBF + re-scoring and TAC (3 views, 2 rounds). Data: qc_collect['per_class'] (recomputed with
pycocotools from the saved detections)."""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import C, OURS, W_FULL, panel_label, layout, save
from tabdata import QC

PC = QC["per_class"]
SETS = [("4c", "Heart-4CC (9 structures)"), ("abdomen", "Abdomen (8 structures)"), ("spine", "Spine (6 structures)")]
ROWS = [("heat", "T$^3$A", C["gold"], "h"), ("wbfar", "18-view WBF + re-scoring", C["teal"], "H"), ("tac", "TAC-3, 2 rounds (ours)", OURS, "o")]
fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.3), sharey=True, gridspec_kw={"width_ratios": [9, 8, 6]})
for ax, (dsn, title) in zip(axes, SETS):
    src = PC[dsn]["src"]; names = [k for k in src if not k.startswith("_")]
    x = np.arange(len(names)); w = 0.26
    for j, (row, lab, col, mk) in enumerate(ROWS):
        d = [PC[dsn][row][k]["AP"] - src[k]["AP"] for k in names]
        ax.bar(x + (j - 1) * w, d, w, color=col, lw=0, label=lab)
    ax.axhline(0, color=C["ink2"], lw=0.6)
    labs = [k.split(":")[1] for k in names]; long = max(len(l) for l in labs) >= 3  # three-letter labels collide when horizontal
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.tick_params(axis="x", length=0); ax.set_xlim(-0.6, len(names) - 0.4)
    src_ap = [src[k]["AP"] for k in names]
    for i, k in enumerate(names):
        ax.text(x[i], -0.9, f"{src_ap[i]:.0f}", ha="center", va="top", fontsize=7, color=C["ink2"])
    ax.set_ylim(-2.4, 14.5)  # headroom keeps the legend off the bars
    ax.set_xlabel("structure (gray: source AP)")
axes[0].set_ylabel("AP gain over source")
axes[2].legend(loc="upper right", handlelength=1.0)
for i, ax in enumerate(axes):
    panel_label(ax, f"({chr(97 + i)})", SETS[i][1])
layout(fig, w_pad=1.0)
save(fig, "fig6_perclass", tight=False)
