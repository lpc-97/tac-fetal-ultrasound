"""Fig. 6 - per-structure AP improvement over the source detector (test, 6 directions x 3 seeds) on the three datasets for the
18-view heat-map fusion, 18-view WBF + re-scoring and TAC (3 views, 2 rounds). Data: qc_collect['per_class'] (recomputed with
pycocotools from the saved detections)."""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import C, OURS, panel_label, save
from tabdata import QC

PC = QC["per_class"]
SETS = [("4c", "Heart-4CC (9 structures)"), ("abdomen", "Abdomen (8 structures)"), ("spine", "Spine (6 structures)")]
ROWS = [("heat", "18-view heat-map fusion", C["gold"], "h"), ("wbfar", "18-view WBF + re-scoring", C["teal"], "H"), ("tac", "TAC-3, 2 rounds (ours)", OURS, "o")]
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.0), gridspec_kw={"width_ratios": [9, 8, 6], "wspace": 0.22})
for ax, (dsn, title) in zip(axes, SETS):
    src = PC[dsn]["src"]; names = [k for k in src if not k.startswith("_")]
    x = np.arange(len(names)); w = 0.26
    for j, (row, lab, col, mk) in enumerate(ROWS):
        d = [PC[dsn][row][k]["AP"] - src[k]["AP"] for k in names]
        ax.bar(x + (j - 1) * w, d, w, color=col, lw=0, label=lab)
    ax.axhline(0, color=C["ink2"], lw=0.6)
    labs = [k.split(":")[1] for k in names]; long = max(len(l) for l in labs) >= 4  # spine labels (VAOC, VOC) collide when horizontal
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.3 if not long else 6.0, rotation=35 if long else 0, ha="right" if long else "center", rotation_mode="anchor" if long else "default")
    ax.set_title(title, fontsize=7.5, pad=8)
    src_ap = [src[k]["AP"] for k in names]
    for i, k in enumerate(names):
        ax.text(x[i], -0.9, f"{src_ap[i]:.0f}", ha="center", va="top", fontsize=5.5, color=C["ink2"])
    ax.set_ylim(-2.4, 12.5)
    ax.set_xlabel("structure (grey: source AP)", fontsize=6.5, labelpad=2)
axes[0].set_ylabel("Δ AP vs. source (%)")
axes[0].legend(loc="upper left", fontsize=6.2, handlelength=1.0, ncol=1)
for i, ax in enumerate(axes):
    panel_label(ax, f"({chr(97 + i)})", x=-0.1 if i == 0 else -0.06, y=1.16)
save(fig, "fig6_perclass")
