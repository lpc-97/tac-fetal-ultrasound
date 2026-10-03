"""Fig. 3 - mechanism: (top) FPN-level assignment of ground-truth objects (detectron2 RoI pooler rule) for the source train
set at its calibrated scale, the target test set at the default scale, and the target test set after TAC canonicalization,
per cross-center direction, with the total-variation distance to the source histogram (mean over seeds 42/43/44);
(bottom) per-image scale factors inferred by TAC (seed 42). Data: results/fpn_levels_s42_2026-09-14.json, paper_aux/fpn_levels_s4{3,4}.json."""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import C, load, load_aux, save

D = {"42": load("fpn_levels_s42_2026-09-14.json"), "43": load_aux("fpn_levels_s43.json"), "44": load_aux("fpn_levels_s44.json")}
DIRS = ["c1_to_c2", "c1_to_c3", "c2_to_c1", "c2_to_c3", "c3_to_c1", "c3_to_c2"]
LEVELS = ["2", "3", "4", "5"]
C_SRC, C_T800, C_TAC = C["blue"], C["orange"], C["cyan"]

fig, axes = plt.subplots(2, 6, figsize=(6.9, 2.9), gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.62, "wspace": 0.3})
w = 0.27
for j, d in enumerate(DIRS):
    r = D["42"][d]; ax = axes[0, j]; x = np.arange(4)
    hs = [r["source_train"][l] * 100 for l in LEVELS]; ht = [r["target_800"][l] * 100 for l in LEVELS]
    hc = np.mean([[D[s][d]["target_tac"][l] * 100 for l in LEVELS] for s in D], axis=0)
    ax.bar(x - w, hs, w, color=C_SRC, lw=0, label="source train @ $s_0$")
    ax.bar(x, ht, w, color=C_T800, lw=0, label="target test @ default 800")
    ax.bar(x + w, hc, w, color=C_TAC, lw=0, hatch="////", edgecolor="white", label="target test after TAC")
    ax.set_xticks(x); ax.set_xticklabels([f"P{l}" for l in LEVELS], fontsize=6.5); ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100])
    a, b = d.split("_to_"); ax.set_title(f"{a} → {b}", pad=3)
    tv800 = r["tv_800"]; tvt = np.mean([D[s][d]["tv_tac"] for s in D])
    ax.text(0.5, 0.97, f"TV {tv800:.2f} → {tvt:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=6.3, color=C["ink2"])
    if j == 0:
        ax.set_ylabel("GT objects per\nFPN level (%)", fontsize=7)
    ax2 = axes[1, j]
    f = np.asarray(r["factors"]); lf = np.log2(np.clip(f, 0.25, 4))
    ax2.hist(lf, bins=np.linspace(-2, 2, 33), color=C_TAC, lw=0)
    ax2.axvline(0, color=C["ink2"], lw=0.7, ls="--")
    med = float(np.median(f)); p10, p90 = np.percentile(f, 10), np.percentile(f, 90)
    ax2.set_title(f"med {med:.2f}  [{p10:.2f}–{p90:.2f}]", fontsize=5.8, pad=2, fontweight="normal", color=C["ink2"])
    ax2.set_xticks([-2, -1, 0, 1, 2]); ax2.set_xticklabels(["¼", "½", "1", "2", "4"], fontsize=6.5); ax2.set_xlim(-2, 2)
    ax2.set_xlabel("scale factor $f$", fontsize=7, labelpad=1)
    if j == 0:
        ax2.set_ylabel("test images", fontsize=7)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.03))
save(fig, "fig3_mechanism")
