"""Module contributions and composition (Heart-4CC, Faster R-CNN, test, three seeds), as two bar charts.
(a) Mean AP as the modules are added to the source detector: the canonical input (ANI), the fusion of the three views
    (SDF) and the acquired bracket (UVA, TAC-7; single round). The trainer applies the structure-aware re-scoring after
    every fusion rule and also to the single canonical view (tta_canon.py, FUSION=last), so the SDF step pairs two
    re-scored runs of the fusion table.
(b) AP of the frozen source model and of three parameter-adaptive methods on the original and on the canonicalized input
    (the rows of the composition table).
Data: tables/numbers.json, tables/numbers_gap.json (generated, nothing typed)."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import C, OURS, OURS_TINT, W_FULL, panel_label, layout, save

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
N = json.load(open(os.path.join(T, "numbers.json"), encoding="utf-8"))
G = json.load(open(os.path.join(T, "numbers_gap.json"), encoding="utf-8"))
src, ind = G["source_cross_domain"], G["source_in_domain"]


def ap(key):
    v = N[key]
    return v[0] if isinstance(v, list) else (v["mean"] if isinstance(v, dict) else v)


Y0, Y1 = 44.0, 62.0
steps = [("Source", src, C["lgray"]),
         ("+ ANI", ap("fusion::TAC canonical view only (no fusion)"), OURS_TINT),
         ("+ SDF", ap("fusion::TAC (WBF, 3 views)"), "#D2716E"),
         ("+ UVA", ap("main_TAC 7 views (quadrature)"), OURS)]
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 1.95), gridspec_kw={"width_ratios": [1.0, 1.55]})

# ---------------------------------------------------------------- (a) modules added in turn
ax = axes[0]
prev = None
for i, (lab, val, col) in enumerate(steps):
    ax.bar(i, val - Y0, 0.58, bottom=Y0, color=col, lw=0)
    ax.text(i, val + 0.25, f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=C["ink"], fontweight="bold" if i == len(steps) - 1 else "normal")
    if prev is not None:
        ax.text(i, val + 1.75, f"+{val - prev:.2f}", ha="center", va="bottom", fontsize=6.5, color=C["ink2"])
    prev = val
ax.axhline(ind, color=C["ink2"], lw=0.7, ls=(0, (4, 2)))
ax.text(-0.42, ind + 0.2, "in-domain", ha="left", va="bottom", fontsize=6.5, color=C["ink2"])
ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps]); ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.5, len(steps) - 0.5); ax.set_ylim(Y0, Y1); ax.set_yticks(np.arange(44, 63, 4)); ax.set_ylabel("AP (%)")
panel_label(ax, "(a)", "Modules added in turn")

# ---------------------------------------------------------------- (b) original against canonicalized input
ax = axes[1]
pairs = [("Frozen source", "Source-only (1 view @800)", "Source + TAC input (1 view, 0 updates)"), ("IoU-Filter", "IoU-Filter", "IoU-Filter + TAC"),
         ("AMROD", "AMROD", "AMROD + TAC"), ("CoTTA-det", "CoTTA-det", "CoTTA-det + TAC")]
w = 0.36
for j, (name, k0, k1) in enumerate(pairs):
    a0, s0 = N["compose::" + k0]["AP"]; a1, s1 = N["compose::" + k1]["AP"]
    ax.bar(j - w / 2, a0 - Y0, w, bottom=Y0, color=C["lgray"], lw=0, yerr=s0, error_kw=dict(lw=0.6, capsize=1.5, ecolor=C["ink2"]))
    ax.bar(j + w / 2, a1 - Y0, w, bottom=Y0, color=OURS, lw=0, yerr=s1, error_kw=dict(lw=0.6, capsize=1.5, ecolor=C["ink2"]))
    ax.text(j - w / 2, a0 + s0 + 0.25, f"{a0:.2f}", ha="center", va="bottom", fontsize=6.5, color=C["ink2"])
    ax.text(j + w / 2, a1 + s1 + 0.25, f"{a1:.2f}", ha="center", va="bottom", fontsize=6.5, color=C["ink"])
    ax.text(j + w / 2, a1 + s1 + 1.6, f"+{a1 - a0:.2f}", ha="center", va="bottom", fontsize=6.5, color=OURS, fontweight="bold")
ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p[0] for p in pairs]); ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.6, len(pairs) - 0.4); ax.set_ylim(Y0, Y1); ax.set_yticks(np.arange(44, 63, 4)); ax.set_ylabel("AP (%)")
ax.legend(handles=[Patch(facecolor=C["lgray"], label="original input"), Patch(facecolor=OURS, label="canonicalized input")],
          loc="upper left", handlelength=1.0, ncol=1)
panel_label(ax, "(b)", "Parameter-adaptive methods on the canonicalized input")
layout(fig, w_pad=2.2)
save(fig, "fig11_layers", tight=False)
print("steps", [(s[0], round(s[1], 2)) for s in steps], "in-domain", round(ind, 2),
      "pairs", [(p[0], round(N["compose::" + p[1]]["AP"][0], 2), round(N["compose::" + p[2]]["AP"][0], 2)) for p in pairs])
