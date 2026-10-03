"""Fig. 11 - what each mechanism of the architecture contributes (Heart-4CC, Faster R-CNN, test, three seeds).
(a) Waterfall from the source detector to TAC-7: geometric canonicalization, cross-view fusion, structure-aware
    resolution, the refinement round and the acquired bracket; the right axis reads the same quantity as the fraction
    of the cross-center loss (in-domain minus cross-domain source AP) that has been closed.
(b) The asymmetry between correcting the input and adapting the parameters: what two parameter-adaptive methods add
    on the original input, what canonicalization alone adds, and what the same methods still add once the input is
    canonicalized.
Data: tables/numbers.json, tables/numbers_gap.json, tables/numbers_reviewer.json (all generated, nothing typed)."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import C, OURS, BASE, panel_label, save

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
N = json.load(open(os.path.join(T, "numbers.json"), encoding="utf-8"))
G = json.load(open(os.path.join(T, "numbers_gap.json"), encoding="utf-8"))
src, ind, gap = G["source_cross_domain"], G["source_in_domain"], G["cross_centre_loss"]


def ap(key):
    v = N[key]
    return v[0] if isinstance(v, list) else (v["mean"] if isinstance(v, dict) else v)


steps = [("source\ndetector", src, None),
         ("geometric\ncanonicalisation", ap("fusion::TAC canonical view only (no fusion)"), "ANI"),
         ("cross-view\nfusion", ap("abl_heart::TAC w/o AR"), "SDF"),
         ("structure\nresolution", ap("abl_heart::TAC-light"), "SDF"),
         ("refinement\nround", ap("abl_heart::TAC (2 rounds)"), "SDF"),
         ("acquired\nbracket", ap("abl_heart::TAC 7 views"), "UVA")]
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.5})

# ---------------------------------------------------------------- (a) waterfall
ax = axes[0]
COL = {"ANI": OURS, "UVA": C["teal"], "SDF": C["sky"] if "sky" in C else C["blue"]}
x = np.arange(len(steps))
ax.bar(0, steps[0][1] - 44, 0.62, bottom=44, color=C["gray"])
ax.text(0, steps[0][1] + 0.12, f"{steps[0][1]:.2f}", ha="center", va="bottom", fontsize=6.2, color=C["ink"])
prev = steps[0][1]
for i, (lab, val, mod) in enumerate(steps[1:], start=1):
    d = val - prev
    ax.bar(i, d, 0.62, bottom=prev, color=COL[mod], lw=0)
    ax.plot([i - 0.31 - 0.38, i - 0.31], [prev, prev], color=C["ink2"], lw=0.5, ls=":")
    ax.text(i, val + 0.12, f"+{d:.2f}", ha="center", va="bottom", fontsize=6.2, color=C["ink"], fontweight="bold")
    prev = val
ax.axhline(ind, color=C["ink2"], lw=0.8, ls="--")
ax.text(len(steps) - 0.5, ind - 0.35, f"in-domain {ind:.2f}", ha="right", va="top", fontsize=6, color=C["ink2"])
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], fontsize=5.6)
ax.set_ylim(44, 61.6); ax.set_ylabel("mean AP (%)")
ax2 = ax.twinx(); ax2.grid(False)
ax2.set_ylim((44 - src) / gap * 100, (61.6 - src) / gap * 100)
ax2.set_ylabel("cross-center loss closed (\\%)", fontsize=6.5); ax2.tick_params(axis="y", labelsize=6)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=COL["ANI"], label="ANI"), Patch(facecolor=COL["SDF"], label="SDF"), Patch(facecolor=COL["UVA"], label="UVA")],
          loc="lower right", fontsize=6, handlelength=1.1, frameon=False, ncol=3, columnspacing=0.8)
panel_label(ax, "(a)", x=-0.15, y=1.06)

# ---------------------------------------------------------------- (b) input correction vs parameter adaptation
ax = axes[1]
rows = [("IoU-Filter", 47.44, 51.77), ("AMROD", 47.40, 52.53)]
ctrl = 51.60
y = np.arange(len(rows)); w = 0.34
for j, (name, orig, canon) in enumerate(rows):
    ax.barh(y[j] + w / 2, orig - src, w, left=0, color=C["gray"], lw=0)
    ax.text(orig - src + 0.12, y[j] + w / 2, f"+{orig - src:.2f}", va="center", fontsize=6, color=C["ink2"])
    ax.barh(y[j] - w / 2, canon - ctrl, w, left=0, color=BASE, lw=0)
    ax.text(canon - ctrl + 0.12, y[j] - w / 2, f"+{canon - ctrl:.2f}", va="center", fontsize=6, color=C["ink2"])
ax.barh(len(rows), ctrl - src, 0.5, left=0, color=OURS, lw=0)
ax.text(ctrl - src + 0.12, len(rows), f"+{ctrl - src:.2f}", va="center", fontsize=6.4, color=C["ink"], fontweight="bold")
ax.set_yticks(list(y) + [len(rows)])
ax.set_yticklabels([r[0] for r in rows] + ["canonicalization\n(frozen detector)"], fontsize=6)
ax.set_xlabel("AP added by the mechanism", fontsize=7)
ax.set_xlim(0, 6.2); ax.set_ylim(len(rows) + 0.75, -0.9); ax.invert_yaxis(); ax.invert_yaxis()
ax.legend(handles=[Patch(facecolor=C["gray"], label="parameter adaptation,\noriginal input"),
                   Patch(facecolor=BASE, label="parameter adaptation,\ncanonicalised input"),
                   Patch(facecolor=OURS, label="input correction alone")],
          loc="upper right", fontsize=5.4, handlelength=1.0, handletextpad=0.4, labelspacing=0.5, frameon=False, borderaxespad=0.2)
panel_label(ax, "(b)", x=-0.42, y=1.06)
save(fig, "fig11_layers")
