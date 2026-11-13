"""Fig. 11 - what each mechanism of the architecture contributes (Heart-4CC, Faster R-CNN, test, three seeds).
(a) Waterfall from the source detector to TAC-7, one step per module: geometric canonicalization (ANI), the fusion of
    the three views (SDF) and the acquired bracket (UVA) (TAC-7 runs a single round, so the optional second round is not
    on this path); the right axis reads the same quantity as the fraction of the cross-center loss (in-domain minus
    cross-domain source AP) that has been closed. The trainer applies the structure-aware re-scoring after every fusion
    rule and also to the single canonical view (tta_canon.py, FUSION=last), so the SDF step pairs two re-scored runs of
    Table 6(a); until 2026-11-10 it was split at "TAC w/o AR", which mixed adding fusion with removing the re-scoring.
(b) The asymmetry between correcting the input and adapting the parameters: what two parameter-adaptive methods add
    on the original input, what canonicalization alone adds, and what the same methods still add once the input is
    canonicalized.
Data: tables/numbers.json, tables/numbers_gap.json, tables/numbers_reviewer.json (all generated, nothing typed)."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import C, OURS, BASE, MOD, W_FULL, panel_label, layout, save

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables")
N = json.load(open(os.path.join(T, "numbers.json"), encoding="utf-8"))
G = json.load(open(os.path.join(T, "numbers_gap.json"), encoding="utf-8"))
src, ind, gap = G["source_cross_domain"], G["source_in_domain"], G["cross_centre_loss"]


def ap(key):
    v = N[key]
    return v[0] if isinstance(v, list) else (v["mean"] if isinstance(v, dict) else v)


steps = [("source\ndetector", src, None),
         ("canonical\ninput", ap("fusion::TAC canonical view only (no fusion)"), "ANI"),
         ("decision\nfusion", ap("fusion::TAC (WBF, 3 views)"), "SDF"),
         ("acquired\nbracket", ap("main_TAC 7 views (quadrature)"), "UVA")]
fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.55), gridspec_kw={"width_ratios": [1.35, 1.0]})

# ---------------------------------------------------------------- (a) waterfall
ax = axes[0]
COL = MOD  # module colours of the overview figure
Y0 = 46.0  # axis floor; the source bar is drawn from it, and the right axis then carries no negative tick
x = np.arange(len(steps))
ax.bar(0, steps[0][1] - Y0, 0.62, bottom=Y0, color=C["gray"])
ax.text(0, steps[0][1] + 0.12, f"{steps[0][1]:.2f}", ha="center", va="bottom", fontsize=7, color=C["ink"])
prev = steps[0][1]
for i, (lab, val, mod) in enumerate(steps[1:], start=1):
    d = val - prev
    ax.bar(i, d, 0.62, bottom=prev, color=COL[mod], lw=0)
    ax.plot([i - 0.31 - 0.38, i - 0.31], [prev, prev], color=C["ink2"], lw=0.5, ls=":")
    ax.text(i, val + 0.12, f"+{d:.2f}", ha="center", va="bottom", fontsize=7, color=C["ink"], fontweight="bold")
    prev = val
ax.axhline(ind, color=C["ink2"], lw=0.8, ls="--")
ax.text(0.985, ind - 0.35, f"in-domain {ind:.2f}", ha="right", va="top", fontsize=7, color=C["ink2"], transform=ax.get_yaxis_transform())  # x in axes fraction: clear of the right spine for any number of steps
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], fontsize=7)
ax.set_ylim(Y0, 61.6); ax.set_ylabel("mean AP (%)")
ax2 = ax.twinx(); ax2.grid(False)
ax2.set_ylim((Y0 - src) / gap * 100, (61.6 - src) / gap * 100)
ax2.set_ylabel("cross-center loss closed (%)"); ax2.spines["right"].set_visible(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=COL["ANI"], label="ANI"), Patch(facecolor=COL["SDF"], label="SDF"), Patch(facecolor=COL["UVA"], label="UVA")],
          loc="lower right", handlelength=1.1, ncol=3, columnspacing=1.0)
ax.tick_params(axis="x", length=0)
panel_label(ax, "(a)", "From the source detector to TAC-7")

# ---------------------------------------------------------------- (b) input correction vs parameter adaptation
ax = axes[1]
def cap(key):
    return N["compose::" + key]["AP"][0]


rows = [("IoU-Filter", cap("IoU-Filter"), cap("IoU-Filter + TAC")), ("AMROD", cap("AMROD"), cap("AMROD + TAC"))]
ctrl = cap("Source + TAC input (1 view, 0 updates)")
y = np.arange(len(rows)); w = 0.34
for j, (name, orig, canon) in enumerate(rows):
    ax.barh(y[j] + w / 2, orig - src, w, left=0, color=C["gray"], lw=0)
    ax.text(orig - src + 0.12, y[j] + w / 2, f"+{orig - src:.2f}", va="center", fontsize=7, color=C["ink2"])
    ax.barh(y[j] - w / 2, canon - ctrl, w, left=0, color=BASE, lw=0)
    ax.text(canon - ctrl + 0.12, y[j] - w / 2, f"+{canon - ctrl:.2f}", va="center", fontsize=7, color=C["ink2"])
ax.barh(len(rows), ctrl - src, 0.5, left=0, color=OURS, lw=0)
ax.text(ctrl - src + 0.12, len(rows), f"+{ctrl - src:.2f}", va="center", fontsize=7, color=C["ink"], fontweight="bold")
ax.set_yticks(list(y) + [len(rows)])
ax.set_yticklabels([r[0] for r in rows] + ["canonicalization\n(frozen detector)"], fontsize=7)
ax.set_xlabel("AP added by the mechanism"); ax.grid(axis="y", visible=False) if False else ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 6.2); ax.set_ylim(len(rows) + 0.75, -0.9); ax.invert_yaxis(); ax.invert_yaxis()
ax.legend(handles=[Patch(facecolor=C["gray"], label="parameter adaptation,\noriginal input"),
                   Patch(facecolor=BASE, label="parameter adaptation,\ncanonicalized input"),
                   Patch(facecolor=OURS, label="input correction alone")],
          loc="upper right", handlelength=1.0, labelspacing=0.5)
panel_label(ax, "(b)", "Input correction against parameter adaptation")
layout(fig, w_pad=2.0)
save(fig, "fig11_layers", tight=False)
