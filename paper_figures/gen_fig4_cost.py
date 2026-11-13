"""Fig. 4 - accuracy versus inference cost (detector forward passes per image) on the four settings; test split, mean of the
six cross-domain directions over three seeds; error bars = population std over seeds. Methods that also back-propagate at test
time are drawn with open markers. Data: results/qc_collect_2026-09-18.json via tabdata."""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import C, OURS, W_FULL, panel_label, layout, save
from tabdata import rows

SETS = [("heart4cc_frcnn", "Heart-4CC, Faster R-CNN"), ("heart4cc_retinanet", "Heart-4CC, RetinaNet"), ("abdomen", "Abdomen, Faster R-CNN"), ("spine", "Spine, Faster R-CNN")]
# name in table -> (label, views, marker, colour, trains?)
SPEC = [
    ("Source-only (FrozenBN)", "source model", 1, "o", C["ink"], False), ("RetinaNet source-only", "source model", 1, "o", C["ink"], False),
    # the eight single-view parameter-adaptive methods share one marker: at one forward pass they overlap, and the
    # tables give them individually
    *[(k, "single-view TTA (up to 8 methods)", 1, "s", C["lgray"], True) for k in
      ("IoU-Filter", "AMROD", "WHW", "CD-Buffer", "SGP", "O-SFDA", "VLOD-TTA objective", "BufferTTA")],
    ("Senior heatmap fusion (18 views)", "T$^3$A", None, "h", C["gold"], False),  # mean of 16 / 18 views over the directions
    ("18-view WBF", "18-view WBF", 18, "p", C["cyan"], False), ("18-view WBF + AR", "18-view WBF + re-scoring", 18, "H", C["teal"], False),
    ("CoTTA-det", "CoTTA-det (18 views + training)", 18, "d", C["blue"], True),
    ("TAC-light", "TAC-3 (ours)", 3.0, "o", OURS, False), ("TAC (2 rounds)", "TAC-3, 2 rounds (ours)", None, "D", OURS, False), ("TAC 7 views (quadrature)", "TAC-7 (ours)", 7.0, "*", OURS, False),
]
SV = [x[0] for x in SPEC if x[3] == "s"]
JIT = {k: o for k, o in zip(SV, np.linspace(-0.2, 0.2, len(SV)))}
fig, axes = plt.subplots(1, 4, figsize=(W_FULL, 2.75))
seen = {}
for ax, (tab, title) in zip(axes, SETS):
    R = rows(tab)
    ours_pts = []
    for name, lab, views, mk, col, trains in SPEC:
        if name not in R:
            continue
        r = R[name]; v = views if views is not None else (r["views"] or 3.3)
        if mk == "s":  # single-view methods: spread around one forward pass so that the markers do not pile up
            v = 2.0 ** JIT[name]
        kw = dict(marker=mk, ms=4.5 if (col != OURS or mk == "D") else 6, mec=col, mfc="white" if trains else col, mew=0.9 if trains else 0.5, ls="none",
                  zorder=4 if col in (OURS, C["ink"]) else 3)
        h = ax.errorbar(v, r["mean"], yerr=r["std"], ecolor=col, elinewidth=0.6, capsize=1.5, color=col, **kw)
        seen.setdefault(lab, h)
        if col == OURS:
            ours_pts.append((v, r["mean"]))
    ours_pts.sort()
    ax.plot([p[0] for p in ours_pts], [p[1] for p in ours_pts], color=OURS, lw=1.0, alpha=0.7, zorder=2)
    ax.set_xscale("log", basex=2); ax.set_xticks([1, 3, 7, 18]); ax.set_xticklabels(["1", "3", "7", "18"]); ax.set_xlim(0.72, 24)
    ax.set_xlabel("forward passes per image"); ax.titles_ = title
    ax.minorticks_off()
axes[0].set_ylabel("mean test AP (%)")
for i, ax in enumerate(axes):
    panel_label(ax, f"({chr(97 + i)})", SETS[i][1])
from matplotlib.lines import Line2D
handles, labels = [], []
for name, lab, views, mk, col, trains in SPEC:
    if lab in labels or lab not in seen:
        continue
    handles.append(Line2D([], [], marker=mk, color=col, mec=col, mfc="white" if trains else col, mew=0.9 if trains else 0.5, ms=4.5 if (col != OURS or mk == "D") else 6, ls="none")); labels.append(lab)
order = [0, 1, 5, 2, 3, 4, 6, 7, 8] if len(labels) == 9 else list(range(len(labels)))  # CoTTA-det (trained) joins the first column
handles = [handles[i] for i in order]; labels = [labels[i] for i in order]
layout(fig, rect=(0, 0.2, 1, 1))
# three columns, filled column by column: single-pass methods, 18-view references, TAC
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=3.0)
save(fig, "fig4_cost", tight=False)
