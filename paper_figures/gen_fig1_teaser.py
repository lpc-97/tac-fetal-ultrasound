"""Fig. 1 (2026-11-12, v2): the problem, its effect on a frozen detector and what correcting it recovers, in real frames.
(a) one four-chamber frame per Heart-4CC center, cropped to a square of the full frame height, so all three are shown at the
    detector input scale (short edge 800 px); the square outline has the side of the median left-ventricle size of that
    center's training annotations at that scale (results/paper_aux/dataset_size_stats.json);
(b) one center-3 test frame (the frame of the qualitative figure, first column): the center-1 source model on the frame as
    acquired and the output of TAC-3 with two rounds; boxes with confidence >= 0.5 (samples/qual_examples_s42.json);
(c) share of the cross-center loss (in-domain minus source AP) closed on Heart-4CC with Faster R-CNN by the best single-view
    parameter-adaptive method, the best 18-view ensemble and TAC-7 (results via tabdata, the rows of the main table).
No number is typed: sizes, the factor and the shares are computed from the files."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image
from pubstyle import C, OURS, OUT, W_FULL, load_aux, save
from tabdata import rows, UPPER

plt.rcParams["savefig.dpi"] = 600
stats = load_aux("dataset_size_stats.json")["4c"]
EX = json.load(open(os.path.join(OUT, "samples", "qual_examples_s42.json")))["c3_small_f285"]
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#8C8C8C", "#264653"]
# file, domain, banner, horizontal position of the crop centre (fraction of the frame width)
FRAMES = [("c1_sample.jpg", "c1", "center 1", 0.40), ("c2_sample.jpg", "c2", "center 2", 0.50), ("c3_small_f254.jpg", "c3", "center 3", 0.50)]

H = 0.98; TOP = 0.25; BOT = 0.05; GAP = 0.05; BLOCK = 0.2; ARW = 0.30; CW = 1.33; CGAP = 0.27
FH = H + TOP + BOT
fig = plt.figure(figsize=(W_FULL, FH))


def banner(ax, s, loc="bl"):
    x, y, ha, va = {"bl": (0.03, 0.04, "left", "bottom"), "tl": (0.03, 0.96, "left", "top")}[loc]
    ax.text(x, y, s, transform=ax.transAxes, fontsize=7, color="white", ha=ha, va=va,
            bbox=dict(facecolor="black", edgecolor="none", pad=1.6, alpha=0.6), zorder=6)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


total = 3 * H + 2 * GAP + BLOCK + 2 * H + ARW + CGAP + CW
x = (W_FULL - total) / 2.0
x_a = x
for f, dom, name, cx in FRAMES:
    im = np.asarray(Image.open(os.path.join(OUT, "samples", f)).convert("L")); h, w = im.shape
    x0 = int(round(min(max(cx * w - h / 2.0, 0), w - h))); im = im[:, x0:x0 + h]
    im = im.copy(); im[:int(0.115 * h), :] = 0  # header band of the scanner display blacked out (residual text fragments)
    ax = fig.add_axes([x / W_FULL, BOT / FH, H / W_FULL, H / FH]); bare(ax)
    ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="hanning")
    med = stats[dom + "_train"]["cls"]["LV"]["median"]      # px at short edge 800
    side = med * h / 800.0                                    # px in this frame (its short edge is the height)
    m = 0.05 * h
    ax.add_patch(Rectangle((h - m - side, h - m - side), side, side, fill=False, ec=C["gold"], lw=1.2, zorder=5))
    ax.text(h - m, h - m - side - 0.03 * h, f"{med:.0f} px", color=C["gold"], fontsize=7, ha="right", va="bottom", zorder=6,
            bbox=dict(facecolor="black", edgecolor="none", pad=0.8, alpha=0.55))
    banner(ax, name, "tl")
    x += H + GAP
x += BLOCK - GAP
x_b = x

# (b) crop around the annotated structures, as in the qualitative figure
xs = [b["bbox"][0] for b in EX["gt"]] + [b["bbox"][0] + b["bbox"][2] for b in EX["gt"]]
ys = [b["bbox"][1] for b in EX["gt"]] + [b["bbox"][1] + b["bbox"][3] for b in EX["gt"]]
s = min(max(max(xs) - min(xs), max(ys) - min(ys)) * 1.3, EX["h"], EX["w"])
x0 = max(0, min((min(xs) + max(xs)) / 2 - s / 2, EX["w"] - s)); y0 = max(0, min((min(ys) + max(ys)) / 2 - s / 2, EX["h"] - s))
im = np.asarray(Image.open(os.path.join(OUT, "samples", "c3_small_f285.jpg")).convert("L"))
for k, (kind, lab) in enumerate([("src", "default scale"), ("tac", f"TAC, $f$ = {EX['factor'][0]:.2f}")]):
    ax = fig.add_axes([(x + k * (H + ARW)) / W_FULL, BOT / FH, H / W_FULL, H / FH]); bare(ax)
    ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="hanning"); ax.set_xlim(x0, x0 + s); ax.set_ylim(y0 + s, y0)
    for b in EX[kind]:
        if b["score"] >= 0.5:
            bx, by, bw, bh = b["bbox"]; col = PAL[(b["cat"] - 1) % len(PAL)]
            ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=col, lw=1.1, zorder=4))
            nm = EX["names"][str(b["cat"])]; nm = {"CR": "CRO", "R": "RIB"}.get(nm, nm)
            ax.text(bx + bw, by, nm, fontsize=7, color="white", ha="right", va="bottom", bbox=dict(facecolor=col, edgecolor="none", pad=0.6), zorder=5)
    banner(ax, lab, "bl")
fig.add_artist(FancyArrowPatch(((x + H + 0.04) / W_FULL, (BOT + H / 2) / FH), ((x + H + ARW - 0.04) / W_FULL, (BOT + H / 2) / FH), transform=fig.transFigure,
                               arrowstyle="-|>", mutation_scale=9, lw=1.2, color=C["ink"]))
fig.text((x + H + ARW / 2) / W_FULL, (BOT + H / 2 + 0.09) / FH, "TAC", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
x += 2 * H + ARW + CGAP
x_c = x

# (c) share of the cross-center loss closed
R = rows("heart4cc_frcnn")
src = R["Source-only (FrozenBN)"]["mean"]; gap = R[UPPER]["mean"] - src
SV = ["IoU-Filter", "AMROD", "WHW", "CD-Buffer", "SGP", "O-SFDA", "VLOD-TTA objective", "BufferTTA"]
ENS = ["Senior heatmap fusion (18 views)", "18-view WBF", "18-view WBF + AR"]
items = [("best single-view adaptation", max(R[k]["mean"] for k in SV), C["gray"]),
         ("best 18-view ensemble", max(R[k]["mean"] for k in ENS), C["teal"]),
         ("TAC, 7 views (ours)", R["TAC 7 views (quadrature)"]["mean"], OURS)]
AXB = 0.27  # room for the tick labels and the axis label
ax = fig.add_axes([x / W_FULL, (BOT + AXB) / FH, CW / W_FULL, (H - AXB) / FH])
for i, (lab, ap, col) in enumerate(items):
    y = 2 - i; v = 100.0 * (ap - src) / gap
    ax.barh(y - 0.2, v, height=0.3, color=col, lw=0)
    ax.text(0, y + 0.02, lab, fontsize=7, ha="left", va="bottom", color=C["ink"])
    ax.text(v + 2.5, y - 0.2, f"{v:.0f}%", fontsize=7, ha="left", va="center", color=col, fontweight="bold" if col == OURS else "normal")
ax.set_xlim(0, 100); ax.set_ylim(-0.45, 2.45); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xticks([0, 50, 100]); ax.set_xlabel("cross-center loss closed (%)", labelpad=1.5)

for xx, letter, title in [(x_a, "(a)", "One plane at three centers, at the detector input scale"), (x_b, "(b)", "The same frozen detector"), (x_c, "(c)", "Loss recovered")]:
    fig.text(xx / W_FULL, (BOT + H + 0.07) / FH, letter, fontsize=9, fontweight="bold", ha="left", va="bottom")
    fig.text((xx + 0.21) / W_FULL, (BOT + H + 0.078) / FH, title, fontsize=8, ha="left", va="bottom")
save(fig, "fig1_teaser", tight=False)
print("LV medians", {d: round(stats[d + "_train"]["cls"]["LV"]["median"], 1) for _, d, _, _ in FRAMES}, "factor", round(EX["factor"][0], 3),
      "boxes", sum(b["score"] >= 0.5 for b in EX["src"]), sum(b["score"] >= 0.5 for b in EX["tac"]),
      "shares", [(l, round(a, 2), round(100 * (a - src) / gap, 1)) for l, a, _ in items], "gap", round(gap, 2))
