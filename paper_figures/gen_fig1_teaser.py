"""Fig. 1 (2026-11-13, v3): motivation, in two blocks of real frames with one plot under each.
(a) the problem: one four-chamber frame per Heart-4CC center, cropped to a square of the full frame height, so all three are
    shown at the detector input scale (short edge 800 px); the corner square has the side of the median left-ventricle size
    of that center's training annotations at that scale. Below: 10-90% range and median of the same size per center
    (results/paper_aux/dataset_size_stats.json).
(b) the correction: one center-3 test frame (the frame of the qualitative figure, first column) with the output of the
    center-1 source model on the frame as acquired, and the region magnified by the factor inferred for this frame with the
    output of TAC-3 with two rounds; boxes with confidence >= 0.5 (samples/qual_examples_s42.json). Below: share of the
    cross-center loss (in-domain minus source AP) closed on Heart-4CC with Faster R-CNN by the best single-view
    parameter-adaptive method, the best 18-view ensemble and TAC-7 (rows of the main table, via tabdata).
No number is typed: sizes, the factor, the magnification and the shares are computed from the files."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from pubstyle import C, DOM, OURS, OUT, W_FULL, load_aux, save
from tabdata import rows, UPPER

plt.rcParams["savefig.dpi"] = 600
stats = load_aux("dataset_size_stats.json")["4c"]
EX = json.load(open(os.path.join(OUT, "samples", "qual_examples_s42.json")))["c3_small_f285"]
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#8C8C8C", "#264653"]
DARK_TEXT = {"#F0E442", "#E69F00", "#56B4E9"}
# file, domain, banner, horizontal position of the crop centre (fraction of the frame width)
FRAMES = [("c1_sample.jpg", "c1", "Center 1", 0.40), ("c2_sample.jpg", "c2", "Center 2", 0.50), ("c3_small_f254.jpg", "c3", "Center 3", 0.50)]

S = 1.27; GAP = 0.05; X0 = 0.11; BGAP = 0.24; ZGAP = 0.34          # frame side, gaps (in)
PB = 0.30; PH = 0.56; RGAP = 0.17; TOP = 0.24                       # plot row: bottom, height; gap to the frames; title strip
FY = PB + PH + RGAP; FH = FY + S + TOP
fig = plt.figure(figsize=(W_FULL, FH))


def frame_axes(x):
    ax = fig.add_axes([x / W_FULL, FY / FH, S / W_FULL, S / FH])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def banner(ax, s, loc="tl"):
    x, y, ha, va = {"bl": (0.035, 0.04, "left", "bottom"), "tl": (0.035, 0.96, "left", "top")}[loc]
    ax.text(x, y, s, transform=ax.transAxes, fontsize=7, color="white", ha=ha, va=va,
            bbox=dict(facecolor="black", edgecolor="none", pad=1.6, alpha=0.62), zorder=6)


# ------------------------------------------------------------------ (a) frames
for i, (f, dom, name, cx) in enumerate(FRAMES):
    im = np.asarray(Image.open(os.path.join(OUT, "samples", f)).convert("L")); h, w = im.shape
    x0 = int(round(min(max(cx * w - h / 2.0, 0), w - h))); im = im[:, x0:x0 + h]
    im = im.copy(); im[:int(0.115 * h), :] = 0  # header band of the scanner display blacked out (residual text fragments)
    ax = frame_axes(X0 + i * (S + GAP))
    ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="hanning")
    med = stats[dom + "_train"]["cls"]["LV"]["median"]      # px at short edge 800
    side = med * h / 800.0                                    # px in this frame (its short edge is the height)
    m = 0.05 * h
    ax.add_patch(Rectangle((h - m - side, h - m - side), side, side, fill=False, ec="white", lw=0.9, zorder=5))
    ax.text(h - m - side / 2.0, h - m - side - 0.025 * h, f"{med:.0f} px", color="white", fontsize=7, ha="center", va="bottom", zorder=6)
    banner(ax, name)

# ------------------------------------------------------------------ (a) size ranges
AW = 3 * S + 2 * GAP
ax = fig.add_axes([(X0 + 0.60) / W_FULL, PB / FH, (AW - 0.66) / W_FULL, PH / FH])
c1 = stats["c1_train"]["cls"]["LV"]
for i, (_, dom, name, _) in enumerate(FRAMES):
    st = stats[dom + "_train"]["cls"]["LV"]; y = 2 - i
    ax.plot([st["p10"], st["p90"]], [y, y], color=DOM[dom], lw=4.2, solid_capstyle="butt", zorder=3)
    ax.plot([st["median"]], [y], marker="o", ms=4.2, mfc="white", mec=DOM[dom], mew=1.1, zorder=4)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels([n for _, _, n, _ in FRAMES]); ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, 2.6); ax.set_xlim(50, 262); ax.set_xticks([50, 100, 150, 200, 250])
ax.spines["left"].set_visible(False)
ax.set_xlabel("left-ventricle size at the detector input (px)", labelpad=1.5)

# ------------------------------------------------------------------ (b) frames
XB = X0 + AW + BGAP
f = EX["factor"][0]
im = np.asarray(Image.open(os.path.join(OUT, "samples", "c3_small_f285.jpg")).convert("L")).copy(); h, w = im.shape
im[:int(0.115 * h), :] = 0
xs = [b["bbox"][0] for b in EX["gt"]] + [b["bbox"][0] + b["bbox"][2] for b in EX["gt"]]
ys = [b["bbox"][1] for b in EX["gt"]] + [b["bbox"][1] + b["bbox"][3] for b in EX["gt"]]
cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
side = h / f                                                  # region whose magnification to the panel is exactly f
assert max(max(xs) - min(xs), max(ys) - min(ys)) < side
fx0 = min(max(cx - h / 2.0, 0), w - h)                        # full-height square around the heart
zx0, zy0 = cx - side / 2.0, cy - side / 2.0


def boxes(ax, dets, lw, fs):
    for b in dets:
        if b["score"] < 0.5:
            continue
        bx, by, bw, bh = b["bbox"]; col = PAL[(b["cat"] - 1) % len(PAL)]
        ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=col, lw=lw, zorder=4))
        nm = EX["names"][str(b["cat"])]; nm = {"CR": "CRO", "R": "RIB"}.get(nm, nm)
        # label outside the box, on the side that points away from the centre of the heart
        up = (by + bh / 2.0) <= cy; left = (bx + bw / 2.0) <= cx
        ax.text(bx if left else bx + bw, by if up else by + bh, nm, fontsize=fs, color="#222222" if col in DARK_TEXT else "white",
                ha="left" if left else "right", va="bottom" if up else "top", bbox=dict(facecolor=col, edgecolor="none", pad=0.6), zorder=5)


ax1 = frame_axes(XB)
ax1.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="hanning"); ax1.set_xlim(fx0, fx0 + h); ax1.set_ylim(h, 0)
boxes(ax1, EX["src"], 0.9, 6.5)
ax1.add_patch(Rectangle((zx0, zy0), side, side, fill=False, ec=OURS, lw=0.9, ls=(0, (3, 2)), zorder=6))
banner(ax1, "As acquired")
ax2 = frame_axes(XB + S + ZGAP)
ax2.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="hanning"); ax2.set_xlim(zx0, zx0 + side); ax2.set_ylim(zy0 + side, zy0)
boxes(ax2, EX["tac"], 1.1, 6.5)
for sp in ax2.spines.values():
    sp.set_visible(True); sp.set_edgecolor(OURS); sp.set_linewidth(1.0)
banner(ax2, f"Canonicalized, $f$ = {f:.2f}")
# zoom connectors from the marked region to the magnified panel
fig.canvas.draw()
inv = fig.transFigure.inverted()
for yy, ya in [(zy0, 1.0), (zy0 + side, 0.0)]:
    p = inv.transform(ax1.transData.transform((zx0 + side, yy))); q = inv.transform(ax2.transAxes.transform((0.0, ya)))
    fig.add_artist(Line2D([p[0], q[0]], [p[1], q[1]], transform=fig.transFigure, color=OURS, lw=0.6, ls=(0, (3, 2))))

# ------------------------------------------------------------------ (b) share of the cross-center loss closed
R = rows("heart4cc_frcnn")
src = R["Source-only (FrozenBN)"]["mean"]; gap = R[UPPER]["mean"] - src
SV = ["IoU-Filter", "AMROD", "WHW", "CD-Buffer", "SGP", "O-SFDA", "VLOD-TTA objective", "BufferTTA"]
ENS = ["Senior heatmap fusion (18 views)", "18-view WBF", "18-view WBF + AR"]
items = [("Single-view adaptation", max(R[k]["mean"] for k in SV), C["lgray"]),
         ("18-view ensemble", max(R[k]["mean"] for k in ENS), C["blue"]),
         ("TAC, 7 views", R["TAC 7 views (quadrature)"]["mean"], OURS)]
BW = 2 * S + ZGAP
ax = fig.add_axes([(XB + 1.12) / W_FULL, PB / FH, (BW - 1.12 - 0.10) / W_FULL, PH / FH])
for i, (lab, apv, col) in enumerate(items):
    y = 2 - i; v = 100.0 * (apv - src) / gap
    ax.barh(y, v, height=0.62, color=col, lw=0)
    ax.text(v + 2.0, y, f"{v:.0f}%", fontsize=7, ha="left", va="center", color=C["ink"], fontweight="bold" if col == OURS else "normal")
ax.set_yticks([2, 1, 0]); ax.set_yticklabels([it[0] for it in items]); ax.tick_params(axis="y", length=0)
ax.get_yticklabels()[2].set_fontweight("bold")
ax.set_xlim(0, 100); ax.set_ylim(-0.6, 2.6); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("cross-center loss closed (%)", labelpad=1.5)

for xx, letter, title in [(X0, "(a)", "Same anatomy, different apparent scale across centers"), (XB, "(b)", "Scale corrected per image, detector frozen")]:
    fig.text(xx / W_FULL, (FY + S + 0.065) / FH, letter, fontsize=9, fontweight="bold", ha="left", va="bottom")
    fig.text((xx + 0.21) / W_FULL, (FY + S + 0.073) / FH, title, fontsize=8, ha="left", va="bottom")
save(fig, "fig1_teaser", tight=False)
print("LV", {d: [round(stats[d + "_train"]["cls"]["LV"][k], 1) for k in ("p10", "median", "p90")] for _, d, _, _ in FRAMES}, "factor", round(f, 3),
      "boxes", sum(b["score"] >= 0.5 for b in EX["src"]), sum(b["score"] >= 0.5 for b in EX["tac"]),
      "shares", [(l, round(a, 2), round(100 * (a - src) / gap, 1)) for l, a, _ in items], "gap", round(gap, 2))
