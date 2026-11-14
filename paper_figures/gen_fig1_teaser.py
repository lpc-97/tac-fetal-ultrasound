"""Fig. 1 (2026-11-13, v4): motivation on a light canvas, three blocks.
(a) the problem: one four-chamber frame per Heart-4CC center, the same window of the detector input (short edge 800 px) cut
    from each, so the three thumbnails share one pixel scale; the corner square has the side of the median left-ventricle
    size of that center's training annotations. Below: 10-90% range and median of this size per center
    (results/paper_aux/dataset_size_stats.json).
(b) three ways to adapt at test time, one row each, drawn as glyphs (views fed to the detector, detector updated or frozen),
    with the share of the cross-center loss (in-domain minus source AP) closed on Heart-4CC with Faster R-CNN: the best
    single-view parameter-adaptive method, the best 18-view ensemble and TAC-7 (rows of the main table, via tabdata).
(c) one center-3 test frame (the frame of the qualitative figure, first column): the same window of the detector input
    before and after resampling by the inferred factor, with the detections of the center-1 source model and of TAC-3 with
    two rounds at confidence >= 0.5 (samples/qual_examples_s42.json).
No number is typed: sizes, the factor, the windows and the shares are computed from the files."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from pubstyle import C, DOM, OURS, OUT, W_FULL, load_aux, save
from tabdata import rows, UPPER

plt.rcParams["savefig.dpi"] = 600
stats = load_aux("dataset_size_stats.json")["4c"]
EX = json.load(open(os.path.join(OUT, "samples", "qual_examples_s42.json")))["c3_small_f285"]
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#8C8C8C", "#264653"]
# file, domain, label, window centre (fractions of the frame width and height)
FRAMES = [("c1_sample.jpg", "c1", "Center 1", 0.42, 0.60), ("c2_sample.jpg", "c2", "Center 2", 0.50, 0.60), ("c3_small_f254.jpg", "c3", "Center 3", 0.49, 0.50)]
WIN = 0.62                    # window side as a fraction of the frame height, the same for the three centers
PANEL = "#F4F5F7"; TINT = "#FBECEB"; FROZEN = "#DCE6F1"; UPDATED = "#F6DFB8"

H = 2.22
fig = plt.figure(figsize=(W_FULL, H))
cv = fig.add_axes([0, 0, 1, 1]); cv.set_xlim(0, W_FULL); cv.set_ylim(0, H); cv.axis("off")


def inset(x, y, w, h):
    return fig.add_axes([x / W_FULL, y / H, w / W_FULL, h / H])


def block(x0, x1, letter, title):
    cv.add_patch(FancyBboxPatch((x0, 0.04), x1 - x0, H - 0.08, boxstyle="round,pad=0,rounding_size=0.07", fc=PANEL, ec="none", zorder=0))
    cv.text(x0 + 0.09, H - 0.20, letter, fontsize=9, fontweight="bold", ha="left", va="center")
    cv.text(x0 + 0.30, H - 0.20, title, fontsize=8, ha="left", va="center")


def thumb(x, y, s, im, extent=None):
    ax = inset(x, y, s, s); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_alpha(0)
    art = ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="hanning", extent=extent)
    clip = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.07", transform=ax.transAxes)
    art.set_clip_path(clip)
    return ax, clip


def arrow(x0, y0, x1, y1, col=None):
    cv.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=col or C["ink2"], shrinkA=0, shrinkB=0, zorder=3))


# ================================================================== (a) the scale shift
XA0, XA1 = 0.04, 2.40
block(XA0, XA1, "(a)", "Scale shift across centers")
TS = 0.68; TG = 0.06; TX = XA0 + (XA1 - XA0 - 3 * TS - 2 * TG) / 2.0; TY = 1.12
for i, (f, dom, name, cxf, cyf) in enumerate(FRAMES):
    im = np.asarray(Image.open(os.path.join(OUT, "samples", f)).convert("L")); h, w = im.shape
    side = WIN * h
    x0 = int(round(min(max(cxf * w - side / 2.0, 0), w - side))); y0 = int(round(min(max(cyf * h - side / 2.0, 0), h - side))); n = int(round(side))
    ax, clip = thumb(TX + i * (TS + TG), TY, TS, im[y0:y0 + n, x0:x0 + n])
    med = stats[dom + "_train"]["cls"]["LV"]["median"]      # px at short edge 800
    b = med * h / 800.0; m = 0.06 * n
    r = Rectangle((n - m - b, n - m - b), b, b, fill=False, ec="white", lw=0.9, zorder=5); ax.add_patch(r); r.set_clip_path(clip)
    xc = TX + i * (TS + TG) + TS / 2.0
    cv.text(xc, TY - 0.09, name, fontsize=7, ha="center", va="center", color=DOM[dom], fontweight="bold")
ax = inset(XA0 + 0.62, 0.40, XA1 - XA0 - 0.76, 0.46); ax.patch.set_alpha(0)
for i, (_, dom, name, _, _) in enumerate(FRAMES):
    st = stats[dom + "_train"]["cls"]["LV"]; y = 2 - i
    ax.plot([st["p10"], st["p90"]], [y, y], color=DOM[dom], lw=3.6, solid_capstyle="butt", zorder=3)
    ax.plot([st["median"]], [y], marker="o", ms=3.8, mfc="white", mec=DOM[dom], mew=1.0, zorder=4)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels([fr[2] for fr in FRAMES]); ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, 2.6); ax.set_xlim(50, 262); ax.set_xticks([50, 100, 150, 200, 250]); ax.spines["left"].set_visible(False)
ax.set_xlabel("left-ventricle size at the detector input (px)", labelpad=1.5)

# ================================================================== (b) three ways to adapt
XB0, XB1 = 2.48, 5.86
block(XB0, XB1, "(b)", "Adapting a detector at test time")
R = rows("heart4cc_frcnn")
src = R["Source-only (FrozenBN)"]["mean"]; gap = R[UPPER]["mean"] - src
SV = ["IoU-Filter", "AMROD", "WHW", "CD-Buffer", "SGP", "O-SFDA", "VLOD-TTA objective", "BufferTTA"]
ENS = ["Senior heatmap fusion (18 views)", "18-view WBF", "18-view WBF + AR"]
items = [("Parameter adaptation", "one view as acquired", max(R[k]["mean"] for k in SV), C["lgray"]),
         ("View ensemble", "18 views at fixed scales", max(R[k]["mean"] for k in ENS), C["blue"]),
         ("TAC (ours)", "7 views at the inferred scale", R["TAC 7 views (quadrature)"]["mean"], OURS)]
RY = [1.58, 1.11, 0.64]; RH = 0.40
GX = XB0 + 1.48           # left edge of the view glyphs
DX0, DX1 = XB0 + 2.12, XB0 + 2.56   # detector box
BX0, BX1 = XB0 + 2.72, XB1 - 0.14   # bars
cv.add_patch(FancyBboxPatch((XB0 + 0.06, RY[2] - RH / 2.0 - 0.02), XB1 - XB0 - 0.12, RH + 0.04, boxstyle="round,pad=0,rounding_size=0.05", fc=TINT, ec="none", zorder=1))
for hx, ht in [(GX + 0.25, "views"), ((DX0 + DX1) / 2.0, "detector"), ((BX0 + BX1) / 2.0, "loss closed (%)")]:
    cv.text(hx, RY[0] + RH / 2.0 + 0.04, ht, fontsize=7, ha="center", va="bottom", color=C["ink2"])
for k, (name, subt, apv, col) in enumerate(items):
    y = RY[k]; ours = col == OURS
    cv.text(XB0 + 0.14, y + 0.075, name, fontsize=7.5, ha="left", va="center", fontweight="bold", color=OURS if ours else C["ink"], zorder=4)
    cv.text(XB0 + 0.14, y - 0.085, subt, fontsize=7, ha="left", va="center", color=C["ink2"], zorder=4)
    if k == 0:      # one view
        s = 0.17; cv.add_patch(Rectangle((GX + 0.25 - s / 2, y - s / 2), s, s, fc="white", ec=C["gray"], lw=0.8, zorder=3))
    elif k == 1:    # a sweep of fixed scales
        for j, s in enumerate([0.08, 0.11, 0.14, 0.17, 0.20]):
            cv.add_patch(Rectangle((GX + 0.02 + j * 0.075, y - 0.10), s, s, fc="white", ec=C["blue"], lw=0.7, zorder=3 + j))
    else:           # the surrogate posterior of the scale and the views at its quadrature nodes
        xx = np.linspace(-0.25, 0.25, 80); sg = 0.085
        cv.plot(GX + 0.25 + xx, y + 0.045 + 0.13 * np.exp(-0.5 * (xx / sg) ** 2), color=OURS, lw=0.9, zorder=4)
        cv.plot([GX, GX + 0.5], [y + 0.045, y + 0.045], color=OURS, lw=0.5, zorder=4)
        for dx in (-np.sqrt(3) * sg, 0.0, np.sqrt(3) * sg):
            s = 0.105 * np.exp(dx / 0.6)
            cv.plot([GX + 0.25 + dx, GX + 0.25 + dx], [y + 0.045, y + 0.01], color=OURS, lw=0.5, zorder=4)
            cv.add_patch(Rectangle((GX + 0.25 + dx - s / 2, y - 0.165), s, s + 0.0, fc="white", ec=OURS, lw=0.8, zorder=4))
    arrow(GX + 0.56, y, DX0 - 0.03, y)
    upd = k == 0
    cv.add_patch(FancyBboxPatch((DX0, y - 0.105), DX1 - DX0, 0.21, boxstyle="round,pad=0,rounding_size=0.04", fc=UPDATED if upd else FROZEN,
                                ec="#B9822B" if upd else C["blue"], lw=0.6, zorder=3))
    cv.text((DX0 + DX1) / 2.0, y, "updated" if upd else "frozen", fontsize=7, ha="center", va="center", color=C["ink"], zorder=4)
ax = inset(BX0, RY[2] - RH / 2.0, BX1 - BX0, RY[0] - RY[2] + RH); ax.patch.set_alpha(0)
for k, (name, subt, apv, col) in enumerate(items):
    v = 100.0 * (apv - src) / gap
    ax.barh(RY[k], v, height=0.19, color=col, lw=0)
    ax.text(v + 4.0, RY[k], f"{v:.0f}%", fontsize=7.5 if col == OURS else 7, ha="left", va="center", color=OURS if col == OURS else C["ink"],
            fontweight="bold" if col == OURS else "normal")
ax.set_ylim(RY[2] - RH / 2.0, RY[0] + RH / 2.0); ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xticks([0, 50, 100])

# ================================================================== (c) one frame before and after canonicalization
XC0, XC1 = 5.94, W_FULL - 0.04
block(XC0, XC1, "(c)", "One test frame")
f = EX["factor"][0]
im = np.asarray(Image.open(os.path.join(OUT, "samples", "c3_small_f285.jpg")).convert("L")); h, w = im.shape
xs = [b["bbox"][0] for b in EX["gt"]] + [b["bbox"][0] + b["bbox"][2] for b in EX["gt"]]
ys = [b["bbox"][1] for b in EX["gt"]] + [b["bbox"][1] + b["bbox"][3] for b in EX["gt"]]
cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
big = 0.8 * h; small = big / f            # the same window of the detector input before and after resampling by f
assert max(max(xs) - min(xs), max(ys) - min(ys)) < small
CS = 0.74; CX = XC0 + (XC1 - XC0 - CS) / 2.0
for (yy, side, kind, lab) in [(1.13, big, "src", "As acquired"), (0.22, small, "tac", f"TAC, $f$ = {f:.2f}")]:
    x0 = min(max(cx - side / 2.0, 0), w - side); y0 = min(max(cy - side / 2.0, 0), h - side)
    ax, clip = thumb(CX, yy, CS, im, extent=(0, w, h, 0)); ax.set_xlim(x0, x0 + side); ax.set_ylim(y0 + side, y0)
    for b in EX[kind]:
        if b["score"] >= 0.5:
            bx, by, bw, bh = b["bbox"]
            r = Rectangle((bx, by), bw, bh, fill=False, ec=PAL[(b["cat"] - 1) % len(PAL)], lw=1.0, zorder=4); ax.add_patch(r); r.set_clip_path(clip)
    cv.text(CX + CS / 2.0, yy - 0.085, lab, fontsize=7, ha="center", va="center", color=OURS if kind == "tac" else C["ink"],
            fontweight="bold" if kind == "tac" else "normal")
save(fig, "fig1_teaser", tight=False)
print("LV", {d: [round(stats[d + "_train"]["cls"]["LV"][k], 1) for k in ("p10", "median", "p90")] for _, d, _, _, _ in FRAMES}, "factor", round(f, 3),
      "boxes", sum(b["score"] >= 0.5 for b in EX["src"]), sum(b["score"] >= 0.5 for b in EX["tac"]),
      "shares", [(n, round(a, 2), round(100 * (a - src) / gap, 1)) for n, _, a, _ in items], "gap", round(gap, 2))
