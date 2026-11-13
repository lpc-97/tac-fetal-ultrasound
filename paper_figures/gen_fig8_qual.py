"""Fig. 8 - qualitative results (seed 42, test): ground truth, source detector at the default scale, TAC-3 with two rounds.
One example per setting, chosen among the images with the largest inferred scale factor. Data: samples/qual_examples_s42.json
(boxes dumped from the saved detections) and the corresponding test images. Box labels are placed greedily so that they do not
overlap each other (candidate positions: above-left, inside top-left, below-left, above-right of the box, then positions slid
along the box edges); when every candidate collides, the one with the smallest overlap area is used."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pubstyle import C, OUT, save

# Elsevier asks for at least 300 dpi for photographs in the final PDF; the figure is scaled up in LaTeX, so the PDF is written at
# 600 dpi. The image filter is fixed to hanning, the filter matplotlib's default (antialiased) chose for every panel at 300 dpi.
plt.rcParams["savefig.dpi"] = 400  # drawn at print size since 2026-11-10, so 400 dpi in the paper
EX =json.load(open(os.path.join(OUT, "samples", "qual_examples_s42.json")))
ORDER = [("c3_small_f285", "Heart-4CC, c1 → c3"), ("c2_small_f033", "Heart-4CC, c3 → c2"), ("abd_ph_f157", "Abdomen, Samsung → Philips"), ("spine_sa_f205", "Spine, GE → Samsung")]
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#8C8C8C", "#264653"]
fig, axes = plt.subplots(3, 4, figsize=(9.53, 6.77), gridspec_kw={"wspace": 0.04, "hspace": 0.12})
renderer = fig.canvas.get_renderer()


def crop_box(ex, pad=0.07):
    xs = [b["bbox"][0] for b in ex["gt"]] + [b["bbox"][0] + b["bbox"][2] for b in ex["gt"]]
    ys = [b["bbox"][1] for b in ex["gt"]] + [b["bbox"][1] + b["bbox"][3] for b in ex["gt"]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys); w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = max(w, h) * (1 + 2 * pad)
    return cx - s / 2, cy - s / 2, s


def overlap_area(a, b):
    """area of the intersection of two display-space bounding boxes (0 when disjoint)"""
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def place_labels(ax, boxes, names, kind, placed=None):
    """draw one label per box, trying several anchor positions until it overlaps no previously placed label and no
    obstacle in `placed` (the panel banner, since 2026-11-10)"""
    placed = list(placed or [])
    for b in sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        x, y, w, h = b["bbox"]; col = PAL[(b["cat"] - 1) % len(PAL)]
        name = names.get(str(b["cat"]), str(b["cat"])); name = ({"CR": "CRO", "R": "RIB"} if "LV" in names.values() else {}).get(name, name)  # heart codes only; the abdominal R stays R
        lab = name if kind == "gt" else f"{name} {b['score']:.2f}"
        cands = [((x + 1, y - 1), "bottom", "left"), ((x + 1, y + 1), "top", "left"), ((x + 1, y + h + 1), "top", "left"), ((x + w - 1, y - 1), "bottom", "right"),
                 ((x + w - 1, y + h + 1), "top", "right"), ((x + w - 1, y + 1), "top", "right"), ((x + 1, y + h - 1), "bottom", "left"), ((x + w - 1, y + h - 1), "bottom", "right")]
        # last resort: slide the label down along the left edge of the box in steps of one label height
        step = h / 8.0
        cands += [((x + 1, y + 1 + k * step), "top", "left") for k in range(1, 8)]
        cands += [((x + w - 1, y + 1 + k * step), "top", "right") for k in range(1, 8)]
        cands += [((x + 1 + k * w / 6.0, y - 1), "bottom", "left") for k in range(1, 6)]
        t = None; best = None
        for (px, py), va, ha in cands:
            t = ax.text(px, py, lab, fontsize=7, color="white", va=va, ha=ha, bbox=dict(facecolor=col, edgecolor="none", pad=0.15, alpha=0.9), zorder=5)
            bb = t.get_window_extent(renderer).expanded(1.02, 1.06)
            ov = sum(overlap_area(bb, q) for q in placed)
            if ov == 0:
                placed.append(bb); break
            if best is None or ov < best[0]:
                best = (ov, (px, py), va, ha)
            t.remove(); t = None
        if t is None:  # every candidate collides: take the candidate with the smallest overlap area
            _, (px, py), va, ha = best
            t = ax.text(px, py, lab, fontsize=7, color="white", va=va, ha=ha, bbox=dict(facecolor=col, edgecolor="none", pad=0.15, alpha=0.88), zorder=5)
            placed.append(t.get_window_extent(renderer))


for j, (key, title) in enumerate(ORDER):
    ex = EX[key]; im = np.asarray(Image.open(os.path.join(OUT, "samples", key + ".jpg")).convert("L"))
    x0, y0, s = crop_box(ex)
    x0 = max(0, min(x0, ex["w"] - s)); y0 = max(0, min(y0, ex["h"] - s))
    for i, (kind, lab) in enumerate([("gt", "GT"), ("src", "source"), ("tac", "ours")]):  # rows named in the caption
        ax = axes[i, j]; ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="hanning"); ax.set_xlim(x0, x0 + s); ax.set_ylim(y0 + s, y0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.apply_aspect()  # final (square) panel box before any label extent is measured
        for sp in ax.spines.values():
            sp.set_visible(False)
        boxes = ex[kind] if kind == "gt" else [b for b in ex[kind] if b["score"] >= 0.5]
        for b in boxes:
            x, y, w, h = b["bbox"]; col = PAL[(b["cat"] - 1) % len(PAL)]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=col, lw=1.1, ls="--" if kind == "gt" else "-"))
        if i == 0:
            ax.set_title(title, fontsize=8.5, pad=4, fontweight="bold")
        n = len(boxes); extra = f", $f$ = {ex['factor'][0]:.2f}" if kind == "tac" and ex.get("factor") else ""
        # the banner is drawn first and its extent (text + bbox pad) is an obstacle for the box labels
        bt = ax.text(0.02, 0.03, f"{lab}: {n} box{'es' if n != 1 else ''}{extra}", transform=ax.transAxes, fontsize=7.5, color="white", ha="left", va="bottom",
                     bbox=dict(facecolor="black", edgecolor="none", pad=0.4, alpha=0.55), zorder=6)
        banner = bt.get_window_extent(renderer).padded(0.4 * 7.5 * fig.dpi / 72.0 + 1.0)
        place_labels(ax, boxes, ex["names"], kind, [banner])
save(fig, "fig8_qual")
