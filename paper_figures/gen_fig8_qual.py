"""Fig. 8 - qualitative results (seed 42, test): ground truth, source detector at the default scale, TAC-3 with two rounds.
One example per setting, chosen among the images with the largest inferred scale factor. Data: samples/qual_examples_s42.json
(boxes dumped from the saved detections) and the corresponding test images. Box labels are placed greedily so that they do not
overlap each other (candidate positions: above-left, inside top-left, below-left, above-right of the box)."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pubstyle import C, OUT, save

EX = json.load(open(os.path.join(OUT, "samples", "qual_examples_s42.json")))
ORDER = [("c3_small_f285", "Heart-4CC, c1 → c3"), ("c2_small_f033", "Heart-4CC, c3 → c2"), ("abd_ph_f157", "Abdomen, Samsung → Philips"), ("spine_sa_f205", "Spine, GE → Samsung")]
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#8C8C8C", "#264653"]
fig, axes = plt.subplots(3, 4, figsize=(6.9, 4.9), gridspec_kw={"wspace": 0.04, "hspace": 0.16})
renderer = fig.canvas.get_renderer()


def crop_box(ex, pad=0.07):
    xs = [b["bbox"][0] for b in ex["gt"]] + [b["bbox"][0] + b["bbox"][2] for b in ex["gt"]]
    ys = [b["bbox"][1] for b in ex["gt"]] + [b["bbox"][1] + b["bbox"][3] for b in ex["gt"]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys); w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = max(w, h) * (1 + 2 * pad)
    return cx - s / 2, cy - s / 2, s


def place_labels(ax, boxes, names, kind):
    """draw one label per box, trying several anchor positions until it overlaps no previously placed label"""
    placed = []
    for b in sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        x, y, w, h = b["bbox"]; col = PAL[(b["cat"] - 1) % len(PAL)]
        name = names.get(str(b["cat"]), str(b["cat"])); name = {"CR": "CRO", "R": "RIB"}.get(name, name)
        lab = name if kind == "gt" else f"{name} {b['score']:.2f}"
        cands = [((x + 1, y - 1), "bottom", "left"), ((x + 1, y + 1), "top", "left"), ((x + 1, y + h + 1), "top", "left"), ((x + w - 1, y - 1), "bottom", "right"),
                 ((x + w - 1, y + h + 1), "top", "right"), ((x + w - 1, y + 1), "top", "right"), ((x + 1, y + h - 1), "bottom", "left"), ((x + w - 1, y + h - 1), "bottom", "right")]
        # last resort: slide the label down along the left edge of the box in steps of one label height
        step = h / 8.0
        cands += [((x + 1, y + 1 + k * step), "top", "left") for k in range(1, 8)]
        cands += [((x + w - 1, y + 1 + k * step), "top", "right") for k in range(1, 8)]
        cands += [((x + 1 + k * w / 6.0, y - 1), "bottom", "left") for k in range(1, 6)]
        t = None
        for (px, py), va, ha in cands:
            t = ax.text(px, py, lab, fontsize=4.8, color="white", va=va, ha=ha, bbox=dict(facecolor=col, edgecolor="none", pad=0.2, alpha=0.9), zorder=5)
            bb = t.get_window_extent(renderer).expanded(1.02, 1.06)
            if not any(bb.overlaps(q) for q in placed):
                placed.append(bb); break
            t.remove(); t = None
        if t is None:  # every candidate collides: fall back to the default position
            t = ax.text(x + 1, y - 1, lab, fontsize=5.0, color="white", va="bottom", ha="left", bbox=dict(facecolor=col, edgecolor="none", pad=0.25, alpha=0.88), zorder=5)
            placed.append(t.get_window_extent(renderer))


for j, (key, title) in enumerate(ORDER):
    ex = EX[key]; im = np.asarray(Image.open(os.path.join(OUT, "samples", key + ".jpg")).convert("L"))
    x0, y0, s = crop_box(ex)
    x0 = max(0, min(x0, ex["w"] - s)); y0 = max(0, min(y0, ex["h"] - s))
    for i, (kind, lab) in enumerate([("gt", "ground truth"), ("src", "source detector, default scale"), ("tac", "TAC-3, 2 rounds (ours)")]):
        ax = axes[i, j]; ax.imshow(im, cmap="gray", vmin=0, vmax=255); ax.set_xlim(x0, x0 + s); ax.set_ylim(y0 + s, y0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        boxes = ex[kind] if kind == "gt" else [b for b in ex[kind] if b["score"] >= 0.5]
        for b in boxes:
            x, y, w, h = b["bbox"]; col = PAL[(b["cat"] - 1) % len(PAL)]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=col, lw=0.9, ls="--" if kind == "gt" else "-"))
        place_labels(ax, boxes, ex["names"], kind)
        if i == 0:
            ax.set_title(title, fontsize=6.8, pad=3)
        n = len(boxes); extra = f"  $f$ = {ex['factor'][0]:.2f}" if kind == "tac" and ex.get("factor") else ""
        ax.text(0.02, 0.03, f"{lab}: {n} box{'es' if n != 1 else ''}{extra}", transform=ax.transAxes, fontsize=5.6, color="white", ha="left", va="bottom",
                bbox=dict(facecolor="black", edgecolor="none", pad=0.4, alpha=0.55), zorder=6)
save(fig, "fig8_qual")
