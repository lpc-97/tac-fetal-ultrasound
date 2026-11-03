"""Failure cases (seed 42, Heart-4CC test, c1 -> c2): the two images with the largest drop of per-image recall at IoU 0.5 under
TAC-3 with two rounds relative to the source detector (results/reviewer/perimage_harm_4c.json, worst10 of tac3_2r at seed 42).
Rows: ground truth, source detector at the default scale, TAC-3 with two rounds; boxes with confidence >= 0.3, the threshold
of the recall statistic. Data: samples/fail/fail_examples_s42.json exported by server/perimage_harm.py; drawing code as in
gen_fig8_qual.py."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pubstyle import OUT, save

EX = json.load(open(os.path.join(OUT, "samples", "fail", "fail_examples_s42.json")))
KEYS = list(EX)
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#8C8C8C", "#264653"]
THR = 0.3
fig, axes = plt.subplots(3, len(KEYS), figsize=(3.35, 5.0), gridspec_kw={"wspace": 0.04, "hspace": 0.16})
renderer = fig.canvas.get_renderer()


def crop_box(ex, pad=0.07):
    """square crop around every drawn box: ground truth and the detections of both methods above the threshold"""
    bx = ex["gt"] + [b for k in ("src", "tac") for b in ex[k] if b["score"] >= THR]
    xs = [b["bbox"][0] for b in bx] + [b["bbox"][0] + b["bbox"][2] for b in bx]
    ys = [b["bbox"][1] for b in bx] + [b["bbox"][1] + b["bbox"][3] for b in bx]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys); w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = max(w, h) * (1 + 2 * pad)
    return cx - s / 2, cy - s / 2, s


def place_labels(ax, boxes, names, kind):
    placed = []
    for b in sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        x, y, w, h = b["bbox"]; col = PAL[(b["cat"] - 1) % len(PAL)]
        name = names.get(str(b["cat"]), str(b["cat"])); name = {"CR": "CRO", "R": "RIB"}.get(name, name)
        lab = name if kind == "gt" else f"{name} {b['score']:.2f}"
        cands = [((x + 1, y - 1), "bottom", "left"), ((x + 1, y + 1), "top", "left"), ((x + 1, y + h + 1), "top", "left"), ((x + w - 1, y - 1), "bottom", "right"),
                 ((x + w - 1, y + h + 1), "top", "right"), ((x + w - 1, y + 1), "top", "right"), ((x + 1, y + h - 1), "bottom", "left"), ((x + w - 1, y + h - 1), "bottom", "right")]
        step = h / 8.0
        cands += [((x + 1, y + 1 + k * step), "top", "left") for k in range(1, 8)]
        cands += [((x + w - 1, y + 1 + k * step), "top", "right") for k in range(1, 8)]
        t = None
        for (px, py), va, ha in cands:
            t = ax.text(px, py, lab, fontsize=5.5, color="white", va=va, ha=ha, bbox=dict(facecolor=col, edgecolor="none", pad=0.15, alpha=0.9), zorder=5, clip_on=True)
            bb = t.get_window_extent(renderer).expanded(1.02, 1.06)
            if not any(bb.overlaps(q) for q in placed):
                placed.append(bb); break
            t.remove(); t = None
        if t is None:
            t = ax.text(x + 1, y - 1, lab, fontsize=5.5, color="white", va="bottom", ha="left", bbox=dict(facecolor=col, edgecolor="none", pad=0.15, alpha=0.88), zorder=5, clip_on=True)
            placed.append(t.get_window_extent(renderer))


for j, key in enumerate(KEYS):
    ex = EX[key]; im = np.asarray(Image.open(os.path.join(OUT, "samples", "fail", key + ".jpg")).convert("L"))
    x0, y0, s = crop_box(ex)
    x0 = max(0, min(x0, ex["w"] - s)); y0 = max(0, min(y0, ex["h"] - s))
    for i, (kind, lab) in enumerate([("gt", "GT"), ("src", "source"), ("tac", "ours")]):
        ax = axes[i, j]; ax.imshow(im, cmap="gray", vmin=0, vmax=255); ax.set_xlim(x0, x0 + s); ax.set_ylim(y0 + s, y0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        boxes = ex[kind] if kind == "gt" else [b for b in ex[kind] if b["score"] >= THR]
        for b in boxes:
            x, y, w, h = b["bbox"]; col = PAL[(b["cat"] - 1) % len(PAL)]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=col, lw=0.9, ls="--" if kind == "gt" else "-"))
        place_labels(ax, boxes, ex["names"], kind)
        if i == 0:
            ax.set_title(f"Heart-4CC, c1 → c2 ({chr(97 + j)})", fontsize=7, pad=3)
        if kind == "gt":
            txt = f"GT: {len(boxes)} boxes"
        else:
            rec = ex["recall_src"] if kind == "src" else ex["recall_tac"]
            txt = f"{lab}: {rec}/{ex['n_gt']} matched" + (f", $f$ = {ex['factor'][0]:.2f}" if kind == "tac" else "")
        ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=6.5, color="white", ha="left", va="bottom",
                bbox=dict(facecolor="black", edgecolor="none", pad=0.4, alpha=0.55), zorder=6)
save(fig, "fig13_fail")
print({k: (v["recall_src"], v["recall_tac"], v["n_gt"], round(v["factor"][0], 2)) for k, v in EX.items()})
