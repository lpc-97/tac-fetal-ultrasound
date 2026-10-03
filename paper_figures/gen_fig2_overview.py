"""Fig. 2 - method overview drawn from real data (Heart-4CC, seed 42, source center 3 -> target center 2, the image of Fig. 8):
(1) the first pass at the calibrated scale yields candidate boxes with confidences; (2) their log-sizes are fused with the
source prior of canonical structure sizes into a robust MAP estimate of the scale factor (each candidate votes for the shift
that brings it to its canonical size; responsibilities down-weight implausible boxes; the posterior over f is shown);
(3) the image is resampled to the canonical scale and the frozen detector is run on the canonicalized views;
(4) the boxes of all views are fused by WBF and one box per structure is selected under the pairwise layout prior.
The scale inference replays adapteacher/engine/tta_canon.py (MIN_SCORE 0.3, SIGMA_OBS 0.1, SIGMA_U 1, OUTLIER 0.2, 3 IRLS steps)."""
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image
from pubstyle import C, OURS, OUT, AUX, save

INK, INK2 = C["ink"], C["ink2"]
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#8C8C8C", "#264653"]  # by class id 1..9
EX = json.load(open(os.path.join(OUT, "samples", "qual_examples_s42.json")))["c2_small_f033"]
PR = json.load(open(os.path.join(AUX, "anat_prior_c3.json")))
NAMES = {int(k): v for k, v in PR["names"].items()}
IM = np.asarray(Image.open(os.path.join(OUT, "samples", "c2_small_f033.jpg")).convert("L"))
W, H = EX["w"], EX["h"]
FW, FH = 6.9, 4.1
ASP = FW / FH  # figure-fraction height that makes a square axes = width * ASP

# ---------------------------------------------------------------- replay the scale inference (round 1) on the pass-1 candidates
S0 = PR.get("indomain_best_scale", 800); k0 = min(S0 / min(W, H), 4000 / max(W, H)); shift = math.log(S0 / 800.0)
CAN = PR["canonical_at_800"]
cands = [b for b in EX["src"] if b["score"] >= 0.3]
z = np.array([math.log(math.sqrt(max(b["bbox"][2], 1) * max(b["bbox"][3], 1)) * k0) for b in cands])
p = np.array([b["score"] for b in cands]); cls = [b["cat"] - 1 for b in cands]
mu = np.array([CAN[str(k)]["logsize_mean"] + shift for k in cls]); sg2 = np.array([CAN[str(k)]["logsize_std"] ** 2 for k in cls]) + 0.1 ** 2
SIG_U, RHO = 1.0, 0.2
u = 0.0
for _ in range(3):
    lik = np.exp(-0.5 * (z + u - mu) ** 2 / sg2) / np.sqrt(2 * np.pi * sg2); r = p * lik / (p * lik + (1 - p) * RHO + 1e-12)
    u = float(np.sum(r * (mu - z) / sg2) / (np.sum(r / sg2) + 1.0 / SIG_U ** 2))
f1 = math.exp(u); f_final = EX["factor"][0]


def neg_log_post(uu):
    lik = np.exp(-0.5 * (z + uu - mu) ** 2 / sg2) / np.sqrt(2 * np.pi * sg2)
    return -np.sum(np.log(p * lik + (1 - p) * RHO)) + uu ** 2 / (2 * SIG_U ** 2)


# ---------------------------------------------------------------- crop window (union of the ground-truth boxes + margin)
xs = [b["bbox"][0] for b in EX["gt"]] + [b["bbox"][0] + b["bbox"][2] for b in EX["gt"]]
ys = [b["bbox"][1] for b in EX["gt"]] + [b["bbox"][1] + b["bbox"][3] for b in EX["gt"]]
cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
CW, CH = min(W, (max(xs) - min(xs)) * 1.12), min(H, (max(ys) - min(ys)) * 1.22)   # rectangular crop window around the structures
X0, Y0 = max(0, min(cx - CW / 2, W - CW)), max(0, min(cy - CH / 2, H - CH))
RATIO = CH / CW   # axes height = width * RATIO * ASP keeps pixels square

fig = plt.figure(figsize=(FW, FH))
renderer = fig.canvas.get_renderer()


def stage_title(x, y, num, text):
    fig.text(x, y, f"{num}  {text}", fontsize=7.0, fontweight="bold", color=INK, ha="left", va="bottom")


def image_axes(x, y, w, boxes=None, flip=False, frame=INK2, lw=0.6):
    """square axes showing the crop window; w in figure fraction, height derived so that pixels are square"""
    ax = fig.add_axes([x, y, w, w * RATIO * ASP]); ax.imshow(IM[:, ::-1] if flip else IM, cmap="gray", vmin=0, vmax=255, aspect="auto")
    if flip:
        ax.set_xlim(W - X0 - CW, W - X0)
    else:
        ax.set_xlim(X0, X0 + CW)
    ax.set_ylim(Y0 + CH, Y0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_edgecolor(frame); s.set_linewidth(lw)
    placed = []
    if boxes:
        for b in sorted(boxes, key=lambda b: (b["bbox"][1], b["bbox"][0])):
            bx, by, bw, bh = b["bbox"]; col = PAL[(b["cat"] - 1) % 9]
            ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=col, lw=0.8))
            lab = f"{NAMES[b['cat'] - 1]} {b['score']:.2f}"
            cand = [((bx + 1, by - 1), "bottom", "left"), ((bx + 1, by + 1), "top", "left"), ((bx + bw - 1, by - 1), "bottom", "right"), ((bx + 1, by + bh + 1), "top", "left"), ((bx + bw - 1, by + bh + 1), "top", "right")]
            cand += [((bx + 1, by + 1 + k * bh / 8), "top", "left") for k in range(1, 8)] + [((bx + bw - 1, by + 1 + k * bh / 8), "top", "right") for k in range(1, 8)]
            for (px, py), va, ha in cand:
                t = ax.text(px, py, lab, fontsize=4.8, color="white", va=va, ha=ha, bbox=dict(facecolor=col, edgecolor="none", pad=0.18, alpha=0.9), zorder=5)
                bb = t.get_window_extent(renderer).expanded(1.02, 1.06)
                if not any(bb.overlaps(q) for q in placed):
                    placed.append(bb); break
                t.remove()
    return ax


def arrow(x0, y0, x1, y1, text=None, col=INK2, dy=0.018):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=col))
    if text:
        fig.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, fontsize=5.2, color=col, ha="center", va="bottom", style="italic")


TOP, IMW = 0.905, 0.205          # top of the image row, width of the stage-1 / stage-4 images
IMH = IMW * RATIO * ASP
lv = 2; can_lv = math.exp(CAN[str(lv)]["logsize_mean"] + shift) / k0   # canonical LV size in original pixels

# ================================================================ stage 1: pass 1 with candidates
stage_title(0.012, 0.935, "1", "Pass 1 at the calibrated scale $s_0$")
ax1 = image_axes(0.012, TOP - IMH, IMW, boxes=cands)
ax1.plot([X0 + CW * 0.03, X0 + CW * 0.03 + can_lv], [Y0 + CH * 0.05] * 2, color=PAL[lv], lw=2.4, solid_capstyle="butt", zorder=6)
ax1.text(X0 + CW * 0.03 + can_lv + CW * 0.015, Y0 + CH * 0.05, "canonical LV size (prior)", fontsize=4.8, color=PAL[lv], ha="left", va="center", zorder=6, bbox=dict(facecolor="black", edgecolor="none", pad=0.15, alpha=0.45))
ax1.text(0.98, 0.03, f"{len(cands)} candidates $(b_i, c_i, p_i)$", transform=ax1.transAxes, fontsize=5.4, color="white", ha="right", va="bottom", bbox=dict(facecolor="black", edgecolor="none", pad=0.3, alpha=0.55), zorder=6)
fig.text(0.012 + IMW / 2, TOP - IMH - 0.03, "frozen source detector, no target labels,\none image at a time; sizes $z_i=\\log\\sqrt{w_ih_i}$", fontsize=5.6, color=INK2, ha="center", va="top", style="italic", linespacing=1.3)
# the source prior as a small table (the only information that leaves the source site)
ty = TOP - IMH - 0.125
fig.text(0.012, ty, f"Source anatomical prior (center 3 training set, {PR['n_images']} images)", fontsize=5.8, fontweight="bold", color=INK, ha="left", va="top")
cols = [0.03, 0.066, 0.152, 0.19, 0.22]
for cx_, h in zip(cols, ["", "structure", "size @ $s_0$", "$\\sigma_k$", "$n_k$"]):
    fig.text(cx_, ty - 0.03, h, fontsize=5.2, color=INK2, ha="left" if cx_ < 0.1 else "right", va="top")
for i, k in enumerate(sorted(range(9), key=lambda k: -CAN[str(k)]["logsize_mean"])):
    yy = ty - 0.06 - i * 0.03
    fig.patches.append(Rectangle((cols[0], yy - 0.007), 0.012, 0.014, transform=fig.transFigure, fc=PAL[k], ec="none"))
    fig.text(cols[1], yy, NAMES[k], fontsize=5.2, color=INK, ha="left", va="center")
    fig.text(cols[2], yy, f"{math.exp(CAN[str(k)]['logsize_mean'] + shift):.0f} px", fontsize=5.2, color=INK, ha="right", va="center")
    fig.text(cols[3], yy, f"{CAN[str(k)]['logsize_std']:.2f}", fontsize=5.2, color=INK, ha="right", va="center")
    fig.text(cols[4], yy, f"{PR['max_count'][k]}", fontsize=5.2, color=INK, ha="right", va="center")
fig.text(0.012, ty - 0.06 - 9 * 0.03 - 0.006, "+ pairwise layout Gaussians $(\\mathbf{m}_{kl},\\mathbf{\\Sigma}_{kl})$, $s_0$ = %d px" % S0, fontsize=5.2, color=INK2, ha="left", va="top")

# ================================================================ stage 2: evidence fusion for the scale factor
stage_title(0.262, 0.935, "2", "Fuse the evidence with the size prior")
ax2 = fig.add_axes([0.295, 0.575, 0.215, 0.33])
rows = sorted(set(cls), key=lambda k: -CAN[str(k)]["logsize_mean"]); ypos = {k: i for i, k in enumerate(rows)}
for k in rows:
    m, s = CAN[str(k)]["logsize_mean"] + shift, CAN[str(k)]["logsize_std"]
    ax2.add_patch(Rectangle((m - s, ypos[k] - 0.32), 2 * s, 0.64, fc=PAL[k], ec="none", alpha=0.22))
    ax2.plot([m, m], [ypos[k] - 0.32, ypos[k] + 0.32], color=PAL[k], lw=0.9)
seen = {}
for i, (k, zi, pi, ri) in enumerate(zip(cls, z, p, r)):
    n_k = sum(1 for kk in cls if kk == k); j = seen.get(k, 0); seen[k] = j + 1
    yy = ypos[k] + (0 if n_k == 1 else (-0.18 + 0.36 * j / (n_k - 1)))
    ax2.annotate("", xy=(zi + u, yy), xytext=(zi, yy), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, mutation_scale=6, alpha=0.7))
    ax2.plot(zi, yy, marker="o", ms=3.0 + 3.5 * pi, mfc="white", mec=PAL[k], mew=0.8, zorder=4)
    ax2.plot(zi + u, yy, marker="o", ms=3.0 + 3.5 * pi, mfc=PAL[k], mec=PAL[k], mew=0.6, alpha=0.25 + 0.75 * ri, zorder=5)
ax2.set_yticks(range(len(rows))); ax2.set_yticklabels([NAMES[k] for k in rows], fontsize=6)
ticks = [50, 100, 200, 400]; ax2.set_xticks([math.log(t) for t in ticks]); ax2.set_xticklabels([str(t) for t in ticks], fontsize=6)
ax2.set_xlim(math.log(35), math.log(560)); ax2.set_ylim(-0.7, len(rows) - 0.3); ax2.invert_yaxis()
ax2.set_xlabel("structure size at $s_0$ (px, log axis)", fontsize=6, labelpad=1)
ax2.grid(axis="x", alpha=0.15); ax2.grid(axis="y", alpha=0)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
fig.text(0.295, 0.912, "band: prior $\\mu_k\\pm\\sigma_k$;  open: observed $z_i$ (size $\\propto p_i$);  filled: $z_i+u^*$ (opacity $=r_i$)", fontsize=4.9, color=INK2, ha="left", va="bottom")
# posterior over f
ax3 = fig.add_axes([0.295, 0.215, 0.215, 0.24])
uu = np.linspace(math.log(0.2), math.log(2.5), 300); nlp = np.array([neg_log_post(v) for v in uu]); nlp -= nlp.min()
i0 = int(np.searchsorted(uu, 0.0)); ytop = nlp[i0] * 1.18
ax3.plot(np.exp(uu), nlp, color=OURS, lw=1.2)
ax3.set_xscale("log"); ax3.set_xticks([0.25, 0.5, 1, 2]); ax3.set_xticklabels(["0.25", "0.5", "1", "2"], fontsize=6); ax3.minorticks_off()
ax3.set_ylim(-ytop * 0.30, ytop); ax3.set_yticks([0, 5, 10, 15]); ax3.tick_params(axis="y", labelsize=5.5)
ax3.set_xlabel("scale factor $f = e^{u}$", fontsize=6, labelpad=1); ax3.set_ylabel("$-\\log$ posterior", fontsize=6, labelpad=1)
ax3.plot([1], [nlp[i0]], marker="o", ms=4, mfc="white", mec=INK2, mew=0.8, zorder=5); ax3.text(1.07, nlp[i0], "start $u=0$", fontsize=5.2, color=INK2, va="center")
ax3.plot([f1], [0], marker="o", ms=4.5, color=OURS, zorder=5)
ax3.text(f1, -ytop * 0.05, f"MAP $f^*$ = {f1:.2f} (3 IRLS steps)\n2nd round: $f$ = {f_final:.2f}", fontsize=5.3, color=OURS, ha="center", va="top", linespacing=1.25)
ax3.axhline(0, color=INK2, lw=0.4, alpha=0.5)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)
fig.text(0.4025, 0.04, "$r_i \\propto p_i\\,\\mathcal{N}(z_i+u;\\mu_{c_i},\\tilde\\sigma_i^2)$ vs. $(1-p_i)\\rho_{\\mathrm{out}}$;   "
         "$u^{*}=\\frac{\\sum_i r_i(\\mu_{c_i}-z_i)/\\tilde\\sigma_i^{2}}{\\sum_i r_i/\\tilde\\sigma_i^{2}+1/\\sigma_u^{2}}$", fontsize=6.0, color=INK, ha="center", va="bottom")

# ================================================================ stage 3: canonicalized views (drawn at f times the pass-1 size)
X3 = 0.545
stage_title(X3, 0.935, "3", "Canonicalized views")
small = IMW * f1; gap = 0.008
ax4 = image_axes(X3, TOP - small * RATIO * ASP, small, frame=PAL[lv], lw=0.9)
ax4.plot([X0 + CW * 0.04, X0 + CW * 0.04 + can_lv / f1], [Y0 + CH * 0.09] * 2, color=PAL[lv], lw=2.2, solid_capstyle="butt", zorder=6)
ax5 = image_axes(X3 + small + gap, TOP - small * RATIO * ASP, small, flip=True, frame=PAL[lv], lw=0.9)
yl = TOP - small * RATIO * ASP - 0.012
fig.text(X3 + small / 2, yl, "$\\times f^*$", fontsize=6, color=INK, ha="center", va="top")
fig.text(X3 + small * 1.5 + gap, yl, "$\\times f^*$, flipped", fontsize=6, color=INK, ha="center", va="top")
fig.text(X3 + small + gap / 2, yl - 0.05, "structures return to the prior\nsize (LV = ruler)", fontsize=5.4, color=INK2, ha="center", va="top", style="italic", linespacing=1.3)
sw = small * 0.72; y7 = yl - 0.13 - sw * RATIO * ASP
for i, lab in enumerate(["0.85$f^*$", "1.15$f^*$"]):
    image_axes(X3 + i * (sw + gap), y7, sw, frame=C["sky"], lw=0.7)
    fig.text(X3 + i * (sw + gap) + sw / 2, y7 - 0.012, lab + " (+flip)", fontsize=5.2, color=INK2, ha="center", va="top")
fig.text(X3 + small + gap / 2, y7 - 0.06, "TAC-3: pass 1 + 2 views\nTAC-7: + 4 views at $\\pm$15%", fontsize=5.8, color=INK, ha="center", va="top", linespacing=1.3)
fig.text(X3 + small + gap / 2, y7 - 0.145, "frozen detector on every view", fontsize=5.6, color=INK2, ha="center", va="top", style="italic")

# ================================================================ stage 4: fusion and re-scoring
X4 = 0.79
stage_title(X4 - 0.014, 0.935, "4", "Fuse the views, re-score with the layout prior")
final = [b for b in EX["tac"] if b["score"] >= 0.5]
ax6 = image_axes(X4, TOP - IMH, IMW, boxes=final)
ax6.text(0.98, 0.03, f"output: {len(final)} structures", transform=ax6.transAxes, fontsize=5.4, color="white", ha="right", va="bottom", bbox=dict(facecolor="black", edgecolor="none", pad=0.3, alpha=0.55), zorder=6)
fig.text(X4 + IMW / 2, TOP - IMH - 0.03, "WBF over all views (IoU 0.55), then one box per\nstructure by ICM on $\\sum_k p_{a_k}+\\lambda\\,\\overline{\\mathrm{plaus}}$;\ncompetitors $\\times\\gamma$", fontsize=5.5, color=INK, ha="center", va="top", linespacing=1.3)
# layout-prior graph: mean position of every structure relative to the LV on the source training set (offsets in pixels of a 720x576 frame)
gw = 0.19; ax7 = fig.add_axes([X4 + (IMW - gw) / 2, 0.11, gw, 0.27]); ax7.axis("off")
pos = {lv: (0.0, 0.0)}
for k in range(9):
    if k != lv and f"{lv}_{k}" in PR["pairs"]:
        m = PR["pairs"][f"{lv}_{k}"]["mean"]; pos[k] = (m[0] * 720, m[1] * 576)
for a in pos:
    for b in pos:
        if a < b:
            ax7.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=INK2, lw=0.35, alpha=0.3, zorder=1)
OFF = {"LA": (4, 0, "left"), "RA": (4, -1, "left"), "LV": (-4, 0, "right"), "RV": (-4, -1, "right"), "VS": (-4, 0, "right"), "CRO": (4, 0, "left"), "SP": (4, 0, "left"), "DAO": (4, 0, "left"), "RIB": (4, 2, "left")}
for k, (xx, yy) in pos.items():
    ax7.plot(xx, yy, marker="o", ms=5, color=PAL[k], mec="white", mew=0.5, zorder=3)
    dx, dy, ha = OFF[NAMES[k]]
    ax7.text(xx + dx, yy + dy, NAMES[k], fontsize=5.0, color=INK, ha=ha, va="center", zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.1, alpha=0.7))
ax7.set_aspect("equal"); ax7.invert_yaxis(); ax7.margins(0.22)
fig.text(X4 + IMW / 2, 0.085, "pairwise layout prior $\\mathcal{N}(\\mathbf{m}_{kl},\\mathbf{\\Sigma}_{kl})$: mean\noffsets w.r.t. the LV on the source training set", fontsize=5.2, color=INK2, ha="center", va="top", linespacing=1.3)

# ================================================================ arrows between stages and footer
ym = TOP - IMH / 2
arrow(0.222, ym, 0.258, ym, "$z_i, p_i, c_i$")
arrow(0.516, ym, 0.544, ym, "$f^*$")
arrow(0.748, ym, 0.783, ym, "boxes of\nall views", dy=0.022)
fig.text(0.5, 0.005, "online, batch size 1, source-free  •  zero learnable parameters  •  only source-domain statistics $(\\mu_k,\\sigma_k,\\mathbf{m}_{kl},\\mathbf{\\Sigma}_{kl},n_k,s_0)$  •  example: center 3 → center 2, seed 42",
         fontsize=5.8, color=INK2, ha="center", va="bottom")
save(fig, "fig2_overview")
