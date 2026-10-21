"""Fig. 2 - framework overview of TAC, drawn from the deployed run on one real test frame.

Example: Heart-4CC, seed 42, source center 3 -> target center 2 (the frame of Fig. 8, the example of the project page).
Data, all read from files, nothing typed by hand:
  samples/qual_examples_s42.json["c2_small_f033"]  pass-1 boxes ("src"), final boxes of TAC-3 with two rounds ("tac"),
                                                    deployed factor after the second round ("factor"[0])
  samples/c2_small_f033.jpg                         the test frame (colour Doppler, as exported; shown with its colour
                                                    saturation reduced by a third so that the boxes stay legible)
  results/paper_aux/anat_prior_c3.json              source size and layout prior of center 3, calibrated s0, image count
The scale inference (ANI) is replayed exactly as adapteacher/engine/tta_canon.py and build/export_demo_data.py do
(MIN_SCORE 0.3, SIGMA_OBS 0.1, SIGMA_U 1, RHO_OUT 0.2, 3 EM alternations, h = clip(sqrt(3) s, 0.15, 0.5)), and the
result is checked against build/demo_data.json["result"] before anything is drawn. The canonicalized views are the crop
of the frame resampled by their factors and are drawn to scale relative to the input frame.
Layout in inches on a full-figure background axes; image panels and plots are separate axes on top of it."""
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image
from pubstyle import C, OUT, AUX, save

INK, INK2 = C["ink"], C["ink2"]
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#8C8C8C", "#264653"]  # class id 1..9, as Fig. 8
MOD = {"src": ("#5F6B73", "#F2F4F5"), "ani": ("#2A7F77", "#EAF4F2"), "uva": ("#2F6DA3", "#EBF1F8"), "sdf": ("#B0601C", "#FBF1E6")}
ARW = "#4A4A4A"
FS, FS_S = 7.0, 6.5           # body and small font size (pt, printed at full text width)

EX = json.load(open(os.path.join(OUT, "samples", "qual_examples_s42.json")))["c2_small_f033"]
PR = json.load(open(os.path.join(AUX, "anat_prior_c3.json")))
DEMO = json.load(open(os.path.join(OUT, "..", "build", "demo_data.json")))["result"]
NAMES = {int(k): v for k, v in PR["names"].items()}
CAN = PR["canonical_at_800"]
W, H = EX["w"], EX["h"]
RGB = np.asarray(Image.open(os.path.join(OUT, "samples", "c2_small_f033.jpg")).convert("RGB")).astype(float)
GRAY = RGB.mean(axis=2, keepdims=True)
IMG = np.clip(0.65 * RGB + 0.35 * GRAY, 0, 255).astype(np.uint8)    # colour saturation reduced by a third, intensities unchanged

# ------------------------------------------------------------------ replay ANI on the pass-1 candidates
S0 = PR.get("indomain_best_scale", 800)
k0 = min(S0 / min(W, H), 4000 / max(W, H)); shift = math.log(S0 / 800.0)
cands = [b for b in EX["src"] if b["score"] >= 0.3]
z = np.array([math.log(math.sqrt(max(b["bbox"][2], 1) * max(b["bbox"][3], 1)) * k0) for b in cands])
p = np.array([b["score"] for b in cands]); cls = [b["cat"] - 1 for b in cands]
mu = np.array([CAN[str(k)]["logsize_mean"] + shift for k in cls])
sg2 = np.array([CAN[str(k)]["logsize_std"] ** 2 for k in cls]) + 0.1 ** 2
SIG_U, RHO = 1.0, 0.2
u = 0.0
for _ in range(3):
    lik = np.exp(-0.5 * (z + u - mu) ** 2 / sg2) / np.sqrt(2 * np.pi * sg2)
    r = p * lik / (p * lik + (1 - p) * RHO + 1e-12)
    den = np.sum(r / sg2) + 1.0 / SIG_U ** 2
    u = float(np.sum(r * (mu - z) / sg2) / den)
s = 1.0 / math.sqrt(den); h = min(max(math.sqrt(3) * s, 0.15), 0.5)
f1 = math.exp(u); f2 = EX["factor"][0]
views = [f1 * math.exp(-h), f1, f1 * math.exp(h)]
assert abs(u - DEMO["u"]) < 1e-3 and abs(s - DEMO["s"]) < 1e-3 and abs(h - DEMO["h"]) < 1e-3, (u, s, h, DEMO)
assert all(abs(a - b) < 2e-3 for a, b in zip(views, DEMO["views"])) and abs(f2 - DEMO["f_deployed"]) < 1e-3
final = [b for b in EX["tac"] if b["score"] >= 0.5]
print(f"replayed: u*={u:.5f} f*={f1:.4f} s={s:.5f} h={h:.5f} views={[round(v, 4) for v in views]} f(2nd round)={f2:.4f} "
      f"candidates={len(cands)} final(>=0.5)={len(final)} n_images={PR['n_images']} s0={S0}")

# ------------------------------------------------------------------ crop window: union of all boxes of the example, with margin
X0, X1, Y0, Y1 = 90, 1015, 160, 750
CROP = IMG[Y0:Y1, X0:X1]
RATIO = (Y1 - Y0) / (X1 - X0)

# ------------------------------------------------------------------ canvas in inches
FW, FH = 7.16, 3.36
fig = plt.figure(figsize=(FW, FH))
renderer = fig.canvas.get_renderer()
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, FW); bg.set_ylim(0, FH); bg.axis("off"); bg.set_zorder(0)
top = fig.add_axes([0, 0, 1, 1], zorder=20); top.set_xlim(0, FW); top.set_ylim(0, FH); top.axis("off"); top.patch.set_alpha(0)


def axes_in(x, y, w, hh, z=5):
    return fig.add_axes([x / FW, y / FH, w / FW, hh / FH], zorder=z)


def text_width_in(t):
    return t.get_window_extent(renderer).width / fig.dpi


def label(x, y, s, ha="center", va="center", fs=FS_S, col=INK, ax=None, **kw):
    return (ax or top).text(x, y, s, fontsize=fs, color=col, ha=ha, va=va, **kw)


def module(x0, x1, y0, y1, key, title, sub):
    hd, tint = MOD[key]
    bg.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.06", fc=tint, ec=hd, lw=0.7))
    hh = 0.18
    bg.add_patch(FancyBboxPatch((x0, y1 - hh), x1 - x0, hh, boxstyle="round,pad=0,rounding_size=0.06", fc=hd, ec=hd, lw=0.7))
    bg.add_patch(Rectangle((x0, y1 - hh), x1 - x0, hh / 2, fc=hd, ec="none"))      # square off the lower corners of the strip
    t = label(x0 + 0.07, y1 - hh / 2, title, ha="left", fs=FS + 0.5, col="white", ax=bg, fontweight="bold")
    label(x0 + 0.07 + text_width_in(t) + 0.06, y1 - hh / 2, sub, ha="left", fs=FS, col="white", ax=bg)


def image_panel(x, y, w, img, boxes=None, scale=1.0, flip=False, edge="#404040", ls="-", lw=0.6, z=6):
    """the crop window, resampled by `scale` and drawn at width w (in); boxes in original pixels"""
    hh = w * RATIO
    ax = axes_in(x, y, w, hh, z)
    if scale != 1.0:      # the canonicalized view: the frame resampled by its factor, as the method does
        img = np.asarray(Image.fromarray(img).resize((max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale))), Image.BILINEAR))
    # display copy at 600 dpi of the printed panel, embedded as is in the PDF (interpolation "none")
    disp = Image.fromarray(img).resize((round(w * 600), round(hh * 600)), Image.LANCZOS)
    disp = np.asarray(disp)
    ax.imshow(disp[:, ::-1] if flip else disp, extent=(X0, X1, Y1, Y0), interpolation="none")
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(edge); sp.set_linewidth(lw); sp.set_linestyle(ls)
    for b in boxes or []:
        bx, by, bw, bh = b["bbox"]
        rect = Rectangle((bx, by), bw, bh, fill=False, ec=PAL[(b["cat"] - 1) % 9], lw=1.0, zorder=4)
        rect.set_path_effects([pe.Stroke(linewidth=2.2, foreground="white", alpha=0.9), pe.Normal()])
        ax.add_patch(rect)
    return ax, hh


def arrow(x0, y0, x1, y1, ls="-", col=ARW, lw=0.8, cs="arc3", ms=7):
    top.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=col,
                                  linestyle=ls, connectionstyle=cs, shrinkA=0, shrinkB=0, zorder=30))


def ruler(ax, scale):
    """prior LV size at s0, drawn in the top-left corner of a panel resampled by `scale`"""
    # the axes keep original-pixel data coordinates while the panel is drawn `scale` times smaller, so a fixed size at
    # the detector input (can_lv pixels of the pass-1 frame) spans can_lv / scale data units here
    x_ = X0 + 20 / scale; y_ = Y0 + 24 / scale
    ax.plot([x_, x_ + can_lv / scale], [y_, y_], color=PAL[2], lw=2.0, solid_capstyle="butt", zorder=6,
            path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])


def snowflake(cx, cy, rr, col="white"):
    for a in range(6):
        t = a * math.pi / 3 + math.pi / 2
        top.plot([cx, cx + rr * math.cos(t)], [cy, cy + rr * math.sin(t)], color=col, lw=0.7, solid_capstyle="round", zorder=27)
        for sgn in (-1, 1):
            bx, by = cx + 0.62 * rr * math.cos(t), cy + 0.62 * rr * math.sin(t)
            top.plot([bx, bx + 0.34 * rr * math.cos(t + sgn * 0.85)], [by, by + 0.34 * rr * math.sin(t + sgn * 0.85)], color=col, lw=0.6, zorder=27)


# ------------------------------------------------------------------ geometry (inches)
IW = 1.08                                  # width of the input and output frames
XI = 0.04
XA0, XA1 = 1.26, 2.74                      # ANI
XU0, XU1 = 2.86, 4.80                      # UVA
XS0, XS1 = 4.92, 5.94                      # SDF
XO0 = FW - 0.04 - IW                       # output frame
YM0, YM1 = 0.20, 2.44                      # module bottom / top
B0, B1 = 2.57, 3.32                        # source-site band
can_lv = math.exp(CAN["2"]["logsize_mean"] + shift) / k0     # prior LV size (class index 2 = LV) at s0, in original pixels

# ================================================================== source site band
hd, tint = MOD["src"]
bg.add_patch(FancyBboxPatch((XA0, B0), FW - 0.04 - XA0, B1 - B0, boxstyle="round,pad=0,rounding_size=0.06", fc=tint, ec=hd, lw=0.7, ls=(0, (4, 2))))
t = label(XA0 + 0.08, B1 - 0.1, "Source site (offline)", ha="left", fs=FS + 0.5, col=hd, ax=bg, fontweight="bold")
label(XA0 + 0.08 + text_width_in(t) + 0.08, B1 - 0.1,
      f"Gaussian statistics of {PR['n_images']} annotated training images; no source image is used at test time",
      ha="left", fs=FS_S, col=INK2, ax=bg)
label(XA0 + 0.08, B0 + 0.2, "calibrated\n$s_0$ = %d px" % S0, ha="left", va="center", fs=FS_S, col=INK, linespacing=1.2)
# size prior: one Gaussian per structure over the log-size at s0
BX0, BX1 = 1.98, 3.72
axp = axes_in(BX0, B0 + 0.15, BX1 - BX0, 0.36)
xx = np.linspace(math.log(28), math.log(330), 400)
for k in sorted(range(9), key=lambda k: CAN[str(k)]["logsize_mean"]):
    m, sd = CAN[str(k)]["logsize_mean"] + shift, CAN[str(k)]["logsize_std"]
    yy = np.exp(-0.5 * (xx - m) ** 2 / sd ** 2)
    axp.fill_between(xx, yy, color=PAL[k], alpha=0.10, lw=0); axp.plot(xx, yy, color=PAL[k], lw=0.9)
axp.set_xlim(xx[0], xx[-1]); axp.set_ylim(0, 1.05); axp.set_yticks([])
axp.set_xticks([math.log(t_) for t_ in (50, 100, 200)]); axp.set_xticklabels(["50", "100", "200 px"], fontsize=FS_S)
axp.tick_params(axis="x", pad=1, length=2); axp.grid(False); axp.patch.set_alpha(0)
for sp in ("left", "top", "right"):
    axp.spines[sp].set_visible(False)
label(BX1 + 0.08, B0 + 0.36, "size prior\n$\\mathcal{N}(\\mu_k,\\sigma_k^2)$", ha="left", va="center", fs=FS_S, col=INK, linespacing=1.25)
# layout prior: mean position of every structure relative to the LV on the source training set
LX0, LX1 = 5.02, 5.84
axl = axes_in(LX0, B0 + 0.03, LX1 - LX0, B1 - B0 - 0.24)
lv = 2; pos = {lv: (0.0, 0.0)}
for k in range(9):
    if k != lv and f"{lv}_{k}" in PR["pairs"]:
        m = PR["pairs"][f"{lv}_{k}"]["mean"]; pos[k] = (m[0] * 720, m[1] * 576)
for a in pos:
    for b in pos:
        if a < b:
            axl.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=INK2, lw=0.35, alpha=0.35, zorder=1)
for k, (px, py) in pos.items():
    axl.plot(px, py, marker="o", ms=3.8, color=PAL[k], mec="white", mew=0.4, zorder=3)
axl.set_aspect("equal"); axl.invert_yaxis(); axl.margins(0.1); axl.axis("off")
label(LX1 + 0.1, B0 + 0.3, "layout prior\n$\\mathcal{N}(\\mathbf{m}_{kl},\\mathbf{\\Sigma}_{kl})$ per pair", ha="left", va="center", fs=FS_S, col=INK, linespacing=1.25)

# ================================================================== input column: test frame -> frozen detector -> pass 1
label(XI, B1 - 0.02, "Test image $x$", ha="left", va="top", fs=FS + 0.5, col=INK, fontweight="bold")
yr = B1 - 0.2 - IW * RATIO
image_panel(XI, yr, IW, CROP)
dx0, dx1, dy0, dy1 = XI + 0.1, XI + IW - 0.1, 1.86, 2.13
top.add_patch(FancyBboxPatch((dx0, dy0), dx1 - dx0, dy1 - dy0, boxstyle="round,pad=0,rounding_size=0.05", fc="#37474F", ec="none", zorder=25))
snowflake(dx0 + 0.12, (dy0 + dy1) / 2, 0.08)
label(dx0 + 0.24, (dy0 + dy1) / 2, "frozen detector $D$", ha="left", fs=FS_S, col="white", zorder=28)
arrow(XI + IW / 2, yr - 0.02, XI + IW / 2, dy1 + 0.01)
yp = dy0 - 0.16 - IW * RATIO
axc, hc = image_panel(XI, yp, IW, CROP, boxes=cands)
ruler(axc, 1.0)
arrow(XI + IW / 2, dy0 - 0.01, XI + IW / 2, yp + hc + 0.02)
label(XI + IW / 2 + 0.05, (dy0 + yp + hc) / 2, "pass 1 at $s_0$", ha="left", fs=FS_S, col=INK2)
label(XI + IW / 2, yp - 0.05, f"{len(cands)} candidates $(b_i,c_i,p_i)$", va="top", fs=FS_S, col=INK)
key = ["LA", "RA", "LV", "RV", "VS", "CRO", "SP", "DAO", "RIB"]
inv = {v: k for k, v in NAMES.items()}
for i, nm in enumerate(key):
    cx_, cy_ = XI + 0.03 + (i % 3) * 0.36, yp - 0.33 - (i // 3) * 0.135
    top.add_patch(Rectangle((cx_, cy_ - 0.035), 0.07, 0.07, fc=PAL[inv[nm]], ec="none"))
    label(cx_ + 0.1, cy_, nm, ha="left", fs=FS_S, col=INK)

# ================================================================== ANI
module(XA0, XA1, YM0, YM1, "ani", "ANI", "nuisance inference")
rows = sorted(set(cls), key=lambda k: -CAN[str(k)]["logsize_mean"]); ypos = {k: i for i, k in enumerate(rows)}
AX0, AXW = XA0 + 0.30, XA1 - XA0 - 0.38
axv = axes_in(AX0, 1.38, AXW, 0.80)
for k in rows:
    m, sd = CAN[str(k)]["logsize_mean"] + shift, CAN[str(k)]["logsize_std"]
    axv.add_patch(Rectangle((m - sd, ypos[k] - 0.34), 2 * sd, 0.68, fc=PAL[k], ec="none", alpha=0.20))
    axv.plot([m, m], [ypos[k] - 0.34, ypos[k] + 0.34], color=PAL[k], lw=0.9)
seen = {}
for k, zi, pi, ri in zip(cls, z, p, r):
    n_k = sum(1 for kk in cls if kk == k); j = seen.get(k, 0); seen[k] = j + 1
    yy = ypos[k] + (0 if n_k == 1 else (-0.2 + 0.4 * j / (n_k - 1)))
    axv.annotate("", xy=(zi + u, yy), xytext=(zi, yy), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.5, mutation_scale=5, alpha=0.75, shrinkA=2, shrinkB=2))
    axv.plot(zi, yy, marker="o", ms=2.6 + 2.6 * pi, mfc="white", mec=PAL[k], mew=0.7, zorder=4)
    axv.plot(zi + u, yy, marker="o", ms=2.6 + 2.6 * pi, mfc=PAL[k], mec=PAL[k], mew=0.5, alpha=0.3 + 0.7 * ri, zorder=5)
axv.set_yticks(range(len(rows))); axv.set_yticklabels([NAMES[k] for k in rows], fontsize=FS_S)
axv.set_xticks([math.log(t_) for t_ in (50, 100, 200)]); axv.set_xticklabels(["50", "100", "200"], fontsize=FS_S)
axv.set_xlim(math.log(40), math.log(330)); axv.set_ylim(len(rows) - 0.5, -0.6)
axv.tick_params(pad=1, length=2); axv.grid(axis="x", alpha=0.25); axv.grid(axis="y", alpha=0); axv.patch.set_alpha(0)
label(AX0 + AXW / 2, 1.13, "log-size votes (px at $s_0$)", fs=FS_S, col=INK2)
arrow(AX0 + AXW / 2, 1.04, AX0 + AXW / 2, 0.93, lw=0.7, ms=5)
label(AX0 + AXW / 2 + 0.06, 0.985, "3 EM steps", ha="left", fs=FS_S, col=INK2)
# Gaussian surrogate posterior over the scale factor
axq = axes_in(AX0, 0.44, AXW, 0.42)
ff = np.exp(np.linspace(math.log(0.22), math.log(1.15), 400)); qq = np.exp(-0.5 * (np.log(ff) - u) ** 2 / s ** 2)
axq.fill_between(ff, qq, color=MOD["ani"][0], alpha=0.18, lw=0); axq.plot(ff, qq, color=MOD["ani"][0], lw=1.1)
axq.plot([f1, f1], [0, 1], color=MOD["ani"][0], lw=0.8, ls="--")
axq.set_xscale("log"); axq.set_xlim(0.22, 1.15); axq.set_ylim(0, 1.12); axq.minorticks_off()
axq.set_xticks([0.25, 0.5, 1.0]); axq.set_xticklabels(["0.25", "0.5", "1"], fontsize=FS_S); axq.set_yticks([])
axq.tick_params(axis="x", pad=1, length=2); axq.grid(False); axq.patch.set_alpha(0)
for sp in ("left", "top", "right"):
    axq.spines[sp].set_visible(False)
label(AX0 + AXW / 2, 0.25, "scale factor $f=e^{u}$", fs=FS_S, col=INK2)
label(XA0 + 0.08, 0.82, "$q(u)$", ha="left", fs=FS_S, col=INK)
label(XA1 - 0.07, 0.74, f"$f^*$ = {f1:.2f}\n$s$ = {s:.2f}", ha="right", va="center", fs=FS_S, col=MOD["ani"][0], linespacing=1.2)

# ================================================================== UVA
module(XU0, XU1, YM0, YM1, "uva", "UVA", "view acquisition")
GAP, FO = 0.05, 0.03                                        # gap between views, offset of the flipped card
wv = [IW * v for v in views]
d = max((wv[0] + wv[1]) / 2 + GAP, (wv[1] + wv[2]) / 2 + GAP + FO)
ext_l, ext_r = -d - wv[0] / 2, d + wv[2] / 2 + FO
c1 = (XU0 + XU1) / 2 - (ext_l + ext_r) / 2
centres = [c1 - d, c1, c1 + d]
YB = 1.20                                                   # common bottom edge of the views
for i, (cx_, wvi, fv) in enumerate(zip(centres, wv, views)):
    centre = i == 1
    edge = MOD["uva"][0]; ls = "-" if centre else (0, (2.5, 1.5))
    image_panel(cx_ - wvi / 2 + FO, YB + FO, wvi, CROP, scale=fv, flip=True, edge=edge, ls=ls, lw=0.6, z=5)
    axv_, hv = image_panel(cx_ - wvi / 2, YB, wvi, CROP, scale=fv, edge=edge, ls=ls, lw=1.0 if centre else 0.8, z=6)
    if centre:
        ruler(axv_, fv)
    lab = "$f^*$" if centre else ("$f^*e^{-h}$" if i == 0 else "$f^*e^{+h}$")
    label(cx_, YB - 0.05, f"×{fv:.2f}", va="top", fs=FS_S, col=INK)
# Gauss-Hermite nodes of q, aligned under the views
D_ = d * 1.32
axn = axes_in(c1 - D_, 0.44, 2 * D_, 0.30)
lo, hi = u - math.sqrt(3) * s * D_ / d, u + math.sqrt(3) * s * D_ / d
fn = np.exp(np.linspace(lo, hi, 300)); qn = np.exp(-0.5 * (np.log(fn) - u) ** 2 / s ** 2)
axn.fill_between(fn, qn, color=MOD["uva"][0], alpha=0.12, lw=0); axn.plot(fn, qn, color=MOD["uva"][0], lw=0.9)
axn.set_xscale("log"); axn.set_xlim(math.exp(lo), math.exp(hi)); axn.set_ylim(0, 1.1); axn.axis("off")
for i, (cx_, fv) in enumerate(zip(centres, views)):
    qv = math.exp(-0.5 * (math.log(fv) - u) ** 2 / s ** 2)
    axn.plot(fv, qv, marker="o", ms=3.2, color=MOD["uva"][0], zorder=5)
    ytop_node = 0.44 + 0.30 * qv / 1.1
    top.plot([cx_, cx_], [ytop_node + 0.04, YB - 0.2], color=MOD["uva"][0], lw=0.6, ls=(0, (1, 1.5)), zorder=29)
    lab = "$f^*$" if i == 1 else ("$f^*e^{-h}$" if i == 0 else "$f^*e^{+h}$")
    label(cx_ + (0.05 if i != 1 else 0.04), YB - 0.29, lab, ha="left", fs=FS_S, col=MOD["uva"][0])
label((XU0 + XU1) / 2, 0.30, "Gauss–Hermite nodes, $h=\\sqrt{3}s$", fs=FS_S, col=INK2)
label(XU1 - 0.07, YB + wv[2] * RATIO + FO + 0.06, "each view + flip;\ndashed: if $s>\\tau_s$", ha="right", va="bottom", fs=FS_S, col=INK2, linespacing=1.15)

# ================================================================== SDF
module(XS0, XS1, YM0, YM1, "sdf", "SDF", "decision fusion")
cxs = (XS0 + XS1) / 2
label(cxs, YM1 - 0.3, "boxes of all views", fs=FS_S, col=INK2)
gcol = PAL[2]
for dxy in ((-0.05, 0.03), (0.0, -0.025), (0.04, 0.02)):
    top.add_patch(Rectangle((cxs - 0.34 + dxy[0], 1.66 + dxy[1]), 0.26, 0.2, fill=False, ec=gcol, lw=0.7, alpha=0.8))
arrow(cxs - 0.03, 1.76, cxs + 0.07, 1.76, lw=0.6, ms=5)
top.add_patch(Rectangle((cxs + 0.09, 1.66), 0.26, 0.2, fill=False, ec=gcol, lw=1.4))
label(cxs, 1.55, "WBF (IoU 0.55)", va="top", fs=FS_S, col=INK)
arrow(cxs, 1.36, cxs, 1.20, lw=0.6, ms=5)
top.add_patch(Rectangle((cxs - 0.33, 0.86), 0.25, 0.2, fill=False, ec=PAL[3], lw=1.4))
top.add_patch(Rectangle((cxs + 0.08, 0.86), 0.25, 0.2, fill=False, ec=PAL[3], lw=0.8, ls=(0, (2, 1.5)), alpha=0.6))
label(cxs + 0.205, 0.96, "$\\times\\gamma$", fs=FS_S, col=INK2)
label(cxs, 0.76, "ICM under the\nlayout prior", va="top", fs=FS_S, col=INK, linespacing=1.15)

# ================================================================== output
ycand = yp + hc / 2
yo = ycand - IW * RATIO / 2
label(XO0, yo + IW * RATIO + 0.05, "Output on $x$", ha="left", va="bottom", fs=FS + 0.5, col=INK, fontweight="bold")
image_panel(XO0, yo, IW, CROP, boxes=final)
label(XO0 + IW / 2, yo - 0.05, f"{len(final)} structures, boxes\nmapped back to $x$", va="top", fs=FS_S, col=INK, linespacing=1.15)

# ================================================================== flow arrows
arrow(XI + IW + 0.02, ycand + 0.22, AX0 - 0.26, ycand + 0.22)                    # candidates -> ANI
arrow(XA1 - 0.02, 0.60, centres[0] - D_ * 0.25, 0.60)                          # f*, s -> UVA
label(XU0 + 0.08, 0.71, "$f^*\\!,\\,s$", ha="left", fs=FS_S, col=INK2)
yv = YB + wv[2] * RATIO / 2
arrow(centres[2] + wv[2] / 2 + FO + 0.03, yv, XS0 + 0.08, 1.76)                 # views -> WBF
arrow(XS1 - 0.02, ycand, XO0 - 0.03, ycand)                                     # SDF -> output
# the pass-1 boxes are fused too: routed under the modules
yl = 0.06
top.plot([XI + IW + 0.02, XA0 - 0.06, XA0 - 0.06, cxs], [yp + 0.12, yp + 0.12, yl, yl], color=ARW, lw=0.8, zorder=29)
arrow(cxs, yl, cxs, YM0 + 0.01)
label((XA0 + XU1) / 2, yl + 0.015, "pass-1 boxes are fused as well", va="bottom", fs=FS_S, col=INK2)
# optional second round: canonical view back to ANI
arrow(centres[1] - wv[1] * 0.1, YB + wv[1] * RATIO + FO + 0.02, XA1 - 0.06, 2.0, ls=(0, (3, 2)), cs="arc3,rad=0.3", col=MOD["uva"][0])
label(XU0 + 0.1, 2.08, f"optional 2nd round: $f$ = {f2:.2f}", ha="left", fs=FS_S, col=MOD["uva"][0])
# source statistics feed the detector scale, ANI and SDF
arrow(2.30, B0, 2.30, YM1 + 0.01, ls=(0, (3, 2)), col=MOD["src"][0])
arrow((LX0 + LX1) / 2, B0, (LX0 + LX1) / 2, YM1 + 0.01, ls=(0, (3, 2)), col=MOD["src"][0])
arrow(XA0 + 0.08, B0 + 0.05, dx1 + 0.02, (dy0 + dy1) / 2, ls=(0, (3, 2)), col=MOD["src"][0], cs="arc3,rad=-0.25")

save(fig, "fig2_overview")
