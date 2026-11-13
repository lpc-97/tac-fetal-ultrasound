"""Fig. 12 - uncertainty-guided view acquisition (UVA) on Heart-4CC with Faster R-CNN (Section 5.4):
(a) test AP against mean forward passes per image: TAC-3, the fixed +-15% bracket (TAC-7, TAC-11), the quadrature bracket
    (TAC-7, ours) and the two gated configurations (tau_s = 0.08, 0.12); error bars = std over three seeds;
(b) validation sweep of the gating threshold tau_s: mean val AP (left axis) and mean views per image (right axis), with the
    TAC-3 and fixed-bracket TAC-7 validation levels as reference lines;
(c) gain of the four extra views of the fixed bracket over TAC-3 per quartile of the surrogate spread s, with the 18
    direction-seed cells overlaid and the mean number of first-pass candidates annotated.
Data: results/reviewer/uncert_summary.json (test, val), results/reviewer/reviewer_summary.json (cross-check only),
      results/reviewer/uva_quartiles.json. Every plotted number is read from these files."""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from pubstyle import C, OURS, OURS_TINT, RES, W_FULL, panel_label, layout, save

REV = os.path.join(RES, "reviewer")
U = json.load(open(os.path.join(REV, "uncert_summary.json"), encoding="utf-8"))
R = json.load(open(os.path.join(REV, "reviewer_summary.json"), encoding="utf-8"))["rows"]
Q = json.load(open(os.path.join(REV, "uva_quartiles.json"), encoding="utf-8"))
T, V = U["test"], U["val"]

# the three fixed-bracket test rows must agree between the two summary files
for k in ("tac", "tac_v7", "tac_v11"):
    assert abs(T[k]["mean"] - R[k]["mean"]) < 1e-9 and abs(T[k]["std"] - R[k]["std"]) < 1e-9, k

LAB = {"tac": "TAC-3", "tac_v7": "TAC-7, fixed bracket", "tac_v11": "TAC-11, fixed bracket",
       "u7_k17": "TAC-7, quadrature (ours)", "ua_t08": "UVA, gated ($\\tau_s$=0.08)", "ua_t12": "UVA, gated ($\\tau_s$=0.12)"}
GRAY = C["gray"]

fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.6), gridspec_kw={"width_ratios": [1.15, 1.0, 0.95]})
audit = {}

# ---------------------------------------------------------------- (a) test: AP vs mean forward passes
ax = axes[0]
curve = [(T[k]["views"], T[k]["mean"]) for k in ("tac", "ua_t12", "ua_t08", "u7_k17")]
ax.plot([p[0] for p in curve], [p[1] for p in curve], color=OURS, lw=1.0, alpha=0.7, zorder=2)
style = {"tac": (C["ink2"], "o"), "tac_v7": (GRAY, "s"), "tac_v11": (GRAY, "s"),
         "u7_k17": (OURS, "*"), "ua_t08": (OURS, "D"), "ua_t12": (OURS, "D")}
# label offsets in points relative to the marker: (dx, dy, ha, va)
ann = {"tac": (7, 0, "left", "center"), "ua_t12": (7, -3, "left", "center"),
       "ua_t08": (7, -11, "left", "center"), "u7_k17": (-7, 5, "right", "bottom"),
       "tac_v7": (8, -2, "left", "center"), "tac_v11": (0, 10, "center", "bottom")}
SHORT = {"tac": "TAC-3", "ua_t12": "gated, $\\tau_s$ = 0.12", "ua_t08": "gated, $\\tau_s$ = 0.08",
         "u7_k17": "TAC-7 (ours)", "tac_v7": "TAC-7, fixed", "tac_v11": "TAC-11, fixed"}
for k in ("tac", "ua_t12", "ua_t08", "u7_k17", "tac_v7", "tac_v11"):
    col, mk = style[k]
    r = T[k]
    ax.errorbar(r["views"], r["mean"], yerr=r["std"], color=col, marker=mk, ms=7.5 if mk == "*" else 4.2, ls="none",
                capsize=1.6, elinewidth=0.7, zorder=4)
    dx, dy, ha, va = ann[k]
    ax.annotate(SHORT[k], xy=(r["views"], r["mean"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7,
                color=OURS if k == "u7_k17" else (C["ink"] if col != GRAY else C["ink2"]),
                fontweight="bold" if k == "u7_k17" else "normal", linespacing=1.0)
    audit[("a", k)] = (r["views"], r["mean"], r["std"])
ax.set_xticks([3, 4, 5, 6, 7, 11])
ax.set_xlim(2.4, 13.2)
ax.set_xlabel("mean forward passes per image")
ax.set_ylabel("mean test AP (%)")
ys = [T[k]["mean"] for k in style]
ax.set_ylim(min(ys) - 0.3, max(ys) + 0.5)
panel_label(ax, "(a)", "Accuracy against cost (test)")

# ---------------------------------------------------------------- (b) validation sweep of tau_s
ax = axes[1]
sweep = sorted((float(k[len("ua_t"):len("ua_t") + 2]) / 100.0, k) for k in V
               if k.startswith("ua_t") and not k.endswith("_w"))
taus = [t for t, _ in sweep]
ap = [V[k]["AP"] for _, k in sweep]
views = [V[k]["views"] for _, k in sweep]
audit["b_sweep"] = list(zip([k for _, k in sweep], taus, ap, views))
audit["b_ref"] = {"tac": V["tac"]["AP"], "tac_v7": V["tac_v7"]["AP"]}
ax.axhline(V["tac_v7"]["AP"], color=GRAY, lw=0.9, ls="--", label="TAC-7, fixed bracket")
ax.axhline(V["tac"]["AP"], color=C["ink2"], lw=0.9, ls=":", label="TAC-3")
ax.plot(taus, ap, color=OURS, marker="o", ms=4, lw=1.2, label="gated", zorder=4)
ax.plot([t for t in taus if round(t, 2) in (0.08, 0.12)], [a_ for t, a_ in zip(taus, ap) if round(t, 2) in (0.08, 0.12)], ls="none", marker="o", ms=5, color=OURS, mec=C["ink"], mew=0.8, zorder=5)  # the two thresholds reported on the test split
for i, (t, a_, v) in enumerate(zip(taus, ap, views)):
    below = i == 1  # this label would sit on the reference line of the fixed bracket
    ax.annotate(f"{v:.2f}", xy=(t, a_), xytext=(-4, -4) if below else (4, 4), textcoords="offset points", ha="right" if below else "left",
                va="top" if below else "bottom", fontsize=7, color=C["ink2"])
ax.set_xlabel("gating threshold $\\tau_s$")
ax.set_ylabel("mean validation AP (%)")
ax.set_xticks(taus)
ax.set_xticklabels([f"{t:.2f}" for t in taus])
ax.set_xlim(taus[0] - 0.012, taus[-1] + 0.022)
ax.set_ylim(min(ap + [V["tac"]["AP"]]) - 0.2, max(ap + [V["tac_v7"]["AP"]]) + 0.65)
h, l = ax.get_legend_handles_labels()
ax.legend([h[2], h[0], h[1]], [l[2], l[0], l[1]], loc="upper right", handlelength=1.8)
panel_label(ax, "(b)", "Gating threshold (validation)")

# ---------------------------------------------------------------- (c) gain of the fixed bracket per quartile of s
ax = axes[2]
qnames = list(Q["bins"].keys())
x = np.arange(len(qnames))
gain = [Q["bins"][q]["delta_tac7_tac3"] for q in qnames]
ndet = [Q["bins"][q]["mean_ndet"] for q in qnames]
mean_s = [Q["bins"][q]["mean_s"] for q in qnames]
cells = [list(Q["bins"][q]["delta_per_cell"].values()) for q in qnames]
audit["c"] = {q: (Q["bins"][q]["delta_tac7_tac3"], Q["bins"][q]["mean_ndet"], Q["bins"][q]["mean_s"],
                  Q["bins"][q]["n_images"], len(Q["bins"][q]["delta_per_cell"])) for q in qnames}
ax.bar(x, gain, 0.6, color=OURS_TINT, lw=0, zorder=2, label="mean over 18 cells")
ax.hlines(gain, x - 0.3, x + 0.3, color=OURS, lw=1.4, zorder=3)
rng = np.random.default_rng(0)
for i, cv in enumerate(cells):
    jit = rng.uniform(-0.17, 0.17, size=len(cv))
    ax.plot(x[i] + jit, cv, ls="none", marker="o", ms=2.6, mfc="white", mec=C["ink2"], mew=0.5, zorder=4,
            label="direction–seed cell" if i == 0 else None)
ymax = max(max(c) for c in cells)
ax.set_xticks(x)
ax.set_xticklabels([f"Q{i + 1}\n{s:.3f}\n{n:.1f}" for i, (s, n) in enumerate(zip(mean_s, ndet))])
ax.tick_params(axis="x", length=0)
ax.set_xlabel("quartile of $s$ (narrow → wide)\nmean $s$ / first-pass candidates per image")
ax.set_ylabel("gain of TAC-7 (fixed)\nover TAC-3 (AP)")
ax.set_xlim(-0.6, len(qnames) - 0.4)
ax.set_ylim(0, ymax + 1.15)
ax.legend(loc="upper left", handlelength=1.0)
panel_label(ax, "(c)", "Gain of the bracket by spread")

layout(fig)
save(fig, "fig12_uva", tight=False)

# audit print-out: every number plotted, straight from the files
print("(a) test rows (views, mean AP, std):")
for key, v in audit.items():
    if isinstance(key, tuple) and key[0] == "a":
        print(f"   {key[1]:8s} views={v[0]:.4f} AP={v[1]:.4f} std={v[2]:.4f}")
print("(b) val sweep (key, tau_s, AP, views):")
for row in audit["b_sweep"]:
    print(f"   {row[0]:8s} tau={row[1]:.2f} AP={row[2]:.4f} views={row[3]:.4f}")
print(f"   reference: tac AP={audit['b_ref']['tac']:.4f}  tac_v7 AP={audit['b_ref']['tac_v7']:.4f}")
print("(c) quartiles (gain, mean_ndet, mean_s, n_images, n_cells):")
for q, v in audit["c"].items():
    print(f"   {q:15s} gain={v[0]:.4f} ndet={v[1]:.3f} s={v[2]:.4f} n_img={v[3]} cells={v[4]}")
