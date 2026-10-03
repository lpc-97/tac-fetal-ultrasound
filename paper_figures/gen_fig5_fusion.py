"""Fig. 5 - fusion analysis on Heart-4CC / Faster R-CNN: (a) fusion rule with the same three views (test, 3 seeds);
(b) accuracy vs number of fused views (test, 3 seeds) against the 18-view ensembles; (c) hyper-parameter sensitivity
(val, seed 42, 6-direction mean AP; dashed = default). Data: qc_collect via tabdata."""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import C, OURS, BASE, RES, panel_label, save
from tabdata import rows, val_variants

R = rows("heart4cc_fusion_views")
fig = plt.figure(figsize=(6.9, 2.35))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.85], wspace=0.35)

# (a) fusion rule
ax = fig.add_subplot(gs[0])
items = [("TAC canonical view only (no fusion)", "no\nfusion"), ("TAC fusion=heatmap (3 views)", "heat-\nmap"), ("TAC fusion=NMS (3 views)", "NMS"), ("TAC (WBF, 3 views)", "WBF\n(ours)"), ("18-view heatmap (senior)", "18-view\nheat-\nmap")]
vals = [R[k]["mean"] for k, _ in items]; err = [R[k]["std"] for k, _ in items]
cols = [BASE, BASE, BASE, OURS, C["gold"]]
b = ax.bar(range(5), vals, 0.62, color=cols, yerr=err, error_kw=dict(lw=0.6, capsize=2, ecolor=C["ink2"]))
for i, v in enumerate(vals):
    ax.text(i, v + err[i] + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3, color=C["ink"])
ax.set_xticks(range(5)); ax.set_xticklabels([l for _, l in items], fontsize=5.7)
ax.set_ylim(50.5, 54.6); ax.set_ylabel("mean AP (%)")
ax.axvline(3.5, color=C["ink2"], lw=0.5, ls=":")
ax.set_xlabel("fusion rule (same 3 views)   |   reference", fontsize=6.8)
panel_label(ax, "(a)", x=-0.2)

# (b) number of views
ax = fig.add_subplot(gs[1])
pts = [("TAC 2 views (no flip)", 2), ("TAC (WBF, 3 views)", 3), ("TAC 7 views (+-15% scales)", 7), ("TAC 11 views (+-10/20/25% scales)", 11)]
xs = [v for _, v in pts]; ys = [R[k]["mean"] for k, _ in pts]; es = [R[k]["std"] for k, _ in pts]
ax.errorbar(xs, ys, yerr=es, color=OURS, marker="o", ms=4.5, lw=1.2, capsize=2, elinewidth=0.6, label="TAC (ours)", zorder=4)
for k, lab, col, mk in [("18-view WBF + AR", "18-view WBF + re-scoring", C["teal"], "H"), ("18-view WBF", "18-view WBF", C["teal"], "p"), ("18-view heatmap (senior)", "18-view heat-map", C["gold"], "h")]:
    ax.errorbar(18, R[k]["mean"], yerr=R[k]["std"], color=col, marker=mk, ms=5, ls="none", capsize=2, elinewidth=0.6, label=lab, mfc=col if "AR" in k or "heat" in k else "white")
# blind fixed multi-scale ensembles with the same number of views (reviewer experiment C; results/reviewer/reviewer_summary.json)
import json as _json, os as _os
_RV = _json.load(open(_os.path.join(RES, "reviewer", "reviewer_summary.json"), encoding="utf-8"))["rows"]
bx = [3, 6, 7]; bk = ["ss_fix3_wbf_ar", "ss_fix3f_wbf_ar", "ss_fix7_wbf_ar"]
ax.errorbar(bx, [_RV[k]["mean"] for k in bk], yerr=[_RV[k]["std"] for k in bk], color=C["gray"], marker="s", ms=4, lw=1.0, ls="--", capsize=2, elinewidth=0.6, label="blind fixed scales + re-scoring", zorder=3)
ax.set_xscale("log", basex=2); ax.set_xticks([2, 3, 7, 11, 18]); ax.set_xticklabels(["2", "3", "7", "11", "18"]); ax.minorticks_off()
ax.set_xlabel("fused views per image"); ax.set_ylim(49.5, 56); ax.set_ylabel("mean AP (%)")
ax.legend(loc="upper center", fontsize=5.8, handlelength=1.2, ncol=2, columnspacing=0.8, handletextpad=0.4, bbox_to_anchor=(0.5, -0.3))
panel_label(ax, "(b)", x=-0.2)

# (c) hyper-parameter sensitivity (val seed 42)
V = val_variants("4c", "canon"); default = V["tac"]
HP = [("$\\sigma_u$", [("0.3", "hp_su0.3"), ("0.5", "hp_su0.5"), ("1", "tac"), ("2", "hp_su2"), ("100", "hp_su100")], "1"),
      ("WBF IoU", [("0.45", "hp_wbf0.45"), ("0.55", "tac"), ("0.65", "hp_wbf0.65"), ("0.75", "hp_wbf0.75")], "0.55"),
      ("$\\gamma$", [("0", "hp_gam0"), ("0.3", "tac"), ("0.6", "hp_gam0.6"), ("1", "hp_gam1")], "0.3"),
      ("$\\lambda$", [("0.25", "hp_lam0.25"), ("0.5", "tac"), ("1", "hp_lam1"), ("2", "hp_lam2")], "0.5"),
      ("IRLS it.", [("1", "hp_irls1"), ("3", "tac"), ("6", "hp_irls6")], "3"),
      ("$\\rho_{\\mathrm{out}}$", [("0", "hp_od0"), ("0.05", "hp_od0.05"), ("0.1", "tac"), ("0.5", "hp_od0.5"), ("1", "hp_od1")], "0.1"),
      ("$\\tau_{\\min}$", [("0.1", "hp_ms0.1"), ("0.3", "tac"), ("0.5", "hp_ms0.5"), ("0.7", "hp_ms0.7")], "0.3"),
      ("$\\sigma_o$", [("0.05", "hp_so0.05"), ("0.1", "tac"), ("0.2", "hp_so0.2"), ("0.4", "hp_so0.4")], "0.1")]
sub = gs[2].subgridspec(2, 4, wspace=0.45, hspace=0.75)
for i, (name, opts, dflt) in enumerate(HP):
    ax = fig.add_subplot(sub[i // 4, i % 4])
    ys = [V[k] for _, k in opts]; xs = range(len(opts))
    ax.plot(xs, ys, color=OURS, marker="o", ms=3, lw=1)
    ax.axhline(default, color=C["ink2"], lw=0.6, ls="--")
    ax.set_xticks(list(xs)); ax.set_xticklabels([l for l, _ in opts], fontsize=5.4, rotation=45)
    ax.set_ylim(53.3, 54.6); ax.set_yticks([53.5, 54.0, 54.5]); ax.tick_params(axis="y", labelsize=5.8)
    ax.set_title(name, fontsize=7, pad=2, fontweight="normal")
    if i % 4:
        ax.set_yticklabels([])
    if i == 0:
        panel_label(ax, "(c)", x=-0.75, y=1.5)
fig.text(0.735, -0.06, "hyper-parameter value (val, seed 42; dashed = default, 54.14 AP)", ha="center", fontsize=6.5, color=C["ink2"])
save(fig, "fig5_fusion")
