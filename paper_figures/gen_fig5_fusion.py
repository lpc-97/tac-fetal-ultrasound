"""Fig. 5 - hyper-parameter sensitivity of TAC on Heart-4CC / Faster R-CNN (val, seed 42, 6-direction mean AP): one small
panel per hyper-parameter; the black-edged marker is the default configuration and the dashed line its AP. The fusion-rule
and view-count comparisons that used to form panels (a) and (b) are tabulated in the paper (tab_views), so the figure carries
only the sensitivity grid at single-column width. Data: qc_collect via tabdata (val_variants)."""
import matplotlib.pyplot as plt
from pubstyle import C, OURS, FIG_SINGLE, save
from tabdata import val_variants

V = val_variants("4c", "canon"); default = V["tac"]
HP = [("$\\sigma_u$", [("0.3", "hp_su0.3"), ("0.5", "hp_su0.5"), ("1", "tac"), ("2", "hp_su2"), ("100", "hp_su100")], "1"),
      ("WBF IoU", [("0.45", "hp_wbf0.45"), ("0.55", "tac"), ("0.65", "hp_wbf0.65"), ("0.75", "hp_wbf0.75")], "0.55"),
      ("$\\gamma$", [("0", "hp_gam0"), ("0.3", "tac"), ("0.6", "hp_gam0.6"), ("1", "hp_gam1")], "0.3"),
      ("$\\lambda$", [("0.25", "hp_lam0.25"), ("0.5", "tac"), ("1", "hp_lam1"), ("2", "hp_lam2")], "0.5"),
      ("IRLS it.", [("1", "hp_irls1"), ("3", "tac"), ("6", "hp_irls6")], "3"),
      ("$\\rho_{\\mathrm{out}}$", [("0", "hp_od0"), ("0.05", "hp_od0.05"), ("0.2", "tac"), ("0.5", "hp_od0.5"), ("1", "hp_od1")], "0.2"),
      ("$\\tau_{\\min}$", [("0.1", "hp_ms0.1"), ("0.3", "tac"), ("0.5", "hp_ms0.5"), ("0.7", "hp_ms0.7")], "0.3"),
      ("$\\sigma_o$", [("0.05", "hp_so0.05"), ("0.1", "tac"), ("0.2", "hp_so0.2"), ("0.4", "hp_so0.4")], "0.1")]
fig, axes = plt.subplots(2, 4, figsize=FIG_SINGLE, gridspec_kw={"wspace": 0.3, "hspace": 0.8})
for i, (name, opts, dflt) in enumerate(HP):
    ax = axes[i // 4, i % 4]
    ys = [V[k] for _, k in opts]; xs = range(len(opts))
    ax.plot(xs, ys, color=OURS, marker="o", ms=3, lw=1)
    j = [l for l, _ in opts].index(dflt)  # the default configuration: black-edged marker
    ax.plot(j, ys[j], marker="o", ms=4.5, color=OURS, markeredgecolor="black", markeredgewidth=0.6, zorder=5)
    ax.axhline(default, color=C["ink2"], lw=0.6, ls="--")
    # only the two ends are labeled at a legible size; every value is listed in the configuration table
    ax.set_xticks(list(xs)); ax.set_xticklabels([l if k in (0, len(opts) - 1) else "" for k, (l, _) in enumerate(opts)], fontsize=7)
    ax.set_ylim(53.3, 54.6); ax.set_yticks([53.5, 54.0, 54.5]); ax.tick_params(axis="y", labelsize=7)
    ax.set_title(name, fontsize=7.5, pad=2, fontweight="normal")
    if i % 4:
        ax.set_yticklabels([])
axes[0, 0].set_ylabel("mean AP (%)", fontsize=7); axes[1, 0].set_ylabel("mean AP (%)", fontsize=7)
fig.text(0.5, -0.04, f"hyper-parameter value (val, seed 42; dashed = default, {default:.2f} AP)", ha="center", fontsize=7, color=C["ink2"])
save(fig, "fig5_fusion")
