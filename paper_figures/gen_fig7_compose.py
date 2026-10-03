"""Fig. 7 - composition: parameter-adaptive TTA methods run on the TAC-canonicalized input (Heart-4CC, Faster R-CNN, test,
3 seeds). Dumbbells: original -> + TAC input; reference lines: source, control (source model on the canonicalized input,
no update), TAC-3 (2 rounds) and TAC-7. Data: qc_collect['compose'] and tables."""
import statistics as st
import matplotlib.pyplot as plt
from pubstyle import C, OURS, BASE, save
from tabdata import QC, rows

CP = QC["compose"]["AP"]
def m(name):
    v = CP[name]["per_seed_mean6"]; return st.mean(v.values()), st.pstdev(v.values())
pairs = [("IoU-Filter", "IoU-Filter + TAC", "IoU-Filter (1 view, 5 updates)"), ("AMROD", "AMROD + TAC", "AMROD (1 view, online)"), ("CoTTA-det", "CoTTA-det + TAC", "CoTTA-det (18 views, online)")]
fig, ax = plt.subplots(figsize=(3.35, 2.6))
for i, (a, b, lab) in enumerate(pairs):
    (ma, sa), (mb, sb) = m(a), m(b)
    ax.plot([ma, mb], [i, i], color=C["ink2"], lw=1.2, zorder=2)
    ax.errorbar(ma, i, xerr=sa, color=BASE, marker="o", ms=6, capsize=2, elinewidth=0.6, ls="none", zorder=3, mec=C["ink2"], mew=0.5)
    ax.errorbar(mb, i, xerr=sb, color=OURS, marker="o", ms=6, capsize=2, elinewidth=0.6, ls="none", zorder=4)
    d = CP[b]["delta_vs_base"]
    ax.text((ma + mb) / 2, i + 0.2, f"+{d[0]:.2f} ± {d[1]:.2f}{'*' if d[2] else ''}", ha="center", va="bottom", fontsize=6.3, color=C["ink"], zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))
ax.set_yticks(range(3)); ax.set_yticklabels([p[2] for p in pairs], fontsize=6.8)
refs = [("Source-only (1 view @800)", "source-only", C["gray"], "--"), ("Source + TAC input (1 view, 0 updates)", "control: source model on canonical input, no update", C["cyan"], "--"),
        ("TAC 2 rounds (ours, 3.3 views)", "TAC-3, 2 rounds (ours)", OURS, "--")]
for name, lab, col, ls in refs:
    mu, _ = m(name); ax.axvline(mu, color=col, lw=0.9, ls=ls, zorder=1, label=lab)
t7 = rows("heart4cc_frcnn")["TAC 7 views"]["mean"]
ax.axvline(t7, color=OURS, lw=0.9, ls=":", zorder=1, label="TAC-7 (ours)")
ax.set_xlim(45.5, 57.5); ax.set_ylim(-0.6, 2.6); ax.set_xlabel("mean AP over 6 directions (%)")
ax.grid(axis="y", alpha=0)
ax.plot([], [], marker="o", color=BASE, mec=C["ink2"], ls="none", label="original input"); ax.plot([], [], marker="o", color=OURS, ls="none", label="TAC-canonicalized input")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=6, handletextpad=0.4, ncol=2, handlelength=1.6, columnspacing=1.0)
save(fig, "fig7_compose")
