"""Shared publication style for all paper figures (Information Fusion, elsarticle two-column: 3.35 in single / 6.9 in full).
Palette: one deep red reserved for our method, a blue family for the multi-view references, ochre and grays."""
import json
import os

import matplotlib
import matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.environ.get("TAC_ROOT", "."), "results")
AUX = os.path.join(RES, "paper_aux")
OUT = os.path.dirname(os.path.abspath(__file__))

# Physical sizes of the journal layout (elsarticle 5p): text width 522 pt, column width 252 pt. Figures are drawn at exactly
# these widths and saved without cropping, so the type sizes below are the sizes in print.
W_FULL = 522.0 / 72.0
W_COL = 252.0 / 72.0

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "normal", "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "legend.handlelength": 1.2, "legend.handletextpad": 0.5, "legend.labelspacing": 0.3, "legend.borderaxespad": 0.3,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222", "text.color": "#222222",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5, "axes.labelpad": 3,
    "axes.grid": False, "axes.grid.axis": "y", "grid.color": "#E6E6E6", "grid.alpha": 1.0, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "patch.edgecolor": "white", "patch.linewidth": 0.4, "hatch.linewidth": 0.6,
    "axes.axisbelow": True,
})

# keys keep their historical names; since 2026-11-12: "coral" = deep red of the method, "orange" = its light tint,
# "teal" / "blue" / "cyan" / "sky" = one blue family from navy to light, "gold" = ochre
C = {"teal": "#1F3A5F", "cyan": "#5B8DB8", "gold": "#C28F2C", "orange": "#DB8A80", "coral": "#B4282E",
     "blue": "#2E5E8C", "sky": "#7FA8CC", "gray": "#8C8C8C", "lgray": "#B5B5B5", "ink": "#222222", "ink2": "#555555"}
OURS = C["coral"]
OURS_TINT = "#E9B7B2"   # light tint of the method colour, for bars that carry overlaid points
BASE = C["lgray"]
BEST = C["teal"]
DOM = {"c1": C["blue"], "c2": C["cyan"], "c3": C["orange"], "ge": C["blue"], "ph": C["cyan"], "sa": C["orange"]}
MOD = {"ANI": "#B07D1E", "UVA": "#2E5E8C", "SDF": "#3F7F7A"}  # module colours of the overview figure (fig2_v7/kit.py REG borders)
FIG_SINGLE = (3.35, 2.4)
FIG_FULL = (6.9, 2.5)
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def load_aux(name):
    with open(os.path.join(AUX, name), encoding="utf-8") as f:
        return json.load(f)


def panel_label(ax, s, title=None, **_):
    """Bold panel letter at the top-left corner of the axes, with an optional descriptive title beside it; both are placed
    at fixed offsets in points, so that every panel of every figure carries them at the same position."""
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points", fontsize=9, fontweight="bold",
                ha="left", va="bottom", annotation_clip=False)
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(15, 6.6), textcoords="offset points", fontsize=8,
                    ha="left", va="bottom", annotation_clip=False)


def layout(fig, rect=(0, 0, 1, 1), w_pad=1.4, h_pad=1.2):
    """Fit the axes into the fixed figure size (optionally inside rect, which leaves room for a figure legend)."""
    fig.tight_layout(pad=0.3, w_pad=w_pad, h_pad=h_pad, rect=rect)


def save(fig, name, tight=True):
    """tight=False keeps the figure at its designed physical size (the data figures); tight=True crops to the content."""
    kw = {} if tight else {"bbox_inches": matplotlib.transforms.Bbox.from_bounds(0, 0, *fig.get_size_inches())}
    fig.savefig(os.path.join(OUT, name + ".pdf"), **kw)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, **kw)
    plt.close(fig)
    print("saved", name)
