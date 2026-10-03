"""Shared publication style for all paper figures (Information Fusion, elsarticle two-column: 3.35 in single / 6.9 in full).
Palette = "Ocean Dusk" (academic-plotting skill) with coral reserved for our method; Okabe-Ito blue/sky for extra series."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.environ.get("TAC_ROOT", "."), "results")
AUX = os.path.join(RES, "paper_aux")
OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "patch.edgecolor": "white", "patch.linewidth": 0.5,
    "axes.axisbelow": True,
})

C = {"teal": "#264653", "cyan": "#2A9D8F", "gold": "#E9C46A", "orange": "#F4A261", "coral": "#E76F51",
     "blue": "#0072B2", "sky": "#56B4E9", "gray": "#8C8C8C", "lgray": "#B0BEC5", "ink": "#222222", "ink2": "#555555"}
OURS = C["coral"]
BASE = C["lgray"]
BEST = C["teal"]
DOM = {"c1": C["blue"], "c2": C["cyan"], "c3": C["orange"], "ge": C["blue"], "ph": C["cyan"], "sa": C["orange"]}
FIG_SINGLE = (3.35, 2.4)
FIG_FULL = (6.9, 2.5)
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def load_aux(name):
    with open(os.path.join(AUX, name), encoding="utf-8") as f:
        return json.load(f)


def panel_label(ax, s, x=-0.08, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)
    print("saved", name)
