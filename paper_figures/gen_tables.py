# -*- coding: utf-8 -*-
"""Generate every LaTeX table body of the manuscript from results/qc_collect_2026-09-18.json (+ dataset stats) so that no
number is typed by hand. Output: $TAC_ROOT/paper/tables/*.tex and $TAC_ROOT/paper/tables/numbers.json (all quantities quoted in
the text, for the provenance audit)."""
import json
import math
import os
import statistics as st

from pubstyle import load_aux
from tabdata import QC, rows, paired, val_variants

ROOT = os.environ.get("TAC_ROOT", ".")
OUTD = os.path.join(ROOT, "paper", "tables"); os.makedirs(OUTD, exist_ok=True)
NUM = {}


def fmt(m, s=None, bold=False, prec=2):
    t = f"{m:.{prec}f}" + (f"\\,$\\pm$\\,{s:.{prec}f}" if s is not None else "")
    return f"\\textbf{{{t}}}" if bold else t


def delta(a, b):
    mu, half, sig = paired(a, b)
    return f"{mu:+.2f}\\,$\\pm$\\,{half:.2f}{'$^{*}$' if sig else ''}"


def write(name, text):
    with open(os.path.join(OUTD, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote", name)


# ------------------------------------------------------------------ Table: datasets
S = load_aux("dataset_size_stats.json")
VAL = {"4c": {"c1": 81, "c2": 116, "c3": 89}, "abdomen": {"GE": 127, "PH": 112, "SA": 166}, "spine": {"GE": 126, "PH": 151, "SA": 182}}
L = []
for dsn, title, doms in [("4c", "Heart-4CC (4-chamber view, 9 structures)", ["c1", "c2", "c3"]), ("abdomen", "Abdomen (transverse plane, 8 structures)", ["GE", "PH", "SA"]), ("spine", "Spine (sagittal plane, 6 structures)", ["GE", "PH", "SA"])]:
    L.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}}\\\\")
    for d in doms:
        tr, te = S[dsn][f"{d}_train"], S[dsn][f"{d}_test"]
        res = ", ".join(f"{w}$\\times${h}" for (w, h), n in tr["res_top"][:2])
        lab = {"c1": "center 1", "c2": "center 2", "c3": "center 3", "GE": "GE", "PH": "Philips", "SA": "Samsung"}[d]
        L.append(f"{lab} & {tr['n_img']} & {VAL[dsn][d]} & {te['n_img']} & {tr['n_ann'] / tr['n_img']:.1f} & {res} & {tr['all_median']:.0f} \\\\")
write("tab_datasets.tex", "\n".join(L) + "\n")
NUM["dataset_median_size"] = {f"{k}_{d}": S[k][f"{d}_train"]["all_median"] for k, ds in [("4c", ["c1", "c2", "c3"]), ("abdomen", ["GE", "PH", "SA"]), ("spine", ["GE", "PH", "SA"])] for d in ds}

# ------------------------------------------------------------------ Table: main Heart-4CC Faster R-CNN
R = rows("heart4cc_frcnn"); B = rows("heart4cc_bn_models"); F = rows("heart4cc_fusion_views")
src = R["Source-only (FrozenBN)"]
MAIN = [  # (row name in QC, label, views/updates, group)
    ("Source-only (FrozenBN)", "Source model (no adaptation)", "1", "src"),
    ("IoU-Filter", "IoU-Filter~\\cite{ruan2024ioufilter}", "1 + 5 upd.", "single"),
    ("AMROD", "AMROD~\\cite{cao2026amrod}", "1 + train", "single"),
    ("WHW", "WHW~\\cite{yoo2024whw}", "1 + train", "single"),
    ("CD-Buffer", "CD-Buffer~\\cite{song2026cdbuffer}", "1 + train", "single"),
    ("SGP", "SGP~\\cite{wang2025sgp}", "1 + train", "single"),
    ("O-SFDA", "O-SFDA data acq.~\\cite{shi2024osfda}", "1 + train", "single"),
    ("VLOD-TTA objective", "VLOD-TTA objective~\\cite{belal2026vlod}", "1 + 1 step", "single"),
    ("BufferTTA", "BufferTTA~\\cite{kim2025buffer}", "1 + train", "single"),
    ("Senior heatmap fusion (18 views)", "18-view heat-map fusion", "18", "multi"),
    ("18-view WBF", "18-view WBF~\\cite{solovyev2021wbf}", "18", "multi"),
    ("18-view WBF + AR", "18-view WBF + re-scoring", "18", "multi"),
    ("CoTTA-det", "CoTTA-det~\\cite{wang2022cotta}", "18 + train", "multi"),
    ("TAC-light", "\\textbf{TAC-3} (ours)", "3", "ours"),
    ("TAC (2 rounds)", "\\textbf{TAC-3, 2 rounds} (ours)", "3.3", "ours"),
    ("TAC 7 views", "\\textbf{TAC-7} (ours)", "7", "ours"),
]
best = max(r["mean"] for r in R.values())
L = []
for key, lab, views, grp in MAIN:
    r = R[key]
    cells = " & ".join(f"{d:.1f}" for d in r["dirs"])
    d = "--" if grp == "src" else delta(r, src)
    spi = "" if r["spi"] is None else f"{r['spi']:.2f}"
    L.append(f"{lab} & {views} & {cells} & {fmt(r['mean'], r['std'], bold=(grp == 'ours'))} & {r['ap50']:.2f} & {d} \\\\")
    NUM[f"main_{key}"] = {"mean": r["mean"], "std": r["std"], "ap50": r["ap50"], "dirs": r["dirs"], "views": r["views"], "spi": r["spi"], "d_src": list(paired(r, src)), "d50_src": [st.mean(r["m50"][s] - src["m50"][s] for s in r["m50"])]}
    if grp == "src":
        L.append("\\midrule")
    if key in ("BufferTTA", "CoTTA-det"):
        L.append("\\midrule")
# BN-model block
L.append("\\midrule")
L.append("\\multicolumn{11}{l}{\\textit{Source models with trainable BatchNorm (required by BN-statistics methods)}}\\\\")
bsrc = B["Source-only (BN)"]
for key, lab in [("Source-only (BN)", "Source model (BN)"), ("Tent", "Tent~\\cite{wang2021tent}"), ("DomainAdaptor", "DomainAdaptor~\\cite{zhang2023domainadaptor}"), ("VPTTA", "VPTTA~\\cite{chen2024vptta}"), ("GraTa", "GraTa~\\cite{chen2025grata}")]:
    r = B[key]; cells = " & ".join(f"{d:.1f}" for d in r["dirs"]); d = "--" if key.startswith("Source") else delta(r, bsrc)
    L.append(f"{lab} & 1{'' if key.startswith('Source') else ' + train'} & {cells} & {fmt(r['mean'], r['std'])} & {r['ap50']:.2f} & {d} \\\\")
    NUM[f"bn_{key}"] = {"mean": r["mean"], "std": r["std"], "ap50": r["ap50"]}
write("tab_main_heart.tex", "\n".join(L) + "\n")

# paired comparisons quoted in the text (Heart FRCNN)
for a, b in [("TAC-light", "Senior heatmap fusion (18 views)"), ("TAC (2 rounds)", "Senior heatmap fusion (18 views)"), ("TAC (2 rounds)", "18-view WBF + AR"), ("TAC (2 rounds)", "CoTTA-det"),
             ("TAC 7 views", "Senior heatmap fusion (18 views)"), ("TAC 7 views", "18-view WBF + AR"), ("TAC 7 views", "CoTTA-det"), ("TAC 7 views", "TAC (2 rounds)"), ("TAC-light", "18-view WBF + AR"),
             ("18-view WBF", "Senior heatmap fusion (18 views)"), ("18-view WBF + AR", "18-view WBF"), ("TAC 7 views", "TAC-light"), ("TAC (2 rounds)", "TAC-light")]:
    mu, half, sig = paired(R[a], R[b]); NUM[f"paired_heart::{a}::{b}"] = [mu, half, sig]

# ------------------------------------------------------------------ Table: other settings (RetinaNet, abdomen, spine) - mean AP / AP50 only
RT, AB, SP = rows("heart4cc_retinanet"), rows("abdomen"), rows("spine")
OTH = [
    ("Source model (no adaptation)", "1", "RetinaNet source-only", "Source-only (FrozenBN)", "Source-only (FrozenBN)"),
    ("IoU-Filter~\\cite{ruan2024ioufilter}", "1 + 5 upd.", "IoU-Filter", "IoU-Filter", "IoU-Filter"),
    ("AMROD~\\cite{cao2026amrod}", "1 + train", None, "AMROD", "AMROD"),
    ("WHW~\\cite{yoo2024whw}", "1 + train", None, "WHW", "WHW"),
    ("CD-Buffer~\\cite{song2026cdbuffer}", "1 + train", None, "CD-Buffer", "CD-Buffer"),
    ("SGP~\\cite{wang2025sgp}", "1 + train", None, "SGP", "SGP"),
    ("O-SFDA data acq.~\\cite{shi2024osfda}", "1 + train", None, "O-SFDA", "O-SFDA"),
    ("VLOD-TTA objective~\\cite{belal2026vlod}", "1 + 1 step", None, "VLOD-TTA objective", "VLOD-TTA objective"),
    ("BufferTTA~\\cite{kim2025buffer}", "1 + train", None, "BufferTTA", "BufferTTA"),
    ("18-view heat-map fusion", "18", "Senior heatmap fusion (18 views)", "Senior heatmap fusion (18 views)", "Senior heatmap fusion (18 views)"),
    ("18-view WBF~\\cite{solovyev2021wbf}", "18", "18-view WBF", "18-view WBF", "18-view WBF"),
    ("18-view WBF + re-scoring", "18", "18-view WBF + AR", "18-view WBF + AR", "18-view WBF + AR"),
    ("CoTTA-det~\\cite{wang2022cotta}", "18 + train", "CoTTA-det", "CoTTA-det", "CoTTA-det"),
    ("\\textbf{TAC-3} (ours)", "3", "TAC-light", "TAC-light", "TAC-light"),
    ("\\textbf{TAC-3, 2 rounds} (ours)", "3--3.4", "TAC (2 rounds)", "TAC (2 rounds)", "TAC (2 rounds)"),
    ("\\textbf{TAC-7} (ours)", "7", "TAC 7 views", "TAC 7 views", "TAC 7 views"),
]
L = []
srcs = (RT["RetinaNet source-only"], AB["Source-only (FrozenBN)"], SP["Source-only (FrozenBN)"])
for lab, views, krt, kab, ksp in OTH:
    cells = []
    for T, k, s in zip((RT, AB, SP), (krt, kab, ksp), srcs):
        if k is None or k not in T:
            cells.append("\multicolumn{3}{c}{n/a}"); continue
        r = T[k]; ours = "ours" in lab
        cells.append(f"{fmt(r['mean'], r['std'], bold=ours)} & {r['ap50']:.2f} & {'--' if r is s else delta(r, s)}")
        NUM[f"{'rt' if T is RT else 'abdomen' if T is AB else 'spine'}_{k}"] = {"mean": r["mean"], "std": r["std"], "ap50": r["ap50"], "dirs": r["dirs"], "views": r["views"], "spi": r["spi"], "d_src": list(paired(r, s))}
    L.append(f"{lab} & {views} & " + " & ".join(cells) + " \\\\")
    if lab.startswith("Source") or "BufferTTA" in lab or "CoTTA" in lab:
        L.append("\\midrule")
write("tab_other.tex", "\n".join(L) + "\n")
for name, T in [("rt", RT), ("abdomen", AB), ("spine", SP)]:
    for a, b in [("TAC-light", "Senior heatmap fusion (18 views)"), ("TAC (2 rounds)", "Senior heatmap fusion (18 views)"), ("TAC (2 rounds)", "18-view WBF + AR"), ("TAC (2 rounds)", "CoTTA-det"),
                 ("TAC 7 views", "Senior heatmap fusion (18 views)"), ("TAC 7 views", "18-view WBF + AR"), ("TAC 7 views", "CoTTA-det"), ("TAC 7 views", "TAC (2 rounds)"), ("TAC 7 views", "TAC-light"), ("TAC-light", "18-view WBF + AR"), ("TAC-light", "CoTTA-det")]:
        if a in T and b in T:
            mu, half, sig = paired(T[a], T[b]); NUM[f"paired_{name}::{a}::{b}"] = [mu, half, sig]
    # best single-view baseline
    singles = [k for k in ("IoU-Filter", "AMROD", "WHW", "CD-Buffer", "SGP", "O-SFDA", "VLOD-TTA objective", "BufferTTA") if k in T]
    kb = max(singles, key=lambda k: T[k]["mean"]); s = srcs[["rt", "abdomen", "spine"].index(name)]
    NUM[f"best_single_{name}"] = [kb, T[kb]["mean"], *paired(T[kb], s)]
singles = [k for k in ("IoU-Filter", "AMROD", "WHW", "CD-Buffer", "SGP", "O-SFDA", "VLOD-TTA objective", "BufferTTA")]
kb = max(singles, key=lambda k: R[k]["mean"]); NUM["best_single_heart"] = [kb, R[kb]["mean"], *paired(R[kb], src)]
for k in singles:
    NUM[f"single_vs_src_heart::{k}"] = list(paired(R[k], src))
    for name, T in [("abdomen", AB), ("spine", SP)]:
        NUM[f"single_vs_src_{name}::{k}"] = list(paired(T[k], T["Source-only (FrozenBN)"]))

# per-direction tables for the appendix (RetinaNet / abdomen / spine)
for name, T, hdr in [("rt", RT, ["c1$\\to$c2", "c1$\\to$c3", "c2$\\to$c1", "c2$\\to$c3", "c3$\\to$c1", "c3$\\to$c2"]), ("abdomen", AB, ["GE$\\to$PH", "GE$\\to$SA", "PH$\\to$GE", "PH$\\to$SA", "SA$\\to$GE", "SA$\\to$PH"]), ("spine", SP, ["GE$\\to$PH", "GE$\\to$SA", "PH$\\to$GE", "PH$\\to$SA", "SA$\\to$GE", "SA$\\to$PH"])]:
    L = ["\\midrule".join([]) ]
    L = []
    for lab, views, krt, kab, ksp in OTH:
        k = {"rt": krt, "abdomen": kab, "spine": ksp}[name]
        if k is None or k not in T:
            continue
        r = T[k]; cells = " & ".join(f"{d:.1f}" for d in r["dirs"])
        L.append(f"{lab} & {views} & {cells} & {fmt(r['mean'], r['std'], bold='ours' in lab)} & {r['ap50']:.2f} \\\\")
    write(f"tab_dirs_{name}.tex", "\n".join(L) + "\n")
    NUM[f"dirhdr_{name}"] = hdr

# ------------------------------------------------------------------ Table: component ablation on test (4 settings)
ABL = [("TAC-light", "TAC-3 (full)"), ("TAC w/o AR", "\\quad w/o structure-aware re-scoring"), ("TAC w/o s0 calib", "\\quad w/o calibrated $s_0$ (default 800)"), ("TAC (2 rounds)", "\\quad + second round"), ("TAC 7 views", "TAC-7 (+4 views at $\\pm$15\\%)")]
L = []
for key, lab in ABL:
    cells = []
    for T in (R, RT, AB, SP):
        if key in T:
            r = T[key]; ref = T["TAC-light"]
            cells.append(f"{fmt(r['mean'], r['std'])} & {'--' if key == 'TAC-light' else delta(r, ref)}")
        else:
            cells.append("\multicolumn{2}{c}{n/a}")
    L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
r = F["TAC 7 views w/o AR"]; ref = F["TAC 7 views (+-15% scales)"]
NA2 = "\\multicolumn{2}{c}{n/a}"
L.append(f"\\quad TAC-7 w/o structure-aware re-scoring & {fmt(r['mean'], r['std'])} & {delta(r, ref)} & {NA2} & {NA2} & {NA2} \\\\")
NUM["tac7_noar_heart"] = [r["mean"], r["std"], *paired(r, ref)]
write("tab_ablation.tex", "\n".join(L) + "\n")
for T, name in [(R, "heart"), (RT, "rt"), (AB, "abdomen"), (SP, "spine")]:
    for key, _ in ABL:
        if key in T:
            NUM[f"abl_{name}::{key}"] = [T[key]["mean"], T[key]["std"], *paired(T[key], T["TAC-light"])]

# ------------------------------------------------------------------ Table: fusion rule / views (test) + val ablations (seed 42)
L = []
for key, lab in [("TAC canonical view only (no fusion)", "canonical view only (no fusion)"), ("TAC fusion=heatmap (3 views)", "Gaussian heat-map fusion"), ("TAC fusion=NMS (3 views)", "concatenation + class-wise NMS"), ("TAC (WBF, 3 views)", "weighted boxes fusion (default)")]:
    r = F[key]; L.append(f"{lab} & 3 & {fmt(r['mean'], r['std'])} & {r['ap50']:.2f} & {'--' if 'WBF' in key else delta(r, F['TAC (WBF, 3 views)'])} \\\\")
    NUM[f"fusion::{key}"] = [r["mean"], r["std"], *paired(r, F["TAC (WBF, 3 views)"])]
L.append("\\midrule")
for key, lab in [("TAC 2 views (no flip)", "pass 1 + canonical view (no flip)"), ("TAC (WBF, 3 views)", "TAC-3"), ("TAC 7 views (+-15% scales)", "TAC-7"), ("TAC 11 views (+-10/20/25% scales)", "TAC-11 ($\\pm$10/20/25\\%)"), ("18-view WBF + AR", "18-view WBF + re-scoring"), ("18-view heatmap (senior)", "18-view heat-map fusion")]:
    r = F[key]; v = r["views"] if r["views"] else 18
    L.append(f"{lab} & {v:.0f} & {fmt(r['mean'], r['std'])} & {r['ap50']:.2f} & {'--' if key == 'TAC (WBF, 3 views)' else delta(r, F['TAC (WBF, 3 views)'])} \\\\")
    NUM[f"views::{key}"] = [r["mean"], r["std"], r["ap50"], *paired(r, F["TAC (WBF, 3 views)"])]
write("tab_fusion.tex", "\n".join(L) + "\n")

V = val_variants("4c", "canon")
VAL_ABL = [("tac", "TAC-3 (default)"), ("tac_nopass1", "\\quad w/o pass-1 view in the fusion"), ("tac_flip0", "\\quad + flipped pass-1 view"), ("tac_aspect", "\\quad + aspect-ratio factor $v$"),
           ("tac_shrink03", "\\quad strong shrinkage ($\\sigma_u=0.3$)"), ("tac_nooutlier", "\\quad w/o outlier component ($r_i=1$)"),
           ("tac_noar", "\\quad w/o structure-aware re-scoring"), ("tac_nocalib", "\\quad w/o calibrated $s_0$"), ("tac_rounds2", "\\quad + second round"), ("tac_full", "\\quad + second round + aspect factor")]
L = [f"{lab} & {V[k]:.2f} & {V[k] - V['tac']:+.2f} \\\\" for k, lab in VAL_ABL if k in V]
write("tab_val_ablation.tex", "\n".join(L) + "\n")
NUM["val_ablation"] = {k: V[k] for k, _ in VAL_ABL if k in V}
NUM["val_hp"] = {k: v for k, v in V.items() if k.startswith("hp_")}
NUM["val_baselines"] = val_variants("4c", "baselines")

# ------------------------------------------------------------------ Table: composition
CP = QC["compose"]
L = []
for key, lab in [("Source-only (1 view @800)", "Source model"), ("Source + TAC input (1 view, 0 updates)", "\\quad canonical input (control)"),
                 ("IoU-Filter", "IoU-Filter~\\cite{ruan2024ioufilter}"), ("IoU-Filter + TAC", "\\quad + canonical input"), ("AMROD", "AMROD~\\cite{cao2026amrod}"), ("AMROD + TAC", "\\quad + canonical input"),
                 ("CoTTA-det", "CoTTA-det~\\cite{wang2022cotta}"), ("CoTTA-det + TAC", "\\quad + canonical input"), ("TAC 2 rounds (ours, 3.3 views)", "TAC-3, 2 rounds (ours)")]:
    a = CP["AP"][key]; b = CP["AP50"][key]
    ma, sa = st.mean(a["per_seed_mean6"].values()), st.pstdev(a["per_seed_mean6"].values()); mb, sb = st.mean(b["per_seed_mean6"].values()), st.pstdev(b["per_seed_mean6"].values())
    def dl(x):
        return "--" if not x else f"{x[0]:+.2f}$\\pm${x[1]:.2f}{'$^{*}$' if x[2] else ''}"
    L.append(f"{lab} & {ma:.2f}$\\pm${sa:.2f} & {mb:.2f} & {dl(a.get('delta_vs_base'))} & {dl(a.get('delta_vs_ctrl'))} \\\\")
    NUM[f"compose::{key}"] = {"AP": [ma, sa], "AP50": [mb, sb], "d_base": a.get("delta_vs_base"), "d_ctrl": a.get("delta_vs_ctrl"), "d_tac2": a.get("delta_vs_tac2"), "d50_base": b.get("delta_vs_base")}
write("tab_compose.tex", "\n".join(L) + "\n")

# ------------------------------------------------------------------ cost numbers (Heart FRCNN, s/img on the shared GPU)
NUM["cost"] = {k: R[k]["spi"] for k in R if R[k]["spi"] is not None}
NUM["cost_views"] = {k: R[k]["views"] for k in R}
# FPN total variation (mean over seeds)
fp = QC["fpn"]; NUM["fpn_tv"] = {d: [fp["42"][d]["tv_800"], st.mean(fp[s][d]["tv_tac"] for s in fp)] for d in fp["42"]}
# per-class
NUM["per_class"] = QC["per_class"]
# source models in-domain val AP
NUM["source_models"] = {k: v["val_AP_final"] for k, v in QC["source_models"].items()}
json.dump(NUM, open(os.path.join(OUTD, "numbers.json"), "w"), indent=1)
print("numbers.json:", len(NUM), "entries")
