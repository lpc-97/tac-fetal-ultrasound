# -*- coding: utf-8 -*-
"""Generate every LaTeX table body of the manuscript from results/qc_collect_2026-09-18.json (+ dataset stats) so that no
number is typed by hand. Output: $TAC_ROOT/paper/tables/*.tex and $TAC_ROOT/paper/tables/numbers.json (all quantities quoted in
the text, for the provenance audit)."""
import json
import math
import os
import statistics as st

from pubstyle import load_aux
from tabdata import QC, T3A, UPPER, rows, paired, val_variants

ROOT = os.environ.get("TAC_ROOT", ".")
OUTD = os.path.join(ROOT, "paper", "tables"); os.makedirs(OUTD, exist_ok=True)
NUM = {}


def fmt(m, s=None, bold=False, prec=2, under=False):
    t = f"{m:.{prec}f}" + (f"\\,$\\pm$\\,{s:.{prec}f}" if s is not None else "")
    return f"\\textbf{{{t}}}" if bold else (f"\\underline{{{t}}}" if under else t)


def argbest(vals):
    """indices holding the largest value (ties included); None entries are skipped"""
    m = max(v for v in vals if v is not None)
    return {i for i, v in enumerate(vals) if v is not None and abs(v - m) < 1e-9}


def argsecond(vals):
    """indices holding the second-largest value (ties included); None entries are skipped"""
    m = max(v for v in vals if v is not None)
    rest = [v for v in vals if v is not None and v < m - 1e-9]
    return set() if not rest else {i for i, v in enumerate(vals) if v is not None and abs(v - max(rest)) < 1e-9}


def mark(t, i, best, second):
    """bold for the best value of a column, underline for the second best"""
    return f"\\textbf{{{t}}}" if i in best else (f"\\underline{{{t}}}" if i in second else t)


def delta(a, b):
    """seed-paired difference; the asterisk marks a 95% interval that excludes zero (the text gives the interval)"""
    mu, half, sig = paired(a, b)
    return f"${mu:+.2f}{'^{*}' if sig else ''}$"


YES, NO = "\\cmark", "\\xmark"  # parameters updated at test time (macros defined in main.tex)


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
# The full seven-view model is the surrogate-guided (quadrature) bracket of the methodology. Those runs
# were collected later; where they are present they are reported as TAC-7, and the fixed bracket
# becomes the control row of the view-count ablation.
QUAD7 = "TAC 7 views (quadrature)"


def k7(tbl, fixed_key):
    """key of the seven-view row to report: the quadrature run if collected, else the fixed bracket"""
    return QUAD7 if QUAD7 in tbl else fixed_key

R = rows("heart4cc_frcnn"); B = rows("heart4cc_bn_models"); F = rows("heart4cc_fusion_views")
src = R["Source-only (FrozenBN)"]
T3K = "Senior heatmap fusion (18 views)"


def vf(v):
    """views per image: an integer, or one decimal for a mean over directions (T^3A selects 16 or 18 views per direction)"""
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


GROUP = {"single": "Single-view test-time adaptation", "multi": "Multi-view fusion",
         "ours": "Test-time anatomical canonicalization (ours)"}
MAIN = [  # (row name in QC, label, views per image, parameters updated at test time, group)
    ("Source-only (FrozenBN)", "Source model (no adaptation)", "1", False, "src"),
    ("IoU-Filter", "IoU-Filter~\\cite{ruan2024ioufilter}", "1", True, "single"),
    ("AMROD", "AMROD~\\cite{cao2026amrod}", "1", True, "single"),
    ("WHW", "WHW~\\cite{yoo2024whw}", "1", True, "single"),
    ("CD-Buffer", "CD-Buffer~\\cite{song2026cdbuffer}", "1", True, "single"),
    ("SGP", "SGP~\\cite{wang2025sgp}", "1", True, "single"),
    ("O-SFDA", "O-SFDA data acq.~\\cite{shi2024osfda}", "1", True, "single"),
    ("VLOD-TTA objective", "VLOD-TTA objective~\\cite{belal2026vlod}", "1", True, "single"),
    ("BufferTTA", "BufferTTA~\\cite{kim2025buffer}", "1", True, "single"),
    (T3K, "T$^3$A~\\cite{pu2026t3a}", vf(R[T3K]["views"]), False, "multi"),
    ("18-view WBF", "18-view WBF~\\cite{solovyev2021wbf}", "18", False, "multi"),
    ("18-view WBF + AR", "18-view WBF + re-scoring", "18", False, "multi"),
    ("CoTTA-det", "CoTTA-det~\\cite{wang2022cotta}", "18", True, "multi"),
    ("TAC-light", "TAC-3 (ours)", "3", False, "ours"),
    ("TAC (2 rounds)", "TAC-3, 2 rounds (ours)", "3.3", False, "ours"),
    (k7(R, "TAC 7 views"), "TAC-7 (ours)", "7", False, "ours"),
]


def heart_block(T, spec, ref, ncol):
    """rows of one block of source models: bold best and underline second best per column, a group header whenever
    the group changes, the paired difference to the block's source model"""
    rs = [T[k] for k, *_ in spec]
    bd = [argbest([r["dirs"][j] for r in rs]) for j in range(6)]; sd = [argsecond([r["dirs"][j] for r in rs]) for j in range(6)]
    bm, sm = argbest([r["mean"] for r in rs]), argsecond([r["mean"] for r in rs])
    b5, s5 = argbest([r["ap50"] for r in rs]), argsecond([r["ap50"] for r in rs])
    out, prev = [], None
    for i, (key, lab, views, upd, grp) in enumerate(spec):
        if grp != prev and grp in GROUP:
            out += ["\\midrule", f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{GROUP[grp]}}}}}\\\\"]
        prev = grp
        r = T[key]
        cells = " & ".join(mark(f"{d:.1f}", i, bd[j], sd[j]) for j, d in enumerate(r["dirs"]))
        out.append(f"{lab} & {views} & {YES if upd else NO} & {cells} & {fmt(r['mean'], r['std'], bold=(i in bm), under=(i in sm))} & "
                   f"{mark(format(r['ap50'], '.2f'), i, b5, s5)} & {'--' if key == ref else delta(r, T[ref])} \\\\")
    return out


L = heart_block(R, MAIN, "Source-only (FrozenBN)", 12)
for key, *_ in MAIN:
    r = R[key]
    NUM[f"main_{key}"] = {"mean": r["mean"], "std": r["std"], "ap50": r["ap50"], "dirs": r["dirs"], "views": r["views"], "spi": r["spi"], "d_src": list(paired(r, src)), "d50_src": [st.mean(r["m50"][s] - src["m50"][s] for s in r["m50"])]}
# in-domain reference: the target-domain detector with one view (not an upper bound for multi-view methods)
r = R[UPPER]
L += ["\\midrule", f"\\textit{{Target-domain detector (in-domain)}} & 1 & {NO} & " + " & ".join(f"{d:.1f}" for d in r["dirs"])
      + f" & {fmt(r['mean'], r['std'])} & {r['ap50']:.2f} & {delta(r, src)} \\\\"]
NUM["main_" + UPPER] = {"mean": r["mean"], "std": r["std"], "ap50": r["ap50"], "dirs": r["dirs"], "d_src": list(paired(r, src))}
# BN-model block
L.append("\\midrule")
L.append("\\multicolumn{12}{l}{\\textit{Source models with trainable BatchNorm (required by BN-statistics methods)}}\\\\")
bsrc = B["Source-only (BN)"]
_BN = [("Source-only (BN)", "Source model (BN)", "1", False, "bn"), ("Tent", "Tent~\\cite{wang2021tent}", "1", True, "bn"),
       ("DomainAdaptor", "DomainAdaptor~\\cite{zhang2023domainadaptor}", "1", True, "bn"), ("VPTTA", "VPTTA~\\cite{chen2024vptta}", "1", True, "bn"),
       ("GraTa", "GraTa~\\cite{chen2025grata}", "1", True, "bn")]
L += heart_block(B, _BN, "Source-only (BN)", 12)
for key, *_ in _BN:
    NUM[f"bn_{key}"] = {"mean": B[key]["mean"], "std": B[key]["std"], "ap50": B[key]["ap50"]}
write("tab_main_heart.tex", "\n".join(L) + "\n")

# paired comparisons quoted in the text (Heart FRCNN)
for a, b in [("TAC-light", "Senior heatmap fusion (18 views)"), ("TAC (2 rounds)", "Senior heatmap fusion (18 views)"), ("TAC (2 rounds)", "18-view WBF + AR"), ("TAC (2 rounds)", "CoTTA-det"),
             ("TAC 7 views", "Senior heatmap fusion (18 views)"), ("TAC 7 views", "18-view WBF + AR"), ("TAC 7 views", "CoTTA-det"), ("TAC 7 views", "TAC (2 rounds)"), ("TAC-light", "18-view WBF + AR"),
             ("18-view WBF", "Senior heatmap fusion (18 views)"), ("18-view WBF + AR", "18-view WBF"), ("TAC 7 views", "TAC-light"), ("TAC (2 rounds)", "TAC-light")]:
    mu, half, sig = paired(R[a], R[b]); NUM[f"paired_heart::{a}::{b}"] = [mu, half, sig]

# ------------------------------------------------------------------ Table: other settings (RetinaNet, abdomen, spine) - mean AP / AP50 only
RT, AB, SP = rows("heart4cc_retinanet"), rows("abdomen"), rows("spine")
_t3v = [T[T3K]["views"] for T in (RT, AB, SP)]
OTH = [  # (label, views per image, parameters updated, row in RetinaNet / abdomen / spine, group)
    ("Source model (no adaptation)", "1", False, "RetinaNet source-only", "Source-only (FrozenBN)", "Source-only (FrozenBN)", "src"),
    ("IoU-Filter~\\cite{ruan2024ioufilter}", "1", True, "IoU-Filter", "IoU-Filter", "IoU-Filter", "single"),
    ("AMROD~\\cite{cao2026amrod}", "1", True, None, "AMROD", "AMROD", "single"),
    ("WHW~\\cite{yoo2024whw}", "1", True, None, "WHW", "WHW", "single"),
    ("CD-Buffer~\\cite{song2026cdbuffer}", "1", True, None, "CD-Buffer", "CD-Buffer", "single"),
    ("SGP~\\cite{wang2025sgp}", "1", True, None, "SGP", "SGP", "single"),
    ("O-SFDA data acq.~\\cite{shi2024osfda}", "1", True, None, "O-SFDA", "O-SFDA", "single"),
    ("VLOD-TTA objective~\\cite{belal2026vlod}", "1", True, None, "VLOD-TTA objective", "VLOD-TTA objective", "single"),
    ("BufferTTA~\\cite{kim2025buffer}", "1", True, None, "BufferTTA", "BufferTTA", "single"),
    ("T$^3$A~\\cite{pu2026t3a}", f"{vf(min(_t3v))}--{vf(max(_t3v))}", False, T3K, T3K, T3K, "multi"),
    ("18-view WBF~\\cite{solovyev2021wbf}", "18", False, "18-view WBF", "18-view WBF", "18-view WBF", "multi"),
    ("18-view WBF + re-scoring", "18", False, "18-view WBF + AR", "18-view WBF + AR", "18-view WBF + AR", "multi"),
    ("CoTTA-det~\\cite{wang2022cotta}", "18", True, "CoTTA-det", "CoTTA-det", "CoTTA-det", "multi"),
    ("TAC-3 (ours)", "3", False, "TAC-light", "TAC-light", "TAC-light", "ours"),
    ("TAC-3, 2 rounds (ours)", "3--3.4", False, "TAC (2 rounds)", "TAC (2 rounds)", "TAC (2 rounds)", "ours"),
    ("TAC-7 (ours)", "7", False, k7(RT, "TAC 7 views"), k7(AB, "TAC 7 views"), k7(SP, "TAC 7 views"), "ours"),
]
L = []
srcs = (RT["RetinaNet source-only"], AB["Source-only (FrozenBN)"], SP["Source-only (FrozenBN)"])
_best = []
for T, col in zip((RT, AB, SP), (3, 4, 5)):
    _vals = [(T[row[col]]["mean"] if row[col] in T else None) for row in OTH]
    _v50 = [(T[row[col]]["ap50"] if row[col] in T else None) for row in OTH]
    _best.append((argbest(_vals), argbest(_v50), argsecond(_vals), argsecond(_v50)))
prev = None
for i, (lab, views, upd, krt, kab, ksp, grp) in enumerate(OTH):
    if grp != prev and grp in GROUP:
        L += ["\\midrule", f"\\multicolumn{{12}}{{l}}{{\\textit{{{GROUP[grp]}}}}}\\\\"]
    prev = grp
    cells = []
    for bi, (T, k, s) in enumerate(zip((RT, AB, SP), (krt, kab, ksp), srcs)):
        if k is None or k not in T:
            cells.append("\\multicolumn{3}{c}{n/a}"); continue
        r = T[k]; bm, b5, sm, s5 = _best[bi]
        cells.append(f"{fmt(r['mean'], r['std'], bold=(i in bm), under=(i in sm))} & {mark(format(r['ap50'], '.2f'), i, b5, s5)} & {'--' if r is s else delta(r, s)}")
        NUM[f"{'rt' if T is RT else 'abdomen' if T is AB else 'spine'}_{k}"] = {"mean": r["mean"], "std": r["std"], "ap50": r["ap50"], "dirs": r["dirs"], "views": r["views"], "spi": r["spi"], "d_src": list(paired(r, s))}
    L.append(f"{lab} & {views} & {YES if upd else NO} & " + " & ".join(cells) + " \\\\")
# in-domain reference (one view): an upper bound for single-view methods only; multi-view methods exceed it on the abdomen
cells = []
for T, s, nm in zip((RT, AB, SP), srcs, ("rt", "abdomen", "spine")):
    r = T[UPPER]; cells.append(f"{fmt(r['mean'], r['std'])} & {r['ap50']:.2f} & {delta(r, s)}")
    NUM[f"{nm}_{UPPER}"] = {"mean": r["mean"], "std": r["std"], "ap50": r["ap50"], "dirs": r["dirs"], "d_src": list(paired(r, s))}
L += ["\\midrule", f"\\textit{{Target-domain detector (in-domain)}} & 1 & {NO} & " + " & ".join(cells) + " \\\\"]
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
    _present = [(lab, views, {"rt": krt, "abdomen": kab, "spine": ksp}[name]) for lab, views, upd, krt, kab, ksp, grp in OTH]
    # T^3A and the second round report their own mean view count in each setting
    _present = [(lab, vf(T[k]["views"]) if k == T3K else (f"{T[k]['views']:.1f}" if k == "TAC (2 rounds)" else views), k)
                for lab, views, k in _present if k is not None and k in T]
    _prow = [T[k] for _, _, k in _present]
    _bdir = [argbest([r["dirs"][j] for r in _prow]) for j in range(6)]; _sdir = [argsecond([r["dirs"][j] for r in _prow]) for j in range(6)]
    _bmean, _b50 = argbest([r["mean"] for r in _prow]), argbest([r["ap50"] for r in _prow])
    _smean, _s50 = argsecond([r["mean"] for r in _prow]), argsecond([r["ap50"] for r in _prow])
    for i, (lab, views, k) in enumerate(_present):
        r = T[k]
        cells = " & ".join(mark(f"{d:.1f}", i, _bdir[j], _sdir[j]) for j, d in enumerate(r["dirs"]))
        L.append(f"{lab} & {views} & {cells} & {fmt(r['mean'], r['std'], bold=(i in _bmean), under=(i in _smean))} & {mark(format(r['ap50'], '.2f'), i, _b50, _s50)} \\\\")
    write(f"tab_dirs_{name}.tex", "\n".join(L) + "\n")
    NUM[f"dirhdr_{name}"] = hdr

# ------------------------------------------------------------------ Table: component ablation on test, two panels
# (a) components removed from TAC-3, on the three settings where they were run;
# (b) a second round and four further views, on all four settings. No cell is missing in either panel.
ABL_A = [("TAC-light", "TAC-3 (full)"), ("TAC w/o AR", "\\quad w/o structure-aware re-scoring"), ("TAC w/o s0 calib", "\\quad w/o calibrated $s_0$ (default 800)")]
ABL_B = [("TAC-light", "TAC-3 (full)"), ("TAC (2 rounds)", "\\quad + second round"), (QUAD7, "TAC-7 (+4 views)")]


def abl_panel(rows, tables):
    best = [argbest([T[k]["mean"] for k, _ in rows]) for T in tables]   # KeyError if a cell is missing
    L = []
    for i, (key, lab) in enumerate(rows):
        cells = []
        for bi, T in enumerate(tables):
            r = T[key]; ref = T["TAC-light"]
            cells.append(f"{fmt(r['mean'], r['std'], bold=(i in best[bi]))} & {'--' if key == 'TAC-light' else delta(r, ref)}")
        L.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    return "\n".join(L) + "\n"


write("tab_ablation_a.tex", abl_panel(ABL_A, (R, AB, SP)))
write("tab_ablation_b.tex", abl_panel(ABL_B, (R, RT, AB, SP)))
r = F["TAC 7 views w/o AR"]; ref = F["TAC 7 views (+-15% scales)"]
NUM["tac7_noar_heart"] = [r["mean"], r["std"], *paired(r, ref)]   # recorded, no longer tabulated
for T, name in [(R, "heart"), (RT, "rt"), (AB, "abdomen"), (SP, "spine")]:
    for key in dict(ABL_A + ABL_B):
        if key in T:
            NUM[f"abl_{name}::{key}"] = [T[key]["mean"], T[key]["std"], *paired(T[key], T["TAC-light"])]

# ------------------------------------------------------------------ fusion rule / views (test) + val ablations (seed 42)
# The fusion-rule and view-count rows are tabulated together with the equal-compute blind ensembles by
# gen_tables_reviewer.py (tab_views.tex); the quantities quoted in the text are recorded here.
for key in ["TAC canonical view only (no fusion)", "TAC fusion=heatmap (3 views)", "TAC fusion=NMS (3 views)", "TAC (WBF, 3 views)"]:
    r = F[key]; NUM[f"fusion::{key}"] = [r["mean"], r["std"], *paired(r, F["TAC (WBF, 3 views)"])]
for key in ["TAC 2 views (no flip)", "TAC (WBF, 3 views)", "TAC 7 views (+-15% scales)", "TAC 11 views (+-10/20/25% scales)", "18-view WBF + AR", "18-view heatmap (senior)"]:
    r = F[key]; NUM[f"views::{key}"] = [r["mean"], r["std"], r["ap50"], *paired(r, F["TAC (WBF, 3 views)"])]

V = val_variants("4c", "canon")
VAL_ABL = [("tac", "TAC-3 (default)"), ("tac_nopass1", "\\quad w/o pass-1 view in the fusion"), ("tac_flip0", "\\quad + flipped pass-1 view"), ("tac_aspect", "\\quad + aspect-ratio factor $v$"),
           ("tac_shrink03", "\\quad strong shrinkage ($\\sigma_u=0.3$)"), ("tac_nooutlier", "\\quad w/o outlier component ($r_i=1$)"),
           ("tac_noar", "\\quad w/o structure-aware re-scoring"), ("tac_nocalib", "\\quad w/o calibrated $s_0$"), ("tac_rounds2", "\\quad + second round"), ("tac_full", "\\quad + second round + aspect factor")]
L = [f"{lab} & {V[k]:.2f} & ${V[k] - V['tac']:+.2f}$ \\\\" for k, lab in VAL_ABL if k in V]
write("tab_val_ablation.tex", "\n".join(L) + "\n")
NUM["val_ablation"] = {k: V[k] for k, _ in VAL_ABL if k in V}
NUM["val_hp"] = {k: v for k, v in V.items() if k.startswith("hp_")}
NUM["val_baselines"] = val_variants("4c", "baselines")
# T^3A (Appendix E): mean validation AP over the six heart directions per configuration and for the per-direction selection
_va, _sel = T3A["best"]["val_ap"]["4c"], T3A["best"]["best"]["4c"]
NUM["val_t3a"] = {v: st.mean(_va[d][v] for d in _va) for v in T3A["best"]["variants"]}
NUM["val_t3a"]["selected"] = st.mean(_va[d][_sel[d]] for d in _va)
NUM["val_t3a"]["selected_variant"] = _sel

# ------------------------------------------------------------------ Table: composition
CP = QC["compose"]
L = []
for key, lab in [("Source-only (1 view @800)", "Source model"), ("Source + TAC input (1 view, 0 updates)", "\\quad canonical input (control)"),
                 ("IoU-Filter", "IoU-Filter~\\cite{ruan2024ioufilter}"), ("IoU-Filter + TAC", "\\quad + canonical input"), ("AMROD", "AMROD~\\cite{cao2026amrod}"), ("AMROD + TAC", "\\quad + canonical input"),
                 ("CoTTA-det", "CoTTA-det~\\cite{wang2022cotta}"), ("CoTTA-det + TAC", "\\quad + canonical input"), ("TAC 2 rounds (ours, 3.3 views)", "TAC-3, 2 rounds (ours)")]:
    a = CP["AP"][key]; b = CP["AP50"][key]
    ma, sa = st.mean(a["per_seed_mean6"].values()), st.pstdev(a["per_seed_mean6"].values()); mb, sb = st.mean(b["per_seed_mean6"].values()), st.pstdev(b["per_seed_mean6"].values())
    def dl(x):
        return "--" if not x else f"${x[0]:+.2f}{'^{*}' if x[2] else ''}$"
    L.append(f"{lab} & {fmt(ma, sa)} & {mb:.2f} & {dl(a.get('delta_vs_base'))} & {dl(a.get('delta_vs_ctrl'))} \\\\")
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
