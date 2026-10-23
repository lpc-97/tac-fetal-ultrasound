# -*- coding: utf-8 -*-
"""Table bodies for the reviewer experiments, generated from results/reviewer/*.json (no number typed by hand):
  tab_scalechain.tex  A. source / val-selected scale / stream-global factor / TAC factor / oracle factor, single view and TAC-3 pipeline
  tab_estimator.tex   B. scale estimator inside TAC-3
  tab_views.tex       C. fusion rule, number of views and equal-compute blind ensembles vs TAC (+ time / peak memory from the
                      idle-GPU timing run when available)
  tab_firstpass.tex   first-pass candidate groups and oracle-shift groups
  tab_indomain.tex    D. in-domain control
  tab_bootstrap.tex   cluster statistics, de-duplicated AP and cluster-level bootstrap CIs (three datasets)
  tab_defaults.tex    E. published-default baseline configurations vs val-selected
Writes tables/numbers_reviewer.json with every quantity quoted in the text."""
import json
import math
import os
import statistics as st

ROOT = os.environ.get("TAC_ROOT", ".")
RES = os.path.join(ROOT, "results", "reviewer")
OUTD = os.path.join(ROOT, "paper", "tables")
T975 = 4.303
R = json.load(open(os.path.join(RES, "reviewer_summary.json"), encoding="utf-8"))["rows"]
# T^3A re-run on 2026-10-22 with its configuration selected per direction on validation (server: *_t3a.py). Its
# summary differs from the original only in the heatmap18 row and adds paper_t3a (published setting, Appendix D);
# its bootstrap files differ only in the comparisons and de-duplicated APs that involve heatmap18.
T3D = os.path.join(RES, "t3a_2026-10-22")
_T3 = json.load(open(os.path.join(T3D, "reviewer_summary_t3a.json"), encoding="utf-8"))["rows"]
R["heatmap18"], R["paper_t3a"] = _T3["heatmap18"], _T3["paper_t3a"]
from tabdata import rows as _qrows  # noqa: E402  (views per image of the selected T^3A runs: 16 or 18 per direction)
R["heatmap18"]["views"] = _qrows("heart4cc_frcnn")["Senior heatmap fusion (18 views)"]["views"]
FP = json.load(open(os.path.join(RES, "firstpass_analysis.json"), encoding="utf-8"))
BOOT = {d: json.load(open(os.path.join(T3D, f"cluster_bootstrap_{d}_t3a.json"), encoding="utf-8")) for d in ("4c", "abdomen", "spine")}
TIM = json.load(open(os.path.join(RES, "timing.json"), encoding="utf-8")) if os.path.exists(os.path.join(RES, "timing.json")) else {}
# the uncertainty-aware runs live in their own collection; the quadrature configuration is the one
# reported as TAC-7, so it is merged into the main row dict for every table that references it
UV = json.load(open(os.path.join(RES, "uncert_summary.json"), encoding="utf-8"))["test"] if os.path.exists(os.path.join(RES, "uncert_summary.json")) else {}
for _k, _v in UV.items():
    R.setdefault(_k, _v)
NUM = {}


def paired(a, b):
    d = [a["m6"][s] - b["m6"][s] for s in a["m6"] if s in b["m6"]]; mu = st.mean(d); half = T975 * st.stdev(d) / math.sqrt(len(d))
    return mu, half, abs(mu) > half


def argbest(vals):
    m = max(v for v in vals if v is not None)
    return {i for i, v in enumerate(vals) if v is not None and abs(v - m) < 1e-9}


def fd(a, b):
    """seed-paired difference; the asterisk marks a 95% interval that excludes zero (the text gives the interval)"""
    mu, half, sig = paired(a, b); return f"${mu:+.2f}{'^{*}' if sig else ''}$"


def fm(r, bold=False):
    t = f"{r['mean']:.2f}\\,$\\pm$\\,{r['std']:.2f}"; return f"\\textbf{{{t}}}" if bold else t


def write(name, lines):
    open(os.path.join(OUTD, name), "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("wrote", name)


src, orc1, orc3 = R["source"], R["fac_oracle_1v"], R["fac_oracle"]
# ------------------------------------------------------------------ A. scale chain
L = []
for lab, k1, k3 in [("default scale ($s_0$)", "source", None), ("val-selected fixed scale per direction", "fac_valscale_1v", "fac_valscale"), ("stream-level factor (median of TAC factors)", "fac_global_1v", "fac_global"),
                    ("TAC per-image factor (ours)", "tac_1v", "tac"), ("oracle per-image factor (GT sizes)", "fac_oracle_1v", "fac_oracle")]:
    a = R[k1]; c = R.get(k3)
    g1 = "--" if k1 == "source" else f"{100 * (a['mean'] - src['mean']) / (orc1['mean'] - src['mean']):.0f}\\%"
    d1 = "--" if k1 == "source" else fd(a, src)
    if c:
        g3 = f"{100 * (c['mean'] - src['mean']) / (orc3['mean'] - src['mean']):.0f}\\%"; c3 = f"{fm(c, k3 == 'tac')} & {fd(c, src)} & {g3}"
    else:
        c3 = "\\multicolumn{3}{c}{--}"
    L.append(f"{lab} & {fm(a, k1 == 'tac_1v')} & {d1} & {g1} & {c3} \\\\")
    NUM[f"chain::{k1}"] = {"mean": a["mean"], "std": a["std"], "gap": None if k1 == "source" else (a["mean"] - src["mean"]) / (orc1["mean"] - src["mean"])}
    if c:
        NUM[f"chain3::{k3}"] = {"mean": c["mean"], "std": c["std"], "gap": (c["mean"] - src["mean"]) / (orc3["mean"] - src["mean"]), "d_src": list(paired(c, src))}
write("tab_scalechain.tex", L)
NUM["chain::oracle_minus_tac_1v"] = list(paired(orc1, R["tac_1v"])); NUM["chain::oracle_minus_tac3"] = list(paired(orc3, R["tac"]))
NUM["chain::tac1v_minus_valscale1v"] = list(paired(R["tac_1v"], R["fac_valscale_1v"])); NUM["chain::tac1v_minus_global1v"] = list(paired(R["tac_1v"], R["fac_global_1v"]))
NUM["chain::tac3_minus_valscale3"] = list(paired(R["tac"], R["fac_valscale"])); NUM["chain::tac3_minus_global3"] = list(paired(R["tac"], R["fac_global"]))
NUM["chain::tac3_2r_vs_oracle3"] = list(paired(R["tac_rounds2"], R["fac_oracle"]))

# ------------------------------------------------------------------ B. estimator
L = []
for lab, k in [("median of the votes (no weights, no prior)", "est_median"), ("confidence-weighted mean of votes", "est_confmean"), ("IRLS, class-agnostic (pooled) prior", "est_noclass"), ("IRLS, per-structure prior (ours)", "tac")]:
    r = R[k]; L.append(f"{lab} & {fm(r, k == 'tac')} & {r['ap50']:.2f} & {'--' if k == 'tac' else fd(r, R['tac'])} \\\\")
    NUM[f"est::{k}"] = {"mean": r["mean"], "std": r["std"], "d_tac": None if k == "tac" else list(paired(r, R["tac"]))}
write("tab_estimator.tex", L)

# ------------------------------------------------------------------ C. blind ensembles (+ timing)
def tcell(key):
    t = TIM.get(key)
    return ("", "") if not t else (f"{t['s_per_img']:.2f}", f"{t['peak_alloc_MB'] / 1024:.1f}")


def vstr(v):
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


# One table for the fusion step: (a) the fusion rule with the same three views (heart4cc_fusion_views rows of the main
# collection), (b) the number of views and the blind ensembles at equal compute (reviewer rows). Every difference pairs
# rows of the same collection: TAC-3 is the reference up to three views (3.3 with a second round), TAC-7 above.
F = _qrows("heart4cc_fusion_views"); F3 = "TAC (WBF, 3 views)"
L = ["\\multicolumn{7}{l}{\\textit{(a) Fusion rule, the same three views}}\\\\"]
_FA = [("canonical view only (no fusion)", "TAC canonical view only (no fusion)"), ("heat-map fusion of T$^3$A", "TAC fusion=heatmap (3 views)"),
       ("concatenation + class-wise NMS", "TAC fusion=NMS (3 views)"), ("weighted boxes fusion (\\tac-3)", F3)]
_ba, _b5 = argbest([F[k]["mean"] for _, k in _FA]), argbest([F[k]["ap50"] for _, k in _FA])
for i, (lab, k) in enumerate(_FA):
    r = F[k]; ap50 = f"\\textbf{{{r['ap50']:.2f}}}" if i in _b5 else f"{r['ap50']:.2f}"
    L.append(f"{lab} & 3 & {fm(r, i in _ba)} & {ap50} & {'--' if k == F3 else fd(r, F[F3])} & & \\\\")
L += ["\\midrule", "\\multicolumn{7}{l}{\\textit{(b) Number of views and blind ensembles at equal compute}}\\\\"]
_BL = [  # (label, row, collection, reference, timing key, compute block)
    ("pass 1 + canonical view (no flip)", "TAC 2 views (no flip)", F, F3, None, 3),
    ("blind: 3 scales \\{600, 800, 1000\\}, WBF", "ss_fix3_wbf", R, "tac", None, 3), ("blind: 3 scales, WBF + re-scoring", "ss_fix3_wbf_ar", R, "tac", "blind3_ar", 3),
    ("\\tac-3 (ours)", "tac", R, None, "tac3", 3), ("\\tac-3, 2 rounds (ours)", "tac_rounds2", R, "tac", "tac3_2r", 3),
    ("blind: 3 scales $\\times$ flip, WBF + re-scoring", "ss_fix3f_wbf_ar", R, "u7_k17", None, 7),
    ("blind: 7 scales \\{500--1200\\}, WBF", "ss_fix7_wbf", R, "u7_k17", None, 7), ("blind: 7 scales, WBF + re-scoring", "ss_fix7_wbf_ar", R, "u7_k17", "blind7_ar", 7),
    ("\\tac, fixed bracket $\\pm15\\%$ (control)", "tac_v7", R, "u7_k17", None, 7), ("\\tac-7 (ours)", "u7_k17", R, None, "tac7", 7),
    ("\\tac-11, fixed bracket", "tac_v11", R, "u7_k17", "tac11", 18),
    ("18-view WBF + re-scoring", "ss_all_flip_wbf_ar", R, "u7_k17", "wbf18_ar", 18), ("T$^3$A~\\cite{pu2026t3a}", "heatmap18", R, "u7_k17", "heatmap18", 18)]
_bb = {}
for blk in (3, 7, 18):
    idx = [i for i, row in enumerate(_BL) if row[5] == blk]
    bm = argbest([_BL[i][2][_BL[i][1]]["mean"] for i in idx]); b5 = argbest([_BL[i][2][_BL[i][1]]["ap50"] for i in idx])
    for j, i in enumerate(idx):
        _bb[i] = (j in bm, j in b5)
for i, (lab, k, T, ref, tk, blk) in enumerate(_BL):
    r = T[k]; v = r["views"] if r["views"] else 18; sp, mem = tcell(tk)
    ap50 = f"\\textbf{{{r['ap50']:.2f}}}" if _bb[i][1] else f"{r['ap50']:.2f}"
    L.append(f"{lab} & {vstr(v)} & {fm(r, _bb[i][0])} & {ap50} & {'--' if ref is None else fd(r, T[ref])} & {sp} & {mem} \\\\")
    if T is R:
        # d_ref keeps the references of the former equal-compute table, which the text quotes
        old = {"ss_fix3_wbf": "tac", "ss_fix3_wbf_ar": "tac", "ss_fix3f_wbf_ar": "u7_k17", "ss_fix7_wbf": "u7_k17", "ss_fix7_wbf_ar": "u7_k17"}.get(k)
        NUM[f"blind::{k}"] = {"mean": r["mean"], "std": r["std"], "views": v, "d_ref": None if old is None else list(paired(r, R[old])),
                              "d_table": None if ref is None else list(paired(r, T[ref])), "timing": TIM.get(tk)}
write("tab_views.tex", L)

# ------------------------------------------------------------------ F. uncertainty-guided view acquisition (UVA)
UV = json.load(open(os.path.join(RES, "uncert_summary.json"), encoding="utf-8"))["test"] if os.path.exists(os.path.join(RES, "uncert_summary.json")) else {}
if UV:
    def urow(lab, key, ref, bold=False, bold50=False):
        r = UV[key] if key in UV else R[key]
        d = "--" if ref is None else fd(r, UV[ref] if ref in UV else R[ref])
        t = fm(r, bold)
        NUM[f"uva::{key}"] = {"mean": r["mean"], "std": r["std"], "views": r["views"], "d_ref": None if ref is None else list(paired(r, UV[ref] if ref in UV else R[ref]))}
        v = r["views"]; vs = f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.2f}"
        ap50 = f"\\textbf{{{r['ap50']:.2f}}}" if bold50 else f"{r['ap50']:.2f}"
        return f"{lab} & {vs} & {t} & {ap50} & {d} \\\\"
    _UR = [("\\tac-3 (no bracket)", "tac", None), ("\\quad fixed bracket $\\pm15\\%$ (control)", "tac_v7", "tac"),
           ("\\tac-11, fixed bracket", "tac_v11", "tac_v7"), None,
           ("\\tac-7, quadrature (ours)", "u7_k17", "tac_v7"), ("UVA, gated ($\\tau_s=0.08$)", "ua_t08", "tac_v7"),
           ("UVA, gated ($\\tau_s=0.12$)", "ua_t12", "tac")]
    _uv = [((UV[k] if k in UV else R[k])["mean"] if row else None) for row in _UR for k in [row[1] if row else None]]
    _ub = argbest(_uv)
    _ub50 = argbest([((UV[k] if k in UV else R[k])["ap50"] if row else None) for row in _UR for k in [row[1] if row else None]])
    L = [("\\midrule" if row is None else urow(row[0], row[1], row[2], i in _ub, i in _ub50)) for i, row in enumerate(_UR)]
    write("tab_uva.tex", L)
    # gated runs: share of images bracketed (each bracketed image costs four more passes than TAC-3) and the
    # share of the gain of full quadrature acquisition (TAC-7) that a gated run keeps
    _get = (lambda k: UV[k] if k in UV else R[k])
    _g7 = _get("u7_k17")["mean"] - _get("tac")["mean"]
    for _k in ("ua_t08", "ua_t12"):
        _r = _get(_k)
        NUM[f"uva::gated::{_k}"] = {"bracket_frac_pct": round(100 * (_r["views"] - 3) / 4, 1),
                                    "share_of_tac7_gain_pct": round(100 * (_r["mean"] - _get("tac")["mean"]) / _g7, 1),
                                    "share_of_extra_passes_pct": round(100 * (_r["views"] - 3) / 4, 1)}
# derived timing numbers quoted in Sec. 5.9 (idle-GPU run, c1->c2 seed 42): s/img to 2 decimals, GB to 1 decimal, speed ratios
if TIM:
    for k, t in TIM.items():
        NUM[f"timing::{k}"] = {"s_per_img_2d": round(t["s_per_img"], 2), "s_per_img_3d": round(t["s_per_img"], 3), "GB": round(t["peak_alloc_MB"] / 1024, 1), "wall_s": t.get("wall_s"), "n_img": t.get("n_img"),
                               "teacher_2d": round(t["s_per_img_teacher"], 2) if "s_per_img_teacher" in t else None, "update_2d": round(t["s_per_img_update"], 2) if "s_per_img_update" in t else None}
    NUM["timing::ratios_vs_tac7"] = {k: round(TIM[k]["s_per_img"] / TIM["tac7"]["s_per_img"], 1) for k in ("wbf18_ar", "heatmap18", "cotta")}
    NUM["timing::ratios_vs_tac3"] = {k: round(TIM[k]["s_per_img"] / TIM["tac3"]["s_per_img"], 1) for k in ("wbf18_ar", "heatmap18", "cotta")}

# ------------------------------------------------------------------ first-pass groups
L = []
for g, title, keys in [("ndet", "first-pass candidates ($p_i \\ge 0.3$)", ["0", "1-2", "3-5", ">5"]), ("shift", "oracle scale shift $|\\log f^{\\star}|$", ["|log f*| <= log 1.25", "log 1.25 - log 1.6", "log 1.6 - log 2", "> log 2"])]:
    L.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{grouped by {title}}}}}\\\\")
    for lab in keys:
        v = FP[g][lab]; nice = {"|log f*| <= log 1.25": "$\\le \\log 1.25$ ($f^{\\star}\\in[0.8,1.25]$)", "log 1.25 - log 1.6": "$\\log 1.25$--$\\log 1.6$", "log 1.6 - log 2": "$\\log 1.6$--$\\log 2$", "> log 2": "$> \\log 2$ ($f^{\\star}<0.5$ or $>2$)", "0": "0", "1-2": "1--2", "3-5": "3--5", ">5": "$>5$"}[lab]
        if "tac7" not in v:
            L.append(f"{nice} & {v['n_images']} & \\multicolumn{{7}}{{c}}{{too few images ($<5$ per test set)}} \\\\"); continue
        L.append(f"{nice} & {v['n_images']} & {v['source']:.1f} & {v['tac3']:.1f} & {v['tac7']:.1f} & {v['oracle3']:.1f} & {v['global3']:.1f} & $\\times${math.exp(v['median_abs_log_err']):.2f} & {100 * v['within15']:.0f}\\% \\\\")
        NUM[f"fp::{g}::{lab}"] = v
write("tab_firstpass.tex", L)

# ------------------------------------------------------------------ G. atypical anatomical layout
LD = json.load(open(os.path.join(RES, "layout_deviation.json"), encoding="utf-8")) if os.path.exists(os.path.join(RES, "layout_deviation.json")) else None
if LD:
    L = []
    for lab, r in LD["bins"].items():
        nice = {"most deviant (lowest 10%)": "most atypical (lowest 10\\%)", "10-25%": "10--25\\%", "25-50%": "25--50\\%", "typical (top 50%)": "typical (top 50\\%)"}[lab]
        L.append(f"{nice} & {r['n_images']} & {r['source']:.2f} & {r['tac3']:.2f} & {r['tac3_noar']:.2f} & {r['tac7']:.2f} & {r['tac3_minus_source']:+.2f} & {r['rescoring_effect']:+.2f} \\\\")
        NUM[f"layout::{lab}"] = r
    write("tab_layout.tex", L)
    NUM["layout::n_scored"] = LD["n_images_scored"]

# ------------------------------------------------------------------ D. in-domain
L = []
_IN = [("source model", "in_source"), ("TAC-3 (ours)", "in_tac"), ("TAC-7, fixed bracket (ours)", "in_tac_v7")]
_ib = argbest([R[k]["mean"] for _, k in _IN]); _i5 = argbest([R[k]["ap50"] for _, k in _IN])
for i, (lab, k) in enumerate(_IN):
    r = R[k]; ap50 = f"\\textbf{{{r['ap50']:.2f}}}" if i in _i5 else f"{r['ap50']:.2f}"
    L.append(f"{lab} & {fm(r, i in _ib)} & {ap50} & {'--' if k == 'in_source' else fd(r, R['in_source'])} \\\\")
    NUM[f"indomain::{k}"] = {"mean": r["mean"], "std": r["std"], "d": None if k == "in_source" else list(paired(r, R["in_source"]))}
write("tab_indomain.tex", L)

# ------------------------------------------------------------------ bootstrap / dedup
L = []
NAMES = {"4c": "Heart-4CC", "abdomen": "Abdomen", "spine": "Spine"}
for d, b in BOOT.items():
    cl = b["clusters"]; n_img = sum(v["n_images"] for v in cl.values()); n_cl = sum(v["n_clusters"] for v in cl.values())
    L.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{NAMES[d]} ({n_cl} clusters / {n_img} images)}}}}\\\\")
    # the seven-view configuration reported as TAC-7 is the quadrature bracket (tac7q) where its
    # bootstrap has been run, and the fixed bracket (tac7) otherwise
    _p7 = (lambda ref: ("tac7q-" + ref) if ("tac7q-" + ref) in b["bootstrap"] else ("tac7-" + ref))
    for key, lab in [(_p7("source"), "TAC-7 vs.\\ source"), (_p7("heatmap18"), "TAC-7 vs.\\ T$^3$A"), (_p7("wbf18_ar"), "TAC-7 vs.\\ WBF-18+RS"), (_p7("cotta"), "TAC-7 vs.\\ CoTTA-det"), ("tac3_2r-wbf18_ar", "TAC-3 (2r) vs.\\ WBF-18+RS"), ("tac3_2r-cotta", "TAC-3 (2r) vs.\\ CoTTA-det")]:
        v = b["bootstrap"][key]; x, y = key.split("-"); dd = b["dedup_ap"]
        L.append(f"{lab} & ${v['point']:+.2f}$ & $[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]$ & ${dd[x]['dedup_mean'] - dd[y]['dedup_mean']:+.2f}$ & {('<0.001' if v['p_le_0'] < 0.001 else format(v['p_le_0'], '.3f'))} \\\\")
        NUM[f"boot::{d}::{key}"] = {**v, "dedup_diff": dd[x]["dedup_mean"] - dd[y]["dedup_mean"], "per_dir": None}
    NUM[f"dedup::{d}"] = b["dedup_ap"]; NUM[f"clusters::{d}"] = cl
write("tab_bootstrap.tex", L)

# ------------------------------------------------------------------ E. published defaults
L = []
for m, lab in [("iouf", "IoU-Filter"), ("amrod", "AMROD"), ("whw", "WHW"), ("cdb", "CD-Buffer"), ("sgp", "SGP"), ("osfda", "O-SFDA"), ("vlod", "VLOD-TTA"), ("buftta", "BufferTTA"), ("t3a", "T$^3$A")]:
    p, b = R.get(f"paper_{m}"), R.get("heatmap18" if m == "t3a" else f"bl_{m}")
    if m == "t3a":
        L.append("\\midrule")
    if p:
        L.append(f"{lab} & {fm(p)} & {fd(p, src)} & {fm(b)} & {fd(p, b)} \\\\"); NUM[f"default::{m}"] = {"paper": p["mean"], "selected": b["mean"], "d_src": list(paired(p, src)), "d_sel": list(paired(p, b))}
    else:
        L.append(f"{lab} & \\multicolumn{{2}}{{c}}{{pending}} & {fm(b)} & \\\\")
write("tab_defaults.tex", L)
json.dump(NUM, open(os.path.join(OUTD, "numbers_reviewer.json"), "w"), indent=1); print("numbers_reviewer.json", len(NUM))

# ------------------------------------------------------------------ H. exactness of the posterior spread (Eq. 4 vs the exact 1-D posterior)
UX = json.load(open(os.path.join(RES, "uncert_exact.json"), encoding="utf-8")) if os.path.exists(os.path.join(RES, "uncert_exact.json")) else None
if UX:
    A = UX["all"]
    L = []
    for lab, key in [(r"EM surrogate, \cref{eq:post}$^{\dagger}$", "eq4"),
                     (r"observed information, \cref{eq:obsinfo}", "obs"),
                     (r"exact posterior, numerical", "grid")]:
        v = A["variants"][key]
        L.append("%s & %.4f & %.3f & %.3f & %.4f & %.1f & %.1f \\\\" % (
            lab, v["median_s"], v["median_ratio_to_eq4"], v["p90_ratio_to_eq4"],
            v["h_mad_vs_eq4"], 100 * v["cov68"], 100 * v["cov95"]))
    write("tab_uncert.tex", L)
    NUM["uncert::all"] = A
    NUM["uncert::n_le_2"] = UX.get("n_le_2")
    NUM["uncert::n_ge_5"] = UX.get("n_ge_5")
    NUM["uncert::dirs"] = [{"dir": r["dir"], "n_img": r["n_img"], "n_reproducible": r["n_reproducible"],
                            "frac": round(r["frac_reproducible"], 3), "s0": r["s0"],
                            "gate": r["gate_max_abs_err"]} for r in UX["per_direction"]]
    NUM["uncert::n_images"] = A["n"]
    NUM["uncert::n_dirs_used"] = sum(1 for r in UX["per_direction"] if r["frac_reproducible"] > 0.5)
    # share of the full-acquisition gain captured by the cheapest gated setting, and its share of the extra cost
    if "uva::ua_t12" in NUM and "uva::tac_v7" in NUM:
        g_full = NUM["uva::tac_v7"]["d_ref"][0]; g_gate = NUM["uva::ua_t12"]["d_ref"][0]
        v3, v7, vg = NUM["uva::tac"]["views"], NUM["uva::tac_v7"]["views"], NUM["uva::ua_t12"]["views"]
        NUM["uncert::gate_gain_share"] = round(100 * g_gate / g_full, 0)
        NUM["uncert::gate_cost_share"] = round(100 * (vg - v3) / (v7 - v3), 0)
json.dump(NUM, open(os.path.join(OUTD, "numbers_reviewer.json"), "w"), indent=1); print("numbers_reviewer.json", len(NUM))

# ------------------------------------------------------------------ I. does the spread identify where extra views pay off?
UQ = json.load(open(os.path.join(RES, "uva_quartiles.json"), encoding="utf-8")) if os.path.exists(os.path.join(RES, "uva_quartiles.json")) else None
if UQ:
    L = []
    for lab, v in UQ["bins"].items():
        nice = lab.replace("Q1 (narrowest)", "Q1 (narrowest $s$)").replace("Q4 (widest)", "Q4 (widest $s$)")
        L.append("%s & %d & %.3f & %.1f & %.3f & %.2f & %.2f & %.2f & %+.2f \\\\" % (
            nice, v["n_images"], v["mean_s"], v["mean_ndet"], v["mean_abs_log_err"],
            v["source"], v["tac3"], v["tac7"], v["delta_tac7_tac3"]))
    write("tab_uvaq.tex", L)
    NUM["uvaq::bins"] = UQ["bins"]
    NUM["uvaq::q4_minus_q1"] = UQ["q4_minus_q1"]
    _q = list(UQ["bins"].values())
    NUM["uvaq::source_q1_minus_q4"] = round(_q[0]["source"] - _q[-1]["source"], 2)
    NUM["uvaq::n_positive_cells"] = int(round(UQ["q4_minus_q1"]["frac_positive"] * UQ["q4_minus_q1"]["n_cells"]))
json.dump(NUM, open(os.path.join(OUTD, "numbers_reviewer.json"), "w"), indent=1); print("numbers_reviewer.json", len(NUM))

# ------------------------------------------------------------------ J. equal-budget gating control
GC = json.load(open(os.path.join(RES, "gating_control.json"), encoding="utf-8")) if os.path.exists(os.path.join(RES, "gating_control.json")) else None
if GC:
    NUM["gate::rules"] = {k: round(v["mean"], 2) for k, v in GC["rules"].items()}
    base = GC["rules"]["none"]["mean"]; full = GC["rules"]["all"]["mean"]
    NUM["gate::gain"] = {k: round(v["mean"] - base, 2) for k, v in GC["rules"].items() if k not in ("none",)}
    NUM["gate::share"] = {k: round(100 * (v["mean"] - base) / (full - base), 0) for k, v in GC["rules"].items() if k not in ("none",)}
    for k, v in GC.items():
        if k.startswith("top_minus"):
            NUM["gate::" + k] = {"mean": round(v["mean"], 2), "ci95_half": round(v["ci95_half"], 2),
                                 "n_cells": v["n_cells"],
                                 "n_positive": int(round(v["frac_positive"] * v["n_cells"]))}
json.dump(NUM, open(os.path.join(OUTD, "numbers_reviewer.json"), "w"), indent=1); print("numbers_reviewer.json", len(NUM))
