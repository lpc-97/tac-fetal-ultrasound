# -*- coding: utf-8 -*-
"""Table bodies for the reviewer experiments, generated from results/reviewer/*.json (no number typed by hand):
  tab_scalechain.tex  A. source / val-selected scale / stream-global factor / TAC factor / oracle factor, single view and TAC-3 pipeline
  tab_estimator.tex   B. scale estimator inside TAC-3
  tab_blind.tex       C. equal-compute blind ensembles vs TAC (+ time / peak memory from the idle-GPU timing run when available)
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
FP = json.load(open(os.path.join(RES, "firstpass_analysis.json"), encoding="utf-8"))
BOOT = {d: json.load(open(os.path.join(RES, f"cluster_bootstrap_{d}.json"), encoding="utf-8")) for d in ("4c", "abdomen", "spine")}
TIM = json.load(open(os.path.join(RES, "timing.json"), encoding="utf-8")) if os.path.exists(os.path.join(RES, "timing.json")) else {}
NUM = {}


def paired(a, b):
    d = [a["m6"][s] - b["m6"][s] for s in a["m6"] if s in b["m6"]]; mu = st.mean(d); half = T975 * st.stdev(d) / math.sqrt(len(d))
    return mu, half, abs(mu) > half


def fd(a, b):
    mu, half, sig = paired(a, b); return f"{mu:+.2f}\\,$\\pm$\\,{half:.2f}{'$^{*}$' if sig else ''}"


def fm(r, bold=False):
    t = f"{r['mean']:.2f}\\,$\\pm$\\,{r['std']:.2f}"; return f"\\textbf{{{t}}}" if bold else t


def write(name, lines):
    open(os.path.join(OUTD, name), "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("wrote", name)


src, orc1, orc3 = R["source"], R["fac_oracle_1v"], R["fac_oracle"]
# ------------------------------------------------------------------ A. scale chain
L = []
for lab, k1, k3 in [("default scale ($s_0$)", "source", None), ("val-selected fixed scale per direction", "fac_valscale_1v", "fac_valscale"), ("stream-level factor (median of TAC factors)", "fac_global_1v", "fac_global"),
                    ("\\textbf{TAC per-image factor}", "tac_1v", "tac"), ("oracle per-image factor (GT sizes)", "fac_oracle_1v", "fac_oracle")]:
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
for lab, k in [("median of the votes (no weights, no prior)", "est_median"), ("confidence-weighted mean of votes", "est_confmean"), ("IRLS, class-agnostic (pooled) prior", "est_noclass"), ("\\textbf{IRLS, per-structure prior (TAC)}", "tac")]:
    r = R[k]; L.append(f"{lab} & {fm(r, k == 'tac')} & {r['ap50']:.2f} & {'--' if k == 'tac' else fd(r, R['tac'])} \\\\")
    NUM[f"est::{k}"] = {"mean": r["mean"], "std": r["std"], "d_tac": None if k == "tac" else list(paired(r, R["tac"]))}
write("tab_estimator.tex", L)

# ------------------------------------------------------------------ C. blind ensembles (+ timing)
def tcell(key):
    t = TIM.get(key)
    return ("", "") if not t else (f"{t['s_per_img']:.2f}", f"{t['peak_alloc_MB'] / 1024:.1f}")


L = []
for lab, k, ref, tk in [("blind: 3 scales \\{600, 800, 1000\\}, WBF", "ss_fix3_wbf", "tac", None), ("blind: 3 scales, WBF + re-scoring", "ss_fix3_wbf_ar", "tac", "blind3_ar"),
                        ("\\textbf{TAC-3} (ours)", "tac", None, "tac3"), ("\\textbf{TAC-3, 2 rounds} (ours)", "tac_rounds2", None, "tac3_2r"),
                        ("blind: 3 scales $\\times$ flip, WBF + re-scoring", "ss_fix3f_wbf_ar", "tac_v7", None),
                        ("blind: 7 scales \\{500--1200\\}, WBF", "ss_fix7_wbf", "tac_v7", None), ("blind: 7 scales, WBF + re-scoring", "ss_fix7_wbf_ar", "tac_v7", "blind7_ar"),
                        ("\\textbf{TAC-7} (ours)", "tac_v7", None, "tac7"), ("\\textbf{TAC-11} (ours)", "tac_v11", None, "tac11"),
                        ("18-view WBF + re-scoring", "ss_all_flip_wbf_ar", None, "wbf18_ar"), ("18-view heat-map fusion", "heatmap18", None, "heatmap18")]:
    r = R[k]; v = r["views"] if r["views"] else 18; sp, mem = tcell(tk)
    L.append(f"{lab} & {v:.0f} & {fm(r, 'ours' in lab)} & {r['ap50']:.2f} & {'--' if ref is None else fd(r, R[ref])} & {sp} & {mem} \\\\")
    NUM[f"blind::{k}"] = {"mean": r["mean"], "std": r["std"], "views": v, "d_ref": None if ref is None else list(paired(r, R[ref])), "timing": TIM.get(tk)}
write("tab_blind.tex", L)

# ------------------------------------------------------------------ F. uncertainty-guided view acquisition (UVA)
UV = json.load(open(os.path.join(RES, "uncert_summary.json"), encoding="utf-8"))["test"] if os.path.exists(os.path.join(RES, "uncert_summary.json")) else {}
if UV:
    def urow(lab, key, ref, bold=False):
        r = UV[key] if key in UV else R[key]
        d = "--" if ref is None else fd(r, UV[ref] if ref in UV else R[ref])
        t = fm(r, bold)
        NUM[f"uva::{key}"] = {"mean": r["mean"], "std": r["std"], "views": r["views"], "d_ref": None if ref is None else list(paired(r, UV[ref] if ref in UV else R[ref]))}
        return f"{lab} & {r['views']:.2f} & {t} & {r['ap50']:.2f} & {d} \\\\"
    L = [urow("\\tac-3 (no bracket)", "tac", None),
         urow("\\tac-7, fixed bracket $\\pm15\\%$", "tac_v7", "tac"),
         urow("\\tac-11, fixed bracket", "tac_v11", "tac_v7"),
         "\\midrule",
         urow("\\textbf{UVA, quadrature bracket}", "u7_k17", "tac_v7", True),
         urow("UVA, gated ($\\tau_s=0.08$)", "ua_t08", "tac_v7"),
         urow("UVA, gated ($\\tau_s=0.12$)", "ua_t12", "tac")]
    write("tab_uva.tex", L)
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
for lab, k in [("source model", "in_source"), ("\\textbf{TAC-3}", "in_tac"), ("\\textbf{TAC-7}", "in_tac_v7")]:
    r = R[k]; L.append(f"{lab} & {fm(r, k != 'in_source')} & {r['ap50']:.2f} & {'--' if k == 'in_source' else fd(r, R['in_source'])} \\\\")
    NUM[f"indomain::{k}"] = {"mean": r["mean"], "std": r["std"], "d": None if k == "in_source" else list(paired(r, R["in_source"]))}
write("tab_indomain.tex", L)

# ------------------------------------------------------------------ bootstrap / dedup
L = []
NAMES = {"4c": "Heart-4CC", "abdomen": "Abdomen", "spine": "Spine"}
for d, b in BOOT.items():
    cl = b["clusters"]; n_img = sum(v["n_images"] for v in cl.values()); n_cl = sum(v["n_clusters"] for v in cl.values())
    L.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{NAMES[d]} ({n_cl} clusters / {n_img} images)}}}}\\\\")
    for key, lab in [("tac7-source", "TAC-7 vs.\\ source"), ("tac7-heatmap18", "TAC-7 vs.\\ HM-18"), ("tac7-wbf18_ar", "TAC-7 vs.\\ WBF-18+RS"), ("tac7-cotta", "TAC-7 vs.\\ CoTTA-det"), ("tac3_2r-wbf18_ar", "TAC-3 (2r) vs.\\ WBF-18+RS"), ("tac3_2r-cotta", "TAC-3 (2r) vs.\\ CoTTA-det")]:
        v = b["bootstrap"][key]; x, y = key.split("-"); dd = b["dedup_ap"]
        L.append(f"{lab} & {v['point']:+.2f} & [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] & {dd[x]['dedup_mean'] - dd[y]['dedup_mean']:+.2f} & {v['p_le_0']:.3f} \\\\")
        NUM[f"boot::{d}::{key}"] = {**v, "dedup_diff": dd[x]["dedup_mean"] - dd[y]["dedup_mean"], "per_dir": None}
    NUM[f"dedup::{d}"] = b["dedup_ap"]; NUM[f"clusters::{d}"] = cl
write("tab_bootstrap.tex", L)

# ------------------------------------------------------------------ E. published defaults
L = []
for m, lab in [("iouf", "IoU-Filter"), ("amrod", "AMROD"), ("whw", "WHW"), ("cdb", "CD-Buffer"), ("sgp", "SGP"), ("osfda", "O-SFDA"), ("vlod", "VLOD-TTA"), ("buftta", "BufferTTA")]:
    p, b = R.get(f"paper_{m}"), R.get(f"bl_{m}")
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
