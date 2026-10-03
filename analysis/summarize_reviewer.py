# -*- coding: utf-8 -*-
"""Summaries of the reviewer experiments (Heart-4CC, Faster R-CNN, TEST, seeds 42/43/44, 6 directions):
  A. scale-shift evidence chain: source @ default | val-selected fixed scale | stream-global factor | TAC per-image factor | ORACLE factor,
     each as a single view and inside the TAC-3 pipeline (pass 1 + canonical + flip, WBF, re-scoring)
  B. scale-estimator ablation inside TAC-3: median-of-votes | confidence-weighted mean | class-agnostic prior | IRLS (default)
  C. equal-compute blind ensembles: fixed 3 / 7 scales (+ re-scoring) vs TAC-3 / TAC-7, with views, s/img and peak GPU memory
  D. in-domain control (source centre = target centre): source vs TAC-3 vs TAC-7
  E. published-default baseline configurations vs the val-selected ones (test, 3 seeds)
Writes output/analysis/reviewer_summary.{json,md}."""
import glob
import json
import math
import os
import statistics as st

os.chdir(os.environ.get("TAC_ROOT", "."))
SEEDS = [42, 43, 44]; DOMS = ["c1", "c2", "c3"]; DIRS = [(a, b) for a in DOMS for b in DOMS if a != b]; T975 = 4.303


def read_ap(d):
    f = os.path.join(d, "result_ap.txt")
    if not os.path.exists(f):
        return None
    return json.loads(open(f).read().strip().splitlines()[-1])


def stats_of(d):
    for n in ("canon_stats.json", "scalesel_stats.json", "iouf_stats.json", "amrod_stats.json", "distill_stats.json", "whw_stats.json", "cdb_stats.json", "sgp_stats.json", "osfda_stats.json", "vlod_stats.json", "buftta_stats.json"):
        p = os.path.join(d, n)
        if os.path.exists(p):
            return json.load(open(p))
    return {}


def mem_of(d):
    p = os.path.join(d, "gpu_mem.json")
    return json.load(open(p))["peak_alloc_MB"] if os.path.exists(p) else None


def row(pat, dirs=DIRS):
    """pat(s, a, b) -> dir; returns per-seed 6-dir means (AP, AP50), mean views, s/img, mem"""
    m6, m50, views, spi, mem = {}, {}, [], [], []
    for s in SEEDS:
        aps = []; ap50 = []
        for a, b in dirs:
            r = read_ap(pat(s, a, b))
            if r is None:
                return None
            aps.append(r["AP"]); ap50.append(r["AP50"]); stt = stats_of(pat(s, a, b))
            if stt.get("views_per_img") is not None: views.append(stt["views_per_img"])
            if stt.get("s_per_img") is not None: spi.append(stt["s_per_img"])
            mm = mem_of(pat(s, a, b))
            if mm: mem.append(mm)
        m6[s] = st.mean(aps); m50[s] = st.mean(ap50)
    return {"m6": m6, "mean": st.mean(m6.values()), "std": st.pstdev(m6.values()), "ap50": st.mean(m50.values()),
            "views": st.mean(views) if views else None, "spi": st.mean(spi) if spi else None, "mem": st.mean(mem) if mem else None}


def paired(a, b):
    d = [a["m6"][s] - b["m6"][s] for s in SEEDS]; mu = st.mean(d); half = T975 * st.stdev(d) / math.sqrt(3)
    return [mu, half, abs(mu) > half]


def fmt(r):
    return "n/a" if r is None else f"{r['mean']:.2f} ± {r['std']:.2f}"


def fd(p):
    return f"{p[0]:+.2f} ± {p[1]:.2f}{'*' if p[2] else ''}"


canon = lambda v: (lambda s, a, b: f"output/canon/{v}_s{s}_{a}_to_{b}_test")
ssel = lambda v: (lambda s, a, b: f"output/scalesel/{v}_s{s}_{a}_to_{b}_test")
R = {}
R["source"] = row(lambda s, a, b: f"output/so_fz_s{s}_{a}_to_{b}")
R["heatmap18"] = row(lambda s, a, b: f"output/tta_heatmap_s{s}_{a}_to_{b}")
for v in ["tac", "tac_rounds2", "tac_v7", "tac_v11", "tac_flast", "tac_1v", "est_median", "est_confmean", "est_noclass", "fac_oracle", "fac_oracle_1v", "fac_global", "fac_global_1v", "fac_valscale", "fac_valscale_1v"]:
    R[v] = row(canon(v))
for v in ["ss_fix3_wbf", "ss_fix3_wbf_ar", "ss_fix7_wbf", "ss_fix7_wbf_ar", "ss_fix3f_wbf_ar", "ss_all_flip_wbf", "ss_all_flip_wbf_ar"]:
    R[v] = row(ssel(v))
best = json.load(open("output/baselines/best_val.json"))
for m in ["iouf", "amrod", "whw", "cdb", "sgp", "osfda", "vlod", "buftta", "cotta"]:
    R[f"bl_{m}"] = row(lambda s, a, b, m=m: f"output/baselines/{best[m][f'{a}_to_{b}']}_s{s}_{a}_to_{b}_test")
PAPER = {"iouf": "iouf_ep_lr1e-3", "amrod": "amrod_lr1e-3", "whw": "whw_lr1e-3", "cdb": "cdb_paper", "sgp": "sgp_paper", "osfda": "osfda_paper", "vlod": "vlod_lr1e-3", "buftta": "buftta_paper"}
for m, v in PAPER.items():
    R[f"paper_{m}"] = row(lambda s, a, b, v=v: f"output/baselines/{v}_s{s}_{a}_to_{b}_test")
IND = [(c, c) for c in DOMS]
R["in_source"] = row(lambda s, a, b: f"output/so_fz_s{s}_{a}_to_{b}", IND)
R["in_tac"] = row(canon("tac"), IND); R["in_tac_v7"] = row(canon("tac_v7"), IND)

md = []; out = {"rows": {k: v for k, v in R.items() if v}}
src, orc, orc1 = R["source"], R["fac_oracle"], R["fac_oracle_1v"]
md.append("## A. Is scale the main shift? (single view = 1 output view; TAC-3 pipeline = pass 1 + canonical + flip, WBF, re-scoring)\n")
md.append("| scale used | single view AP | Δ vs source | recovered gap (oracle=100%) | TAC-3 pipeline AP | Δ vs source | recovered gap |\n|---|---|---|---|---|---|---|")
for lab, k1, k3 in [("default scale (800 / s0)", "source", None), ("val-selected fixed scale per direction", "fac_valscale_1v", "fac_valscale"), ("stream-global factor (median of TAC factors)", "fac_global_1v", "fac_global"), ("TAC per-image factor (ours)", "tac_1v", "tac"), ("ORACLE per-image factor (GT sizes)", "fac_oracle_1v", "fac_oracle")]:
    a = R.get(k1); c = R.get(k3) if k3 else None
    g1 = f"{100 * (a['mean'] - src['mean']) / (orc1['mean'] - src['mean']):.0f}%" if a and orc1 and k1 != "source" else "–"
    g3 = f"{100 * (c['mean'] - src['mean']) / (orc['mean'] - src['mean']):.0f}%" if c and orc else "–"
    md.append(f"| {lab} | {fmt(a)} | {fd(paired(a, src)) if a and k1 != 'source' else '–'} | {g1} | {fmt(c)} | {fd(paired(c, src)) if c else '–'} | {g3} |")
if R.get("tac_flast"):
    md.append(f"\n(TAC canonical view only, older run `tac_flast`: {fmt(R['tac_flast'])}; TAC-3 2 rounds {fmt(R['tac_rounds2'])}; TAC-7 {fmt(R['tac_v7'])})")
md.append("\n## B. Scale estimator inside TAC-3 (same views and fusion)\n\n| estimator | AP | Δ vs IRLS (ours) | views | s/img |\n|---|---|---|---|---|")
for lab, k in [("median of per-candidate votes (heuristic)", "est_median"), ("confidence-weighted mean of votes", "est_confmean"), ("IRLS with class-agnostic (pooled) prior", "est_noclass"), ("IRLS with per-structure prior + outlier mixture + shrinkage (TAC)", "tac")]:
    r = R.get(k); md.append(f"| {lab} | {fmt(r)} | {fd(paired(r, R['tac'])) if r and k != 'tac' else '–'} | {r['views']:.1f} | {r['spi']:.2f} |" if r else f"| {lab} | n/a | | | |")
md.append("\n## C. Equal-compute comparison: blind fixed multi-scale ensembles vs TAC\n\n| method | views | AP | AP50 | Δ vs TAC with same #views | s/img | peak GPU mem (MB) |\n|---|---|---|---|---|---|---|")
for lab, k, ref in [("blind 3 scales {600,800,1000}, WBF", "ss_fix3_wbf", "tac"), ("blind 3 scales + re-scoring", "ss_fix3_wbf_ar", "tac"), ("blind 3 scales x flip (6 views) + re-scoring", "ss_fix3f_wbf_ar", "tac_v7"), ("TAC-3 (ours)", "tac", None), ("TAC-3, 2 rounds (ours)", "tac_rounds2", None),
                    ("blind 7 scales {500..1200}, WBF", "ss_fix7_wbf", "tac_v7"), ("blind 7 scales + re-scoring", "ss_fix7_wbf_ar", "tac_v7"), ("TAC-7 (ours)", "tac_v7", None), ("18-view WBF", "ss_all_flip_wbf", None), ("18-view WBF + re-scoring", "ss_all_flip_wbf_ar", None), ("18-view heat-map", "heatmap18", None), ("TAC-11 (ours)", "tac_v11", None)]:
    r = R.get(k)
    if not r:
        md.append(f"| {lab} | | n/a | | | | |"); continue
    md.append(f"| {lab} | {r['views'] if r['views'] is None else round(r['views'], 1)} | {fmt(r)} | {r['ap50']:.2f} | {fd(paired(r, R[ref])) if ref and R.get(ref) else '–'} | {'' if r['spi'] is None else f'{r[chr(115)+chr(112)+chr(105)]:.2f}'} | {'' if r['mem'] is None else f'{r[chr(109)+chr(101)+chr(109)]:.0f}'} |")
md.append("\n## D. In-domain control (source centre = target centre; physiological size variation only)\n\n| row | AP (3 centres x 3 seeds) | Δ vs in-domain source |\n|---|---|---|")
for lab, k in [("source model", "in_source"), ("TAC-3", "in_tac"), ("TAC-7", "in_tac_v7")]:
    r = R.get(k); md.append(f"| {lab} | {fmt(r)} | {fd(paired(r, R['in_source'])) if r and k != 'in_source' else '–'} |")
md.append("\n## E. Baselines with their published default configuration vs the val-selected configuration (test, 3 seeds)\n\n| method | published default | val-selected (paper table) | Δ default − selected |\n|---|---|---|---|")
for m in PAPER:
    p, b = R.get(f"paper_{m}"), R.get(f"bl_{m}")
    md.append(f"| {m} | {fmt(p)} | {fmt(b)} | {fd(paired(p, b)) if p and b else 'pending'} |")
text = "\n".join(md); print(text)
json.dump(out, open("output/analysis/reviewer_summary.json", "w"), indent=1); open("output/analysis/reviewer_summary.md", "w").write(text + "\n")
print("REVIEWER_SUMMARY_DONE")
