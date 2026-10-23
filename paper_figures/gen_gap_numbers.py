"""Derived quantities quoted in the abstract, highlights and introduction: the cross-center loss (in-domain minus cross-domain
source AP on the Heart-4CC test splits) and the fraction of that loss each method or oracle recovers. Writes
tables/numbers_gap.json so that every percentage in the front matter traces to the measured APs it is computed from.
Sources: tables/numbers.json (main tables), tables/numbers_reviewer.json (oracle / in-domain rows),
results/per_class_ap_2026-09-18.json (per-structure gains)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "..", "tables")
N = json.load(open(os.path.join(T, "numbers.json"), encoding="utf-8"))
R = json.load(open(os.path.join(T, "numbers_reviewer.json"), encoding="utf-8"))
PC = json.load(open(os.path.join(HERE, "..", "..", "results", "per_class_ap_2026-09-18.json"), encoding="utf-8"))
PC = PC.get("per_class", PC)

src_x = R["chain::source"]["mean"]          # cross-domain source, 6 directions x 3 seeds
src_in = R["indomain::in_source"]["mean"]   # in-domain source, 3 centers x 3 seeds
gap = src_in - src_x
out = {"source_cross_domain": round(src_x, 2), "source_in_domain": round(src_in, 2), "cross_centre_loss": round(gap, 2)}


def closes(ap, name):
    out[name] = {"AP": round(ap, 2), "delta": round(ap - src_x, 2), "pct_of_loss": round(100 * (ap - src_x) / gap, 1)}
    return out[name]


closes(R["chain::fac_oracle_1v"]["mean"], "oracle_scale_single_view")
closes(R["chain3::fac_oracle"]["mean"], "oracle_scale_tac3_pipeline")
closes(R["chain::tac_1v"]["mean"], "tac_factor_single_view")
closes(R["chain::fac_valscale_1v"]["mean"], "best_fixed_scale_single_view")
closes(R["chain::fac_global_1v"]["mean"], "stream_level_factor_single_view")
closes(R["blind::tac"]["mean"], "tac3")
# the fixed +-15% bracket (control row of the view-count ablation); the reviewer generator no longer
# exports it as blind::tac_v7, since the equal-compute table now carries the quadrature bracket
closes(N["views::TAC 7 views (+-15% scales)"][0], "tac7_fixed_bracket")
# the configuration reported as TAC-7: the surrogate-guided quadrature bracket
closes(N["main_TAC 7 views (quadrature)"]["mean"], "tac7")
# best evaluated parameter-adaptive baseline on the frozen-BN heart models
best_pa = max((v["selected"], m) for m, v in ((m, R[f"default::{m}"]) for m in ("iouf", "amrod", "whw", "cdb", "sgp", "osfda", "vlod", "buftta")))
out["best_parameter_adaptive_baseline"] = {"method": best_pa[1], **closes(best_pa[0], "_tmp")}
del out["_tmp"]
# fraction of the oracle's gain recovered by the inferred factor, at equal view budget
out["tac_over_oracle_single_view"] = round(100 * (R["chain::tac_1v"]["mean"] - src_x) / (R["chain::fac_oracle_1v"]["mean"] - src_x), 1)
out["tac_over_oracle_tac3_pipeline"] = round(100 * (R["blind::tac"]["mean"] - src_x) / (R["chain3::fac_oracle"]["mean"] - src_x), 1)
# forward passes saved against the 18-view ensembles
out["passes_saved_vs_18view_pct"] = round(100 * (18 - 7) / 18, 1)
# per-structure gains of TAC (2 rounds) over the source detector, all planes
ps = {}
for dsn, d in PC.items():
    src = d["src"]; names = [k for k in src if not k.startswith("_")]
    dl = {k.split(":")[-1]: round(d["tac"][k]["AP"] - src[k]["AP"], 2) for k in names}
    ps[dsn] = {"n_structures": len(dl), "min_gain": min(dl.values()), "n_negative": sum(v <= 0 for v in dl.values()), "gains": dl}
out["per_structure_gains"] = ps
# per-structure gains of T^3A over the source detector, from its val-selected re-run (results/per_class_ap_t3a_2026-10-22.json)
PH = json.load(open(os.path.join(HERE, "..", "..", "results", "per_class_ap_t3a_2026-10-22.json"), encoding="utf-8"))
PH = PH.get("per_class", PH)
out["per_structure_gains_t3a"] = {dsn: {k.split(":")[-1]: round(d["heat"][k]["AP"] - d["src"][k]["AP"], 2) for k in d["src"] if not k.startswith("_")}
                                  for dsn, d in PH.items()}
out["n_structures_total"] = sum(v["n_structures"] for v in ps.values())
out["n_structures_negative"] = sum(v["n_negative"] for v in ps.values())
# quantities quoted in the text whose only earlier record was a run log or a raw result file (traced 2026-10-22)
RES = os.path.join(HERE, "..", "..", "results")
QC = json.load(open(os.path.join(RES, "qc_collect_2026-09-18.json"), encoding="utf-8"))
PROV = os.path.join(RES, "reviewer", "provenance_2026-10-22")
aux = {}
# second round: share of test images on which it is triggered (each trigger adds two passes), pooled over cells
for t in ("heart4cc_frcnn", "heart4cc_retinanet", "abdomen", "spine"):
    c = QC["tables"][t]["TAC (2 rounds)"].values()
    aux[f"round2_trigger_pct::{t}"] = round(100 * sum((x["views"] - 3) / 2 * x["n_img"] for x in c) / sum(x["n_img"] for x in c), 1)
# per-structure log-size standard deviations of the heart priors
sig = [v["logsize_std"] for p in ("c1", "c2", "c3")
       for v in json.load(open(os.path.join(RES, "paper_aux", f"anat_prior_{p}.json"), encoding="utf-8"))["canonical_at_800"].values()]
aux["prior_sigma_range_heart"] = [round(min(sig), 2), round(max(sig), 2)]
# mean validation AP over the six heart directions of the frozen-BN and the BN source models (Appendix E)
srconly = [e["AP"] for e in QC["val_sweeps"]["4c"] if e["folder"] == "scalesel" and e["variant"] == "srconly"]
aux["val_source_frozenbn"] = round(sum(srconly) / len(srconly), 2)
SW = json.load(open(os.path.join(RES, "sweep_best_lr_2026-09-09.json"), encoding="utf-8"))
bn = {k.split("_", 1)[1]: v["src_val_AP"] for k, v in SW.items()}
aux["val_source_bn"] = round(sum(bn.values()) / len(bn), 2)
# median inferred factors of the TAC runs on the abdomen and the spine (server canon_stats.json, copied)
fm = json.load(open(os.path.join(PROV, "f_median_abdomen_spine_tac_runs.json"), encoding="utf-8"))
aux["f_median_range_abdomen_spine"] = [round(min(fm.values()), 2), round(max(fm.values()), 2)]
# largest per-image factor of the TAC-3 timing run (c1->c2, seed 42) and O-SFDA key frames with its published setting
tf = json.load(open(os.path.join(PROV, "timing_tac3_canon_factors.json"), encoding="utf-8"))["factors"]
tf = [x[0] if isinstance(x, list) else x for x in (tf.values() if isinstance(tf, dict) else tf)]
aux["timing_tac3_max_factor"] = round(max(tf), 2)
os_ = json.load(open(os.path.join(PROV, "osfda_paper_s42_c1_to_c2_test_stats.json"), encoding="utf-8"))
aux["osfda_paper_c1c2_key_frames"] = [int(os_["key_frames"]), int(os_["n_img"])]
out["provenance_aux"] = aux
json.dump(out, open(os.path.join(T, "numbers_gap.json"), "w"), indent=1)
for k, v in out.items():
    if k not in ("per_structure_gains", "per_structure_gains_t3a"):
        print(f"{k:34s} {v}")
print("per-structure:", {k: (v["n_structures"], v["min_gain"], v["n_negative"]) for k, v in ps.items()})
