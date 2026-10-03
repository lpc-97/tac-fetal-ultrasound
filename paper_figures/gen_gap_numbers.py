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
closes(R["blind::tac_v7"]["mean"], "tac7")
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
out["n_structures_total"] = sum(v["n_structures"] for v in ps.values())
out["n_structures_negative"] = sum(v["n_negative"] for v in ps.values())
json.dump(out, open(os.path.join(T, "numbers_gap.json"), "w"), indent=1)
for k, v in out.items():
    if k not in ("per_structure_gains",):
        print(f"{k:34s} {v}")
print("per-structure:", {k: (v["n_structures"], v["min_gain"], v["n_negative"]) for k, v in ps.items()})
