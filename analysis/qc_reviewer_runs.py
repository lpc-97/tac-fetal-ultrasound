# -*- coding: utf-8 -*-
"""Integrity check of every run behind the reviewer tables (results/reviewer/*): for each run directory read config.yaml and the
launcher log and verify (i) DATASETS.TEST is the target's test split, (ii) MODEL.WEIGHTS is the source model of that seed,
(iii) the number of evaluated images equals the size of the target test split, (iv) result_ap.txt exists.
Writes output/analysis/qc_reviewer_runs.json and prints a summary."""
import glob
import json
import os
import re

import yaml

os.chdir(os.environ.get("TAC_ROOT", "."))
SEEDS = ["42", "43", "44"]; DOMS = ["c1", "c2", "c3"]
N_TEST = {"c1": 162, "c2": 235, "c3": 179}  # Table 1 of the paper (images per test split)
CANON = ["tac", "tac_rounds2", "tac_v7", "tac_v11", "tac_1v", "est_median", "est_confmean", "est_noclass",
         "fac_oracle", "fac_oracle_1v", "fac_global", "fac_global_1v", "fac_valscale", "fac_valscale_1v"]
SSEL = ["ss_fix3_wbf", "ss_fix3_wbf_ar", "ss_fix7_wbf", "ss_fix7_wbf_ar", "ss_fix3f_wbf_ar", "ss_all_flip_wbf", "ss_all_flip_wbf_ar"]
PAPER = ["iouf_ep_lr1e-3", "amrod_lr1e-3", "whw_lr1e-3", "cdb_paper", "sgp_paper", "osfda_paper", "vlod_lr1e-3", "buftta_paper"]

runs = []
for s in SEEDS:
    for a in DOMS:
        for b in DOMS:
            if a != b:
                runs += [(f"canon/{v}", f"output/canon/{v}_s{s}_{a}_to_{b}_test", s, a, b) for v in CANON]
                runs += [(f"scalesel/{v}", f"output/scalesel/{v}_s{s}_{a}_to_{b}_test", s, a, b) for v in SSEL]
                runs += [(f"baselines/{v}", f"output/baselines/{v}_s{s}_{a}_to_{b}_test", s, a, b) for v in PAPER]
                runs.append(("source", f"output/so_fz_s{s}_{a}_to_{b}", s, a, b))
            else:
                runs += [(f"indomain/{v}", f"output/canon/{v}_s{s}_{a}_to_{b}_test", s, a, b) for v in ["tac", "tac_v7"]]
                runs.append(("indomain/source", f"output/so_fz_s{s}_{a}_to_{b}", s, a, b))


def n_images(d):
    for lg in [d.rstrip("/") + ".log", os.path.join(d, "log.txt")]:
        if os.path.exists(lg):
            m = re.findall(r"Start inference on (\d+) batches", open(lg, errors="ignore").read())
            if m:
                return int(m[-1])
    for st in glob.glob(os.path.join(d, "*_stats.json")):
        j = json.load(open(st))
        if "n_img" in j:
            return int(j["n_img"])
    return None


out = []; bad = []
for group, d, s, a, b in runs:
    rec = {"group": group, "dir": d, "seed": s, "src": a, "tgt": b, "exists": os.path.isdir(d)}
    if not rec["exists"]:
        bad.append((d, "missing dir")); out.append(rec); continue
    rec["result"] = os.path.exists(os.path.join(d, "result_ap.txt"))
    cfgf = os.path.join(d, "config.yaml")
    if os.path.exists(cfgf):
        cfg = yaml.safe_load(open(cfgf))
        test = cfg.get("DATASETS", {}).get("TEST", []); test = list(test) if not isinstance(test, str) else [test]
        w = str(cfg.get("MODEL", {}).get("WEIGHTS", ""))
        rec["test"] = test; rec["weights"] = w
        rec["ok_split"] = len(test) == 1 and (f"_{b}_test" in test[0] or test[0].endswith(f"{b}_test"))
        rec["ok_weights"] = f"fs_{a}_fz_s{s}" in w or (s == "42" and f"fs_{a}_fz" in w and "_s4" not in w)
    else:
        rec["ok_split"] = rec["ok_weights"] = None
    n = n_images(d); rec["n_img"] = n; rec["ok_n"] = (n == N_TEST[b]) if n is not None else None
    if not rec["result"] or rec["ok_split"] is False or rec["ok_weights"] is False or rec["ok_n"] is False:
        bad.append((d, {k: rec.get(k) for k in ("result", "ok_split", "ok_weights", "ok_n", "test", "weights", "n_img")}))
    out.append(rec)

json.dump(out, open("output/analysis/qc_reviewer_runs.json", "w"), indent=1)
groups = {}
for r in out:
    g = groups.setdefault(r["group"], [0, 0, 0, 0]); g[0] += 1
    g[1] += bool(r.get("result")); g[2] += bool(r.get("ok_split") and r.get("ok_weights")); g[3] += bool(r.get("ok_n"))
print(f"{'group':26s} runs result cfg_ok n_ok")
for g, (n, r, c, k) in sorted(groups.items()):
    print(f"{g:26s} {n:4d} {r:6d} {c:6d} {k:4d}")
print("total runs", len(out), "| problems", len(bad))
for d, why in bad[:25]:
    print("  BAD", d, why)
noinfo = [r["dir"] for r in out if r.get("exists") and (r.get("ok_split") is None or r.get("ok_n") is None)]
print("runs without config or image count:", len(noinfo), noinfo[:8])
print("QC_REVIEWER_RUNS_DONE")
