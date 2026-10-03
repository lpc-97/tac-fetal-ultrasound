#!/usr/bin/env python
"""Exactness and calibration of the ANI posterior spread (paper Eq. (4) vs. the exact 1-D posterior).

CPU only. The first-pass candidates are read from the dumps written by the p1dump runs
(patch_canon5.py, SEMISUPNET.CANON.DUMP_PASS1), so the analysis sees exactly the quantities
estimate_theta consumed at deployment, for every test image of every direction. No reconstruction
from a separate source-only evaluation is involved, and no image is excluded.

Compared on each image:
  s_eq4   : 1/sqrt( sum_i r_i/sig2_i + 1/sigma_u^2 )              (paper Eq. (4), EM surrogate curvature)
  s_obs   : 1/sqrt( I(u*) ), I = sum r/sig2 - sum r(1-r) g^2/sig2^2 + 1/sigma_u^2   (exact observed information)
  s_grid  : std of the exact posterior of u, numerically on [log F_MIN, log F_MAX]

Self-check: the recomputed s_eq4 must reproduce the s logged by the deployed run to 1e-9.

Calibration target: the GT-derived anatomical scale factor f* in
output/analysis/oracle_factors_<dir>_test.json.

usage: python analysis_uncert_exact.py [--out output/analysis/uncert_exact.json]
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- deployed defaults (adapteacher/config.py, SEMISUPNET.CANON) --------------------------------
SIGMA_OBS = 0.10
SIGMA_U = 1.0
OUTLIER_DENSITY = 0.2
IRLS_ITERS = 3
F_MIN, F_MAX = 0.3, 3.0
ADAPT_K, ADAPT_MIN, ADAPT_MAX = 1.732, 0.0, 0.5

DIRS = ["c1_to_c2", "c1_to_c3", "c2_to_c1", "c2_to_c3", "c3_to_c1", "c3_to_c2"]
DUMP = "output/canon/p1dump_s42_%s_test/pass1_candidates.json"
GRID_N = 4001


def load_prior(src):
    pj = json.load(open(os.path.join(ROOT, "output", "anat_prior_%s.json" % src)))
    s0 = int(pj.get("indomain_best_scale", 800))
    shift = math.log(s0 / 800.0)
    c = pj["canonical_at_800"]
    mu = {int(k): v["logsize_mean"] + shift for k, v in c.items()}
    sig = {int(k): v["logsize_std"] for k, v in c.items()}
    return mu, sig, s0


def ani(z, p, mu, sg2):
    """Reproduce estimate_theta: 3 alternations of E and M from u=0, then the Eq. (4) spread."""
    u = 0.0
    den = 1.0 / SIGMA_U ** 2
    for _ in range(IRLS_ITERS):
        lik = np.exp(-0.5 * (z + u - mu) ** 2 / sg2) / np.sqrt(2 * np.pi * sg2)
        r = (p * lik) / (p * lik + (1 - p) * OUTLIER_DENSITY + 1e-12)
        num = np.sum(r * (mu - z) / sg2)
        den = np.sum(r / sg2) + 1.0 / SIGMA_U ** 2
        u = float(num / den)
    return u, float(1.0 / math.sqrt(den))


def resp(u, z, p, mu, sg2):
    lik = np.exp(-0.5 * (z + u - mu) ** 2 / sg2) / np.sqrt(2 * np.pi * sg2)
    return (p * lik) / (p * lik + (1 - p) * OUTLIER_DENSITY + 1e-12)


def nlp(ugrid, z, p, mu, sg2):
    """negative log-posterior of u on a grid (mixture likelihood + Gaussian prior), up to a constant."""
    g = z[None, :] + ugrid[:, None] - mu[None, :]
    lik = np.exp(-0.5 * g ** 2 / sg2[None, :]) / np.sqrt(2 * np.pi * sg2[None, :])
    mix = p[None, :] * lik + (1 - p[None, :]) * OUTLIER_DENSITY
    return -np.sum(np.log(np.maximum(mix, 1e-300)), axis=1) + ugrid ** 2 / (2 * SIGMA_U ** 2)


def exact_posterior(z, p, mu, sg2):
    lo, hi = math.log(F_MIN), math.log(F_MAX)
    ug = np.linspace(lo, hi, GRID_N)
    L = nlp(ug, z, p, mu, sg2)
    L -= L.min()
    w = np.exp(-L)
    Z = np.trapz(w, ug)
    if not np.isfinite(Z) or Z <= 0:
        return None
    pdf = w / Z
    m = float(np.trapz(ug * pdf, ug))
    var = float(np.trapz((ug - m) ** 2 * pdf, ug))
    mode = float(ug[int(np.argmin(L))])
    return m, math.sqrt(max(var, 0.0)), mode


def halfwidth(s):
    return float(np.clip(ADAPT_K * s, ADAPT_MIN, ADAPT_MAX))


def run_dir(d):
    src = d.split("_to_")[0]
    mu_d, sig_d, s0 = load_prior(src)
    f = os.path.join(ROOT, DUMP % d)
    if not os.path.exists(f):
        return {"dir": d, "skipped": "no first-pass dump"}
    cand = json.load(open(f))["candidates"]
    orc = os.path.join(ROOT, "output", "analysis", "oracle_factors_%s_test.json" % d)
    oracle = json.load(open(orc))["factors"] if os.path.exists(orc) else {}

    rows, gate_err = [], []
    for fn, c in cand.items():
        p = np.array(c["p"], dtype=np.float64)
        if p.size < 1:
            rec = {"file": fn, "n": 0, "u": 0.0, "s_eq4": SIGMA_U, "s_surr": SIGMA_U,
                   "s_obs": SIGMA_U, "s_grid": SIGMA_U, "neg_curv": 0,
                   "u_pmean": 0.0, "u_pmode": 0.0}
        else:
            w = np.array(c["w"], dtype=np.float64)
            h = np.array(c["h"], dtype=np.float64)
            z = np.log(np.sqrt(w * h))
            cls = c["c"]
            mu = np.array([mu_d.get(int(k), 0.0) for k in cls])
            sg2 = np.array([sig_d.get(int(k), 0.3) ** 2 for k in cls]) + SIGMA_OBS ** 2
            u_raw, s_eq4 = ani(z, p, mu, sg2)
            u = float(np.clip(u_raw, math.log(F_MIN), math.log(F_MAX)))
            r = resp(u, z, p, mu, sg2)
            g = z + u - mu
            info = float(np.sum(r / sg2) - np.sum(r * (1 - r) * g ** 2 / sg2 ** 2) + 1.0 / SIGMA_U ** 2)
            s_obs = float(1.0 / math.sqrt(info)) if info > 0 else float("nan")
            s_surr = float(1.0 / math.sqrt(np.sum(r / sg2) + 1.0 / SIGMA_U ** 2))
            ep = exact_posterior(z, p, mu, sg2)
            rec = {"file": fn, "n": int(p.size), "u": u, "s_eq4": s_eq4, "s_surr": s_surr,
                   "s_obs": s_obs, "neg_curv": int(info <= 0),
                   "s_grid": (ep[1] if ep else float("nan")),
                   "u_pmean": (ep[0] if ep else float("nan")),
                   "u_pmode": (ep[2] if ep else float("nan"))}
        rec["gate"] = abs(c["s_post"] - rec["s_eq4"])
        gate_err.append(rec["gate"])
        if fn in oracle and oracle[fn][0] > 0:
            rec["u_star"] = math.log(oracle[fn][0])
        rows.append(rec)

    keep = [r for r in rows if r["gate"] < 1e-9 and "u_star" in r]
    return {"dir": d, "s0": s0, "n_img": len(rows), "n_gate": len(gate_err),
            "gate_max_abs_err": float(np.max(gate_err)) if gate_err else float("nan"),
            "n_reproducible": len(keep),
            "frac_reproducible": (len(keep) / len(rows) if rows else 0.0),
            "rows": keep}


def summarise(all_rows, tag):
    def arr(k):
        return np.array([r[k] for r in all_rows], dtype=np.float64)

    se, so_, sg = arr("s_eq4"), arr("s_obs"), arr("s_grid")
    ok = np.isfinite(so_) & np.isfinite(sg)
    res = {"tag": tag, "n": int(len(all_rows)), "n_finite": int(ok.sum()),
           "neg_curv_frac": float(np.mean(arr("neg_curv")))}

    err_all = np.array([(r["u"] - r["u_star"]) if "u_star" in r else np.nan for r in all_rows])
    h_e_ok = np.array([halfwidth(x) for x in se[ok]])
    res["variants"] = {}
    for lab, s_all in (("eq4", se), ("obs", so_), ("grid", sg)):
        s = s_all[ok]
        h = np.array([halfwidth(x) for x in s])
        c = np.isfinite(err_all) & np.isfinite(s_all)
        res["variants"][lab] = {
            "median_s": float(np.median(s)),
            "median_ratio_to_eq4": float(np.median(s / se[ok])),
            "p90_ratio_to_eq4": float(np.percentile(s / se[ok], 90)),
            "p99_ratio_to_eq4": float(np.percentile(s / se[ok], 99)),
            "max_ratio_to_eq4": float(np.max(s / se[ok])),
            "h_mean": float(np.mean(h)),
            "h_mad_vs_eq4": float(np.mean(np.abs(h - h_e_ok))),
            "h_max_abs_dev_vs_eq4": float(np.max(np.abs(h - h_e_ok))),
            "h_clip_frac": float(np.mean(h >= ADAPT_MAX - 1e-12)),
            "cov68": float(np.mean(np.abs(err_all[c]) <= 1.0 * s_all[c])),
            "cov95": float(np.mean(np.abs(err_all[c]) <= 1.96 * s_all[c]))}

    have = [r for r in all_rows if "u_star" in r]
    if have:
        err = np.array([r["u"] - r["u_star"] for r in have])
        res["n_cal"] = int(len(have))
        res["median_abs_err"] = float(np.median(np.abs(err)))
        s = np.array([r["s_eq4"] for r in have])
        a, b = np.abs(err), s
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        res["spearman_s_vs_abserr"] = float(np.corrcoef(ra, rb)[0, 1])
        hi = b >= np.median(b)
        res["median_abs_err_high_s"] = float(np.median(a[hi]))
        res["median_abs_err_low_s"] = float(np.median(a[~hi]))
        pm = np.array([r["u_pmean"] for r in have], dtype=np.float64)
        us = np.array([r["u_star"] for r in have], dtype=np.float64)
        um = np.array([r["u"] for r in have], dtype=np.float64)
        m = np.isfinite(pm)
        res["median_abs_err_pmean"] = float(np.median(np.abs(pm[m] - us[m])))
        res["median_abs_u_map_minus_pmean"] = float(np.median(np.abs(um[m] - pm[m])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "analysis", "uncert_exact.json"))
    args = ap.parse_args()

    per_dir, keep = [], []
    for d in DIRS:
        r = run_dir(d)
        rows = r.pop("rows", [])
        per_dir.append(r)
        print("[%s] s0=%s n=%s gate_max=%.2e used=%s (%.1f%%)" %
              (d, r.get("s0"), r.get("n_img"), r.get("gate_max_abs_err", float("nan")),
               r.get("n_reproducible"), 100.0 * r.get("frac_reproducible", 0.0)))
        for x in rows:
            x["dir"] = d
        keep += rows

    summ = {"per_direction": per_dir,
            "source": "first-pass candidate dumps (SEMISUPNET.CANON.DUMP_PASS1, seed 42)",
            "defaults": {"SIGMA_OBS": SIGMA_OBS, "SIGMA_U": SIGMA_U,
                         "OUTLIER_DENSITY": OUTLIER_DENSITY, "IRLS_ITERS": IRLS_ITERS,
                         "ADAPT_K": ADAPT_K, "ADAPT_MAX": ADAPT_MAX, "GRID_N": GRID_N},
            "all": summarise(keep, "all") if keep else None}
    if keep:
        few = [r for r in keep if r["n"] <= 2]
        many = [r for r in keep if r["n"] >= 5]
        summ["n_le_2"] = summarise(few, "n<=2") if few else None
        summ["n_ge_5"] = summarise(many, "n>=5") if many else None
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summ, open(args.out, "w"), indent=1)
    for k in ("all", "n_le_2", "n_ge_5"):
        if summ.get(k):
            print("=== " + k)
            print(json.dumps(summ[k], indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
