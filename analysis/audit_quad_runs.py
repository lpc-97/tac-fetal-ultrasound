# -*- coding: utf-8 -*-
"""Leakage / provenance audit of the runs behind the reported TAC-7 (quadrature) numbers.

For every run directory it reads the configuration that was actually in effect, from the run's own
log, and checks it against what the directory name claims: the evaluated dataset must be the target
of the direction and its TEST split, the weights must be the source model of that direction and that
seed, the anatomical prior must be the source one, and no external factor file or constant factor may
be in force.
"""
import json
import os
import re
import sys

ROOT = os.environ.get("TAC_ROOT", "/public/mvicgs5/Lvxg/TTA-OD")
os.chdir(ROOT)

SETTINGS = {
    "heart4cc_frcnn": ("output/canon/u7_k17_s{s}_{a}_to_{b}_test", "Heart-4cc", ["c1", "c2", "c3"]),
    "heart4cc_retinanet": ("output/canon/rt_u7_k17_s{s}_{a}_to_{b}_test", "Heart-4cc", ["c1", "c2", "c3"]),
    "abdomen": ("output/canon/abdomen_u7_k17_s{s}_{a}_to_{b}_test", "abdomen", ["ge", "ph", "sa"]),
    "spine": ("output/canon/spine_u7_k17_s{s}_{a}_to_{b}_test", "FUSSD", ["ge", "ph", "sa"]),
}
SEEDS = [42, 43, 44]

problems = []
rows = []


def read_text(p, limit=400000):
    if not os.path.exists(p):
        return ""
    with open(p, "r", errors="ignore") as f:
        return f.read(limit)


def find(pat, txt, grp=1):
    m = re.search(pat, txt)
    return m.group(grp) if m else None


for setting, (tpl, dsdir, doms) in SETTINGS.items():
    for a in doms:
        for b in doms:
            if a == b:
                continue
            for s in SEEDS:
                d = tpl.format(s=s, a=a, b=b)
                tag = "%s %s->%s s%d" % (setting, a, b, s)
                if not os.path.isdir(d):
                    problems.append("MISSING RUN   " + tag + "  " + d)
                    continue
                log = read_text(d + "/log.txt") or read_text(d + ".log")
                if not log:
                    problems.append("NO LOG        " + tag)
                    continue

                test_ds = find(r"DATASETS:.*?TEST:\s*\(?'([^']+)'", log) or find(r"'(fetus_[a-z0-9_]+)'", log)
                # the command line recorded in the log is the authoritative record of the override
                weights = find(r"'MODEL\.WEIGHTS',\s*'([^']+)'", log)
                cfgfile = find(r"config_file='([^']+)'", log)
                canon = find(r"\[CANON\] cfg: (\{.*?\})\n", log)
                ap_f = d + "/result_ap.txt"
                ap = None
                if os.path.exists(ap_f):
                    try:
                        ap = json.loads(read_text(ap_f).strip().splitlines()[-1])["AP"]
                    except Exception:
                        pass

                prior = ff = fc = views = None
                if canon:
                    prior = find(r"'ANAT_PRIOR':\s*'([^']*)'", canon)
                    ff = find(r"'FACTORS_FILE':\s*'([^']*)'", canon)
                    fc = find(r"'FACTOR_CONST':\s*([0-9.eE+-]+)", canon)
                    adapt = find(r"'ADAPT':\s*'([^']*)'", canon)
                    kk = find(r"'ADAPT_K':\s*([0-9.]+)", canon)
                else:
                    adapt = kk = None
                views = find(r"views_per_img[\"']?:\s*([0-9.]+)", log)

                # --- the checks
                if test_ds and b not in test_ds.lower():
                    problems.append("WRONG TARGET  %s  evaluated on %s" % (tag, test_ds))
                if test_ds and "test" not in test_ds.lower():
                    problems.append("WRONG SPLIT   %s  evaluated on %s" % (tag, test_ds))
                if not weights:
                    problems.append("NO WEIGHTS    %s" % tag)
                else:
                    if a not in weights.lower():
                        problems.append("WRONG SOURCE  %s  weights %s" % (tag, weights))
                    if b in weights.lower() and a not in weights.lower():
                        problems.append("TARGET MODEL  %s  weights %s" % (tag, weights))
                    if ("_s%d" % s) not in weights:
                        problems.append("WRONG SEED    %s  weights %s (expected seed %d)" % (tag, weights, s))
                if cfgfile and a not in cfgfile.lower():
                    problems.append("WRONG CONFIG  %s  config_file %s" % (tag, cfgfile))
                if prior is not None and a not in prior.lower():
                    problems.append("WRONG PRIOR   %s  prior %s" % (tag, prior))
                if prior is not None and b in prior.lower() and a not in prior.lower():
                    problems.append("TARGET PRIOR  %s  prior %s" % (tag, prior))
                if ff:
                    problems.append("EXTERNAL FACTORS %s  FACTORS_FILE=%s" % (tag, ff))
                if fc is not None and float(fc) > 0:
                    problems.append("CONST FACTOR  %s  FACTOR_CONST=%s" % (tag, fc))
                if adapt is not None and adapt != "bracket":
                    problems.append("WRONG ADAPT   %s  ADAPT=%s" % (tag, adapt))
                if kk is not None and abs(float(kk) - 1.732) > 1e-6:
                    problems.append("WRONG ADAPT_K %s  ADAPT_K=%s" % (tag, kk))
                if views is not None and abs(float(views) - 7.0) > 0.01:
                    problems.append("VIEW COUNT    %s  views_per_img=%s" % (tag, views))

                rows.append((tag, test_ds, weights, cfgfile, prior, ff, fc, adapt, kk, views, ap))

print("runs inspected      : %d of %d expected" % (len(rows), 4 * 6 * 3))
print("problems            : %d" % len(problems))
for p in problems[:30]:
    print("   ", p)
print()
print("--- sample of the configuration actually in force ---")
for r in rows[:3] + rows[18:20] + rows[36:38] + rows[-2:]:
    print("  %-30s test=%-24s w=%-36s prior=%-34s FF=%r FC=%s adapt=%s K=%s views=%s AP=%s"
          % (r[0], r[1], r[2], r[4], r[5], r[6], r[7], r[8], r[9],
             ("%.3f" % r[10]) if r[10] is not None else None))

json.dump({"rows": [list(r) for r in rows], "problems": problems},
          open("output/analysis/audit_quad_runs.json", "w"), indent=1)
print()
print("wrote output/analysis/audit_quad_runs.json")
