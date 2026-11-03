"""Collect the reviewer controls of 2026-11-01 in the row format of summarize_reviewer.py (Heart-4CC, Faster R-CNN, test):
  ss_conf3  : prior-free per-image scale selection by detector self-confidence over 3 scales (600/800/1000), 3 passes, 1 view output
  ss_conf7  : the same over 7 scales (500-1200), 7 passes
The row() helper and its dependencies are taken verbatim from summarize_reviewer.py by executing its source up to the
first row definition, so means, seeds and the view statistics are computed exactly as for every other row.
Writes output/analysis/controls_2026-11-01.json = {"rows": {name: row}}.
usage (project root, TAC_ROOT): python collect_controls.py
"""
import json
import os

os.chdir(os.environ.get("TAC_ROOT", "."))
src = open("summarize_reviewer.py").read()
exec(src[:src.index("R = {}")] if "R = {}" in src else src[:src.index('R["')])  # imports, SEEDS, DIRS, read_ap, stats_of, mem_of, row

rows = {
    "ss_conf3": row(lambda s, a, b: f"output/scalesel/ss_conf3_s{s}_{a}_to_{b}_test"),
    "ss_conf7": row(lambda s, a, b: f"output/scalesel/ss_conf7_s{s}_{a}_to_{b}_test"),
}
missing = [k for k, v in rows.items() if v is None]
json.dump({"rows": {k: v for k, v in rows.items() if v}}, open("output/analysis/controls_2026-11-01.json", "w"), indent=1)
for k, v in rows.items():
    print(k, "MISSING" if v is None else f"AP {v['mean']:.2f} +- {v['std']:.2f}  AP50 {v['ap50']:.2f}  views {v.get('views')}")
print("missing:", missing)
