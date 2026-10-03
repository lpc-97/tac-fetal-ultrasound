# -*- coding: utf-8 -*-
"""Collect the idle-GPU timing run (output/timing/<name>): seconds per image from the method's own stats file (canon / scalesel /
baseline trainers) or from the detectron2 evaluator log (source, heat-map fusion), peak GPU memory from gpu_mem.json, wall-clock
seconds of the whole process from output/timing/wall.txt. Writes output/analysis/timing.json."""
import glob
import json
import os
import re

os.chdir(os.environ.get("TAC_ROOT", "."))
wall = {l.split()[0]: float(l.split()[1]) for l in open("output/timing/wall.txt") if l.strip()}
out = {}
for d in sorted(glob.glob("output/timing/*/")):
    n = os.path.basename(d.rstrip("/")); rec = {"wall_s": wall.get(n)}
    for st in glob.glob(os.path.join(d, "*_stats.json")):
        j = json.load(open(st))
        sp = j.get("s_per_img", j.get("s_per_img_total"))
        if sp is not None:
            rec["s_per_img"] = sp; rec["views_per_img"] = j.get("views_per_img"); rec["stats_file"] = os.path.basename(st)
            for k in ("s_per_img_teacher", "s_per_img_update", "n_img"):
                if k in j: rec[k] = j[k]
    if "s_per_img" not in rec:
        m = re.findall(r"Total inference time: [\d:.]+ \(([\d.]+) s / iter per device", open(d.rstrip("/") + ".log", errors="ignore").read() if os.path.exists(d.rstrip("/") + ".log") else "")
        if not m and os.path.exists(os.path.join(d, "log.txt")):
            m = re.findall(r"Total inference time: [\d:.]+ \(([\d.]+) s / iter per device", open(os.path.join(d, "log.txt"), errors="ignore").read())
        if m:
            rec["s_per_img"] = float(m[-1]); rec["stats_file"] = "evaluator log"
    if os.path.exists(os.path.join(d, "gpu_mem.json")):
        rec.update(json.load(open(os.path.join(d, "gpu_mem.json"))))
    r = os.path.join(d, "result_ap.txt")
    if os.path.exists(r):
        rec["AP"] = json.loads(open(r).read().strip().splitlines()[-1])["AP"]
    out[n] = rec
json.dump(out, open("output/analysis/timing.json", "w"), indent=1)
for n, r in out.items():
    print(f"{n:10s} s/img {r.get('s_per_img', float('nan')):.3f} ({r.get('stats_file', '?')}) wall {r.get('wall_s')} peak_alloc {r.get('peak_alloc_MB', float('nan')):.0f} MB AP {r.get('AP', float('nan')):.2f}")
print("TIMING_COLLECT_DONE")
