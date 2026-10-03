# -*- coding: utf-8 -*-
"""Sanity check of the case-clustering rule used by cluster_bootstrap.py.

The rule turns a DICOM-style file name into a cluster key by dropping the last two dot-separated fields, which is
correct when the trailing fields are series/instance counters of a real study UID. Some exports, however, carry a
generic implementation root (e.g. 1.2.826.0.1.3680043.x.y, the dcm4che / Medical Connections root) followed by random
fields, and for those the rule collapses unrelated images into one huge cluster. This script reports, for every test
split actually used in the paper, the cluster-size distribution and the largest clusters, so that over-merging can be
detected. It changes nothing; it only prints.
"""
import json
import os
import re
from collections import Counter, defaultdict

os.chdir(os.environ.get("TAC_ROOT", "."))
SPLITS = [("Heart-4cc", ["c1", "c2", "c3"], None),
          ("abdomen", ["GE", "PH", "SA"], None),
          ("FUSSD", ["GE", "PH", "SA"], "datasets/spine/{d}/test.json")]


def cluster_keys(names):
    keys = {}; ts = []
    for n in names:
        base = os.path.splitext(n)[0]
        m = re.match(r"^(\d{4})\.(\d{2})\.(\d{2}) (\d{2})\.(\d{2})\.(\d{2})$", base)
        if m:
            y, mo, d, h, mi, se = map(int, m.groups()); ts.append((n, ((y * 12 + mo) * 31 + d) * 86400 + h * 3600 + mi * 60 + se)); continue
        m = re.match(r"^(\d{8})-(\d{6})$", base)
        if m:
            ts.append((n, int(m.group(1)) * 86400 + int(m.group(2)[:2]) * 3600 + int(m.group(2)[2:4]) * 60 + int(m.group(2)[4:]))); continue
        m = re.match(r"^(\d+)_(\d+)$", base)
        if m:
            keys[n] = "id:" + m.group(1); continue
        if base.count(".") >= 6:
            keys[n] = "uid:" + ".".join(base.split(".")[:-2]); continue
        keys[n] = "img:" + base
    ts.sort(key=lambda t: t[1]); cur = None; last = None
    for n, t in ts:
        if last is None or t - last > 60:
            cur = f"ts:{t}"
        keys[n] = cur; last = t
    return keys


for ds, doms, alt in SPLITS:
    for d in doms:
        p = alt.format(d=d) if alt else f"datasets/{ds}/{d}/test.json"
        j = json.load(open(p))
        names = [os.path.basename(im["file_name"]) for im in j["images"]]
        keys = cluster_keys(names)
        cl = defaultdict(list)
        for n in names:
            cl[keys[n]].append(n)
        sizes = Counter(len(v) for v in cl.values())
        big = sorted(cl.items(), key=lambda kv: -len(kv[1]))[:3]
        frac = max(len(v) for v in cl.values()) / len(names)
        flag = "  <-- OVER-MERGED?" if frac > 0.15 else ""
        print(f"{ds}/{d}: {len(names)} images -> {len(cl)} clusters; largest {max(len(v) for v in cl.values())} "
              f"({100 * frac:.0f}% of the split){flag}")
        print(f"    size histogram: {sizes.most_common(5)}")
        for k, v in big:
            print(f"    {len(v):4d}  {k[:70]}")
print("CLUSTER_ROOT_CHECK_DONE")
