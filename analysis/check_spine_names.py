# -*- coding: utf-8 -*-
"""Can the anonymised FUSSD spine file names be mapped back to the original names in datasets/spine/?
Both directories hold the same splits with the same number of images and annotations, but FUSSD uses 'te_00001.jpg'
style names while spine/ keeps the acquisition names (DICOM UIDs, capture time stamps, '<case>_<frame>'). If the two
json files list the same images in the same order with identical geometry and boxes, the mapping is by image id and
the spine bootstrap can be performed at case level instead of falling back to image level.

Writes output/analysis/spine_name_map.json (anonymised name -> original name, per split) when the match is exact.
"""
import json
import os
import re
from collections import defaultdict

os.chdir(os.environ.get("TAC_ROOT", "."))
DOMS = ["GE", "PH", "SA"]; SPLITS = ["train", "val", "test"]


def load(p):
    j = json.load(open(p))
    by = {i["id"]: i for i in j["images"]}
    anns = defaultdict(list)
    for a in j["annotations"]:
        anns[a["image_id"]].append((a["category_id"], tuple(round(x, 3) for x in a["bbox"])))
    for k in anns:
        anns[k].sort()
    return by, anns


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


ok = True
mapping = {}
for d in DOMS:
    for sp in SPLITS:
        fb, fa = load(f"datasets/FUSSD/{d}/{sp}.json")
        sb, sa = load(f"datasets/spine/{d}/{sp}.json")
        if set(fb) != set(sb):
            print(f"{d}/{sp}: image id sets differ ({len(fb)} vs {len(sb)})"); ok = False; continue
        bad_geom = [i for i in fb if (fb[i]["width"], fb[i]["height"]) != (sb[i]["width"], sb[i]["height"])]
        bad_ann = [i for i in fb if fa.get(i, []) != sa.get(i, [])]
        print(f"{d}/{sp}: n={len(fb)} geometry mismatches={len(bad_geom)} annotation mismatches={len(bad_ann)}")
        if bad_geom or bad_ann:
            ok = False; continue
        mapping[f"{d}/{sp}"] = {fb[i]["file_name"]: sb[i]["file_name"] for i in fb}

if ok:
    json.dump(mapping, open("output/analysis/spine_name_map.json", "w"), indent=1)
    print("\nEXACT MATCH: the anonymised names map to the original names by image id.")
    for d in DOMS:
        names = list(mapping[f"{d}/test"].values())
        keys = cluster_keys([os.path.basename(n) for n in names])
        cl = defaultdict(int)
        for n in names:
            cl[keys[os.path.basename(n)]] += 1
        kinds = defaultdict(int)
        for k in cl:
            kinds[k.split(":")[0]] += 1
        print(f"  spine {d} test: {len(names)} images -> {len(cl)} clusters (max {max(cl.values())}, "
              f"multi-frame {sum(1 for v in cl.values() if v > 1)}), kinds {dict(kinds)}")
else:
    print("\nNO exact match: the two directories are not aligned by image id; do not use the mapping.")
print("SPINE_NAME_CHECK_DONE")
