# -*- coding: utf-8 -*-
"""Recompute the leakage checks from the image files themselves, rather than trusting the collection
written earlier.

Three levels, from weakest to strongest:
  1. file name overlap between splits;
  2. image content (md5) overlap, within a domain and between the train set of one domain and the
     test set of another, the pair that would inflate a cross-domain result;
  3. examination-level overlap: frames exported from one acquisition share the leading fields of
     their DICOM-derived file name, so grouping by that prefix and asking whether a group spans
     train and test detects near-duplicate frames that differ in content and in name.
"""
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.environ.get("TAC_ROOT", "/public/mvicgs5/Lvxg/TTA-OD")
os.chdir(ROOT)

DATASETS = {
    "Heart-4cc": ("datasets/Heart-4cc", ["c1", "c2", "c3"]),
    "abdomen": ("datasets/abdomen", ["GE", "PH", "SA"]),
    "FUSSD(spine)": ("datasets/FUSSD", ["GE", "PH", "SA"]),
}
SPLITS = ["train", "val", "test"]


def index_images(dom_dir):
    """basename -> full path, over everything under the domain directory."""
    idx = {}
    for dp, _dn, fns in os.walk(dom_dir):
        for fn in fns:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                idx.setdefault(fn, os.path.join(dp, fn))
    return idx


def md5(p, buf=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def exam_key(fn, drop):
    """Strip the extension and the last `drop` dot-separated fields: frames of one acquisition keep
    the same leading UID fields and differ only in the trailing instance/frame numbers."""
    stem = os.path.splitext(fn)[0]
    parts = stem.split(".")
    return ".".join(parts[:-drop]) if len(parts) > drop else stem


summary = {}
for dsname, (root, doms) in DATASETS.items():
    print("=" * 80)
    print(dsname)
    content = defaultdict(dict)   # (dom, split) -> {md5: fn}
    names = defaultdict(set)      # (dom, split) -> {fn}
    missing_total = 0
    for dom in doms:
        idx = index_images(os.path.join(root, dom))
        for sp in SPLITS:
            j = os.path.join(root, dom, sp + ".json")
            if not os.path.exists(j):
                print("  MISSING %s" % j)
                continue
            d = json.load(open(j))
            miss = 0
            for im in d["images"]:
                fn = os.path.basename(im["file_name"])
                names[(dom, sp)].add(fn)
                p = idx.get(fn)
                if p is None:
                    miss += 1
                    continue
                content[(dom, sp)][md5(p)] = fn
            missing_total += miss
            print("  %-4s %-5s listed=%5d  hashed=%5d  distinct content=%5d%s"
                  % (dom, sp, len(d["images"]), len(d["images"]) - miss, len(content[(dom, sp)]),
                     ("   NOT FOUND: %d" % miss) if miss else ""))
    if missing_total:
        print("  !! %d listed images were not found on disk; the content checks below are incomplete"
              % missing_total)

    print("  --- 1+2. within a domain ---")
    for dom in doms:
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                nm = names[(dom, a)] & names[(dom, b)]
                ct = set(content[(dom, a)]) & set(content[(dom, b)])
                print("    %-4s %-5s vs %-5s : name %3d   content %3d%s"
                      % (dom, a, b, len(nm), len(ct), "   <-- LEAK" if (nm or ct) else ""))

    print("  --- 2. across domains: SOURCE train vs TARGET test ---")
    worst = 0
    for a in doms:
        for b in doms:
            if a == b:
                continue
            ct = set(content[(a, "train")]) & set(content[(b, "test")])
            worst = max(worst, len(ct))
            if ct:
                print("    %s train INTERSECTS %s test : %d  e.g. %s" % (a, b, len(ct), list(ct)[:2]))
    print("    maximum source-train / target-test content overlap over all ordered pairs: %d" % worst)

    print("  --- 3. examination-level: one acquisition split across train and test ---")
    for dom in doms:
        for drop in (1, 2):
            g = defaultdict(set)
            for sp in SPLITS:
                for fn in names[(dom, sp)]:
                    g[exam_key(fn, drop)].add(sp)
            n_groups = len(g)
            n_files = sum(len(names[(dom, sp)]) for sp in SPLITS)
            spanning_tt = sum(1 for v in g.values() if "train" in v and "test" in v)
            spanning_tv = sum(1 for v in g.values() if "train" in v and "val" in v)
            print("    %-4s drop=%d : %5d groups for %5d files (%.2f frames/group) | "
                  "groups spanning train&test %4d   train&val %4d%s"
                  % (dom, drop, n_groups, n_files, n_files / max(n_groups, 1),
                     spanning_tt, spanning_tv, "   <-- EXAMINATION LEAK" if spanning_tt else ""))
        summary.setdefault(dsname, {})[dom] = True

print()
print("note: a 'drop' of 1 or 2 removes the trailing instance/frame fields of the DICOM-derived name;")
print("if frames/group is ~1.0 the names carry no acquisition grouping and this test is uninformative.")
