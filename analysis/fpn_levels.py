"""Mechanism analysis: where do ground-truth objects fall on the FPN pyramid (detectron2 ROI-pooler level assignment,
level = clamp(floor(4 + log2(sqrt(area)/224)), 2, 5), sizes in network-input pixels) for
  (a) SOURCE train at the source's calibrated scale s0,
  (b) TARGET test at the default scale 800 (what source-only sees),
  (c) TARGET test after TAC canonicalisation (per-image factor from output/canon/tacr2_s<seed>_<src>_to_<tgt>_test).
Also the object-size (log sqrt-area) distributions and the total-variation distance of the level histograms to the
source, per direction. usage: python fpn_levels.py [seed] -> output/analysis/fpn_levels.json"""
import json
import math
import os
import sys
from collections import Counter

import numpy as np

os.chdir(os.environ.get("TAC_ROOT", "."))
seed = sys.argv[1] if len(sys.argv) > 1 else "42"
DIRS = [("c1", "c2"), ("c1", "c3"), ("c2", "c1"), ("c2", "c3"), ("c3", "c1"), ("c3", "c2")]
LEVELS = [2, 3, 4, 5]


def load(c, split):
    d = json.load(open(f"datasets/Heart-4cc/{c}/{split}.json"))
    imgs = {im["id"]: im for im in d["images"]}
    return d, imgs


def scale_k(W, H, s, max_size=1333):
    return min(s / min(W, H), max_size / max(W, H))


def level_of(size_px):
    return int(np.clip(math.floor(4 + math.log2(max(size_px, 1e-3) / 224.0)), 2, 5))


def hist_and_sizes(d, imgs, fac):
    """fac(img) -> scale factor from original pixels to network-input pixels"""
    cnt = Counter(); sizes = []
    for an in d["annotations"]:
        x, y, w, h = an["bbox"]
        if w <= 0 or h <= 0:
            continue
        im = imgs[an["image_id"]]
        k = fac(im)
        s = math.sqrt(w * h) * k
        cnt[level_of(s)] += 1; sizes.append(math.log(s))
    n = sum(cnt.values())
    return {l: cnt[l] / n for l in LEVELS}, sizes


def tv(p, q):
    return 0.5 * sum(abs(p[l] - q[l]) for l in LEVELS)


out = {}
print(f"## FPN-level histograms (fractions on p2/p3/p4/p5) and TV distance to source, seed {seed}")
print("  direction | source@s0 (train)        | target@800 (test)         TV  | target@TAC (test)         TV  | median log-size src/tgt800/tgtTAC")
for a, b in DIRS:
    pr = json.load(open(f"output/anat_prior_{a}.json")); s0 = int(pr.get("indomain_best_scale", 800))
    ds, ims = load(a, "train")
    hs, zs = hist_and_sizes(ds, ims, lambda im: scale_k(im["width"], im["height"], s0))
    dt, imt = load(b, "test")
    ht, zt = hist_and_sizes(dt, imt, lambda im: scale_k(im["width"], im["height"], 800))
    fp = f"output/canon/tacr2_s{seed}_{a}_to_{b}_test/canon_factors.json"
    if not os.path.exists(fp):
        print(f"  {a}->{b}: missing {fp}"); continue
    fj = json.load(open(fp)); F = fj["factors"]; s0c = fj["s0"]

    def fac_tac(im):
        f = F.get(os.path.basename(im["file_name"]), [1.0])[0]
        # TAC's canonical view = base at s0c (calibrated) times f, capped at MAX_SIZE 4000
        k = min(s0c / min(im["width"], im["height"]), 4000 / max(im["width"], im["height"])) * f
        return k
    hc, zc = hist_and_sizes(dt, imt, fac_tac)
    fmt = lambda h: "/".join(f"{h[l]*100:4.1f}" for l in LEVELS)
    print(f"  {a}->{b}   | {fmt(hs)} | {fmt(ht)} {tv(hs, ht):.3f} | {fmt(hc)} {tv(hs, hc):.3f} | {np.median(zs):.2f}/{np.median(zt):.2f}/{np.median(zc):.2f}")
    out[f"{a}_to_{b}"] = {"s0": s0, "source_train": hs, "target_800": ht, "target_tac": hc, "tv_800": tv(hs, ht), "tv_tac": tv(hs, hc),
                          "logsize_median": [float(np.median(zs)), float(np.median(zt)), float(np.median(zc))],
                          "logsize_src_hist": np.histogram(zs, bins=np.linspace(2.5, 6.5, 41))[0].tolist(),
                          "logsize_t800_hist": np.histogram(zt, bins=np.linspace(2.5, 6.5, 41))[0].tolist(),
                          "logsize_tac_hist": np.histogram(zc, bins=np.linspace(2.5, 6.5, 41))[0].tolist(),
                          "factors": sorted(v[0] for v in F.values())}
os.makedirs("output/analysis", exist_ok=True)
json.dump(out, open(f"output/analysis/fpn_levels_s{seed}.json", "w"), indent=1)
print(f"saved output/analysis/fpn_levels_s{seed}.json"); print("FPN_LEVELS_DONE")
