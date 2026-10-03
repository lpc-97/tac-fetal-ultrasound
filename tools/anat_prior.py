"""Anatomical layout prior from a SOURCE-domain train.json (zero learnable parameters).
For every ordered class pair (a, b) present together in an image: v = [dcx/W, dcy/H, log(w_b/w_a), log(h_b/h_a)]
-> diagonal Gaussian (mean, var). Also per-class presence probability and max count.
Plausibility of a box i in a candidate set = mean over other boxes j of the pair log-density; the 5th percentile of
this statistic on the source train set is stored as the rejection threshold.
usage: python tools/anat_prior.py --json datasets/Heart-4cc/c3/train.json --out output/anat_prior_c3.json
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--json", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--pct", type=float, default=5.0)
a = ap.parse_args()
d = json.load(open(a.json))
cat_ids = sorted(c["id"] for c in d["categories"])
cid2idx = {cid: i for i, cid in enumerate(cat_ids)}          # detectron2 contiguous ids
names = {cid2idx[c["id"]]: c["name"] for c in d["categories"]}
K = len(cat_ids)
imgs = {im["id"]: im for im in d["images"]}
per_img = defaultdict(list)
for an in d["annotations"]:
    x, y, w, h = an["bbox"]
    if w <= 0 or h <= 0:
        continue
    per_img[an["image_id"]].append((cid2idx[an["category_id"]], x + w / 2, y + h / 2, w, h))

pair_feats = defaultdict(list)
counts = np.zeros((K,), dtype=np.int64); present = np.zeros((K,), dtype=np.int64); maxcnt = np.zeros((K,), dtype=np.int64)


def feat(bi, bj, W, H):
    return [(bj[1] - bi[1]) / W, (bj[2] - bi[2]) / H, math.log(bj[3] / bi[3]), math.log(bj[4] / bi[4])]


for iid, boxes in per_img.items():
    W, H = imgs[iid]["width"], imgs[iid]["height"]
    cc = np.bincount([b[0] for b in boxes], minlength=K)
    counts += cc; present += cc > 0; maxcnt = np.maximum(maxcnt, cc)
    for i, bi in enumerate(boxes):
        for j, bj in enumerate(boxes):
            if i != j:
                pair_feats[(bi[0], bj[0])].append(feat(bi, bj, W, H))
n_img = len(per_img)
pairs = {}
for (ca, cb), fs in pair_feats.items():
    fs = np.asarray(fs, dtype=np.float64)
    mu = fs.mean(0); var = fs.var(0) + 1e-4
    pairs[f"{ca}_{cb}"] = {"mean": mu.tolist(), "var": var.tolist(), "n": int(len(fs))}


def logpdf(v, mu, var):
    v = np.asarray(v); mu = np.asarray(mu); var = np.asarray(var)
    return float(-0.5 * (((v - mu) ** 2) / var).sum() - 0.5 * np.log(2 * math.pi * var).sum())


# calibration: per-box mean pair log-density on the source train set
scores = []
for iid, boxes in per_img.items():
    W, H = imgs[iid]["width"], imgs[iid]["height"]
    for i, bi in enumerate(boxes):
        ll = [logpdf(feat(bi, bj, W, H), pairs[f"{bi[0]}_{bj[0]}"]["mean"], pairs[f"{bi[0]}_{bj[0]}"]["var"])
              for j, bj in enumerate(boxes) if j != i and f"{bi[0]}_{bj[0]}" in pairs]
        if ll:
            scores.append(float(np.mean(ll)))
thr = float(np.percentile(scores, a.pct))
# reference object size at the detector's training scale (short edge 800): median sqrt(w*h) after resizing
sizes = []
for iid, boxes in per_img.items():
    W, H = imgs[iid]["width"], imgs[iid]["height"]
    f = 800.0 / min(W, H)
    if max(W, H) * f > 1333:
        f = 1333.0 / max(W, H)
    sizes += [math.sqrt(b[3] * b[4]) * f for b in boxes]
ref_size = float(np.median(sizes))
per_class_ref = {}
for k in range(K):
    v = []
    for iid, boxes in per_img.items():
        W, H = imgs[iid]["width"], imgs[iid]["height"]
        f = min(800.0 / min(W, H), 1333.0 / max(W, H))
        v += [math.sqrt(b[3] * b[4]) * f for b in boxes if b[0] == k]
    per_class_ref[str(k)] = float(np.median(v)) if v else ref_size
# canonical per-class log-size / log-aspect statistics at scale 800 (for the probabilistic canonicalisation)
logsize = {str(k): [] for k in range(K)}; logasp = {str(k): [] for k in range(K)}
for iid, boxes in per_img.items():
    W, H = imgs[iid]["width"], imgs[iid]["height"]
    f = min(800.0 / min(W, H), 1333.0 / max(W, H))
    for b in boxes:
        logsize[str(b[0])].append(math.log(math.sqrt(b[3] * b[4]) * f)); logasp[str(b[0])].append(math.log(b[3] / b[4]))
canon = {k: {"logsize_mean": float(np.mean(v)), "logsize_std": float(np.std(v) + 1e-3),
             "logaspect_mean": float(np.mean(logasp[k])), "logaspect_std": float(np.std(logasp[k]) + 1e-3), "n": len(v)}
         for k, v in logsize.items() if v}
out = {"source_json": a.json, "num_classes": K, "names": names, "n_images": n_img,
       "ref_size_at_800": ref_size, "ref_size_per_class_at_800": per_class_ref, "canonical_at_800": canon,
       "presence": (present / max(n_img, 1)).tolist(), "max_count": maxcnt.tolist(), "pairs": pairs,
       "box_ll_percentiles": {str(p): float(np.percentile(scores, p)) for p in (1, 5, 10, 25, 50)},
       "reject_threshold": thr, "threshold_pct": a.pct}
json.dump(out, open(a.out, "w"), indent=1)
print(f"images={n_img} pairs={len(pairs)} max_count={maxcnt.tolist()} presence={np.round(present / n_img, 2).tolist()}")
print(f"box log-density percentiles: {out['box_ll_percentiles']}  -> reject below {thr:.2f} (p{a.pct:g})")
print(f"reference object size at scale 800: median sqrt(area) = {ref_size:.1f} px; per class {dict((names[int(k)], round(v, 1)) for k, v in per_class_ref.items())}")
print("canonical log-size std per class: " + " ".join(f"{names[int(k)]}={v['logsize_std']:.2f}" for k, v in canon.items())
      + " | log-aspect mean/std: " + " ".join(f"{names[int(k)]}={v['logaspect_mean']:+.2f}/{v['logaspect_std']:.2f}" for k, v in canon.items()))
print("ANAT_PRIOR_DONE", a.out)
