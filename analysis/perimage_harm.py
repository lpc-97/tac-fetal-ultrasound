"""Per-image harm analysis on Heart-4CC with Faster R-CNN (2026-11-01, reviewer request).

For every test image of the six directions and three seeds, the recall of the annotated structures at IoU 0.5 (a prediction
of the same class with score >= 0.3, greedy by score) under the source detector and under TAC-3, TAC-3 with two rounds and
TAC-7 (quadrature bracket). Reports the share of images on which each variant lowers, keeps or raises the recall of the source
detector, the mean change, the largest drops, and the same by first-pass candidate count. Also exports the two images with the
largest recall drop under TAC-3 with two rounds (seed 42) in the format of qual_examples_s42.json, for a failure panel.
usage (project root, TAC_ROOT): python perimage_harm.py
"""
import json
import os
import shutil
from collections import Counter

os.chdir(os.environ.get("TAC_ROOT", "."))
SEEDS = [42, 43, 44]
DIRS = [("c1", "c2"), ("c1", "c3"), ("c2", "c1"), ("c2", "c3"), ("c3", "c1"), ("c3", "c2")]
VAR = {"tac3": "output/canon/tac_s{s}_{a}_to_{b}_test", "tac3_2r": "output/canon/tacr2_s{s}_{a}_to_{b}_test",
       "tac7": "output/canon/u7_k17_s{s}_{a}_to_{b}_test"}
SRC = "output/so_fz_s{s}_{a}_to_{b}"
SCORE_T = 0.3


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy; u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def preds(d):
    f = os.path.join(d, "inference", "coco_instances_results.json")
    if not os.path.exists(f):
        f = os.path.join(d, "coco_instances_results.json")
    by = {}
    for p in json.load(open(f)):
        by.setdefault(p["image_id"], []).append(p)
    return by


def recall(gts, ps):
    """matched GT count at IoU 0.5, same class, predictions with score >= SCORE_T greedy by score"""
    used = set(); m = 0
    for p in sorted([q for q in ps if q["score"] >= SCORE_T], key=lambda q: -q["score"]):
        best, bj = 0.5, None
        for j, g in enumerate(gts):
            if j in used or g["category_id"] != p["category_id"]:
                continue
            o = iou(p["bbox"], g["bbox"])
            if o >= best:
                best, bj = o, j
        if bj is not None:
            used.add(bj); m += 1
    return m


OUT = {"score_threshold": SCORE_T, "iou": 0.5, "variants": {}, "n_images": 0}
rows = {v: [] for v in VAR}   # (seed, dir, image file, n_gt, rec_src, rec_var, ndet_pass1, factor)
for s in SEEDS:
    for a, b in DIRS:
        gt = json.load(open(f"datasets/Heart-4cc/{b}/test.json"))
        gt_by = {}
        for an in gt["annotations"]:
            gt_by.setdefault(an["image_id"], []).append(an)
        files = {x["id"]: os.path.basename(x["file_name"]) for x in gt["images"]}
        P_src = preds(SRC.format(s=s, a=a, b=b))
        for v, pat in VAR.items():
            d = pat.format(s=s, a=a, b=b); P = preds(d)
            ff = os.path.join(d, "canon_factors.json")  # the plain TAC-3 runs predate this file; the two-round run of the same seed has the same first pass
            if not os.path.exists(ff):
                ff = os.path.join(VAR["tac3_2r"].format(s=s, a=a, b=b), "canon_factors.json")
            fac = json.load(open(ff))["factors"]
            for iid, name in files.items():
                g = gt_by.get(iid, [])
                if not g:
                    continue
                fr = fac.get(name, [None, None, None, None])
                rows[v].append((s, f"{a}_to_{b}", name, len(g), recall(g, P_src.get(iid, [])), recall(g, P.get(iid, [])), fr[2], fr[0]))
for v, R in rows.items():
    d = [(r[5] - r[4]) / r[3] for r in R]
    worse = sum(1 for x in d if x < 0); same = sum(1 for x in d if x == 0); better = sum(1 for x in d if x > 0)
    by_n = {}
    for r, x in zip(R, d):
        grp = "1-2" if (r[6] is not None and r[6] <= 2) else ("3-5" if (r[6] is not None and r[6] <= 5) else ">5")
        by_n.setdefault(grp, []).append(x)
    worst = sorted(zip(d, R))[:10]
    OUT["variants"][v] = {
        "n": len(R), "pct_worse": round(100 * worse / len(R), 1), "pct_same": round(100 * same / len(R), 1), "pct_better": round(100 * better / len(R), 1),
        "mean_delta_recall_pct": round(100 * sum(d) / len(d), 2),
        "mean_recall_src_pct": round(100 * sum(r[4] / r[3] for r in R) / len(R), 2), "mean_recall_var_pct": round(100 * sum(r[5] / r[3] for r in R) / len(R), 2),
        "pct_worse_by_lost_structures": dict(Counter(int(round(-x * r[3])) for x, r in zip(d, R) if x < 0)),
        "by_first_pass_candidates": {k: {"n": len(x), "pct_worse": round(100 * sum(1 for y in x if y < 0) / len(x), 1), "mean_delta_pct": round(100 * sum(x) / len(x), 2)} for k, x in by_n.items()},
        "worst10": [{"delta_recall": round(x, 3), "seed": r[0], "dir": r[1], "file": r[2], "n_gt": r[3], "rec_src": r[4], "rec_var": r[5], "ndet_pass1": r[6], "factor": r[7]} for x, r in worst],
    }
OUT["n_images"] = len(rows["tac3"]) // len(SEEDS)
json.dump(OUT, open("output/analysis/perimage_harm_4c.json", "w"), indent=1)
for v, o in OUT["variants"].items():
    print(f"{v:8s} n={o['n']} worse {o['pct_worse']}% same {o['pct_same']}% better {o['pct_better']}%  mean d-recall {o['mean_delta_recall_pct']:+.2f} pts  by cand: {o['by_first_pass_candidates']}")

# failure examples: the two largest recall drops of TAC-3 with two rounds at seed 42, exported like qual_examples_s42.json
EX = {}
cands = [w for w in OUT["variants"]["tac3_2r"]["worst10"] if w["seed"] == 42][:2]
os.makedirs("output/figs/fail", exist_ok=True)
for i, w in enumerate(cands):
    a, b = w["dir"].split("_to_")
    gt = json.load(open(f"datasets/Heart-4cc/{b}/test.json")); names = {c["id"]: c["name"] for c in gt["categories"]}
    img = next(x for x in gt["images"] if os.path.basename(x["file_name"]) == w["file"])
    gts = [an for an in gt["annotations"] if an["image_id"] == img["id"]]
    so = SRC.format(s=42, a=a, b=b); tac = VAR["tac3_2r"].format(s=42, a=a, b=b)
    P_so, P_tac = preds(so), preds(tac)
    fac = json.load(open(os.path.join(tac, "canon_factors.json")))["factors"][w["file"]]
    key = f"fail{i + 1}_{a}{b}_f{fac[0]:.2f}".replace(".", "")
    EX[key] = {"dsn": "4c", "file": w["file"], "w": img["width"], "h": img["height"], "names": {str(k): v for k, v in names.items()},
               "gt": [{"bbox": g["bbox"], "cat": g["category_id"]} for g in gts],
               "src": [{"bbox": p["bbox"], "cat": p["category_id"], "score": p["score"]} for p in P_so.get(img["id"], [])],
               "tac": [{"bbox": p["bbox"], "cat": p["category_id"], "score": p["score"]} for p in P_tac.get(img["id"], [])],
               "factor": fac, "so_dir": so, "tac_dir": tac, "recall_src": w["rec_src"], "recall_tac": w["rec_var"], "n_gt": w["n_gt"]}
    shutil.copyfile(os.path.join(f"datasets/Heart-4cc/{b}/src", img["file_name"]), f"output/figs/fail/{key}.jpg")
json.dump(EX, open("output/figs/fail/fail_examples_s42.json", "w"), indent=1)
print("exported", list(EX))
