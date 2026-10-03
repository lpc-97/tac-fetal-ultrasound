# -*- coding: utf-8 -*-
"""Collect every number behind the paper tables with its provenance for the QC workbook.
Writes output/qc_collect.json:
  tables[<table>][<row>][<seed>|<dir>] = {path, AP.., n_img (from log), test_json (from log), weights, trainer, traceback, views, s_per_img}
  val_sweeps[<dsn>] = [{variant, dir, AP, AP50, path}]      (seed 42, val)
  best_val[<dsn>] = best_val json
  source_models[<name>] = {val_AP_final, iters, bs, lr, dataset}
  data_qc[<dataset>][<domain>][<split>] = counts / resolutions / classes; overlap checks
  compose = output/compose/summary_test.json ; fpn = per-seed FPN-level summaries
"""
import collections
import glob
import hashlib
import json
import os
import re

os.chdir(os.environ.get("TAC_ROOT", "."))
OUT = {"tables": {}, "val_sweeps": {}, "best_val": {}, "source_models": {}, "data_qc": {}, "compose": None, "fpn": {}}
SEEDS = [42, 43, 44]


def read_ap(p):
    f = os.path.join(p, "result_ap.txt")
    if not os.path.exists(f):
        return None
    lines = open(f).read().strip().splitlines()
    try:
        return json.loads(lines[-1])
    except Exception:
        return None


def log_info(p):
    """weights / test json / n images / trainer / traceback from the run's log.txt (or <dir>.log)."""
    info = {"weights": None, "test_json": None, "n_img": None, "trainer": None, "traceback": False, "log": None}
    for lf in [os.path.join(p, "log.txt"), p + ".log"]:
        if not os.path.exists(lf):
            continue
        info["log"] = lf
        txt = open(lf, errors="ignore").read()
        ws = re.findall(r"^\s*WEIGHTS: (\S+)", txt, re.M)          # last occurrence = merged (eval-only) config, first = the training config echo
        if ws:
            info["weights"] = ws[-1]
        m = re.search(r"Loading from (\S+)", txt)
        if m:
            info["weights"] = m.group(1)
        m = re.search(r"Loaded (\d+) images in COCO format from (\S+)", txt)
        if m:
            info["n_img"] = int(m.group(1)); info["test_json"] = m.group(2)
        m = re.search(r"^\s*Trainer: (\S+)", txt, re.M); info["trainer"] = m.group(1) if m else info["trainer"]
        m2 = re.findall(r"^\s*Trainer: (\S+)", txt, re.M)
        if m2:
            info["trainer"] = m2[-1]
        info["traceback"] = info["traceback"] or ("Traceback" in txt)
        m = re.findall(r"Total inference pure compute time: [0-9:]+ \(([0-9.]+) s / iter", txt)
        if m:
            info["s_per_img_log"] = float(m[-1])
    return info


def stats_info(p):
    for name in ["canon_stats.json", "scalesel_stats.json", "iouf_stats.json", "amrod_stats.json", "distill_stats.json", "whw_stats.json",
                 "cdb_stats.json", "sgp_stats.json", "osfda_stats.json", "vlod_stats.json", "buftta_stats.json"]:
        q = os.path.join(p, name)
        if os.path.exists(q):
            try:
                d = json.load(open(q)); return {"views": d.get("views_per_img"), "s_per_img": d.get("s_per_img"), "stats_file": name}
            except Exception:
                return {}
    return {}


def cell(p, variant=None):
    ap = read_ap(p); r = {"path": p, "exists": ap is not None, "variant": variant}
    if ap:
        r.update({k: ap.get(k) for k in ["AP", "AP50", "AP75", "APs", "APm", "APl"]})
    r.update(log_info(p)); r.update(stats_info(p))
    return r


def table(name, rows, dirs, seeds=SEEDS):
    t = {}
    for label, fn in rows:
        t[label] = {}
        for s in seeds:
            for a, b in dirs:
                p, variant = fn(s, a, b)
                t[label][f"s{s}|{a}_to_{b}"] = cell(p, variant)
    OUT["tables"][name] = t


def bl(best, method, DP=""):
    def f(s, a, b):
        v = best.get(method, {}).get(f"{a}_to_{b}", "none")
        return f"output/baselines/{DP}{v}_s{s}_{a}_to_{b}_test", v
    return f


# ---------------------------------------------------------------- Heart-4cc, Faster R-CNN (main table)
best4c = json.load(open("output/baselines/best_val.json"))
D4 = [("c1", "c2"), ("c1", "c3"), ("c2", "c1"), ("c2", "c3"), ("c3", "c1"), ("c3", "c2")]
rows = [
    ("Source-only (FrozenBN)", lambda s, a, b: (f"output/so_fz_s{s}_{a}_to_{b}", "baseline@800")),
    ("IoU-Filter", bl(best4c, "iouf")), ("AMROD", bl(best4c, "amrod")), ("CoTTA-det", bl(best4c, "cotta")),
    ("WHW", bl(best4c, "whw")), ("CD-Buffer", bl(best4c, "cdb")), ("SGP", bl(best4c, "sgp")), ("O-SFDA", bl(best4c, "osfda")),
    ("VLOD-TTA objective", bl(best4c, "vlod")), ("BufferTTA", bl(best4c, "buftta")),
    ("Senior heatmap fusion (18 views)", lambda s, a, b: (f"output/tta_heatmap_s{s}_{a}_to_{b}", "TEST.TTA True")),
    ("18-view WBF", lambda s, a, b: (f"output/scalesel/ss_all_flip_wbf_s{s}_{a}_to_{b}_test", "ss_all_flip_wbf")),
    ("18-view WBF + AR", lambda s, a, b: (f"output/scalesel/ss_all_flip_wbf_ar_s{s}_{a}_to_{b}_test", "ss_all_flip_wbf_ar")),
    ("Ours-heuristic v0.3", lambda s, a, b: (f"output/scalesel/snc_fuse_flip_ar_s{s}_{a}_to_{b}_test", "snc_fuse_flip_ar")),
    ("TAC-light", lambda s, a, b: (f"output/canon/tac_s{s}_{a}_to_{b}_test", "tac")),
    ("TAC (2 rounds)", lambda s, a, b: (f"output/canon/tac_rounds2_s{s}_{a}_to_{b}_test", "tac_rounds2")),
    ("TAC w/o AR", lambda s, a, b: (f"output/canon/tac_noar_s{s}_{a}_to_{b}_test", "tac_noar")),
    ("TAC w/o s0 calib", lambda s, a, b: (f"output/canon/tac_nocalib_s{s}_{a}_to_{b}_test", "tac_nocalib")),
    ("TAC 7 views", lambda s, a, b: (f"output/canon/tac_v7_s{s}_{a}_to_{b}_test", "tac_v7")),
]
table("heart4cc_frcnn", rows, D4)
# fusion-strategy / view-count ablations (Heart-4cc, FRCNN)
table("heart4cc_fusion_views", [
    ("TAC (WBF, 3 views)", lambda s, a, b: (f"output/canon/tac_s{s}_{a}_to_{b}_test", "tac")),
    ("TAC fusion=NMS (3 views)", lambda s, a, b: (f"output/canon/tac_fnms_s{s}_{a}_to_{b}_test", "tac_fnms")),
    ("TAC fusion=heatmap (3 views)", lambda s, a, b: (f"output/canon/tac_fheat_s{s}_{a}_to_{b}_test", "tac_fheat")),
    ("TAC canonical view only (no fusion)", lambda s, a, b: (f"output/canon/tac_flast_s{s}_{a}_to_{b}_test", "tac_flast")),
    ("TAC 2 views (no flip)", lambda s, a, b: (f"output/canon/tac_v2_s{s}_{a}_to_{b}_test", "tac_v2")),
    ("TAC 7 views (+-15% scales)", lambda s, a, b: (f"output/canon/tac_v7_s{s}_{a}_to_{b}_test", "tac_v7")),
    ("TAC 7 views w/o AR", lambda s, a, b: (f"output/canon/tac_v7_noar_s{s}_{a}_to_{b}_test", "tac_v7_noar")),
    ("TAC 11 views (+-10/20/25% scales)", lambda s, a, b: (f"output/canon/tac_v11_s{s}_{a}_to_{b}_test", "tac_v11")),
    ("18-view WBF", lambda s, a, b: (f"output/scalesel/ss_all_flip_wbf_s{s}_{a}_to_{b}_test", "ss_all_flip_wbf")),
    ("18-view WBF + AR", lambda s, a, b: (f"output/scalesel/ss_all_flip_wbf_ar_s{s}_{a}_to_{b}_test", "ss_all_flip_wbf_ar")),
    ("18-view heatmap (senior)", lambda s, a, b: (f"output/tta_heatmap_s{s}_{a}_to_{b}", "TEST.TTA True")),
], D4)
# BN-model group
def bn_src(s, a, b):
    return (f"output/so_bn_{a}_to_{b}" if s == 42 else f"output/so_bn_s{s}_{a}_to_{b}", "baseline@800 (BN model)")
def bn_m(m):
    return lambda s, a, b: (f"output/tta_tuned_{m}_{a}_to_{b}" if s == 42 else f"output/tta_tuned_s{s}_{m}_{a}_to_{b}", f"{m} (val-tuned lr)")
table("heart4cc_bn_models", [("Source-only (BN)", bn_src), ("Tent", bn_m("tent")), ("VPTTA", bn_m("vptta")), ("DomainAdaptor", bn_m("dadaptor")), ("GraTa", bn_m("grata"))], D4)
# ---------------------------------------------------------------- Heart-4cc, RetinaNet
rows_rt = [
    ("RetinaNet source-only", lambda s, a, b: (f"output/so_rt_s{s}_{a}_to_{b}", "baseline@800")),
    ("IoU-Filter", bl(best4c, "rt_iouf")), ("CoTTA-det", bl(best4c, "rt_cotta")),
    ("Senior heatmap fusion (18 views)", lambda s, a, b: (f"output/tta_heatmap_rt_s{s}_{a}_to_{b}", "TEST.TTA True")),
    ("18-view WBF", lambda s, a, b: (f"output/scalesel/rt_ss_all_flip_wbf_s{s}_{a}_to_{b}_test", "ss_all_flip_wbf")),
    ("18-view WBF + AR", lambda s, a, b: (f"output/scalesel/rt_ss_all_flip_wbf_ar_s{s}_{a}_to_{b}_test", "ss_all_flip_wbf_ar")),
    ("TAC-light", lambda s, a, b: (f"output/canon/rt_tac_s{s}_{a}_to_{b}_test", "tac")),
    ("TAC (2 rounds)", lambda s, a, b: (f"output/canon/rt_tacr2_s{s}_{a}_to_{b}_test", "tacr2")),
    ("TAC 7 views", lambda s, a, b: (f"output/canon/rt_tac_v7_s{s}_{a}_to_{b}_test", "tac_v7")),
]
table("heart4cc_retinanet", rows_rt, D4)
# ---------------------------------------------------------------- abdomen / spine
DD = [("ge", "ph"), ("ge", "sa"), ("ph", "ge"), ("ph", "sa"), ("sa", "ge"), ("sa", "ph")]
for dsn in ["abdomen", "spine"]:
    DP = dsn + "_"; best = json.load(open(f"output/baselines/best_val_{dsn}.json")); OUT["best_val"][dsn] = best
    rows_d = [
        ("Source-only (FrozenBN)", lambda s, a, b, DP=DP: (f"output/so_fz_{DP}s{s}_{a}_to_{b}", "baseline@800")),
        ("IoU-Filter", bl(best, "iouf", DP)), ("AMROD", bl(best, "amrod", DP)), ("CoTTA-det", bl(best, "cotta", DP)), ("WHW", bl(best, "whw", DP)),
        ("CD-Buffer", bl(best, "cdb", DP)), ("SGP", bl(best, "sgp", DP)), ("O-SFDA", bl(best, "osfda", DP)), ("VLOD-TTA objective", bl(best, "vlod", DP)),
        ("BufferTTA", bl(best, "buftta", DP)),
        ("Senior heatmap fusion (18 views)", lambda s, a, b, DP=DP: (f"output/tta_heatmap_{DP}s{s}_{a}_to_{b}", "TEST.TTA True")),
        ("18-view WBF", lambda s, a, b, DP=DP: (f"output/scalesel/{DP}ss_all_flip_wbf_s{s}_{a}_to_{b}_test", "ss_all_flip_wbf")),
        ("18-view WBF + AR", lambda s, a, b, DP=DP: (f"output/scalesel/{DP}ss_all_flip_wbf_ar_s{s}_{a}_to_{b}_test", "ss_all_flip_wbf_ar")),
        ("TAC-light", lambda s, a, b, DP=DP: (f"output/canon/{DP}tac_s{s}_{a}_to_{b}_test", "tac")),
        ("TAC (2 rounds)", lambda s, a, b, DP=DP: (f"output/canon/{DP}tacr2_s{s}_{a}_to_{b}_test", "tacr2")),
        ("TAC w/o AR", lambda s, a, b, DP=DP: (f"output/canon/{DP}tac_noar_s{s}_{a}_to_{b}_test", "tac_noar")),
        ("TAC w/o s0 calib", lambda s, a, b, DP=DP: (f"output/canon/{DP}tac_nocalib_s{s}_{a}_to_{b}_test", "tac_nocalib")),
        ("TAC 7 views", lambda s, a, b, DP=DP: (f"output/canon/{DP}tac_v7_s{s}_{a}_to_{b}_test", "tac_v7")),
    ]
    table(dsn, rows_d, DD)
OUT["best_val"]["4c"] = best4c
# in-domain source (3x3) for all datasets
table("indomain_source", [("Source-only FRCNN 4c", lambda s, a, b: (f"output/so_fz_s{s}_{a}_to_{b}", "")),
                          ("Source-only RetinaNet 4c", lambda s, a, b: (f"output/so_rt_s{s}_{a}_to_{b}", ""))],
      [(a, b) for a in ["c1", "c2", "c3"] for b in ["c1", "c2", "c3"]])
for dsn in ["abdomen", "spine"]:
    table(f"indomain_source_{dsn}", [(f"Source-only FRCNN {dsn}", lambda s, a, b, dsn=dsn: (f"output/so_fz_{dsn}_s{s}_{a}_to_{b}", ""))],
          [(a, b) for a in ["ge", "ph", "sa"] for b in ["ge", "ph", "sa"]])

# ---------------------------------------------------------------- val sweeps (seed 42) for every dataset incl. all TAC ablation variants
for dsn, DP, doms in [("4c", "", "c?"), ("rt_4c", "rt_", "c?"), ("abdomen", "abdomen_", "??"), ("spine", "spine_", "??")]:
    rows_v = []
    for folder in ["baselines", "canon", "scalesel"]:
        for d in sorted(glob.glob(f"output/{folder}/{DP}*_s42_{doms}_to_{doms}_val")):
            base = os.path.basename(d)
            if dsn == "4c" and (base.startswith("rt_") or base.startswith("abdomen_") or base.startswith("spine_")):
                continue
            m = re.match(re.escape(DP) + r"(.+)_s42_(\w\w_to_\w\w)_val$", base)
            if not m:
                continue
            ap = read_ap(d)
            if ap:
                rows_v.append({"folder": folder, "variant": m.group(1), "dir": m.group(2), "AP": ap["AP"], "AP50": ap["AP50"], "path": d})
    OUT["val_sweeps"][dsn] = rows_v
# ---------------------------------------------------------------- source models
for d in sorted(glob.glob("output/fs_*_fz_s4?") + glob.glob("output/rt_*_fz_s4?") + glob.glob("output/fs_c?_bn*")):
    lf = os.path.join(d, "log.txt")
    if not os.path.exists(lf):
        continue
    txt = open(lf, errors="ignore").read()
    aps = re.findall(r"copypaste: ([0-9.]+),([0-9.]+)", txt)
    cfg = {}
    for k in ["IMS_PER_BATCH", "MAX_ITER", "BASE_LR"]:
        m = re.search(rf"^\s*{k}: (\S+)", txt, re.M); cfg[k] = m.group(1) if m else None
    m = re.search(r"^\s*TRAIN:\n\s*- (\S+)", txt, re.M); cfg["train_dataset"] = m.group(1) if m else None
    it = re.findall(r"iter: (\d+)", txt)
    OUT["source_models"][os.path.basename(d)] = {"val_AP_final": float(aps[-1][0]) if aps else None, "val_AP50_final": float(aps[-1][1]) if aps else None,
                                                 "n_evals": len(aps), "last_iter": int(it[-1]) if it else None, "final_ckpt": sorted(glob.glob(os.path.join(d, "model_*.pth")))[-1:] , **cfg}
# ---------------------------------------------------------------- data QC
def qc_dataset(name, root, doms):
    res = {}
    H = {}
    for dom in doms:
        res[dom] = {}; names = {}
        for sp in ["train", "val", "test"]:
            j = json.load(open(f"{root}/{dom}/{sp}.json")); ims = j["images"]; anns = j["annotations"]
            names[sp] = {x["file_name"] for x in ims}
            cats = collections.Counter(a["category_id"] for a in anns)
            resm = collections.Counter((x["width"], x["height"]) for x in ims).most_common(4)
            missing = sum(1 for x in ims if not os.path.exists(f"{root}/{dom}/src/{x['file_name']}"))
            res[dom][sp] = {"n_images": len(ims), "n_boxes": len(anns), "boxes_per_img": len(anns) / max(len(ims), 1), "missing_files": missing,
                            "per_class": dict(sorted(cats.items())), "top_resolutions": [f"{w}x{h}:{c}" for (w, h), c in resm],
                            "categories": [c["name"] for c in j["categories"]]}
        res[dom]["name_overlap_train_val"] = len(names["train"] & names["val"]); res[dom]["name_overlap_train_test"] = len(names["train"] & names["test"]); res[dom]["name_overlap_val_test"] = len(names["val"] & names["test"])
        hs = {}
        for f in os.listdir(f"{root}/{dom}/src"):
            hs[hashlib.md5(open(f"{root}/{dom}/src/{f}", "rb").read()).hexdigest()] = f
        H[dom] = set(hs)
        res[dom]["unique_contents"] = len(hs); res[dom]["files"] = len(os.listdir(f"{root}/{dom}/src"))
    for i, a in enumerate(doms):
        for b in doms[i + 1:]:
            res[f"md5_overlap_{a}_{b}"] = len(H[a] & H[b])
    return res
OUT["data_qc"]["Heart-4cc"] = qc_dataset("Heart-4cc", "datasets/Heart-4cc", ["c1", "c2", "c3"])
OUT["data_qc"]["abdomen"] = qc_dataset("abdomen", "datasets/abdomen", ["GE", "PH", "SA"])
OUT["data_qc"]["FUSSD(spine)"] = qc_dataset("FUSSD", "datasets/FUSSD", ["GE", "PH", "SA"])
# Heart-4cc c3 timestamp near-duplicates (file name = capture time) between test/val and train
def ts_of(n):
    m = re.match(r"(\d{4})\.(\d{2})\.(\d{2}) (\d{2})\.(\d{2})\.(\d{2})", n)
    if not m:
        return None
    import datetime
    return datetime.datetime(*map(int, m.groups()))
for c in ["c1", "c2", "c3"]:
    tr = [ts_of(x["file_name"]) for x in json.load(open(f"datasets/Heart-4cc/{c}/train.json"))["images"]]; tr = sorted(t for t in tr if t)
    out = {}
    for sp in ["val", "test"]:
        ts = [ts_of(x["file_name"]) for x in json.load(open(f"datasets/Heart-4cc/{c}/{sp}.json"))["images"]]
        n_ts = sum(1 for t in ts if t)
        cnt = {60: 0, 300: 0, 1800: 0}
        for t in ts:
            if not t:
                continue
            dmin = min((abs((t - u).total_seconds()) for u in tr), default=1e9)
            for k in cnt:
                cnt[k] += int(dmin <= k)
        out[sp] = {"n_with_timestamp": n_ts, "n_total": len(ts), "within_60s_of_train": cnt[60], "within_5min": cnt[300], "within_30min": cnt[1800]}
    OUT["data_qc"]["Heart-4cc"][c]["timestamp_near_duplicates"] = out
# ---------------------------------------------------------------- composition + FPN mechanism
if os.path.exists("output/compose/summary_test.json"):
    OUT["compose"] = json.load(open("output/compose/summary_test.json"))
for s in SEEDS:
    f = f"output/analysis/fpn_levels_s{s}.json"
    if os.path.exists(f):
        d = json.load(open(f)); OUT["fpn"][str(s)] = {k: {kk: v[kk] for kk in ["tv_800", "tv_tac"] if kk in v} for k, v in d.items() if isinstance(v, dict)}
if os.path.exists("output/per_class_ap.json"):
    OUT["per_class"] = json.load(open("output/per_class_ap.json"))
json.dump(OUT, open("output/qc_collect.json", "w"), indent=1, default=str)
n = sum(len(v) for t in OUT["tables"].values() for v in t.values())
print("COLLECT_DONE cells", n, "val rows", {k: len(v) for k, v in OUT["val_sweeps"].items()}, "source models", len(OUT["source_models"]))
