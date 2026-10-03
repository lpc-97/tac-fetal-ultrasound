# Test-time Anatomical Canonicalisation (TAC, v0.4, 2026-09-13) — probabilistic version of the size-normalisation
# recipe (method_design.md §3). Per test image: pass 1 at the calibrated reference scale s0 -> MAP estimate of the
# image-level geometric nuisance theta = (u = log scale factor, v = log aspect factor) by IRLS under per-structure
# canonical log-size / log-aspect Gaussians (source train stats), a confidence-weighted inlier/outlier mixture and a
# shrinkage prior N(0, sigma_u^2) -> pass 2 on the canonicalised (anisotropically resized) image (+flip) -> WBF fusion of
# all views -> structure-assignment MAP (unary score + size term + pairwise layout prior, ICM) that re-scores
# competitors. Optional second round: re-estimate theta from the assigned boxes and re-run if it moved.
# Zero learnable parameters; only source-domain statistics.
import json
import logging
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
from detectron2.data.transforms import RandomFlip, ResizeTransform, NoOpTransform, apply_augmentations
from detectron2.structures import Boxes, Instances

from adapteacher.data.build import build_detection_test_loader
from adapteacher.engine.trainer import BaselineTrainer
from adapteacher.engine.tta_distill import AnatPrior
from adapteacher.modeling.tta_aug import GeneralizedRCNNWithTTA

logger = logging.getLogger("detectron2")


class CanonPrior:
    def __init__(self, path, s0_calib=True):
        pj = json.load(open(path))
        self.K = pj["num_classes"]
        self.anat = AnatPrior(path)
        c = pj["canonical_at_800"]
        self.s0 = int(pj.get("indomain_best_scale", 800)) if s0_calib else 800
        shift = math.log(self.s0 / 800.0)  # canonical stats measured at 800 -> at s0 the objects are s0/800 larger
        self.mu = {int(k): v["logsize_mean"] + shift for k, v in c.items()}
        self.sig = {int(k): v["logsize_std"] for k, v in c.items()}
        self.nu = {int(k): v["logaspect_mean"] for k, v in c.items()}
        self.tau = {int(k): v["logaspect_std"] for k, v in c.items()}
        self.max_count = pj["max_count"]


def make_view(inp, sx, sy, flip, max_size):
    """resize the input image by (sx, sy) (short-edge scale already folded in) and optionally flip; boxes mapped back with tfm.inverse()."""
    img = inp["image"].permute(1, 2, 0).numpy()
    shape = img.shape
    orig = (inp["height"], inp["width"])
    pre = ResizeTransform(orig[0], orig[1], shape[0], shape[1]) if shape[:2] != orig else NoOpTransform()
    new_h, new_w = int(round(shape[0] * sy)), int(round(shape[1] * sx))
    if max(new_h, new_w) > max_size:
        r = max_size / max(new_h, new_w); new_h, new_w = int(new_h * r), int(new_w * r)
    augs = [ResizeTransform(shape[0], shape[1], new_h, new_w)] + ([RandomFlip(prob=1.0)] if flip else [])
    new_img, tf = apply_augmentations(augs, np.copy(img))
    d = dict(inp); d["image"] = torch.from_numpy(np.ascontiguousarray(new_img.transpose(2, 0, 1)))
    return d, pre + tf


class CanonTTATrainer(BaselineTrainer):

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        C = cfg.SEMISUPNET.CANON
        logger.info(f"[CANON] cfg: {dict(C)}")
        K = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        prior = CanonPrior(C.ANAT_PRIOR, C.S0_CALIB)
        s0 = prior.s0
        model.eval()
        tta = GeneralizedRCNNWithTTA(cfg, model)
        from adapteacher.modeling.anat_fusion import AnatRescorer
        rescorer = AnatRescorer(prior.anat, lam=C.ANAT_LAM, gamma=C.ANAT_GAMMA) if C.ANAT_RESCORE else None
        loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        evaluator = cls.build_evaluator(cfg, cfg.DATASETS.TEST[0]); evaluator.reset()
        stats = defaultdict(float); t_start = time.perf_counter(); n_views = 0
        pass1 = {}
        hist_u, hist_v, hist_s = [], [], []
        per_image = {}  # file_name -> (f, v, ndet): dumped for mechanism analysis / composition experiments
        logger.info(f"[CANON] s0={s0} (calibrated), prior classes={len(prior.mu)}")

        def run(inp, views):
            outs = []
            with torch.no_grad(), tta._turn_off_roi_heads(["mask_on", "keypoint_on"]):
                dets = tta._batch_inference([v for v, _ in views])
            for (v, tf), o in zip(views, dets):
                pb = o.pred_boxes.tensor
                ob = torch.from_numpy(tf.inverse().apply_box(pb.cpu().numpy())).to(pb.device)
                outs.append((ob, o.scores, o.pred_classes))
            return outs

        def base_view(inp, extra_x=1.0, extra_y=1.0, flip=False):
            # short edge s0 (relative to the loader's default 800 resize already applied in inp["image"])
            hh, ww = inp["image"].shape[-2:]
            k = s0 / min(inp["height"], inp["width"])
            kk = min(k, C.MAX_SIZE / max(inp["height"], inp["width"]))
            # inp image is at loader scale; compute factor from loader-scaled image to target
            fx = kk * inp["width"] / ww * extra_x; fy = kk * inp["height"] / hh * extra_y
            return make_view(inp, fx, fy, flip, C.MAX_SIZE)

        def estimate_theta(ob, sc, cl, img_scale_x, img_scale_y, r_fixed=None):
            """IRLS MAP of (u, v) from boxes in ORIGINAL coords; img_scale_* = factor from original to the analysed view."""
            m = sc >= C.MIN_SCORE
            if int(m.sum()) < 1:
                return 0.0, 0.0, 0, 0.0, float(C.SIGMA_U)   # no evidence: posterior = prior
            b = ob[m]; c = cl[m].tolist(); p = sc[m].double().cpu().numpy()
            w = ((b[:, 2] - b[:, 0]).clamp_min(1) * img_scale_x).double().cpu().numpy()
            h = ((b[:, 3] - b[:, 1]).clamp_min(1) * img_scale_y).double().cpu().numpy()
            z = np.log(np.sqrt(w * h)); rho = np.log(w / h)
            mu = np.array([prior.mu.get(k, 0.0) for k in c]); sg2 = np.array([prior.sig.get(k, 0.3) ** 2 for k in c]) + C.SIGMA_OBS ** 2
            nu = np.array([prior.nu.get(k, 0.0) for k in c]); tau2 = np.array([prior.tau.get(k, 0.3) ** 2 for k in c]) + C.SIGMA_OBS ** 2
            u = 0.0; v = 0.0
            if C.ESTIMATOR == 'median':        # median of per-candidate votes, no weights, no outlier model, no prior
                return float(np.median(mu - z)), 0.0, int(m.sum()), float(len(z)), -1.0
            if C.ESTIMATOR == 'confmean':      # confidence-weighted mean of the votes, no variance weighting, no outlier model, no prior
                return float(np.sum(p * (mu - z)) / max(np.sum(p), 1e-6)), 0.0, int(m.sum()), float(np.sum(p)), -1.0
            if C.ESTIMATOR == 'noclass':       # class-agnostic prior: every candidate uses the pooled log-size distribution
                mu_all = np.array(list(prior.mu.values())); sd_all = np.array(list(prior.sig.values()))
                mu = np.full_like(mu, mu_all.mean()); sg2 = np.full_like(sg2, (sd_all ** 2).mean() + mu_all.var() + C.SIGMA_OBS ** 2)
            r = np.ones_like(p) if r_fixed is None else r_fixed
            for it in range(C.IRLS_ITERS):
                if r_fixed is None:
                    # responsibilities: inlier likelihood vs uniform outlier density
                    lik = np.exp(-0.5 * (z + u - mu) ** 2 / sg2) / np.sqrt(2 * np.pi * sg2)
                    r = (p * lik) / (p * lik + (1 - p) * C.OUTLIER_DENSITY + 1e-12)
                num = np.sum(r * (mu - z) / sg2); den = np.sum(r / sg2) + 1.0 / (C.SIGMA_U ** 2)
                u = float(num / den)
                if C.ASPECT:
                    numv = np.sum(r * (nu - rho) / tau2); denv = np.sum(r / tau2) + 1.0 / (C.SIGMA_V ** 2)
                    v = float(numv / denv)
            return u, v, int(m.sum()), float(np.sum(r)), float(1.0 / math.sqrt(den))   # Laplace posterior std of u at the MAP

        def fuse(views_out, H, W, w=None):
            # fusion strategy ablation (Information Fusion): wbf (default) | nms (concatenate + class-wise NMS) | heatmap (senior's
            # Gaussian heat-map fusion, fast implementation) | last (no fusion: only the first canonical view)
            if C.FUSION == "last" or len(views_out) == 1:
                ob, sc, cl = views_out[-1] if C.FUSION == "last" else views_out[0]
                if C.FUSION == "last" and len(views_out) > 1:
                    ob, sc, cl = views_out[1]          # views_out[0] = pass 1, views_out[1] = canonical (un-flipped) view
                inst = Instances((H, W)); inst.pred_boxes = Boxes(ob); inst.scores = sc; inst.pred_classes = cl
                return inst
            if C.FUSION == "nms":
                from torchvision.ops import batched_nms
                b = torch.cat([o for o, _, _ in views_out]); sc = torch.cat([x for _, x, _ in views_out]); cl = torch.cat([x for _, _, x in views_out])
                keep = batched_nms(b, sc, cl, cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
                inst = Instances((H, W)); inst.pred_boxes = Boxes(b[keep]); inst.scores = sc[keep]; inst.pred_classes = cl[keep]
                return inst
            if C.FUSION == "heatmap":
                from adapteacher.modeling.fusion_fast import gaussian_fusion_fast
                b = torch.cat([o for o, _, _ in views_out]); sc = [x for _, s_, _ in views_out for x in s_]; cl = [x for _, _, c_ in views_out for x in c_]
                inst = gaussian_fusion_fast((b, sc, cl), (H, W), num_classes=K)
                inst.scores = inst.scores / len(views_out)
                return inst
            from ensemble_boxes import weighted_boxes_fusion
            bl, sl, ll = [], [], []
            for ob, sc, cl in views_out:
                if len(sc) == 0:
                    bl.append(np.zeros((0, 4))); sl.append(np.zeros((0,))); ll.append(np.zeros((0,))); continue
                bl.append(np.clip(ob.cpu().numpy() / np.array([W, H, W, H]), 0, 1)); sl.append(sc.cpu().numpy()); ll.append(cl.cpu().numpy())
            assert w is None or len(w) == len(bl)
            fb, fs, fl = weighted_boxes_fusion(bl, sl, ll, weights=w, iou_thr=C.WBF_IOU, skip_box_thr=0.05, conf_type="avg")
            dev = model.device
            inst = Instances((H, W)); inst.pred_boxes = Boxes(torch.as_tensor(fb * np.array([W, H, W, H]), dtype=torch.float32, device=dev))
            inst.scores = torch.as_tensor(fs, dtype=torch.float32, device=dev); inst.pred_classes = torch.as_tensor(fl, dtype=torch.long, device=dev)
            return inst

        ext = json.load(open(C.FACTORS_FILE))["factors"] if C.FACTORS_FILE else None
        if ext is not None:
            logger.info(f"[CANON] external factors from {C.FACTORS_FILE}: {len(ext)} images")
        for t, inputs in enumerate(loader):
            inp = inputs[0]; H, W = inp["height"], inp["width"]
            k0 = min(s0 / min(H, W), C.MAX_SIZE / max(H, W))
            # ---- pass 1 at s0
            v0 = [base_view(inp)]
            out0 = run(inp, v0); n_views += 1
            u, v, ndet, reff, s_post = estimate_theta(out0[0][0], out0[0][1], out0[0][2], k0, k0)
            if C.DUMP_PASS1:
                _ob, _sc, _cl = out0[0]
                _m = (_sc >= C.MIN_SCORE)
                _b = _ob[_m]
                pass1[os.path.basename(inp['file_name'])] = {
                    'w': ((_b[:, 2] - _b[:, 0]).clamp_min(1) * k0).double().cpu().numpy().tolist(),
                    'h': ((_b[:, 3] - _b[:, 1]).clamp_min(1) * k0).double().cpu().numpy().tolist(),
                    'p': _sc[_m].double().cpu().numpy().tolist(),
                    'c': _cl[_m].cpu().numpy().tolist(),
                    'k0': float(k0), 's_post': float(s_post), 'H': int(H), 'W': int(W)}
            u = float(np.clip(u, math.log(C.F_MIN), math.log(C.F_MAX))); v = float(np.clip(v, -C.V_MAX, C.V_MAX))
            if C.FACTOR_CONST > 0:
                u, v = math.log(C.FACTOR_CONST), 0.0
            elif ext is not None:
                e = ext.get(os.path.basename(inp["file_name"]))
                u, v = (math.log(max(e[0], 1e-3)), 0.0) if e is not None else (0.0, 0.0)
                if e is None: stats["ext_missing"] += 1
            views_out = list(out0); view_w = [C.PASS1_WEIGHT]
            fx, fy = math.exp(u + v / 2), math.exp(u - v / 2)
            for rnd in range(1 if (ext is not None or C.FACTOR_CONST > 0) else C.ROUNDS):
                # ---- pass 2 on the canonicalised image (+flip)
                v1 = [base_view(inp, fx, fy, False)] + ([base_view(inp, fx, fy, True)] if C.FLIP else [])
                if C.FLIP0 and rnd == 0:
                    v1.append(base_view(inp, 1.0, 1.0, True))
                extra = list(C.EXTRA_SCALES) if rnd == 0 else []
                if rnd == 0 and C.ADAPT != 'none' and s_post >= 0:      # uncertainty-aware acquisition: bracket from the posterior std of u
                    hw = float(np.clip(C.ADAPT_K * s_post, C.ADAPT_MIN, C.ADAPT_MAX))
                    extra = [math.exp(-hw), math.exp(hw)] if (C.ADAPT == 'bracket' or s_post > C.ADAPT_TAU) else []
                    stats['n_extra'] += int(len(extra) > 0); stats['hw_sum'] += hw if extra else 0.0
                view_w += [1.0] + ([1.0] if C.FLIP else []) + ([C.PASS1_WEIGHT] if (C.FLIP0 and rnd == 0) else [])
                for mlt in extra:
                    mlt = float(math.exp(np.clip(math.log(mlt), math.log(C.F_MIN) - u, math.log(C.F_MAX) - u)))   # stay inside [F_MIN, F_MAX]
                    v1.append(base_view(inp, fx * mlt, fy * mlt, False)); view_w.append(0.25)
                    if C.FLIP:
                        v1.append(base_view(inp, fx * mlt, fy * mlt, True)); view_w.append(0.25)
                out1 = run(inp, v1); n_views += len(v1)
                views_out += out1
                if rnd + 1 < C.ROUNDS:
                    # re-estimate from the canonical view (its sizes are already normalised: analysed at scale k0*f)
                    u2, v2, nd2, _, _ = estimate_theta(out1[0][0], out1[0][1], out1[0][2], k0 * fx, k0 * fy)
                    if nd2 >= 1 and abs(u2) > math.log(C.ROUND_TOL):
                        u = float(np.clip(u + u2, math.log(C.F_MIN), math.log(C.F_MAX))); v = float(np.clip(v + v2, -C.V_MAX, C.V_MAX))
                        fx, fy = math.exp(u + v / 2), math.exp(u - v / 2); stats["rounds2"] += 1
                    else:
                        break
            use = views_out if C.FUSE_PASS1 else views_out[1:]
            out = fuse(use, H, W, (view_w if C.FUSE_PASS1 else view_w[1:]) if C.POST_WEIGHTS else None)
            if rescorer is not None:
                out = rescorer(out, W, H)
            evaluator.process(inputs, [{"instances": out}])
            hist_u.append(u); hist_v.append(v); stats["ndet"] += ndet; stats["reff"] += reff; hist_s.append(s_post)
            per_image[os.path.basename(inp["file_name"])] = [float(math.exp(u)), float(v), int(ndet), float(s_post)]
            if (t + 1) % C.LOG_EVERY == 0:
                el = time.perf_counter() - t_start
                logger.info(f"[CANON] {t+1}/{len(loader)} f={math.exp(u):.2f} (v={v:+.2f}) s={s_post:.3f} ndet={ndet} r_eff={reff:.1f} views/img={n_views/(t+1):.1f} {el/(t+1):.2f}s/img")
        results = evaluator.evaluate() or {}
        n = t + 1
        summary = {"n_img": n, "views_per_img": n_views / n, "s_per_img": (time.perf_counter() - t_start) / n, "s0": s0,
                   "f_median": float(np.exp(np.median(hist_u))), "f_p10": float(np.exp(np.percentile(hist_u, 10))), "f_p90": float(np.exp(np.percentile(hist_u, 90))),
                   "v_median": float(np.median(hist_v)), "rounds2": stats["rounds2"], "ndet_mean": stats["ndet"] / n, "reff_mean": stats["reff"] / n,
                   "AP": results.get("bbox", {}).get("AP"), "AP50": results.get("bbox", {}).get("AP50"),
                   "s_median": float(np.median([x for x in hist_s if x >= 0])) if any(x >= 0 for x in hist_s) else None,
                   "extra_frac": stats["n_extra"] / n, "hw_mean": (stats["hw_sum"] / stats["n_extra"]) if stats["n_extra"] else None}
        logger.info(f"[CANON] summary: {json.dumps(summary)}")
        json.dump(summary, open(os.path.join(cfg.OUTPUT_DIR, "canon_stats.json"), "w"), indent=1)
        json.dump({"s0": s0, "factors": per_image}, open(os.path.join(cfg.OUTPUT_DIR, "canon_factors.json"), "w"))
        if C.DUMP_PASS1:
            json.dump({"s0": s0, "candidates": pass1}, open(os.path.join(cfg.OUTPUT_DIR, "pass1_candidates.json"), "w"))
            logger.info(f"[CANON] dumped first-pass candidates for {len(pass1)} images")
        return results
