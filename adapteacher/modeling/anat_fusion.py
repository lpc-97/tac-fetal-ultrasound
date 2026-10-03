# Anatomy-structured re-scoring of fused detections (v0.1, 2026-09-11).
# The 4-chamber view contains a fixed set of structures (one box each, ribs up to two). Given fused candidate boxes
# (from WBF / heatmap fusion over views), pick one candidate per structure that jointly maximises
#     sum_k score(x_k) + lam * mean_{k<l} plaus(x_k, x_l)
# where plaus is the source-domain layout prior (pair log-density, squashed to [0,1] via the prior's percentiles).
# Chosen candidates keep their score; competing candidates of the same structure are down-weighted by gamma.
# Solved by iterated conditional modes from the top-scoring initialisation (9 variables, few candidates each).
import math

import numpy as np
import torch
from detectron2.structures import Instances


class AnatRescorer:
    def __init__(self, prior, lam=0.5, gamma=0.3, max_cand=5, min_score=0.05):
        self.p = prior
        self.lam, self.gamma, self.max_cand, self.min_score = lam, gamma, max_cand, min_score
        pct = prior.thr  # 5th percentile of per-box mean pair log-density on source train
        # squash: log-density above the source median -> ~1, at the 5th percentile -> ~0.5, far below -> 0
        self.lo = pct
        self.hi = pct + 3.0

    def _plaus(self, ll):
        return float(np.clip((ll - self.lo) / (self.hi - self.lo), 0.0, 1.0))

    def _pair(self, bi, ci, bj, cj, W, H):
        key = f"{ci}_{cj}"
        if key not in self.p.pairs:
            return None
        mu, var = self.p.pairs[key]
        v = self.p._feat(bi, bj, W, H)
        return self._plaus(self.p._logpdf(v, mu, var))

    @torch.no_grad()
    def __call__(self, inst, W, H):
        n = len(inst)
        if n == 0:
            return inst
        boxes = inst.pred_boxes.tensor.cpu().numpy(); scores = inst.scores.cpu().numpy(); cls = inst.pred_classes.cpu().numpy()
        cxcywh = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], 1e-3), max(b[3] - b[1], 1e-3)) for b in boxes]
        K = self.p.K
        # candidates per structure (top max_cand by score); structures with max_count 2 (ribs) get 2 slots
        slots = []  # list of (class, candidate indices)
        for k in range(K):
            idx = [i for i in np.argsort(-scores) if cls[i] == k and scores[i] >= self.min_score][: self.max_cand]
            if not idx:
                continue
            for _ in range(max(1, int(self.p.max_count[k]))):
                slots.append((k, idx))
        if len(slots) < 2:
            return inst
        # assignment state: chosen candidate index per slot (or -1 = absent)
        state = [s[1][0] for s in slots]
        # ribs: second slot takes the 2nd candidate if available
        seen = {}
        for j, (k, idx) in enumerate(slots):
            seen.setdefault(k, 0)
            state[j] = idx[seen[k]] if seen[k] < len(idx) else -1
            seen[k] += 1

        def unary(j, i):
            return 0.0 if i < 0 else float(scores[i])

        def pairwise(j, i, others):
            if i < 0:
                return 0.0
            vals = []
            for jj, ii in others:
                if jj == j or ii < 0 or ii == i:
                    continue
                pl = self._pair(cxcywh[i], cls[i], cxcywh[ii], cls[ii], W, H)
                if pl is not None:
                    vals.append(pl)
            return float(np.mean(vals)) if vals else 0.5

        def energy(j, i, st):
            others = list(enumerate(st))
            return unary(j, i) + self.lam * pairwise(j, i, others)

        for _ in range(5):  # ICM sweeps
            changed = False
            for j, (k, idx) in enumerate(slots):
                used = {st for jj, st in enumerate(state) if jj != j}
                cands = [i for i in idx if i not in used] + [-1]
                best = max(cands, key=lambda i: energy(j, i, state))
                if best != state[j]:
                    state[j] = best; changed = True
            if not changed:
                break
        chosen = {i for i in state if i >= 0}
        new_scores = scores.copy()
        for i in range(n):
            if i not in chosen:
                new_scores[i] *= self.gamma
        out = Instances(inst.image_size)
        out.pred_boxes = inst.pred_boxes; out.pred_classes = inst.pred_classes
        out.scores = torch.as_tensor(new_scores, dtype=inst.scores.dtype, device=inst.scores.device)
        return out
