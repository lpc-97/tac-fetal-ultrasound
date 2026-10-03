# Composition helper: apply a pre-computed per-image canonicalisation factor (from a TAC run's canon_factors.json)
# to a loader input dict, so that parameter-adapting TTA methods (AMROD / IoU-Filter / CoTTA-det) operate on the
# canonicalised image. detectron2 maps outputs back to (height, width) automatically, so evaluation is unchanged.
import json
import os

import torch
import torch.nn.functional as F


class CanonInput:
    def __init__(self, path, max_size=4000):
        d = json.load(open(path))
        self.f = d["factors"]; self.s0 = d.get("s0", 800); self.max_size = max_size
        self.n_hit = 0; self.n_miss = 0

    def factor(self, inp):
        key = os.path.basename(inp["file_name"])
        if key in self.f:
            self.n_hit += 1; return float(self.f[key][0])
        self.n_miss += 1; return 1.0

    def __call__(self, inp):
        """returns a shallow copy of inp whose 'image' is rescaled by the per-image factor (and the s0/800 calibration)."""
        f = self.factor(inp) * self.s0 / 800.0
        if abs(f - 1.0) < 1e-3:
            return dict(inp), f
        img = inp["image"]
        h, w = img.shape[-2:]
        nh, nw = int(round(h * f)), int(round(w * f))
        if max(nh, nw) > self.max_size:
            r = self.max_size / max(nh, nw); nh, nw = int(nh * r), int(nw * r)
        x = F.interpolate(img[None].float(), size=(nh, nw), mode="bilinear", align_corners=False)[0]
        out = dict(inp); out["image"] = x.clamp(0, 255).to(img.dtype) if img.dtype == torch.uint8 else x
        return out, f
