# Faster re-implementation of the senior's `gaussian_fusion` (adapteacher/modeling/tta_aug.py) that keeps the
# heat-map construction bit-for-bit identical (same per-box expression, same sequential accumulation order) so the
# peak set is identical; only the per-peak box weighting is batched (float-order differences ~1e-5 px).
# Extra output field `support` = number of boxes contributing to each fused box.
import torch
import torch.nn.functional as F
from detectron2.structures import Boxes, Instances


def _empty(H, W, device):
    inst = Instances(image_size=(H, W))
    inst.pred_boxes = Boxes(torch.empty((0, 4), device=device))
    inst.scores = torch.empty((0,), device=device)
    inst.pred_classes = torch.empty((0,), dtype=torch.long, device=device)
    inst.support = torch.empty((0,), dtype=torch.long, device=device)
    return inst


@torch.no_grad()
def gaussian_fusion_fast(instances_list, image_shape, num_classes=1, sigma_scale=0.2, conf_thresh=0.05,
                         nms_thresh=0.05, window_size=3):
    H, W = int(image_shape[0]), int(image_shape[1])
    all_boxes, all_scores, all_classes = instances_list
    device = all_boxes.device
    if not torch.is_tensor(all_scores):
        all_scores = torch.stack(list(all_scores)) if len(all_scores) else torch.empty(0, device=device)
    if not torch.is_tensor(all_classes):
        all_classes = torch.stack(list(all_classes)) if len(all_classes) else torch.empty(0, dtype=torch.long, device=device)
    N = all_boxes.shape[0]
    if N == 0:
        return _empty(H, W, device)
    b = all_boxes.float()
    s = all_scores.float().reshape(-1)
    c = all_classes.long().reshape(-1)
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    sx = sigma_scale * (b[:, 2] - b[:, 0])
    sy = sigma_scale * (b[:, 3] - b[:, 1])
    # identical grid construction to the original gaussian_2d (meshgrid of y, x)
    xs = torch.arange(0, W).float().to(device)
    ys = torch.arange(0, H).float().to(device)
    yy, xx = torch.meshgrid(ys, xs)
    heatmaps = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
    for i in range(N):  # same expression and same accumulation order as the original -> identical heat maps
        g = torch.exp(-((xx - cx[i]) ** 2 / (2 * sx[i] ** 2) + (yy - cy[i]) ** 2 / (2 * sy[i] ** 2))) * s[i]
        heatmaps[c[i]] += g

    out_b, out_s, out_c, out_sup = [], [], [], []
    for k in range(num_classes):
        heat = heatmaps[k]
        pooled = F.max_pool2d(heat[None, None], kernel_size=window_size, stride=1, padding=window_size // 2)[0, 0]
        peaks = (pooled == heat) & (heat > nms_thresh)
        py, px = torch.where(peaks)
        if py.numel() == 0:
            continue
        idx = (c == k).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        bx, by, bsx, bsy = cx[idx][None, :], cy[idx][None, :], sx[idx][None, :], sy[idx][None, :]
        gval = torch.exp(-((px[:, None] - bx) ** 2 / (2 * bsx ** 2) + (py[:, None] - by) ** 2 / (2 * bsy ** 2)))  # [P, Nk]
        wgt = gval * s[idx][None, :]
        mask = wgt > conf_thresh
        wgt = wgt * mask
        wsum = wgt.sum(1)
        keep = wsum > 0
        if int(keep.sum()) == 0:
            continue
        wn = wgt[keep] / wsum[keep, None]
        out_b.append(wn @ b[idx])
        out_s.append(heat[py[keep], px[keep]])
        out_c.append(torch.full((int(keep.sum()),), k, dtype=torch.long, device=device))
        out_sup.append(mask[keep].sum(1))
    if not out_b:
        return _empty(H, W, device)
    inst = Instances(image_size=(H, W))
    inst.pred_boxes = Boxes(torch.cat(out_b))
    inst.scores = torch.cat(out_s)
    inst.pred_classes = torch.cat(out_c)
    inst.support = torch.cat(out_sup)
    return inst
