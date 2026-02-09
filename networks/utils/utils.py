import torch
import torch.nn.functional as F


def convex_upsample(x, mask, rate=4):
    """
    Upsample a field [N, C, H, W] -> [N, C, rate*H, rate*W]
    using convex combination (RAFT-style).
    Works for C=1 (depth/disp) or C=2 (flow), etc.
    """
    N, C, H, W = x.shape

    # mask: [N, 9*rate*rate, H, W] -> [N, 1, 9, rate, rate, H, W]
    mask = mask.view(N, 1, 9, rate, rate, H, W)
    mask = torch.softmax(mask, dim=2)

    # unfold: [N, C, H, W] -> [N, C*9, H*W]
    up = F.unfold(rate * x, kernel_size=3, padding=1)
    up = up.view(N, C, 9, 1, 1, H, W)

    # convex combination
    up = torch.sum(mask * up, dim=2)              # [N, C, 1, rate, H, rate, W]
    up = up.permute(0, 1, 4, 2, 5, 3)             # [N, C, H, rate, W, rate]
    return up.reshape(N, C, rate * H, rate * W)


def depth_to_disp(depth, focal_t, baseline_t, maxd_t=None, sparse=False):
    if sparse:
        valid = depth > 0
    if maxd_t is not None:
        depth = depth.clamp(max=maxd_t)
    eps = torch.finfo(depth.dtype).eps
    disp = focal_t * baseline_t / depth.clamp(min=eps)
    if sparse:
        disp = disp * valid
    return disp


def disp_to_depth(disp, focal_t, baseline_t, maxd_t=None, sparse=False):
    if sparse:
        valid = disp > 0
    eps = torch.finfo(disp.dtype).eps
    depth = focal_t * baseline_t / disp.clamp(min=eps)
    if maxd_t is not None:
        depth = depth.clamp(max=maxd_t)
    if sparse:
        depth = depth * valid
    return depth

def depth_norm_to_disp(depth_norm, focal_t, baseline_t, maxd_t, sparse=False):
    eps = torch.finfo(depth_norm.dtype).eps
    if sparse:
        valid = depth_norm > 0
        dn = torch.where(valid, depth_norm.clamp(max=1.0), depth_norm)  # keep zeros
        depth_metric = dn * maxd_t
        return depth_to_disp(depth_metric, focal_t, baseline_t, maxd_t=maxd_t, sparse=True)
    else:
        dn = depth_norm.clamp(min=eps, max=1.0)
        depth_metric = dn * maxd_t
        return depth_to_disp(depth_metric, focal_t, baseline_t, maxd_t=maxd_t, sparse=False)


def disp_to_depth_norm(disp, focal_t, baseline_t, maxd_t, sparse=False):
    depth_metric = disp_to_depth(disp, focal_t, baseline_t, maxd_t=maxd_t, sparse=sparse)
    depth_norm = (depth_metric / maxd_t).clamp(0.0, 1.0)
    return depth_norm, depth_metric


def make_sparse_gt_targets_from_depth(
    gt_depth_full,          # [B,1,H,W] metric, sparse with invalid==0
    focal_t, baseline_t, maxd_t,
    min_depth_t=None,       # [B,1,1,1] metric
):
    """
    Full-res sparse-safe GT only.
    Returns:
      - gt_valid: [B,1,H,W] bool
      - gt_depth: [B,1,H,W] metric (clamped only on valid)
      - gt_disp:  [B,1,H,W] disparity in pixel units (invalid stays 0; clamped only on valid)
    """
    eps = torch.finfo(gt_depth_full.dtype).eps
    gt_valid = gt_depth_full > 0

    # clamp depth only where valid
    depth = gt_depth_full
    if min_depth_t is not None:
        depth = torch.where(gt_valid, depth.clamp(min=min_depth_t), depth)
    depth = torch.where(gt_valid, depth.clamp(max=maxd_t), depth)

    # disparity bounds per-sample
    fB = focal_t * baseline_t
    disp_min = fB / maxd_t.clamp(min=eps)
    disp_max = None
    if min_depth_t is not None:
        disp_max = fB / min_depth_t.clamp(min=eps)

    # sparse disparity (invalid stays 0)
    disp = depth_to_disp(depth, focal_t, baseline_t, maxd_t=maxd_t, sparse=True)

    # clamp disparity only on valid pixels (avoid turning zeros into disp_min)
    if disp_max is not None:
        disp_clamped = disp.clamp(min=disp_min, max=disp_max)
        disp = torch.where(gt_valid, disp_clamped, disp)

    return {
        "gt_valid": gt_valid,
        "gt_depth": depth,
        "gt_disp": disp,
    }