import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_base_grid(B, H, W, device, dtype):
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H,W,2]
    return grid.unsqueeze(0).repeat(B, 1, 1, 1)   # [B,H,W,2]


def _resize_disp_to(d, H, W):
    """Resize disparity to (H,W) and scale values (pixel units)."""
    if d is None or (not torch.is_tensor(d)):
        return d
    Hd, Wd = d.shape[-2], d.shape[-1]
    if (Hd, Wd) == (H, W):
        return d
    d_rs = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
    d_rs = d_rs * (float(W) / float(Wd))
    return d_rs


def _is_inference_tensor(t: torch.Tensor) -> bool:
    # torch tensors created under torch.inference_mode() have t.is_inference() == True
    return hasattr(t, "is_inference") and bool(t.is_inference())


def _safe_grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True):
    """
    If x or grid are inference tensors, grid_sample w/ grad can crash.
    Cloning turns them into normal tensors.
    """
    if _is_inference_tensor(x):
        x = x.clone()
    if _is_inference_tensor(grid):
        grid = grid.clone()
    return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def _warp_with_disp(x, disp, direction, return_mask=False, mask_mode="bilinear"):
    """
    x:    [B,C,H,W]
    disp: [B,1,H,W] in pixels (positive)
    direction:
      - "R2L": sample x(u - d)
      - "L2R": sample x(u + d)

    If return_mask=True: also returns coverage cov in [0,1] by warping ones.
    """
    B, _, H, W = x.shape
    device, dtype = x.device, x.dtype

    disp = _resize_disp_to(disp, H, W)

    grid = _make_base_grid(B, H, W, device, dtype)
    dx = 2.0 * disp.squeeze(1) / max(W - 1, 1)  # [B,H,W]

    if direction == "R2L":
        grid[..., 0] = grid[..., 0] - dx
    elif direction == "L2R":
        grid[..., 0] = grid[..., 0] + dx
    else:
        raise ValueError(f"Unknown direction: {direction}")

    y = _safe_grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    if not return_mask:
        return y

    ones = torch.ones((B, 1, H, W), device=device, dtype=dtype)
    cov = _safe_grid_sample(ones, grid, mode=mask_mode, padding_mode="zeros", align_corners=True)
    return y, cov


def _dilate_invalid(valid, dilate_px=3, horizontal_only=True):
    """
    valid: [B,1,H,W] float {0,1}
    Dilate invalid region (1-valid) by dilate_px.
    For rectified stereo, horizontal-only dilation usually makes most sense.
    """
    if dilate_px <= 0:
        return valid

    invalid = 1.0 - valid
    k = 2 * dilate_px + 1

    if horizontal_only:
        invalid = F.max_pool2d(invalid, kernel_size=(1, k), stride=1, padding=(0, k // 2))
    else:
        invalid = F.max_pool2d(invalid, kernel_size=k, stride=1, padding=k // 2)

    return 1.0 - invalid


def _ssim(x, y, C1=0.01**2, C2=0.03**2):
    """
    SSIM over 3x3 window, returns per-pixel map in [0,1] approx.
    x,y: [B,C,H,W] expected in [0,1]
    """
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x  = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y  = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_n / (ssim_d + 1e-12)
    return ssim_map.clamp(0, 1)


class PhotometricLRConsistencyLoss(nn.Module):
    def __init__(
        self,
        ssim_weight=0.85,
        lr_thresh=5.0,      # threshold for LR consistency (pixels)
        eps=1e-12,
        strict=True,
        # border-smear handling
        cov_thresh=0.999,   # how close to 1.0 coverage must be to be considered fully valid
        dilate_px=3,        # how much to expand invalid region
        horizontal_only=True,
        # occlusion mask config
        occ_thresh=None,    # if None, uses lr_thresh; you can set smaller for stricter masking
    ):
        super().__init__()
        self.ssim_weight = float(ssim_weight)
        self.lr_thresh = float(lr_thresh)
        self.eps = float(eps)
        self.strict = bool(strict)

        self.cov_thresh = float(cov_thresh)
        self.dilate_px = int(dilate_px)
        self.horizontal_only = bool(horizontal_only)

        self.occ_thresh = float(occ_thresh) if occ_thresh is not None else None

    def _valid_from_cov(self, cov):
        valid = (cov > self.cov_thresh).float()
        valid = _dilate_invalid(valid, dilate_px=self.dilate_px, horizontal_only=self.horizontal_only)
        return valid

    def forward(self, out_a, out_b, prev_total=None, losses=None, alpha=1.0):
        if losses is None:
            losses = {}

        device = next((v.device for v in out_a.values() if torch.is_tensor(v)), torch.device("cpu"))
        if prev_total is None:
            prev_total = torch.zeros((), device=device)

        req = ["disp", "ref_input", "ref_input_2", "chosen_ref_side"]
        for k in req:
            if self.strict and k not in out_a:
                raise KeyError(f"Missing {k} in out_a")
            if self.strict and k not in out_b:
                raise KeyError(f"Missing {k} in out_b")

        disp_a = out_a["disp"]
        disp_b = out_b["disp"]

        def to01(x):
            # assumes input in [-1, 1]
            return (x + 1.0) * 0.5

        # Map tensors to L/R consistently
        if out_a["chosen_ref_side"] == "left":
            I_L = out_a["ref_input"]
            I_R = out_a["ref_input_2"]
            d_L = disp_a
            d_R = disp_b
        else:
            I_R = out_a["ref_input"]
            I_L = out_a["ref_input_2"]
            d_R = disp_a
            d_L = disp_b

        B, _, H, W = I_L.shape
        d_L = _resize_disp_to(d_L, H, W)
        d_R = _resize_disp_to(d_R, H, W)

        # ============================================================
        # 1) Photometric warp (R -> L)  (KEEP GRAD through d_L)
        # ============================================================
        I_R_warp, cov_I = _warp_with_disp(I_R, d_L, "R2L", return_mask=True)
        valid_I = self._valid_from_cov(cov_I).detach()  # mask only

        # ============================================================
        # 2) Mutual LR consistency occlusion-ish mask (NO GRAD)
        #    Everything is converted to LEFT coordinates at the end.
        # ============================================================
        occ_thr = self.lr_thresh if self.occ_thresh is None else self.occ_thresh

        with torch.no_grad():
            # a) Right disparity into Left frame: d_R(x - d_L(x))
            d_R_in_L, cov_dRinL = _warp_with_disp(d_R, d_L, "R2L", return_mask=True)
            valid_dRinL = self._valid_from_cov(cov_dRinL)

            lr_err_L = (d_L - d_R_in_L).abs()
            lr_ok_L = (lr_err_L < occ_thr).float()  # LEFT coords

            # b) Left disparity into Right frame: d_L(x + d_R(x))
            d_L_in_R, cov_dLinR = _warp_with_disp(d_L, d_R, "L2R", return_mask=True)
            valid_dLinR = self._valid_from_cov(cov_dLinR)

            lr_err_R = (d_R - d_L_in_R).abs()
            lr_ok_R = (lr_err_R < occ_thr).float() * valid_dLinR  # RIGHT coords, include right validity

            # c) Bring right-frame ok mask back to LEFT frame using d_L
            lr_ok_R_in_L, cov_lrRinL = _warp_with_disp(lr_ok_R, d_L, "R2L", return_mask=True)
            valid_lrRinL = self._valid_from_cov(cov_lrRinL)

            # Warp makes it fractional; threshold back to binary
            lr_ok_R_in_L = (lr_ok_R_in_L > 0.999).float() * valid_lrRinL

            # Final occlusion/consistency mask in LEFT coords
            occ_mask_L = lr_ok_L * lr_ok_R_in_L * valid_dRinL

        # Final mask for LEFT photometric loss
        mask = valid_I * occ_mask_L  # [B,1,H,W]

        # ============================================================
        # 3) Photometric loss
        # ============================================================
        I_L01 = to01(I_L)
        I_Rw01 = to01(I_R_warp)

        l1 = (I_L01 - I_Rw01).abs().mean(dim=1, keepdim=True)
        ssim_map = _ssim(I_L01, I_Rw01).mean(dim=1, keepdim=True)
        photometric = self.ssim_weight * (1.0 - ssim_map) * 0.5 + (1.0 - self.ssim_weight) * l1

        denom = mask.sum().clamp(min=self.eps)
        loss_photo = (photometric * mask).sum() / denom

        total = prev_total + float(alpha) * loss_photo

        # ============================================================
        # 4) Logging (NO undefined vars)
        # ============================================================
        losses["loss/photo"] = loss_photo.detach()
        losses["loss/mask_frac"] = mask.mean().detach()
        losses["loss/valid_I_frac"] = valid_I.mean().detach()
        losses["loss/occ_mask_L_frac"] = occ_mask_L.mean().detach()
        losses["dbg/lr_ok_L_frac"] = lr_ok_L.mean().detach()
        losses["dbg/lr_ok_R_in_L_frac"] = lr_ok_R_in_L.mean().detach()
        losses["dbg/lr_err_L_mean"] = lr_err_L.mean().detach()
        losses["dbg/lr_err_R_mean"] = lr_err_R.mean().detach()
        losses["loss/total"] = total.detach()

        return total, losses, out_a


