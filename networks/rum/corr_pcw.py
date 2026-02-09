import torch
import torch.nn.functional as F

# ============================================================
# Helpers (CSM-style PCW warping, align_corners=True)
# ============================================================

def ensure_batched(mat: torch.Tensor, B: int) -> torch.Tensor:
    """(3,3)/(4,4) -> (B,3,3)/(B,4,4); (1,*,*) -> (B,*,*)."""
    if mat.dim() == 2:
        return mat.unsqueeze(0).expand(B, -1, -1)
    if mat.dim() == 3 and mat.shape[0] == 1 and B > 1:
        return mat.expand(B, -1, -1)
    return mat


def as_intrinsics_3x3(K: torch.Tensor, B: int, device, dtype) -> torch.Tensor:
    """Accept (3,3)/(4,4)/(B,3,3)/(B,4,4) and return (B,3,3)."""
    K = ensure_batched(K, B).to(device=device, dtype=dtype)
    if K.shape[-2:] == (4, 4):
        K = K[:, :3, :3]
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"Expected K to be 3x3 or 4x4, got {tuple(K.shape)}")
    return K


def scale_intrinsics(K: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale fx,fy,cx,cy for feature-map resolution."""
    Ks = K.clone()
    Ks[..., 0, 0] *= scale
    Ks[..., 1, 1] *= scale
    Ks[..., 0, 2] *= scale
    Ks[..., 1, 2] *= scale
    return Ks


def se3_inv(T: torch.Tensor) -> torch.Tensor:
    """Invert SE(3) batched transform (B,4,4) or (4,4)."""
    if T.dim() == 2:
        T = T.unsqueeze(0)
    R = T[:, :3, :3]
    t = T[:, :3, 3:4]
    Rinv = R.transpose(1, 2)
    tinv = -Rinv @ t
    Tout = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
    Tout[:, :3, :3] = Rinv
    Tout[:, :3, 3:4] = tinv
    return Tout


def create_pixel_grid_flat(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    """Return homogeneous pixel grid (B,3,HW) with u in [0..W-1], v in [0..H-1]."""
    u = torch.arange(0, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    v = torch.arange(0, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    ones = torch.ones_like(u)
    pix = torch.cat((u, v, ones), dim=1)  # (B,3,H,W)
    return pix.view(B, 3, -1)             # (B,3,HW)


def pix_to_norm(coords_pix: torch.Tensor, H_in: int, W_in: int) -> torch.Tensor:
    """
    Pixel coords (..,2) -> normalized coords for grid_sample, align_corners=True.
    IMPORTANT: normalization uses INPUT (H_in, W_in), not output shape.
    """
    denom_w = max(W_in - 1, 1)
    denom_h = max(H_in - 1, 1)
    x = coords_pix[..., 0]
    y = coords_pix[..., 1]
    x_norm = 2.0 * (x / denom_w) - 1.0
    y_norm = 2.0 * (y / denom_h) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def grid_sample_pix(fmap: torch.Tensor, coords_pix: torch.Tensor) -> torch.Tensor:
    """
    fmap: (B,C,H,W)
    coords_pix: (B,H,W,2) in pixel units of fmap coordinates
    returns: (B,C,H,W)
    """
    B, C, H, W = fmap.shape
    grid = pix_to_norm(coords_pix, H, W)
    return F.grid_sample(
        fmap, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def grid_sample_many_pix(fmap: torch.Tensor, coords_pix: torch.Tensor) -> torch.Tensor:
    """
    fmap: (B,C,H,W)
    coords_pix: (B,S,H,W,2) in pixel units of fmap coordinates
    returns: (B,C,S,H,W)
    """
    B, C, H, W = fmap.shape
    S = coords_pix.shape[1]
    coords_pix = coords_pix.reshape(B, S * H, W, 2)   # treat as output height S*H
    grid = pix_to_norm(coords_pix, H, W)              # normalize using INPUT H,W
    out = F.grid_sample(
        fmap, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )                                                 # (B,C,S*H,W)
    return out.reshape(B, C, S, H, W)


def manual_pad(x: torch.Tensor, pady: int, padx: int) -> torch.Tensor:
    """Replicate padding for correlation windows."""
    return F.pad(x, (padx, padx, pady, pady), mode="replicate")


def build_base_offsets(psize=(1, 9), dilate=(1, 1), device=None, dtype=None) -> torch.Tensor:
    """Return (9,2) offsets (dx,dy) for (1,9) or (3,3) windows (both yield 9)."""
    psizey, psizex = psize
    dy, dx = dilate
    ry = (psizey // 2) * dy
    rx = (psizex // 2) * dx

    x_grid, y_grid = torch.meshgrid(
        torch.arange(-rx, rx + 1, dx, device=device, dtype=dtype),
        torch.arange(-ry, ry + 1, dy, device=device, dtype=dtype),
        indexing="xy",
    )
    offsets = torch.stack((x_grid, y_grid), dim=0).reshape(2, -1).permute(1, 0)  # (9,2)
    return offsets


# ============================================================
# AGCL_PCW (Pre-Correlation Warping; NO disparity/flow input)
# ============================================================

class AGCL_PCW:
    """
    AGCL_PCW = Pre-Correlation Warping + Grouped Correlation.

    PCW:
      - Uses depth_ref_left + extrinsics + intrinsics to warp BOTH complementary features
        (comp-left and comp-right) into the ref-left grid BEFORE correlation.

    No `flow` (disparity) is used here, because your depth already came from disparity.

    Modes:
      - iter_mode=True:
          fixed local correlation (like AGCL.corr_iter style, BUT without flow-warp)
      - iter_mode=False:
          if extra_offset is provided: adaptive offset correlation on ref grid
          else: fixed local correlation

    Output:
      - (B,36,H,W) = 4 groups * 9 offsets
    """

    def __init__(
        self,
        fmap_comp_left: torch.Tensor,
        fmap_comp_right: torch.Tensor,
        att=None,
        scale: float = 0.25,
    ):
        self.fmap_cl = fmap_comp_left
        self.fmap_cr = fmap_comp_right
        self.att = att
        self.scale = scale

        B, C, H, W = self.fmap_cl.shape
        assert self.fmap_cr.shape == (B, C, H, W)
        assert C % 4 == 0, "Channels must be divisible by 4 (group=4)."

    # ---------------------- PCW warping (CSM-style) ---------------------- #

    def warp_comp_to_refleft(
        self,
        fmap_comp: torch.Tensor,          # (B,C,H,W) in comp camera
        depth_ref: torch.Tensor,          # (B,1,*,*) depth in ref-left frame
        K_ref_fullres: torch.Tensor,      # full-res intrinsics
        K_comp_fullres: torch.Tensor,     # full-res intrinsics
        camComp2camRefLeft: torch.Tensor, # (B,4,4) comp->ref
    ):
        """
        Warp fmap_comp into ref-left grid using depth in ref-left.

        Returns:
          fmap_warp: (B,C,H,W)
          valid:     (B,1,H,W)
        """
        B, C, H, W = fmap_comp.shape
        device = fmap_comp.device
        dtype = fmap_comp.dtype

        if depth_ref.dim() == 3:
            depth_ref = depth_ref.unsqueeze(1)
        if depth_ref.shape[-2:] != (H, W):
            depth_ref = F.interpolate(depth_ref, size=(H, W), mode="bilinear", align_corners=True)
        depth_ref = depth_ref.to(device=device, dtype=dtype)

        # intrinsics: full-res -> feature-res
        K_ref = as_intrinsics_3x3(K_ref_fullres, B, device, dtype)
        K_cmp = as_intrinsics_3x3(K_comp_fullres, B, device, dtype)
        K_ref_s = scale_intrinsics(K_ref, self.scale)
        K_cmp_s = scale_intrinsics(K_cmp, self.scale)
        K_ref_inv = torch.inverse(K_ref_s)

        # extrinsics: given comp->ref, need ref->comp
        T_comp2ref = ensure_batched(camComp2camRefLeft, B).to(device=device, dtype=dtype)
        T_ref2comp = se3_inv(T_comp2ref)

        # pixel grid in ref-left
        pix_ref = create_pixel_grid_flat(B, H, W, device=device, dtype=K_ref_inv.dtype)  # (B,3,HW)

        # backproject to 3D in ref: X_ref = depth * K_ref^{-1} * p
        X_ref = (K_ref_inv @ pix_ref) * depth_ref.view(B, 1, -1)  # (B,3,HW)

        # transform to comp: X_comp = R*X_ref + t
        R = T_ref2comp[:, :3, :3]
        t = T_ref2comp[:, :3, 3:4]
        X_cmp = R @ X_ref + t                                     # (B,3,HW)

        Zc = X_cmp[:, 2]                                          # (B,HW)

        # project to comp pixels
        p2 = K_cmp_s @ X_cmp                                      # (B,3,HW)
        x = p2[:, 0] / (p2[:, 2] + 1e-7)
        y = p2[:, 1] / (p2[:, 2] + 1e-7)

        x_img = x.view(B, 1, H, W)
        y_img = y.view(B, 1, H, W)

        # valid mask: inside bounds + in front
        z_pos = (Zc > 0).view(B, 1, H, W)
        in_x = (x_img >= 0) & (x_img <= (W - 1))
        in_y = (y_img >= 0) & (y_img <= (H - 1))
        valid = (in_x & in_y & z_pos).float()

        coords_pix = torch.stack([x_img.squeeze(1), y_img.squeeze(1)], dim=-1)  # (B,H,W,2)
        fmap_warp = grid_sample_pix(fmap_comp, coords_pix)

        return fmap_warp, valid

    # ---------------------- Correlation building ---------------------- #

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):
        """
        Fixed-window correlation (like AGCL.get_correlation).
        left/right: (B,Cg,H,W) -> out: (B, psize_y*psize_x, H, W)
        """
        B, Cg, H, W = left_feature.shape
        dy, dx = dilate
        pady, padx = (psize[0] // 2) * dy, (psize[1] // 2) * dx

        right_pad = manual_pad(right_feature, pady, padx)

        corr_list = []
        for h in range(0, pady * 2 + 1, dy):
            for w in range(0, padx * 2 + 1, dx):
                right_crop = right_pad[:, :, h:h + H, w:w + W]
                corr = torch.mean(left_feature * right_crop, dim=1, keepdim=True)  # (B,1,H,W)
                corr_list.append(corr)

        return torch.cat(corr_list, dim=1)

    def corr_iter_no_flow(self, left_feature, right_feature, small_patch: bool):
        """
        iter_mode for PCW:
          - NO flow warp (because PCW already aligned features onto ref-left grid)
          - just fixed local correlation per group
        """
        B, C, H, W = left_feature.shape

        if small_patch:
            psize_list = [(3, 3)] * 4
            dilate_list = [(1, 1)] * 4
        else:
            psize_list = [(1, 9)] * 4
            dilate_list = [(1, 1)] * 4

        lefts = torch.split(left_feature, C // 4, dim=1)
        rights = torch.split(right_feature, C // 4, dim=1)

        corrs = []
        for i in range(4):
            corrs.append(self.get_correlation(lefts[i], rights[i], psize_list[i], dilate_list[i]))

        return torch.cat(corrs, dim=1)  # (B,36,H,W)

    def corr_att_offset_no_flow(self, left_feature, right_feature, extra_offset, small_patch: bool):
        """
        Adaptive-offset correlation on ref-left grid, NO flow term.

        extra_offset accepted shapes:
          - (B, 18, H, W)  (i.e., 9*2)
          - (B, 9, 2, H, W)
          - (B, 9, H, W, 2)
        """
        B, C, H, W = left_feature.shape
        Cg = C // 4
        search_num = 9

        # extra_offset -> (B,9,H,W,2)
        if extra_offset.dim() == 5:
            eo = extra_offset
            if eo.shape[-1] == 2 and eo.shape[1] == 9:
                pass  # already (B,9,H,W,2)
            elif eo.shape[1] == 9 and eo.shape[2] == 2:
                eo = eo.permute(0, 1, 3, 4, 2)  # (B,9,H,W,2)
            else:
                raise ValueError(f"Unsupported extra_offset 5D shape: {tuple(extra_offset.shape)}")
        else:
            eo = extra_offset.reshape(B, search_num, 2, H, W).permute(0, 1, 3, 4, 2)

        if small_patch:
            psize_list = [(3, 3)] * 4
            dilate_list = [(1, 1)] * 4
        else:
            psize_list = [(1, 9)] * 4
            dilate_list = [(1, 1)] * 4

        lefts = torch.split(left_feature, Cg, dim=1)
        rights = torch.split(right_feature, Cg, dim=1)

        # base grid coords in pixel units (B,H,W,2)
        u = torch.arange(0, W, device=left_feature.device, dtype=left_feature.dtype).view(1, 1, W).expand(B, H, W)
        v = torch.arange(0, H, device=left_feature.device, dtype=left_feature.dtype).view(1, H, 1).expand(B, H, W)
        uv = torch.stack([u, v], dim=-1).unsqueeze(1)  # (B,1,H,W,2)

        corrs = []
        for i in range(4):
            base = build_base_offsets(
                psize_list[i], dilate_list[i],
                device=left_feature.device,
                dtype=left_feature.dtype,
            )  # (9,2)
            base = base.view(1, search_num, 1, 1, 2).expand(B, -1, H, W, -1)  # (B,9,H,W,2)

            coords = uv + base + eo  # (B,9,H,W,2)
            rf_s = grid_sample_many_pix(rights[i], coords)  # (B,Cg,9,H,W)
            lf_r = lefts[i].unsqueeze(2).expand(-1, -1, search_num, -1, -1)

            corr = torch.mean(lf_r * rf_s, dim=1)  # (B,9,H,W)
            corrs.append(corr)

        return torch.cat(corrs, dim=1)  # (B,36,H,W)

    # ------------------------------ Call ------------------------------ #

    def __call__(
        self,
        depth_ref_left: torch.Tensor,
        camCompRight2camRefLeft: torch.Tensor,
        camCompLeft2camRefLeft: torch.Tensor,
        K_ref_left_fullres: torch.Tensor,
        K_comp_left_fullres: torch.Tensor,
        K_comp_right_fullres: torch.Tensor,
        *,
        extra_offset: torch.Tensor = None,
        small_patch: bool = False,
        iter_mode: bool = True,
        use_valid_mask: bool = True,
    ):
        """
        Returns:
          corr: (B,36,H,W)

        Notes:
          - iter_mode=True: fixed local correlation (PCW already did the warping)
          - iter_mode=False: if extra_offset is given => adaptive offsets, else fixed local correlation
        """
        # 1) PCW: warp both complementary features into ref-left grid
        cl_w, cl_m = self.warp_comp_to_refleft(
            self.fmap_cl, depth_ref_left,
            K_ref_left_fullres, K_comp_left_fullres,
            camCompLeft2camRefLeft
        )
        cr_w, cr_m = self.warp_comp_to_refleft(
            self.fmap_cr, depth_ref_left,
            K_ref_left_fullres, K_comp_right_fullres,
            camCompRight2camRefLeft
        )

        # 2) optional mask (recommended)
        if use_valid_mask:
            mask = cl_m * cr_m  # (B,1,H,W)
            cl_w = cl_w * mask
            cr_w = cr_w * mask

        # 3) optional attention (after warping, on same grid)
        if self.att is not None:
            B, C, H, W = cl_w.shape
            cl_t = cl_w.permute(0, 2, 3, 1).reshape(B, H * W, C)
            cr_t = cr_w.permute(0, 2, 3, 1).reshape(B, H * W, C)
            cl_t, cr_t = self.att(cl_t, cr_t)
            cl_w = cl_t.reshape(B, H, W, C).permute(0, 3, 1, 2)
            cr_w = cr_t.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 4) correlation
        if iter_mode:
            # “iter_mode” here means: same as AGCL.corr_iter style, BUT without flow warp
            return self.corr_iter_no_flow(cl_w, cr_w, small_patch)

        # non-iter mode: optionally adaptive offsets
        if extra_offset is not None:
            return self.corr_att_offset_no_flow(cl_w, cr_w, extra_offset, small_patch)

        # fallback: fixed local correlation
        return self.corr_iter_no_flow(cl_w, cr_w, small_patch)