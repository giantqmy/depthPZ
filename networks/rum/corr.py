import torch
import torch.nn.functional as F
from networks.rum.utils import bilinear_sampler, coords_grid, manual_pad

class AGCL:
    """
    Adaptive Group Correlation Layer (AGCL).

    Vanilla usage (UNCHANGED):
        corr_fn = AGCL(fmapL, fmapR)
        corr = corr_fn(flow, extra_offset=None, small_patch=False, iter_mode=True)

    New: secondary warp mode (opt-in):
        corr = corr_fn(
            flow, extra_offset=None, small_patch=False, iter_mode=True,
            warp_secondary=True,
            depth_ref=depth_stage,            # (B,1,H,W) in REF cam
            K_ref=K_ref_stage,                # (B,3,3) at THIS stage resolution
            K_sec_left=K_sec_left_stage,      # (B,3,3)
            K_sec_right=K_sec_right_stage,    # (B,3,3)
            T_ref_to_sec_left=T_ref_to_sec_L, # (B,4,4)
            T_ref_to_sec_right=T_ref_to_sec_R # (B,4,4)
        )

    Notes:
      - We warp BOTH fmap1 (sec-left) and fmap2 (sec-right) into ref grid using depth_ref.
      - After warping, we compute correlation exactly as usual, on the ref grid.
      - No reference_map / ds_factor.
    """

    def __init__(self, fmap1, fmap2, att=None):
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.att = att
        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device)

    # --------------------------- geometry warp helpers ---------------------------

    @staticmethod
    def _ensure_batch(mat: torch.Tensor, B: int) -> torch.Tensor:
        if mat is None:
            return None
        if mat.dim() == 2:
            return mat.unsqueeze(0).expand(B, -1, -1)
        return mat

    @staticmethod
    def _ensure_B33(K: torch.Tensor, B: int) -> torch.Tensor:
        if K is None:
            return None
        K = AGCL._ensure_batch(K, B)
        if K.shape[-2:] == (4, 4):
            K = K[:, :3, :3]
        return K

    @staticmethod
    def _pixel_grid(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
        # pixel coords in REF image grid (stage grid)
        u = torch.arange(0, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        v = torch.arange(0, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        ones = torch.ones_like(u)
        pix = torch.cat((u, v, ones), dim=1)  # (B,3,H,W)
        return pix.view(B, 3, -1)             # (B,3,HW)

    @staticmethod
    def _warp_cam_into_ref_grid(
        fmap_cam: torch.Tensor,
        depth_ref: torch.Tensor,
        K_ref: torch.Tensor,
        K_cam: torch.Tensor,
        T_ref_to_cam: torch.Tensor,
    ):
        """
        Warp fmap_cam (camera "cam") into REF pixel grid using depth_ref in REF cam frame.

        fmap_cam:     (B,C,H,W)  features at stage resolution
        depth_ref:    (B,1,H,W)  depth in REF cam coords (resized if needed)
        K_ref:        (B,3,3)    intrinsics for REF at stage resolution
        K_cam:        (B,3,3)    intrinsics for CAM at stage resolution
        T_ref_to_cam: (B,4,4)    transform points REF->CAM

        Returns:
            fmap_warp:  (B,C,H,W)
            valid:      (B,1,H,W)
        """
        B, C, H, W = fmap_cam.shape
        device = fmap_cam.device
        dtype = fmap_cam.dtype

        # Match depth resolution to feature resolution (you referenced align_corners=True usage)
        if depth_ref.shape[2:] != (H, W):
            depth_ref = F.interpolate(depth_ref, size=(H, W), mode="bilinear", align_corners=True)

        K_ref = AGCL._ensure_B33(K_ref, B)
        K_cam = AGCL._ensure_B33(K_cam, B)
        T_ref_to_cam = AGCL._ensure_batch(T_ref_to_cam, B)

        # If anything missing, fall back to identity (no warp)
        if (K_ref is None) or (K_cam is None) or (T_ref_to_cam is None):
            valid = torch.ones((B, 1, H, W), device=device, dtype=dtype)
            return fmap_cam, valid

        K_ref_inv = torch.inverse(K_ref)

        pix = AGCL._pixel_grid(B, H, W, device=device, dtype=K_ref.dtype)  # math in K dtype

        # X_ref = depth * K_ref^{-1} * p
        cam = K_ref_inv @ pix
        cam = cam * depth_ref.view(B, 1, -1)

        # X_cam = R * X_ref + t
        R = T_ref_to_cam[:, :3, :3]
        t = T_ref_to_cam[:, :3, 3:4]
        cam2 = R @ cam + t

        Z2 = cam2[:, 2]  # (B,HW)
        p2 = K_cam @ cam2
        x = p2[:, 0] / (p2[:, 2] + 1e-7)
        y = p2[:, 1] / (p2[:, 2] + 1e-7)

        x_img = x.view(B, 1, H, W)
        y_img = y.view(B, 1, H, W)
        z_pos = (Z2 > 0).view(B, 1, H, W)

        in_x = (x_img >= 0) & (x_img <= (W - 1))
        in_y = (y_img >= 0) & (y_img <= (H - 1))
        valid = (in_x & in_y & z_pos).to(dtype)

        # grid_sample normalized coords for THIS fmap_cam resolution
        x_norm = 2.0 * (x_img / (W - 1)) - 1.0
        y_norm = 2.0 * (y_img / (H - 1)) - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1).squeeze(1)  # (B,H,W,2)

        fmap_warp = F.grid_sample(
            fmap_cam,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return fmap_warp, valid

    # --------------------------- correlation code (as before) ---------------------------

    def __call__(
        self,
        flow,
        extra_offset,
        small_patch=False,
        iter_mode=False,
        # NEW (optional) secondary-warp arguments:
        warp_secondary=False,
        depth_ref=None,
        K_ref=None,
        K_sec_left=None,
        K_sec_right=None,
        T_ref_to_sec_left=None,
        T_ref_to_sec_right=None,
    ):
        left_feature = self.fmap1
        right_feature = self.fmap2

        # NEW: secondary warp mode (opt-in)
        if warp_secondary:
            # Warp BOTH left and right secondary features into the REF grid using depth_ref
            # (This is the key behavior you asked for.)
            left_feature, _ = self._warp_cam_into_ref_grid(
                fmap_cam=left_feature,
                depth_ref=depth_ref,
                K_ref=K_ref,
                K_cam=K_sec_left,
                T_ref_to_cam=T_ref_to_sec_left,
            )
            right_feature, _ = self._warp_cam_into_ref_grid(
                fmap_cam=right_feature,
                depth_ref=depth_ref,
                K_ref=K_ref,
                K_cam=K_sec_right,
                T_ref_to_cam=T_ref_to_sec_right,
            )

        # vanilla paths unchanged
        if iter_mode:
            return self.corr_iter(left_feature, right_feature, flow, small_patch)
        else:
            # NOTE: if you want offset-mode in your new model later, pass a real extra_offset.
            return self.corr_att_offset(left_feature, right_feature, flow, extra_offset, small_patch)

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):
        N, C, H, W = left_feature.shape
        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x
        right_pad = manual_pad(right_feature, pady, padx)

        corr_list = []
        for h in range(0, pady * 2 + 1, di_y):
            for w in range(0, padx * 2 + 1, di_x):
                right_crop = right_pad[:, :, h: h + H, w: w + W]
                corr = torch.mean(left_feature * right_crop, dim=1, keepdims=True)
                corr_list.append(corr)

        return torch.cat(corr_list, dim=1)

    def corr_iter(self, left_feature, right_feature, flow, small_patch):
        coords = self.coords + flow
        coords = coords.permute(0, 2, 3, 1)
        right_feature = bilinear_sampler(right_feature, coords)

        if small_patch:
            psize_list = [(3, 3)] * 4
            dilate_list = [(1, 1)] * 4
        else:
            psize_list = [(1, 9)] * 4
            dilate_list = [(1, 1)] * 4

        N, C, H, W = left_feature.shape
        assert (C % 4) == 0, f"AGCL expects channels divisible by 4, got {C}"
        lefts = torch.split(left_feature, C // 4, dim=1)
        rights = torch.split(right_feature, C // 4, dim=1)

        corrs = []
        for i in range(4):
            corrs.append(self.get_correlation(lefts[i], rights[i], psize_list[i], dilate_list[i]))

        return torch.cat(corrs, dim=1)

    def corr_att_offset(self, left_feature, right_feature, flow, extra_offset, small_patch):
        # unchanged from your new AGCL (kept, but your model currently uses iter_mode=True)
        N, C, H, W = left_feature.shape

        if self.att is not None:
            left_feature = left_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)
            right_feature = right_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)
            left_feature, right_feature = self.att(left_feature, right_feature)
            left_feature, right_feature = [
                x.reshape(N, H, W, C).permute(0, 3, 1, 2) for x in [left_feature, right_feature]
            ]

        lefts = torch.split(left_feature, C // 4, dim=1)
        rights = torch.split(right_feature, C // 4, dim=1)

        if small_patch:
            psize_list = [(3, 3)] * 4
            dilate_list = [(1, 1)] * 4
        else:
            psize_list = [(1, 9)] * 4
            dilate_list = [(1, 1)] * 4

        search_num = 9
        if extra_offset is None:
            raise ValueError("AGCL corr_att_offset requires extra_offset, got None.")

        extra_offset = extra_offset.reshape(N, search_num, 2, H, W).permute(0, 1, 3, 4, 2)

        corrs = []
        for i in range(4):
            lf, rf = lefts[i], rights[i]
            psize, dilate = psize_list[i], dilate_list[i]

            psizey, psizex = psize[0], psize[1]
            dilatey, dilatex = dilate[0], dilate[1]

            ry = psizey // 2 * dilatey
            rx = psizex // 2 * dilatex

            x_grid, y_grid = torch.meshgrid(
                torch.arange(-rx, rx + 1, dilatex, device=lf.device),
                torch.arange(-ry, ry + 1, dilatey, device=lf.device),
                indexing="xy",
            )

            offsets = torch.stack((x_grid, y_grid)).reshape(2, -1).permute(1, 0)
            for d in sorted((0, 2, 3)):
                offsets = offsets.unsqueeze(d)
            offsets = offsets.repeat_interleave(N, dim=0)
            offsets = offsets + extra_offset

            coords = self.coords + flow
            coords = coords.permute(0, 2, 3, 1)
            coords = torch.unsqueeze(coords, 1) + offsets
            coords = coords.reshape(N, -1, W, 2)

            rf_samp = bilinear_sampler(rf, coords)  # (N,C,search_num*H,W)
            Cq = rf.shape[1]
            rf_samp = rf_samp.reshape(N, Cq, -1, H, W)
            lf_rep = lf.unsqueeze(2).repeat_interleave(rf_samp.shape[2], dim=2)

            corr = torch.mean(lf_rep * rf_samp, dim=1)
            corrs.append(corr)

        return torch.cat(corrs, dim=1)
