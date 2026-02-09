import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.csm.attention_fusion import iAFF
from networks.csm.pose_decoder import PoseDecoder


class CSM(nn.Module):
    """
    Cross-Stereo Matching (CSM) layer.

    Args:
        scale (float): feature map scale w.r.t. original image resolution (default 0.25).
        pose_update (bool): if True, run PoseDecoder to refine cam1_to_cam2 before fusion.

    Forward inputs:
        fmap1:        (B, C, H, W)  feature map from cam1 (left)
        fmap2:        (B, C, H, W)  feature map from cam2 (left of second stereo rig)
        depth:        (B, 1, H?, W?) depth map in cam1 frame (will be resized to fmap size if needed)
        K1:           (B, 3, 3) or (3, 3) intrinsics of cam1
        K2:           (B, 3, 3) or (3, 3) intrinsics of cam2
        T_cam1_to_cam2:
                      (B, 4, 4) or (4, 4) extrinsics: transform points from cam1 to cam2.
        timedifference:
                      (B, 1) or (B,) optional time difference between cam1 and cam2 timestamps.

    Returns:
        out: dict with keys:
            "fused":        fused feature map (B, C, H, W)
            "fmap2_warp":   warped fmap2 into cam1 frame (B, C, H, W)
            "warp_mask":    valid warp mask (B, 1, H, W)
            "T_updated":    updated extrinsic (B, 4, 4)
            "axisangle":    residual axis-angle from PoseDecoder (if pose_update)
            "translation":  residual translation from PoseDecoder (if pose_update)
    """

    def __init__(self, scale: float = 0.25, pose_update: bool = True):
        super().__init__()
        self.scale = scale
        self.pose_update = pose_update

        # Fusion block, as requested
        self.fusion = iAFF(channels=128)

        # Optional pose refinement module
        if self.pose_update:
            self.pose = PoseDecoder(
                num_ch_enc=[128, 256],
                num_input_features=1,
                num_frames_to_predict_for=1,
                stride=2,
            )
        else:
            self.pose = None

    # ---------------------------- Helpers ---------------------------- #

    @staticmethod
    def _scale_intrinsics(K: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Scale intrinsics for feature map resolution.
        Assumes K = [[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]]
        """
        K_scaled = K.clone()
        K_scaled[..., 0, 0] *= scale  # fx
        K_scaled[..., 1, 1] *= scale  # fy
        K_scaled[..., 0, 2] *= scale  # cx
        K_scaled[..., 1, 2] *= scale  # cy
        return K_scaled

    @staticmethod
    def _ensure_batch(mat: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Ensure matrix is batch-shaped: (B, ..., ...).
        If given (3,3) or (4,4), expand to (B, 3,3)/(B,4,4).
        """
        if mat.dim() == 2:
            mat = mat.unsqueeze(0).expand(batch_size, -1, -1)
        return mat

    @staticmethod
    def _create_pixel_grid(B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        u = torch.arange(0, W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        v = torch.arange(0, H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        ones = torch.ones_like(u)
        pix = torch.cat((u, v, ones), dim=1)  # (B, 3, H, W)
        return pix.view(B, 3, -1)  # (B, 3, HW)

    @staticmethod
    def _se3_from_rt(axisangle: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """
        Convert axis-angle and translation into SE3 matrix (B, 4, 4).

        axisangle: (B, 1, 3) or (B, 3) or (B, 3, 1)
        translation: same 'B' dimension, 3-vector.
        """
        if axisangle.dim() == 3:
            axisangle_vec = axisangle[..., 0:3]
            axisangle_vec = axisangle_vec.view(axisangle_vec.size(0), 3)
        elif axisangle.dim() == 2:
            axisangle_vec = axisangle
        else:  # (B, 3, 1)
            axisangle_vec = axisangle.view(axisangle.size(0), 3)

        if translation.dim() == 3:
            t_vec = translation[..., 0:3].view(translation.size(0), 3)
        elif translation.dim() == 2:
            t_vec = translation
        else:  # (B, 3, 1)
            t_vec = translation.view(translation.size(0), 3)

        B = axisangle_vec.size(0)

        angle = torch.norm(axisangle_vec + 1e-7, dim=1, keepdim=True)  # (B, 1)
        axis = axisangle_vec / (angle + 1e-7)                          # (B, 3)

        # Rodrigues' formula
        ca = torch.cos(angle).view(B, 1, 1)
        sa = torch.sin(angle).view(B, 1, 1)

        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

        zeros = torch.zeros_like(x)
        K = torch.stack(
            [
                zeros, -z, y,
                z, zeros, -x,
                -y, x, zeros
            ],
            dim=1
        ).view(B, 3, 3)

        I = torch.eye(3, device=axis.device, dtype=axis.dtype).unsqueeze(0).expand(B, -1, -1)
        R = I + sa * K + (1 - ca) * (K @ K)  # (B, 3, 3)

        T = torch.zeros(B, 4, 4, device=axis.device, dtype=axis.dtype)
        T[:, :3, :3] = R
        T[:, :3, 3] = t_vec
        T[:, 3, 3] = 1.0

        return T

    @staticmethod
    def _compose(T2: torch.Tensor, T1: torch.Tensor) -> torch.Tensor:
        """
        Compose SE3 transforms: return T = T2 * T1
        i.e., apply T1, then T2.
        """
        return T2 @ T1

    # -------------------------- Warping core ------------------------- #

    def _warp_fmap2_to_cam1(
        self,
        fmap2: torch.Tensor,
        depth: torch.Tensor,
        K1: torch.Tensor,
        K2: torch.Tensor,
        T_cam1_to_cam2: torch.Tensor,
    ):
        """
        Warp fmap2 into cam1 frame using depth (in cam1), cam1 intrinsics K1,
        cam2 intrinsics K2, and cam1->cam2 transform.

        Returns:
            fmap2_warp: (B, C, H, W)
            valid_mask: (B, 1, H, W)
        """
        B, C, H, W = fmap2.shape
        device = fmap2.device

        # Ensure depth resolution matches feature map
        if depth.shape[2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)

        # Scale intrinsics for feature resolution
        K1 = self._ensure_batch(K1, B)
        K2 = self._ensure_batch(K2, B)

        if K1.shape[-2:] == (4, 4): K1 = K1[:, :3, :3]
        if K2.shape[-2:] == (4, 4): K2 = K2[:, :3, :3]

        K1_scaled = self._scale_intrinsics(K1, self.scale)
        K2_scaled = self._scale_intrinsics(K2, self.scale)
        K1_inv = torch.inverse(K1_scaled)

        # Ensure extrinsics are batched
        T_cam1_to_cam2 = self._ensure_batch(T_cam1_to_cam2, B)

        # Pixel grid in cam1 image
        pix_cam1 = self._create_pixel_grid(B, H, W, device=fmap2.device, dtype=K1_inv.dtype)

        # 3D points in cam1 frame: X1 = depth * K1^{-1} * p1
        cam_points = K1_inv @ pix_cam1                 # (B, 3, HW)
        depth_flat = depth.view(B, 1, -1)
        cam_points = cam_points * depth_flat           # (B, 3, HW)

        # Transform to cam2 frame: X2 = R * X1 + t
        R = T_cam1_to_cam2[:, :3, :3]                  # (B, 3, 3)
        t = T_cam1_to_cam2[:, :3, 3:4]                 # (B, 3, 1)
        cam_points2 = R @ cam_points + t               # (B, 3, HW)

        X2, Y2, Z2 = cam_points2[:, 0], cam_points2[:, 1], cam_points2[:, 2]  # (B, HW)

        # Project into cam2 image plane using K2
        p2 = K2_scaled @ cam_points2                   # (B, 3, HW)
        x = p2[:, 0] / (p2[:, 2] + 1e-7)               # (B, HW)
        y = p2[:, 1] / (p2[:, 2] + 1e-7)               # (B, HW)

        # Validity mask: in front of camera and inside bounds
        x_img = x.view(B, 1, H, W)
        y_img = y.view(B, 1, H, W)
        z_pos = (Z2 > 0).view(B, 1, H, W)

        in_x = (x_img >= 0) & (x_img <= (W - 1))
        in_y = (y_img >= 0) & (y_img <= (H - 1))
        valid = in_x & in_y & z_pos
        valid = valid.float()

        # Normalize to [-1, 1] for grid_sample (coordinates for fmap2)
        x_norm = 2.0 * (x_img / (W - 1)) - 1.0
        y_norm = 2.0 * (y_img / (H - 1)) - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1)  # (B, 1, H, W, 2)
        grid = grid.squeeze(1)                        # (B, H, W, 2)

        fmap2_warp = F.grid_sample(
            fmap2,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return fmap2_warp, valid

    # ----------------------------- Forward --------------------------- #

    def forward(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        depth: torch.Tensor,
        K1: torch.Tensor,
        K2: torch.Tensor,
        T_cam1_to_cam2: torch.Tensor,
        timedifference: torch.Tensor = None,
        pose_update: bool = None,
    ):
        """
        Run CSM fusion with optional pose refinement.

        pose_update:
            If not None, overrides self.pose_update for this forward pass.
        """
        B, C, H, W = fmap1.shape
        assert fmap2.shape == fmap1.shape, "fmap1 and fmap2 must have same shape"

        if pose_update is None:
            pose_update = self.pose_update

        # First warp with the given extrinsic
        fmap2_warp, warp_mask = self._warp_fmap2_to_cam1(
            fmap2=fmap2,
            depth=depth,
            K1=K1,
            K2=K2,
            T_cam1_to_cam2=T_cam1_to_cam2,
        )

        T_updated = self._ensure_batch(T_cam1_to_cam2, B)
        axisangle_pred = None
        translation_pred = None

        # Optional pose refinement step
        if pose_update and self.pose is not None:
            # PoseDecoder expects list of lists as you described
            pose_input = [[torch.cat((fmap1, fmap2_warp), dim=1)]]

            # Mask: positive where warped fmap2 is non-zero on average
            pose_mask = (fmap2_warp.mean(dim=1, keepdim=True) > 0).float()

            axisangle_pred, translation_pred = self.pose(
                pose_input,
                mask=pose_mask,
                timedifference=timedifference,
            )

            # axisangle_pred, translation_pred are residuals -> SE3
            T_residual = self._se3_from_rt(axisangle_pred, translation_pred)

            # Compose residual with original extrinsic
            T_cam1_to_cam2_batched = self._ensure_batch(T_cam1_to_cam2, B)
            T_updated = self._compose(T_residual, T_cam1_to_cam2_batched)

            # Re-warp with updated extrinsic
            fmap2_warp, warp_mask = self._warp_fmap2_to_cam1(
                fmap2=fmap2,
                depth=depth,
                K1=K1,
                K2=K2,
                T_cam1_to_cam2=T_updated,
            )

        # Feature fusion with iAFF
        fused = self.fusion(fmap1, fmap2_warp)

        out = {
            "fused": fused,
            "fmap2_warp": fmap2_warp,
            "warp_mask": warp_mask,
            "T_updated": T_updated,
        }

        if axisangle_pred is not None:
            out["axisangle"] = axisangle_pred
            out["translation"] = translation_pred

        return out