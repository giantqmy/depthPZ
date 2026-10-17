
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.backbone.image_backbone import MPViTEncoder
from networks.rum.corr import AGCL
from networks.rum.update import UpdateBlock
from networks.csm.csm import CSM
from networks.utils.utils import make_sparse_gt_targets_from_depth, disp_to_depth
from networks.rum.attention import PositionEncodingSine, LocalFeatureTransformer


class GatedRCCBStereo(nn.Module):
    """
    Two-modality stereo model: reference modality (_ref) + complementary modality (_comp).

    Outputs:
      - disp_preds: list of disparity predictions appended at EVERY inner iteration (and now after both passes).
      - depth_preds: list of depth predictions corresponding to disp_preds (if geometry present)
      - disp, depth: final predictions
    """

    def __init__(self, model_args):
        super().__init__()

        def _get(cfg, key, default=None):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        self.args = model_args

        input_cams = _get(model_args, "input_cams", ["rgb", "gated", "rccb"])
        input_channels = _get(model_args, "input_channels", None)
        encoder_lightweight = _get(model_args, "encoder_lightweight", False)
        if input_channels is None:
            raise ValueError("model_args must define 'input_channels' dict for the encoders.")

        self.backbone = nn.ModuleDict()
        for cam in input_cams:
            if cam not in input_channels:
                continue
            self.backbone[cam] = MPViTEncoder(
                num_image_channels=input_channels[cam],
                lightweight=encoder_lightweight,
                lidar_input=False,
                out_channels=model_args.out_channels,  # expect 256
            )

        hidden_dim = _get(model_args, "hidden_dim")
        cor_planes = _get(model_args, "rum_cor_planes", None)
        mask_size = _get(model_args, "rum_mask_size", 4)
        if hidden_dim is None or cor_planes is None:
            raise ValueError("model_args must define hidden_dim and rum_cor_planes.")

        self.rum = UpdateBlock(hidden_dim=hidden_dim, cor_planes=cor_planes, mask_size=mask_size)

        if _get(model_args, "csm_init", True):
            self.csm = CSM(scale=0.25, pose_update=_get(model_args, "do_pose_update", False))
        else:
            self.csm = None


        d_model = 2 * int(hidden_dim)  # 256 when hidden_dim=128
        self.search_num = int(_get(model_args, "search_num", 9))
        self.range_16 = float(_get(model_args, "range_16", 1.0))
        self.range_8 = float(_get(model_args, "range_8", 1.0))

        self.conv_offset_16 = nn.Conv2d(d_model, self.search_num * 2, kernel_size=3, stride=1, padding=1)
        self.conv_offset_8 = nn.Conv2d(d_model, self.search_num * 2, kernel_size=3, stride=1, padding=1)

        # LoFTR attention blocks
        self.self_att_fn = LocalFeatureTransformer(
            d_model=d_model, nhead=8, layer_names=["self"] * 1, attention="linear"
        )
        self.cross_att_fn = LocalFeatureTransformer(
            d_model=d_model, nhead=8, layer_names=["cross"] * 1, attention="linear"
        )

    @staticmethod
    def _infer_integer_factor(h_ref, w_ref, h_comp, w_comp, tol=0.05):
        if h_ref <= 0 or w_ref <= 0:
            return 1
        rh = h_comp / float(h_ref)
        rw = w_comp / float(w_ref)
        f = int(round((rh + rw) * 0.5))
        if f < 1:
            return 1
        if abs(rh - f) > tol or abs(rw - f) > tol:
            return 1
        return f

    @staticmethod
    def _map_get(m, key):
        if m is None:
            raise KeyError(key)
        try:
            return m[key]
        except Exception:
            if hasattr(m, key):
                return getattr(m, key)
            raise KeyError(key)

    def forward(self, inputs, metas, **kwargs):
        def _get(cfg, key, default=None):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        B = len(inputs)
        device = next(self.parameters()).device

        outputs = {}

        # ---------------------------------------------------------------------
        # Choose modalities/sides
        # ---------------------------------------------------------------------
        aug = metas[0].get("augmentations", {}) if metas and isinstance(metas[0], dict) else {}
        modality = aug.get("ref_modality", _get(self.args, "ref_modality", "gated"))
        comp_mod = aug.get("comp_modality", _get(self.args, "comp_modality", "rccb"))
        ref_side = kwargs.get("ref_side", aug.get("ref_side", _get(self.args, "ref_side", "left")))

        iters_per_stage = int(kwargs.get("iters", _get(self.args, "refinement_iterations", 4)))
        iters_per_stage_hd = int(
            kwargs.get(
                "iters_hd",
                kwargs.get(
                    "iters_per_stage_hd",
                    _get(self.args, "refinement_iterations_hd", iters_per_stage),
                ),
            )
        )

        do_csm = (self.csm is not None) and aug.get("do_csm", True) and kwargs.get("do_csm", True)
        do_secondary = kwargs.get("do_secondary", True)

        outputs["chosen_modality"] = modality
        outputs["chosen_ref_side"] = ref_side
        outputs["augmentations"] = aug if aug else None
        outputs["iters_per_stage"] = iters_per_stage
        outputs["iters_per_stage_hd"] = iters_per_stage_hd
        outputs["do_secondary"] = bool(do_secondary)

        # ---------------------------------------------------------------------
        # GLOBAL vs LOCAL indices
        # ---------------------------------------------------------------------
        have_view_names = metas and isinstance(metas[0], dict) and ("view_names" in metas[0])

        def gidx(mod, side):
            return metas[0]["view_names"].index(f"{mod}_{side}")

        def lr_indices(mod):
            gl = gidx(mod, "left")
            gr = gidx(mod, "right")
            if gl < gr:
                return gl, gr, 0, 1
            else:
                return gl, gr, 1, 0

        if have_view_names:
            g_ref_left, g_ref_right, l_ref_left, l_ref_right = lr_indices(modality)
            g_ref_index = g_ref_left if ref_side == "left" else g_ref_right

            g_comp_left, g_comp_right, l_comp_left, l_comp_right = lr_indices(comp_mod)
            g_comp_index = g_comp_left if ref_side == "left" else g_comp_right
        else:
            l_ref_left, l_ref_right = 0, 1
            l_comp_left, l_comp_right = 0, 1
            g_ref_index = None
            g_comp_index = None
            g_comp_left = None
            g_comp_right = None

        # ---------------------------------------------------------------------
        # Stack images
        # ---------------------------------------------------------------------
        def _img_key(d, mod, which):
            if f"img_{which}" in d:
                return f"img_{which}"
            return f"img_{mod}"

        imgs_ref = torch.stack(
            [inputs[b][_img_key(inputs[b], modality, "ref")] for b in range(B)], dim=0
        ).to(device=device, dtype=torch.float32)  # [B,2,C,H,W]

        imgs_comp = torch.stack(
            [inputs[b][_img_key(inputs[b], comp_mod, "comp")] for b in range(B)], dim=0
        ).to(device=device, dtype=torch.float32)  # [B,2,C,Hc,Wc]

        dtype = imgs_ref.dtype
        eps = torch.finfo(dtype).eps

        imgs_ref = 2.0 * (imgs_ref / 255.0) - 1.0
        imgs_comp = 2.0 * (imgs_comp / 255.0) - 1.0

        left_raw_ref = imgs_ref[:, l_ref_left].contiguous()
        right_raw_ref = imgs_ref[:, l_ref_right].contiguous()
        left_raw_comp = imgs_comp[:, l_comp_left].contiguous()
        right_raw_comp = imgs_comp[:, l_comp_right].contiguous()

        # Right-trick
        do_right_trick = (ref_side == "right")
        if do_right_trick:
            left_inp = torch.flip(right_raw_ref, dims=[-1]).contiguous()
            right_inp = torch.flip(left_raw_ref, dims=[-1]).contiguous()

            left_inp_comp = torch.flip(right_raw_comp, dims=[-1]).contiguous()
            right_inp_comp = torch.flip(left_raw_comp, dims=[-1]).contiguous()
        else:
            left_inp, right_inp = left_raw_ref, right_raw_ref
            left_inp_comp, right_inp_comp = left_raw_comp, right_raw_comp

        # ---------------------------------------------------------------------
        # Intrinsics / baseline / depth bounds
        # ---------------------------------------------------------------------
        K_ref_b = None
        K_comp_b = None
        K_comp_left_b = None
        K_comp_right_b = None
        focal_t = None
        baseline_t = None
        have_geom = False

        try:
            if metas and isinstance(metas[0], dict) and ("cam2img" in metas[0]) and (g_ref_index is not None):
                K_ref_b = torch.stack(
                    [torch.as_tensor(metas[b]["cam2img"][g_ref_index], device=device, dtype=dtype) for b in range(B)],
                    dim=0
                )
                focal_t = K_ref_b[:, 0, 0].view(B, 1, 1, 1)

                baseline_t = torch.stack(
                    [torch.as_tensor(metas[b][f"{modality}_baseline"][0], device=device, dtype=dtype) for b in range(B)],
                    dim=0
                ).view(B, 1, 1, 1)

                if do_csm and (g_comp_index is not None):
                    K_comp_b = torch.stack(
                        [torch.as_tensor(metas[b]["cam2img"][g_comp_index], device=device, dtype=dtype) for b in range(B)],
                        dim=0
                    )

                if have_view_names and (g_comp_left is not None) and (g_comp_right is not None):
                    K_comp_left_b = torch.stack(
                        [torch.as_tensor(metas[b]["cam2img"][g_comp_left], device=device, dtype=dtype) for b in range(B)],
                        dim=0
                    )
                    K_comp_right_b = torch.stack(
                        [torch.as_tensor(metas[b]["cam2img"][g_comp_right], device=device, dtype=dtype) for b in range(B)],
                        dim=0
                    )

                have_geom = True
        except Exception:
            have_geom = False

        max_depth_map = _get(self.args, "max_depth", None)
        min_depth_map = _get(self.args, "min_depth", None)

        try:
            maxd = torch.as_tensor(self._map_get(max_depth_map, modality), device=device, dtype=dtype)
            mind = torch.as_tensor(self._map_get(min_depth_map, modality), device=device, dtype=dtype)
        except Exception:
            print(f"[WARN] max_depth/min_depth missing for '{modality}'. Using fallbacks maxd=100, mind=0.1")
            maxd = torch.as_tensor(100.0, device=device, dtype=dtype)
            mind = torch.as_tensor(0.1, device=device, dtype=dtype)

        maxd_t = maxd.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        min_depth_t = mind.view(1, 1, 1, 1).expand(B, 1, 1, 1)

        disp_min = disp_max = None
        if have_geom:
            fB = focal_t * baseline_t
            disp_min = fB / maxd_t.clamp(min=eps)
            disp_max = fB / min_depth_t.clamp(min=eps)

        outputs["focal"] = focal_t
        outputs["baseline"] = baseline_t
        outputs["maxd"] = maxd_t
        outputs["min_depth_t"] = min_depth_t
        outputs["disp_min"] = disp_min
        outputs["disp_max"] = disp_max

        # ---------------------------------------------------------------------
        # Feature extraction (native 1/4 resolution)
        # ---------------------------------------------------------------------
        if modality not in self.backbone or comp_mod not in self.backbone:
            raise KeyError(f"Missing encoder for modality='{modality}' or comp_mod='{comp_mod}' in self.backbone.")

        # Optional cross-encoder stereo smoke test: use the reference modality's
        # encoder for the left view and the complementary modality's encoder for
        # the right view.  This is useful for RGB-left/NIR-right transfer where
        # neither native modality pair represents the requested input.  The
        # complementary maps are aliases because CSM/secondary matching may be
        # disabled and the primary matcher only consumes fmapL_ref/fmapR_ref.
        if kwargs.get("hetero_encoders", False):
            if modality not in self.backbone or comp_mod not in self.backbone:
                raise KeyError(
                    f"Heterogeneous encoders require '{modality}' and '{comp_mod}' in self.backbone."
                )
            fmapL_ref = self.backbone[modality]([left_inp])[0]
            fmapR_ref = self.backbone[comp_mod]([right_inp_comp])[0]
            fmapL_comp, fmapR_comp = fmapL_ref, fmapR_ref
        else:
            fmapL_ref, fmapR_ref = self.backbone[modality]([left_inp, right_inp])
            fmapL_comp, fmapR_comp = self.backbone[comp_mod]([left_inp_comp, right_inp_comp])

        # infer resolution imbalance factor from ORIGINAL image sizes
        H_ref, W_ref = left_inp.shape[-2], left_inp.shape[-1]
        H_comp, W_comp = left_inp_comp.shape[-2], left_inp_comp.shape[-1]
        res_imbalance_factor = self._infer_integer_factor(H_ref, W_ref, H_comp, W_comp)
        outputs["res_imbalance_factor"] = res_imbalance_factor

        # build stage sizes (on feature grid)
        H4, W4 = fmapL_ref.shape[-2], fmapL_ref.shape[-1]
        stage_16 = (max(1, H4 // 4), max(1, W4 // 4))  # ~1/16 original
        stage_8 = (max(1, H4 // 2), max(1, W4 // 2))   # ~1/8 original
        stage_4 = (H4, W4)                              # ~1/4 original
        stages = [stage_16, stage_8, stage_4]

        high_stage = (fmapL_comp.shape[-2], fmapL_comp.shape[-1])
        appended_hd_stage = False
        if res_imbalance_factor != 1 and (high_stage[0] > stage_4[0] or high_stage[1] > stage_4[1]):
            stages.append(high_stage)
            appended_hd_stage = True

        outputs["stages"] = stages
        outputs["has_hd_stage"] = bool(appended_hd_stage)

        # ---------------------------------------------------------------------
        # Extrinsics for CSM + secondary warp
        # ---------------------------------------------------------------------
        T_ref_to_comp_left = None
        T_ref_to_comp_right = None
        dt_ref_to_comp_left = None
        dt_ref_to_comp_right = None

        if have_geom and metas and isinstance(metas[0], dict) and ("extrinsics" in metas[0]):
            try:
                ref_key = f"{modality}_{ref_side}"
                comp_left_key = f"{comp_mod}_left"
                comp_right_key = f"{comp_mod}_right"

                T_ref_to_comp_left = torch.stack(
                    [torch.as_tensor(metas[b]["extrinsics"][ref_key][comp_left_key], device=device, dtype=dtype) for b in range(B)],
                    dim=0
                )
                T_ref_to_comp_right = torch.stack(
                    [torch.as_tensor(metas[b]["extrinsics"][ref_key][comp_right_key], device=device, dtype=dtype) for b in range(B)],
                    dim=0
                )

                # timedifference optional
                if ("timedifference" in metas[0]) and (ref_key in metas[0]["timedifference"]):
                    dt_ref_to_comp_left = torch.stack(
                        [torch.as_tensor(metas[b]["timedifference"][ref_key][comp_left_key], device=device, dtype=dtype) for b in range(B)],
                        dim=0
                    ).view(B, 1)
                    dt_ref_to_comp_right = torch.stack(
                        [torch.as_tensor(metas[b]["timedifference"][ref_key][comp_right_key], device=device, dtype=dtype) for b in range(B)],
                        dim=0
                    ).view(B, 1)
            except Exception:
                T_ref_to_comp_left = None
                T_ref_to_comp_right = None
                dt_ref_to_comp_left = None
                dt_ref_to_comp_right = None

        # ---------------------------------------------------------------------
        # Helpers
        # ---------------------------------------------------------------------
        def resize_feat(x, size_hw):
            if x is None:
                return None
            if x.shape[-2:] == tuple(size_hw):
                return x
            return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

        def stride_from_stage(stage_hw):
            _, stage_w = stage_hw
            return W_ref / float(stage_w)

        def clamp_disp(disp_full):
            if disp_min is None or disp_max is None:
                return disp_full
            return disp_full.clamp(min=disp_min, max=disp_max)

        def depth_from_disp(disp_full):
            if not have_geom:
                return None
            fB_local = focal_t * baseline_t
            return fB_local / disp_full.clamp(min=eps)

        def is_hd_stage_index(si: int):
            is_last = (si == (len(stages) - 1))
            return bool(appended_hd_stage and is_last)

        def out_hw_for_stage(si: int):
            return (H_comp, W_comp) if is_hd_stage_index(si) else (H_ref, W_ref)

        def disp_scale_for_out_hw(out_hw):
            return float(out_hw[1]) / float(W_ref)

        # intrinsics helpers
        def _scale_K_uniform(Kb: torch.Tensor, s: float) -> torch.Tensor:
            K2 = Kb.clone()
            K2[:, 0, 0] *= s
            K2[:, 1, 1] *= s
            K2[:, 0, 2] *= s
            K2[:, 1, 2] *= s
            return K2

        def _flip_Kx(Kb: torch.Tensor, W: int) -> torch.Tensor:
            K2 = Kb.clone()
            K2[:, 0, 2] = (float(W) - 1.0) - K2[:, 0, 2]
            return K2

        # stereo: convert 1ch disparity-flow to 2ch (dx, dy=0) for AGCL coords math
        def _flow_to_xy(flow_1ch: torch.Tensor) -> torch.Tensor:
            return torch.cat([flow_1ch, torch.zeros_like(flow_1ch)], dim=1)

        # ---------------------------------------------------------------------
        # Init outputs
        # ---------------------------------------------------------------------
        outputs["disp_preds"] = []
        outputs["depth_preds"] = []

        # ---------------------------------------------------------------------
        # Hierarchical refinement
        # ---------------------------------------------------------------------
        flow = None
        net = None
        prev_stride = None

        hidden_dim = int(self.args.hidden_dim)

        for si, stage_hw in enumerate(stages):
            fL = resize_feat(fmapL_ref, stage_hw)     # (B,256,Hs,Ws)
            fR = resize_feat(fmapR_ref, stage_hw)

            fL_comp_s = resize_feat(fmapL_comp, stage_hw)  # (B,256,Hs,Ws)
            fR_comp_s = resize_feat(fmapR_comp, stage_hw)

            # -----------------------------------------------------------------
            # split 256 -> 128/128 for net/inp at this stage
            # -----------------------------------------------------------------
            assert fL.shape[1] == 2 * hidden_dim, f"Expected ref fmap channels {2*hidden_dim}, got {fL.shape[1]}"
            assert fL_comp_s.shape[1] == 2 * hidden_dim, f"Expected comp fmap channels {2*hidden_dim}, got {fL_comp_s.shape[1]}"

            net_raw, inp_raw = torch.split(fL, [hidden_dim, hidden_dim], dim=1)
            inp = F.relu(inp_raw)

            net_comp_raw, inp_comp_raw = torch.split(fL_comp_s, [hidden_dim, hidden_dim], dim=1)
            inp_comp = F.relu(inp_comp_raw)

            cur_stride = stride_from_stage(stage_hw)

            # -----------------------------------------------------------------
            # init / rescale net + flow between stages
            # -----------------------------------------------------------------
            if si == 0:
                net = torch.tanh(net_raw)  # (B,128,Hs,Ws)
                flow = torch.zeros((B, 1, stage_hw[0], stage_hw[1]), device=device, dtype=dtype)
            else:
                net = resize_feat(net, stage_hw)
                scale = (prev_stride / cur_stride) if (prev_stride is not None) else 1.0
                flow = resize_feat(flow, stage_hw) * float(scale)

            prev_stride = cur_stride

            # -----------------------------------------------------------------
            # stage_16 + stage_8 offsets (+ attention at stage_16)
            # -----------------------------------------------------------------
            is_stage_16 = (si == 0)
            is_stage_8 = (si == 1)

            corr_iter_mode = True
            extra_offset = None

            fL_corr = fL
            fR_corr = fR

            if is_stage_16:
                # offsets
                off = self.conv_offset_16(fL)
                extra_offset = self.range_16 * (torch.sigmoid(off) - 0.5) * 2.0
                corr_iter_mode = False  # use corr_att_offset

                # positional encoding + self-attention (old dw16 path)
                Hs, Ws = int(stage_hw[0]), int(stage_hw[1])
                pos_enc = PositionEncodingSine(d_model=fL.shape[1], max_shape=(Hs, Ws))
                fL_pe = pos_enc(fL)
                fR_pe = pos_enc(fR)

                B0, C0, _, _ = fL_pe.shape
                fL_seq = fL_pe.permute(0, 2, 3, 1).reshape(B0, Hs * Ws, C0)
                fR_seq = fR_pe.permute(0, 2, 3, 1).reshape(B0, Hs * Ws, C0)

                fL_seq, fR_seq = self.self_att_fn(fL_seq, fR_seq)

                fL_corr = fL_seq.reshape(B0, Hs, Ws, C0).permute(0, 3, 1, 2).contiguous()
                fR_corr = fR_seq.reshape(B0, Hs, Ws, C0).permute(0, 3, 1, 2).contiguous()

                corr_fn = AGCL(fL_corr, fR_corr, att=self.cross_att_fn)

            elif is_stage_8:
                off = self.conv_offset_8(fL)
                extra_offset = self.range_8 * (torch.sigmoid(off) - 0.5) * 2.0
                corr_iter_mode = False  # use corr_att_offset
                corr_fn = AGCL(fL_corr, fR_corr, att=None)

            else:
                # stage_4 (+ optional hd)
                corr_fn = AGCL(fL_corr, fR_corr, att=None)

            # secondary corr function
            corr_fn_sec = AGCL(fL_comp_s, fR_comp_s)

            iters_this_stage = iters_per_stage_hd if is_hd_stage_index(si) else iters_per_stage

            for it in range(iters_this_stage):
                small_patch = ((it % 2) == 1)

                # ---------------------------
                # PASS 1: primary stereo corr
                # ---------------------------
                flow_in = flow.detach()
                flow_xy = _flow_to_xy(flow_in)

                out_corrs = corr_fn(
                    flow_xy,
                    extra_offset,
                    small_patch=small_patch,
                    iter_mode=corr_iter_mode,
                )

                rum_inp = inp

                # ---------------------------
                # CSM fusion
                # ---------------------------
                if (
                    do_csm and have_geom
                    and (T_ref_to_comp_left is not None)
                    and (K_ref_b is not None) and (K_comp_b is not None)
                ):
                    disp_stage = -cur_stride * flow
                    disp_stage = clamp_disp(disp_stage)
                    depth_stage = depth_from_disp(disp_stage)

                    if depth_stage is not None:
                        Hs, Ws = int(stage_hw[0]), int(stage_hw[1])

                        s_ref_w = Ws / float(W_ref)
                        s_ref_h = Hs / float(H_ref)
                        s_ref = 0.5 * (s_ref_w + s_ref_h)

                        s_comp_w = Ws / float(W_comp)
                        s_comp_h = Hs / float(H_comp)
                        s_comp = 0.5 * (s_comp_w + s_comp_h)

                        csm_scale = float(getattr(self.csm, "scale", 0.25))
                        csm_scale = max(csm_scale, 1e-8)

                        K1 = K_ref_b
                        K2 = K_comp_b

                        if do_right_trick:
                            K1 = _flip_Kx(K1, W_ref)
                            K2 = _flip_Kx(K2, W_comp)

                        K1_pass = _scale_K_uniform(K1, s_ref / csm_scale)
                        K2_pass = _scale_K_uniform(K2, s_comp / csm_scale)

                        csm_out = self.csm(
                            fmap1=inp,
                            fmap2=inp_comp,
                            depth=depth_stage,
                            K1=K1_pass,
                            K2=K2_pass,
                            T_cam1_to_cam2=T_ref_to_comp_left,
                            timedifference=dt_ref_to_comp_left,
                        )
                        rum_inp = csm_out["fused"]

                net, flow, up_mask = self.rum(net=net, inp=rum_inp, corr=out_corrs, flow=flow_in)

                # clamp disparity then convert back to flow
                disp_stage = -cur_stride * flow
                disp_stage = clamp_disp(disp_stage)
                flow = -(disp_stage / cur_stride)

                # store prediction after pass 1
                out_hw = out_hw_for_stage(si)
                s = disp_scale_for_out_hw(out_hw)
                disp_out = F.interpolate(disp_stage, size=out_hw, mode="bilinear", align_corners=False) * s

                if do_right_trick:
                    disp_out = torch.flip(disp_out, dims=[-1]).contiguous()

                outputs["disp_preds"].append(disp_out)
                if have_geom:
                    focal_out = focal_t * s
                    depth_out = disp_to_depth(disp_out, focal_out, baseline_t, maxd_t)
                    outputs["depth_preds"].append(depth_out)

                # ---------------------------
                # PASS 2: secondary stereo corr (warped into ref grid)
                # ---------------------------
                if (
                    do_secondary and have_geom
                    and (T_ref_to_comp_left is not None) and (T_ref_to_comp_right is not None)
                    and (K_ref_b is not None) and (K_comp_left_b is not None) and (K_comp_right_b is not None)
                ):
                    # depth from latest disp_stage (ref grid)
                    depth_stage = depth_from_disp(disp_stage)
                    if depth_stage is not None:
                        Hs, Ws = int(stage_hw[0]), int(stage_hw[1])

                        # stage scaling for warping K (NOT CSM K logic)
                        s_ref_w = Ws / float(W_ref)
                        s_ref_h = Hs / float(H_ref)
                        s_ref = 0.5 * (s_ref_w + s_ref_h)

                        s_comp_w = Ws / float(W_comp)
                        s_comp_h = Hs / float(H_comp)
                        s_comp = 0.5 * (s_comp_w + s_comp_h)

                        K_ref_use = K_ref_b
                        K_sec_left_use = K_comp_left_b
                        K_sec_right_use = K_comp_right_b

                        # right-trick principal point correction on ORIGINAL grids
                        if do_right_trick:
                            K_ref_use = _flip_Kx(K_ref_use, W_ref)
                            K_sec_left_use = _flip_Kx(K_sec_left_use, W_comp)
                            K_sec_right_use = _flip_Kx(K_sec_right_use, W_comp)

                        K_ref_stage = _scale_K_uniform(K_ref_use, s_ref)
                        K_sec_left_stage = _scale_K_uniform(K_sec_left_use, s_comp)
                        K_sec_right_stage = _scale_K_uniform(K_sec_right_use, s_comp)

                        flow_in2 = flow.detach()
                        flow_xy2 = _flow_to_xy(flow_in2)

                        out_corrs2 = corr_fn_sec(
                            flow_xy2, None,
                            small_patch=small_patch,
                            iter_mode=True,
                            warp_secondary=True,
                            depth_ref=depth_stage,
                            K_ref=K_ref_stage,
                            K_sec_left=K_sec_left_stage,
                            K_sec_right=K_sec_right_stage,
                            T_ref_to_sec_left=T_ref_to_comp_left,
                            T_ref_to_sec_right=T_ref_to_comp_right,
                        )

                        net, flow, up_mask = self.rum(net=net, inp=inp, corr=out_corrs2, flow=flow_in2)

                        disp_stage = -cur_stride * flow
                        disp_stage = clamp_disp(disp_stage)
                        flow = -(disp_stage / cur_stride)

                        disp_out = F.interpolate(disp_stage, size=out_hw, mode="bilinear", align_corners=False) * s
                        if do_right_trick:
                            disp_out = torch.flip(disp_out, dims=[-1]).contiguous()

                        outputs["disp_preds"].append(disp_out)
                        if have_geom:
                            focal_out = focal_t * s
                            depth_out = disp_to_depth(disp_out, focal_out, baseline_t, maxd_t)
                            outputs["depth_preds"].append(depth_out)

        # final outputs
        outputs["disp"] = outputs["disp_preds"][-1] if len(outputs["disp_preds"]) > 0 else None
        outputs["depth"] = outputs["depth_preds"][-1] if len(outputs["depth_preds"]) > 0 else None

        # ---------------------------------------------------------------------
        # Sparse GT targets
        # ---------------------------------------------------------------------
        outputs["gt_depth"] = None
        outputs["gt_depth_hd"] = None
        outputs["gt_disp_hd"] = None

        gt_key = f"gt_sparse_depth_{modality}"
        hd_stage_active = bool(appended_hd_stage)

        if B > 0 and (gt_key in inputs[0]) and have_geom:
            gt_depth_full = torch.stack([inputs[b][gt_key] for b in range(B)], dim=0).to(device=device, dtype=dtype)
            gt_depth_full = gt_depth_full[:, l_ref_left if not do_right_trick else l_ref_right]
            outputs["gt_depth"] = gt_depth_full

            gt = make_sparse_gt_targets_from_depth(
                gt_depth_full=gt_depth_full,
                focal_t=focal_t,
                baseline_t=baseline_t,
                maxd_t=maxd_t,
                min_depth_t=min_depth_t,
            )
            outputs.update(gt)

            if hd_stage_active:
                gt_hd_key = f"gt_sparse_depth_{modality}_hd"
                if gt_hd_key not in inputs[0]:
                    gt_hd_key = "gt_sparse_depth_hd" if "gt_sparse_depth_hd" in inputs[0] else None

                if gt_hd_key is not None:
                    gt_depth_full_hd = torch.stack([inputs[b][gt_hd_key] for b in range(B)], dim=0).to(device=device, dtype=dtype)
                    gt_depth_full_hd = gt_depth_full_hd[:, l_ref_left if not do_right_trick else l_ref_right]
                    outputs["gt_depth_hd"] = gt_depth_full_hd

                    W_hd = int(gt_depth_full_hd.shape[-1])
                    s_hd = float(W_hd) / float(W_ref)
                    focal_hd = focal_t * s_hd
                    fB_hd = focal_hd * baseline_t

                    valid_hd = gt_depth_full_hd > 0
                    gt_disp_hd = torch.zeros_like(gt_depth_full_hd)
                    gt_disp_hd[valid_hd] = fB_hd.expand_as(gt_depth_full_hd)[valid_hd] / gt_depth_full_hd[valid_hd].clamp(min=eps)
                    outputs["gt_disp_hd"] = gt_disp_hd

        # ---------------------------------------------------------------------
        # debugging / compatibility
        # ---------------------------------------------------------------------
        if do_right_trick:
            left_inp_vis = torch.flip(left_inp, dims=[-1]).contiguous()
            right_inp_vis = torch.flip(right_inp, dims=[-1]).contiguous()
        else:
            left_inp_vis = left_inp
            right_inp_vis = right_inp

        outputs["ref_input"] = left_inp_vis
        outputs["ref_input_2"] = right_inp_vis
        outputs["ref_modality"] = modality

        return outputs
