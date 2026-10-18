"""Run a GatedRCCBStereo checkpoint as an RGB-left/NIR-right stereo model.

This experiment keeps the pretrained RCCB image encoder and shared stereo
matching/update blocks, while disabling CSM and the secondary cross-modal pass.
Raw RGB-left/NIR-right images are undistorted and stereo-rectified by default
with the calibration from depth-视差图代码/sgbmS0.py. Pass
--skip-rectification only when the inputs are already rectified.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT
DATA_ROOT = ROOT.parent / "depth-视差图代码" / "data" / "10"
RESULT_DIR = DATA_ROOT / "gatedrccbstereo_hetero_result"
LEFT_DEFAULT = DATA_ROOT / "2000-01-01-12-29-20_637.png"
RIGHT_DEFAULT = DATA_ROOT / "2000-01-01-12-29-20_637_hsi_nir.png"
CHECKPOINT_DEFAULT = MODEL_ROOT / "gatedrccbstereo.pt"


def install_optional_dependency_shims() -> None:
    """Provide the small mmcv/mmseg API surface used by this model's UPerHead.

    The inference environment has PyTorch/timm but not mmcv or mmseg. These
    shims preserve the module names/layout needed to load the checkpoint.
    """
    if "mmcv.cnn" not in sys.modules:
        mmcv = types.ModuleType("mmcv")
        cnn = types.ModuleType("mmcv.cnn")

        def build_norm_layer(cfg, channels):
            cfg = cfg or {"type": "BN"}
            norm_type = cfg.get("type", "BN")
            if norm_type in ("BN", "BN2d", "SyncBN"):
                norm = nn.BatchNorm2d(channels, eps=cfg.get("eps", 1e-5), momentum=cfg.get("momentum", 0.1))
            else:
                raise ValueError(f"Unsupported norm type in compatibility shim: {norm_type}")
            if not cfg.get("requires_grad", True):
                for parameter in norm.parameters():
                    parameter.requires_grad_(False)
            return "bn", norm

        class ConvModule(nn.Module):
            def __init__(
                self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias="auto",
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=True,
                **kwargs,
            ):
                super().__init__()
                if conv_cfg is not None:
                    raise ValueError("This test shim only supports standard Conv2d")
                use_norm = norm_cfg is not None
                if bias == "auto":
                    bias = not use_norm
                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                )
                self.bn = build_norm_layer(norm_cfg, out_channels)[1] if use_norm else None
                if act_cfg is None or act_cfg.get("type", "ReLU") == "ReLU":
                    self.activate = nn.ReLU(inplace=inplace)
                elif act_cfg.get("type") == "GELU":
                    self.activate = nn.GELU()
                else:
                    raise ValueError(f"Unsupported activation in compatibility shim: {act_cfg}")

            def forward(self, x):
                x = self.conv(x)
                if self.bn is not None:
                    x = self.bn(x)
                return self.activate(x) if self.activate is not None else x

        cnn.build_norm_layer = build_norm_layer
        cnn.ConvModule = ConvModule
        mmcv.cnn = cnn
        sys.modules["mmcv"] = mmcv
        sys.modules["mmcv.cnn"] = cnn

    if "mmseg.models.decode_heads.decode_head" not in sys.modules:
        mmseg = types.ModuleType("mmseg")
        models = types.ModuleType("mmseg.models")
        heads = types.ModuleType("mmseg.models.decode_heads")
        decode_head = types.ModuleType("mmseg.models.decode_heads.decode_head")
        psp_head = types.ModuleType("mmseg.models.decode_heads.psp_head")
        ConvModule = sys.modules["mmcv.cnn"].ConvModule

        class BaseDecodeHead(nn.Module):
            def __init__(
                self,
                in_channels,
                channels,
                num_classes,
                in_index,
                input_transform=None,
                dropout_ratio=0.1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                align_corners=False,
                **kwargs,
            ):
                super().__init__()
                self.in_channels = in_channels
                self.channels = channels
                self.num_classes = num_classes
                self.in_index = in_index
                self.input_transform = input_transform
                self.conv_cfg = conv_cfg
                self.norm_cfg = norm_cfg
                self.act_cfg = act_cfg or {"type": "ReLU"}
                self.align_corners = align_corners
                self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
                self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

            def _transform_inputs(self, inputs):
                return [inputs[i] for i in self.in_index]

            def cls_seg(self, feature):
                if self.dropout is not None:
                    feature = self.dropout(feature)
                return self.conv_seg(feature)

        class PPM(nn.ModuleList):
            def __init__(self, pool_scales, in_channels, channels, conv_cfg=None, norm_cfg=None, act_cfg=None, align_corners=False):
                modules = [
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(scale),
                        ConvModule(
                            in_channels,
                            channels,
                            kernel_size=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                        ),
                    )
                    for scale in pool_scales
                ]
                super().__init__(modules)
                self.align_corners = align_corners

            def forward(self, x):
                return [
                    F.interpolate(module(x), size=x.shape[2:], mode="bilinear", align_corners=self.align_corners)
                    for module in self
                ]

        decode_head.BaseDecodeHead = BaseDecodeHead
        psp_head.PPM = PPM
        sys.modules.update(
            {
                "mmseg": mmseg,
                "mmseg.models": models,
                "mmseg.models.decode_heads": heads,
                "mmseg.models.decode_heads.decode_head": decode_head,
                "mmseg.models.decode_heads.psp_head": psp_head,
            }
        )


def read_image(path: Path) -> np.ndarray:
    # OpenCV's Windows imread does not reliably accept non-ASCII paths.
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] == 4:
        image = image[..., :3]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel image after conversion; got {image.shape} from {path}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    # The author's RCCB transform converts its camera image from BGR to RGB
    # before the model sees it. Match that channel convention for the RGB input.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def stereo_calibration(size):
    """Build rectification maps from the calibration used by sgbmS0.py."""
    width, height = size
    if (width, height) != (640, 512):
        raise ValueError(
            f"Calibration is for 640x512 images, but inputs are {width}x{height}. "
            "Resize/recalculate calibration for another resolution first."
        )
    k_left = np.array(
        [[1265.5616099448073, 0.0, 492.5894472471048],
         [0.0, 1265.0089455071945, 257.24307296897143],
         [0.0, 0.0, 1.0]], dtype=np.float64)
    k_right = np.array(
        [[1277.6494552887766, 0.0, 474.185263116495],
         [0.0, 1278.5279034618504, 260.09879728682546],
         [0.0, 0.0, 1.0]], dtype=np.float64)
    d_left = np.array([[-0.02096660215918474, 0.2409083928738356,
                        -0.005298800804070935, 0.007648334809400866, 0.0]], dtype=np.float64)
    d_right = np.array([[-0.06681173268876799, 0.39362342592776267,
                         -0.007450119715294455, 0.0034144197720179475, 0.0]], dtype=np.float64)
    rotation = np.array(
        [[0.9998762017154619, 0.003542133347215963, 0.015330836063682721],
         [-0.003479358861003513, 0.9999854622282568, -0.004119392437720074],
         [-0.01534520462481131, 0.004065540983698188, 0.9998739900965185]], dtype=np.float64)
    translation = np.array(
        [-0.055078917279073276, 0.00011619425753116915,
         0.0017826427597072247], dtype=np.float64)
    r_left, r_right, p_left, p_right, q, roi_left, roi_right = cv2.stereoRectify(
        k_left, d_left, k_right, d_right, (width, height), rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9,
    )
    map_left = cv2.initUndistortRectifyMap(
        k_left, d_left, r_left, p_left, (width, height), cv2.CV_32FC1)
    map_right = cv2.initUndistortRectifyMap(
        k_right, d_right, r_right, p_right, (width, height), cv2.CV_32FC1)
    return {
        "map_left": map_left,
        "map_right": map_right,
        "P1": p_left,
        "P2": p_right,
        "Q": q,
        "roi_left": roi_left,
        "roi_right": roi_right,
        "baseline_m": float(-p_right[0, 3] / p_right[0, 0]),
    }


def rectify_pair(left_rgb, right_rgb, calibration):
    """Undistort and rectify RGB-left and NIR-right images."""
    right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
    left_rectified = cv2.remap(
        left_rgb, *calibration["map_left"], cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    right_rectified = cv2.remap(
        right_gray, *calibration["map_right"], cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return left_rectified, right_rectified


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode output image: {path}")
    encoded.tofile(str(path))


def make_model(checkpoint_path: Path | None, device: torch.device,
               reinit_encoders: bool = False, gated_channels: int | None = None):
    sys.path.insert(0, str(MODEL_ROOT))
    install_optional_dependency_shims()
    from networks.gatedrccbstereo import GatedRCCBStereo

    checkpoint = (torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
                  if checkpoint_path is not None else {})
    model_state = checkpoint.get("model", checkpoint)
    if checkpoint_path is None or reinit_encoders:
        gated_channels = 1 if gated_channels is None else int(gated_channels)
    else:
        rgb_key = "backbone.rccb.mpvit.stem.0.conv.weight"
        nir_key = "backbone.gated.mpvit.stem.0.conv.weight"
        if rgb_key not in model_state or nir_key not in model_state:
            raise RuntimeError("Inference requires checkpoint weights for both image encoders.")
        if model_state[rgb_key].shape[1] != 3:
            raise ValueError("Expected a 3-channel RGB encoder in the checkpoint.")
        saved_channels = int(model_state[nir_key].shape[1])
        if gated_channels is not None and int(gated_channels) != saved_channels:
            raise ValueError(f"Requested NIR channels {gated_channels}, checkpoint has {saved_channels}.")
        gated_channels = saved_channels
    if gated_channels not in (1, 5):
        raise ValueError(f"Supported NIR channel counts are 1 and 5; got {gated_channels}.")

    args = SimpleNamespace(
        model_name="gatedrccbstereo_heterogeneous_stereo_only",
        input_cams=["rccb", "gated"],
        input_channels={"rccb": 3, "gated": int(gated_channels)},
        encoder_lightweight=False,
        hidden_dim=128,
        out_channels=256,
        rum_cor_planes=36,
        rum_mask_size=4,
        csm_init=False,
        do_pose_update=False,
        ref_modality="rccb",
        comp_modality="gated",
        ref_side="left",
        refinement_iterations=4,
        refinement_iterations_hd=2,
        max_depth={"rccb": 220.0},
        min_depth={"rccb": 1.0},
    )
    model = GatedRCCBStereo(model_args=args)
    if checkpoint_path is None:
        model.to(device).eval()
        return model, 0

    encoder_prefixes = (
        "backbone.rccb.",
        "backbone.gated.",
    )
    prefixes = (
        "rum.",
        "self_att_fn.",
        "cross_att_fn.",
        "conv_offset_16.",
        "conv_offset_8.",
    )
    load_prefixes = prefixes if reinit_encoders else encoder_prefixes + prefixes
    selected = {key: value for key, value in model_state.items() if key.startswith(load_prefixes)}
    expected_keys = {key for key in model.state_dict()
                     if not (reinit_encoders and key.startswith(encoder_prefixes))}
    if set(selected) != expected_keys:
        missing = sorted(expected_keys - set(selected))
        extra = sorted(set(selected) - expected_keys)
        raise RuntimeError(
            "Checkpoint/model parameter mismatch; refusing incomplete checkpoint loading. "
            f"Missing={missing[:12]}, extra={extra[:12]}"
        )
    # In transfer-training mode the two image encoders intentionally retain
    # PyTorch's random initialization; only the stereo/update trunk is loaded.
    model.load_state_dict(selected, strict=not reinit_encoders)
    model.to(device).eval()
    return model, len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=LEFT_DEFAULT)
    parser.add_argument("--right", type=Path, default=RIGHT_DEFAULT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output", type=Path, default=RESULT_DIR)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--skip-rectification", action="store_true",
                        help="Input pair is already rectified; skip calibration/remapping")
    args = parser.parse_args()

    left_raw = read_image(args.left)
    right_raw = read_image(args.right)
    if left_raw.shape != right_raw.shape:
        raise ValueError(f"Left/right sizes differ: {left_raw.shape} vs {right_raw.shape}")
    if left_raw.shape[:2] != (512, 640):
        raise ValueError(f"Expected 640x512 input images, got {left_raw.shape[:2]}")

    args.output.mkdir(parents=True, exist_ok=True)
    rectification = None
    if args.skip_rectification:
        left = left_raw
        right = cv2.cvtColor(right_raw, cv2.COLOR_RGB2GRAY)
        rectification_status = "skipped_input_declared_already_rectified"
    else:
        rectification = stereo_calibration((left_raw.shape[1], left_raw.shape[0]))
        left, right = rectify_pair(left_raw, right_raw, rectification)
        rectification_status = "applied_undistortion_and_stereo_rectification"

    # Save the exact inputs consumed by the network for inspection/reuse.
    write_image(args.output / "left_rectified.png", cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
    write_image(args.output / "right_rectified.png", right)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, loaded_count = make_model(args.checkpoint, device)
    nir_channels = model.backbone["gated"].in_channels
    print(f"Loaded checkpoint: {args.checkpoint.resolve()}", flush=True)
    print(f"RGB channels: 3; NIR channels: {nir_channels}; both encoders restored from checkpoint", flush=True)

    # ``right`` is already the rectified NIR grayscale image above. Only legacy
    # five-channel checkpoints need the replicated-luminance adapter.
    nir_input = right[..., None]
    if nir_channels == 5:
        nir_input = np.repeat(nir_input, 5, axis=2)
    left_pair = np.stack([left, left], axis=0).transpose(0, 3, 1, 2).copy()
    right_pair = np.stack([nir_input, nir_input], axis=0).transpose(0, 3, 1, 2).copy()
    inputs = [{
        "img_ref": torch.from_numpy(left_pair).to(device=device, dtype=torch.float32),
        "img_comp": torch.from_numpy(right_pair).to(device=device, dtype=torch.float32),
    }]
    metas = [{}]  # No camera geometry => CSM and geometry-based secondary warp stay off.

    with torch.inference_mode():
        output = model(
            inputs,
            metas,
            do_csm=False,
            do_secondary=False,
            hetero_encoders=True,
            iters=4,
            iters_hd=2,
        )

    disparity = output["disp"][0, 0].detach().float().cpu().numpy()
    height, width = disparity.shape
    xx = np.arange(width, dtype=np.float32)[None, :]
    valid = np.isfinite(disparity) & (disparity > 0) & ((xx - disparity) >= 0)
    disparity_clean = np.where(np.isfinite(disparity), disparity, 0).astype(np.float32)

    np.save(args.output / "disparity_pixels.npy", disparity_clean)
    write_image(args.output / "valid_mask.png", valid.astype(np.uint8) * 255)

    if np.any(valid):
        upper = float(np.percentile(disparity[valid], 98))
        upper = max(upper, 1e-6)
        gray = np.clip(disparity_clean / upper * 255.0, 0, 255).astype(np.uint8)
        gray[~valid] = 0
        color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        color[~valid] = 0
        percentiles = np.percentile(disparity[valid], [0, 25, 50, 75, 95, 98, 100]).tolist()
    else:
        upper = 0.0
        gray = np.zeros((height, width), dtype=np.uint8)
        color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        percentiles = []
    write_image(args.output / "disparity_gray.png", gray)
    write_image(args.output / "disparity_color.png", color)

    metadata = {
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "loaded_parameter_tensors": loaded_count,
        "model_stream": "RGB-left and NIR-right through their checkpoint-restored encoders",
        "input_channels": {"rgb": 3, "nir": nir_channels},
        "encoders_loaded_from_checkpoint": True,
        "csm_enabled": False,
        "secondary_cross_modal_matching_enabled": False,
        "rectification": rectification_status,
        "raw_left": str(args.left.resolve()),
        "raw_right": str(args.right.resolve()),
        "rectified_outputs": {
            "left": str((args.output / "left_rectified.png").resolve()),
            "right": str((args.output / "right_rectified.png").resolve()),
        },
        "calibration_image_size_wh": [640, 512],
        "calibration_baseline_m": (rectification["baseline_m"] if rectification is not None else None),
        "input_size_wh": [width, height],
        "disparity_units": "pixels at input resolution",
        "disparity_convention": "left_x - right_x; positive values retained in valid mask",
        "valid_fraction": float(valid.mean()),
        "valid_disparity_percentiles_0_25_50_75_95_98_100": percentiles,
        "visualization_scale_pixels_p98": upper,
        "input_adapter": ("NIR converted to single-channel grayscale" if nir_channels == 1
                          else "NIR grayscale replicated to 5 channels"),
        "transfer_note": "All inference parameters restored from the supplied checkpoint; accuracy requires ground-truth evaluation.",
        "rectification_note": "Calibration matches sgbmS0.py. Use --skip-rectification only for already rectified inputs.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"device: {device}")
    print(f"loaded parameter tensors: {loaded_count}")
    print(f"input: {width}x{height}")
    print(f"valid positive disparity fraction: {valid.mean():.3%}")
    print(f"valid disparity percentiles [0,25,50,75,95,98,100]: {percentiles}")
    print(f"saved results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
