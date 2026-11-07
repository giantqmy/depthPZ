"""Infer rectified RGB/S0-left/NIR-right disparity with a trained DoLP-PIM checkpoint.

RGB and mono-S0 checkpoints are supported. Four raw-camera inputs are required:
left RGB or mono S0, right NIR, left DoLP and right NIR-DoLP. The script
rectifies all inputs, reconstructs the checkpoint architecture, and saves
float32 disparity (pixels) and depth (metres).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from train_rgb_nir_lidar_finetune import calibration, read_image


ROOT = Path(__file__).resolve().parent
FOG_ROOT = Path(r"I:\KEDADATA\data\fog")
DEFAULT_KEY = "2000-01-01-12-15-33_340"
DEFAULT_CHECKPOINT = (
    ROOT / "rgb_nir_lidar_dolp_pim_finetune_correct_scale_sparse_gt" / "latest.pt"
)
DEFAULT_MONO_S0 = FOG_ROOT / "rgbs0_mono" / f"{DEFAULT_KEY}_S0_rgb.png"


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def infer_key(left_path: Path) -> str:
    suffix = "_S0_rgb.png"
    if not left_path.name.endswith(suffix):
        raise ValueError(f"Left RGB filename must end with {suffix!r}: {left_path}")
    return left_path.name.removesuffix(suffix)


def resolve_auxiliary_paths(args, key: str) -> tuple[Path, Path]:
    rgb_suffix = "_dolp_rgb.png" if args.mono_s0 else "_dolp_color_rgb.png"
    rgb_dolp = args.rgb_dolp or args.rgb_dolp_dir / f"{key}{rgb_suffix}"
    nir_dolp = args.nir_dolp or args.nir_dolp_dir / f"{key}_dolp_nir.png"
    return rgb_dolp, nir_dolp


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    config = checkpoint.get("run_config", {})
    if config.get("fusion_type") != "pim":
        raise ValueError(f"Expected PIM checkpoint, got {config.get('fusion_type')!r}")
    architecture = config.get("architecture")
    mono_s0 = architecture == "mono_s0_5ch_nir_5ch_dolp_1ch_wavelet_pim256_v1"
    rgb_model = architecture == "rgb_nir_dolp_4encoders_wavelet_pim256_v1"
    if not (mono_s0 or rgb_model):
        raise ValueError(f"Unsupported architecture: {architecture!r}")

    model_args = config.get("model_args")
    if not isinstance(model_args, dict):
        raise ValueError("Checkpoint is missing run_config.model_args")

    from run_gatedrccbstereo_stereo import install_optional_dependency_shims

    install_optional_dependency_shims()
    from networks.gatedrccbstereo import GatedRCCBStereo

    model_args = dict(model_args)
    if mono_s0:
        model_args["input_channels"] = dict(model_args["input_channels"], rccb=5, gated=5)
    model = GatedRCCBStereo(model_args=SimpleNamespace(**model_args))
    model.enable_dolp_fusion(rgb_channels=1 if mono_s0 else 3,
                             nir_channels=1, fusion_type="pim")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint, mono_s0


def load_and_rectify_inputs(args, key: str, calib: dict, nir_channels: int, mono_s0: bool):
    rgb_dolp_path, nir_dolp_path = resolve_auxiliary_paths(args, key)
    required = (args.left, args.right, rgb_dolp_path, nir_dolp_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inference inputs: {missing}")

    rgb = read_image(args.left, cv2.IMREAD_UNCHANGED if mono_s0 else cv2.IMREAD_COLOR)
    nir = read_image(args.right, cv2.IMREAD_UNCHANGED)
    rgb_dolp = read_image(rgb_dolp_path, cv2.IMREAD_UNCHANGED)
    nir_dolp = read_image(nir_dolp_path, cv2.IMREAD_UNCHANGED)

    expected_left_shape = (512, 640) if mono_s0 else (512, 640, 3)
    if rgb.dtype != np.uint8 or rgb.shape != expected_left_shape:
        raise ValueError(f"Expected left uint8 {expected_left_shape}, got {rgb.dtype} {rgb.shape} from {args.left}")
    if nir.ndim == 3:
        nir = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)
    if nir.shape != (512, 640):
        raise ValueError(f"Expected NIR 640x512, got {nir.shape} from {args.right}")
    expected_dolp_shape = (512, 640) if mono_s0 else (512, 640, 3)
    if rgb_dolp.dtype != np.uint8 or rgb_dolp.shape != expected_dolp_shape:
        raise ValueError(f"Expected left DoLP uint8 {expected_dolp_shape}, got {rgb_dolp.dtype} {rgb_dolp.shape}")
    if nir_dolp.ndim == 3:
        nir_dolp = cv2.cvtColor(nir_dolp, cv2.COLOR_BGR2GRAY)
    if nir_dolp.dtype != np.uint8 or nir_dolp.shape != (512, 640):
        raise ValueError(f"Expected NIR-DoLP uint8 640x512, got {nir_dolp.dtype} {nir_dolp.shape}")

    if not mono_s0:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_dolp = cv2.cvtColor(rgb_dolp, cv2.COLOR_BGR2RGB)
    rgb_rect = cv2.remap(rgb, *calib["map_left"], cv2.INTER_LINEAR)
    nir_rect = cv2.remap(nir, *calib["map_right"], cv2.INTER_LINEAR)
    rgb_dolp_rect = cv2.remap(rgb_dolp, *calib["map_left"], cv2.INTER_LINEAR)
    nir_dolp_rect = cv2.remap(nir_dolp, *calib["map_right"], cv2.INTER_LINEAR)

    left_rect_for_view = rgb_rect.copy()
    left_dolp_for_view = rgb_dolp_rect.copy()
    if mono_s0:
        rgb_rect = np.repeat(rgb_rect[..., None], 5, axis=2)
        rgb_dolp_rect = rgb_dolp_rect[..., None]
    nir_model = np.repeat(nir_rect[..., None], nir_channels, axis=2)
    nir_dolp_model = nir_dolp_rect[..., None]

    def pair(image):
        return torch.from_numpy(
            np.stack([image, image]).transpose(0, 3, 1, 2).copy()
        ).float()

    inputs = {
        "img_ref": pair(rgb_rect),
        "img_comp": pair(nir_model),
        "img_ref_dolp": pair(rgb_dolp_rect),
        "img_comp_dolp": pair(nir_dolp_model),
    }
    paths = {"rgb": args.left, "nir": args.right,
             "rgb_dolp": rgb_dolp_path, "nir_dolp": nir_dolp_path}
    images = {"rgb": left_rect_for_view, "nir": nir_rect,
              "rgb_dolp": left_dolp_for_view, "nir_dolp": nir_dolp_rect}
    return inputs, paths, images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--left", type=Path,
                        default=FOG_ROOT / "RGBS0" / f"{DEFAULT_KEY}_S0_rgb.png")
    parser.add_argument("--right", type=Path,
                        default=FOG_ROOT / "NIRS0" / f"{DEFAULT_KEY}_S0_nir.png")
    parser.add_argument("--rgb-dolp", type=Path, default=None)
    parser.add_argument("--nir-dolp", type=Path, default=None)
    parser.add_argument("--rgb-dolp-dir", type=Path, default=FOG_ROOT / "RGBDOLP_color")
    parser.add_argument("--nir-dolp-dir", type=Path,
                        default=FOG_ROOT / "NIRuseI4" / "output_dolp")
    parser.add_argument("--output", type=Path, default=ROOT / "dolp_pim_inference")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    key = infer_key(args.left)
    expected_right = f"{key}_S0_nir.png"
    if args.right.name != expected_right:
        raise ValueError(f"RGB/NIR prefixes differ: expected right filename {expected_right!r}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, checkpoint, mono_s0 = load_model(args.checkpoint, device)
    args.mono_s0 = mono_s0
    if mono_s0 and args.left == FOG_ROOT / "RGBS0" / f"{key}_S0_rgb.png":
        args.left = DEFAULT_MONO_S0
    if mono_s0 and args.rgb_dolp is None:
        args.rgb_dolp_dir = FOG_ROOT / "RGBuseI4" / "output_dolp"
    nir_channels = int(model.backbone["gated"].in_channels)
    calib = calibration()
    inputs, paths, images = load_and_rectify_inputs(args, key, calib, nir_channels, mono_s0)
    inputs = [{name: tensor.to(device, non_blocking=True) for name, tensor in inputs.items()}]
    metas = [{
        "augmentations": {
            "ref_modality": "rccb",
            "comp_modality": "gated",
            "ref_side": "left",
            "do_csm": False,
        },
        "view_names": ["rccb_left", "rccb_right", "gated_left", "gated_right"],
        "cam2img": [calib["P1"].tolist(), calib["P2"].tolist(),
                    calib["P1"].tolist(), calib["P2"].tolist()],
        "rccb_baseline": [calib["baseline"]],
        "sample_key": key,
    }]

    with torch.inference_mode():
        outputs = model(
            inputs,
            metas,
            do_csm=False,
            do_secondary=False,
            hetero_encoders=True,
            dolp_fusion=True,
            iters=4,
            iters_hd=2,
        )

    disparity = outputs["disp"][0, 0].detach().float().cpu().numpy().astype(np.float32)
    depth = outputs["depth"][0, 0].detach().float().cpu().numpy().astype(np.float32)
    height, width = disparity.shape
    xx = np.arange(width, dtype=np.float32)[None, :]
    valid = (np.isfinite(disparity) & (disparity > 0.0)
             & np.isfinite(depth) & (depth > 0.0) & ((xx - disparity) >= 0.0))

    disparity_saved = np.where(np.isfinite(disparity), disparity, 0.0).astype(np.float32)
    depth_saved = np.where(valid, depth, 0.0).astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "disparity_pixels.npy", disparity_saved, allow_pickle=False)
    np.save(args.output / "depth_meters.npy", depth_saved, allow_pickle=False)
    write_image(args.output / "valid_mask.png", valid.astype(np.uint8) * 255)
    if images["rgb"].ndim == 2:
        write_image(args.output / "left_rectified.png", images["rgb"])
    else:
        write_image(args.output / "left_rectified.png", cv2.cvtColor(images["rgb"], cv2.COLOR_RGB2BGR))
    write_image(args.output / "right_rectified.png", images["nir"])
    if images["rgb_dolp"].ndim == 2:
        write_image(args.output / "left_dolp_rectified.png", images["rgb_dolp"])
    else:
        write_image(args.output / "left_dolp_rectified.png",
                    cv2.cvtColor(images["rgb_dolp"], cv2.COLOR_RGB2BGR))
    write_image(args.output / "right_dolp_rectified.png", images["nir_dolp"])

    valid_disp = disparity[valid]
    if valid_disp.size:
        upper = max(float(np.percentile(valid_disp, 98)), 1e-6)
        gray = np.clip(disparity_saved / upper * 255.0, 0, 255).astype(np.uint8)
        percentiles = np.percentile(valid_disp, [0, 25, 50, 75, 95, 98, 100]).tolist()
    else:
        upper = 0.0
        gray = np.zeros((height, width), dtype=np.uint8)
        percentiles = []
    gray[~valid] = 0
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    write_image(args.output / "disparity_gray.png", gray)
    write_image(args.output / "disparity_color.png", color)

    metadata = {
        "sample_key": key,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "architecture": checkpoint.get("run_config", {}).get("architecture"),
        "training_trainer_version": checkpoint.get("run_config", {}).get("trainer_version"),
        "training_depth_scale": checkpoint.get("run_config", {}).get(
            "depth_scale_m_per_uint16",
            checkpoint.get("run_config", {}).get("depth_scale"),
        ),
        "inputs": {name: str(path.resolve()) for name, path in paths.items()},
        "device": str(device),
        "input_channels": {"left_s0": 5, "nir": nir_channels,
                           "left_dolp": 1, "nir_dolp": 1} if mono_s0 else
                          {"rgb": 3, "nir": nir_channels, "rgb_dolp": 3, "nir_dolp": 1},
        "rectification": "applied to all four raw-camera inputs",
        "reference_view": "rectified left camera (mono S0 intensity input)" if mono_s0 else
                          "rectified left RGB camera",
        "disparity_convention": "left_x - right_x",
        "disparity_units": "pixels at 640x512 input resolution",
        "depth_units": "metres",
        "fx_rectified_px": calib["fx"],
        "baseline_m": calib["baseline"],
        "focal_times_baseline_px_m": calib["fx"] * calib["baseline"],
        "valid_fraction": float(valid.mean()),
        "disparity_percentiles_0_25_50_75_95_98_100": percentiles,
        "visualization_scale_pixels_p98": upper,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved inference results to: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
