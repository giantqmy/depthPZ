"""Fine-tune RGB/NIR/DoLP PIM stereo with correctly scaled depth supervision.

Training supervision and evaluation ground truth are intentionally separate:

* Dense supervision: ``fog/depth/*_depth_dense_u16.png``, uint16 metres * 256.
  It is rectified with the left image and used only by the BerHu training loss.
* Sparse evaluation GT: ``fog/depth_gt_sparse/*_depth_gt_sparse_u16.png``,
  uint16 metres * 256, already projected into the rectified left RGB grid.
  It never participates in backpropagation. Metrics use only its nonzero pixels.

This script does not accept a configurable depth scale. Both sources are fixed
to 1/256 metre per stored unit to prevent accidental millimetre decoding.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_rgb_nir_lidar_finetune import (
    ROOT,
    BerHuSparseLoss,
    calibration,
    format_mae,
    move_sample,
    read_image,
    write_json,
)
from train_rgb_nir_lidar_dolp_finetune import (
    RGBNIRDoLPLiDARDataset,
    make_model as make_fusion_model,
    set_training_mode,
)


CS_STEREO_ROOT = ROOT.parent / "cs-stereo"
if not CS_STEREO_ROOT.is_dir():
    raise FileNotFoundError(f"Cannot find cs-stereo STN source directory: {CS_STEREO_ROOT}")
sys.path.insert(0, str(CS_STEREO_ROOT))
from stn import STN as CSSTereoSTN, stn_init as cs_stn_init


DATA_ROOT = Path(r"I:\KEDADATA\data\fog")
DEFAULT_RGB = DATA_ROOT / "RGBS0"
DEFAULT_NIR = DATA_ROOT / "NIRS0"
DEFAULT_DENSE_DEPTH = DATA_ROOT / "depth"
DEFAULT_SPARSE_GT = DATA_ROOT / "depth_gt_sparse"
DEFAULT_RGB_DOLP = DATA_ROOT / "RGBDOLP_color"
DEFAULT_NIR_DOLP = DATA_ROOT / "NIRuseI4" / "output_dolp"
DEFAULT_OUTPUT = ROOT / "rgb_stn_pseudonir_nir_dolp_pim_finetune_correct_scale_sparse_gt"

DEPTH_METRES_PER_UINT16 = 1.0 / 256.0


def cs_stereo_pyramid(image: torch.Tensor, levels: int = 4) -> list[torch.Tensor]:
    """Match cs-stereo's anti-aliased image pyramid for the STN filter network."""
    _, _, height, width = image.shape
    pyramid = [image]
    current = image
    for level in range(1, levels):
        current = (
            current[:, :, :-2, :-2] / 16.0
            + current[:, :, 1:-1, :-2] / 8.0
            + current[:, :, 2:, :-2] / 16.0
            + current[:, :, :-2, 1:-1] / 8.0
            + current[:, :, 1:-1, 1:-1] / 4.0
            + current[:, :, 2:, 1:-1] / 8.0
            + current[:, :, :-2, 2:] / 16.0
            + current[:, :, 1:-1, 2:] / 8.0
            + current[:, :, 2:, 2:] / 16.0
        )
        current = torch.nn.functional.pad(current, (1, 1, 1, 1), mode="replicate")
        current = torch.nn.functional.interpolate(
            current,
            size=(height // (2 ** level), width // (2 ** level)),
            mode="bilinear",
            align_corners=False,
        )
        pyramid.append(current)
    return pyramid


def transform_left_rgb_to_pseudo_nir(inputs: list[dict], stn: torch.nn.Module) -> torch.Tensor:
    """Apply cs-stereo STN to the left RGB image and replace model RGB input."""
    rgb = torch.cat([sample["img_ref"][0:1] for sample in inputs], dim=0)
    nir = torch.cat([sample["img_comp"][0:1, :1] for sample in inputs], dim=0)
    rgb = (rgb.float() / 255.0).clamp(0.0, 1.0)
    nir = (nir.float() / 255.0).clamp(0.0, 1.0)
    rgb_mean = rgb.mean(dim=(1, 2, 3), keepdim=True)
    nir_mean = nir.mean(dim=(1, 2, 3), keepdim=True)
    # Match cs-stereo's target/source exposure direction: NIR / RGB.
    exposure_ratio = nir_mean / rgb_mean.clamp(min=1e-4)
    pseudo_nir = stn(cs_stereo_pyramid(rgb), exposure_ratio)[0].clamp(0.0, 1.0)
    pseudo_nir_rgb = (pseudo_nir * 255.0).repeat(1, 3, 1, 1)
    for index, sample in enumerate(inputs):
        sample_pseudo = pseudo_nir_rgb[index:index + 1]
        sample["img_ref"] = torch.cat((sample_pseudo, sample_pseudo), dim=0)
    return pseudo_nir


class CorrectScalePIMDataset(RGBNIRDoLPLiDARDataset):
    """Load dense supervision and independent rectified sparse evaluation GT."""

    def __init__(
        self,
        rgb_dir: Path,
        nir_dir: Path,
        dense_depth_dir: Path,
        sparse_gt_dir: Path,
        calib: dict,
        rgb_dolp_dir: Path,
        nir_dolp_dir: Path,
        max_train_depth: float = 220.0,
    ):
        self.rgb_dir = Path(rgb_dir)
        self.nir_dir = Path(nir_dir)
        self.depth_dir = Path(dense_depth_dir)
        self.sparse_gt_dir = Path(sparse_gt_dir)
        self.calib = calib
        self.depth_scale = DEPTH_METRES_PER_UINT16
        self.max_depth = float(max_train_depth)
        self.nir_channels = 5

        rgb = {
            p.name.removesuffix("_S0_rgb.png"): p
            for p in self.rgb_dir.glob("*_S0_rgb.png")
        }
        nir = {
            p.name.removesuffix("_S0_nir.png"): p
            for p in self.nir_dir.glob("*_S0_nir.png")
        }
        dense = {
            p.name.removesuffix("_depth_dense_u16.png"): p
            for p in self.depth_dir.glob("*_depth_dense_u16.png")
        }
        sparse = {
            p.name.removesuffix("_depth_gt_sparse_u16.png"): p
            for p in self.sparse_gt_dir.glob("*_depth_gt_sparse_u16.png")
        }
        rgb_dolp = {
            p.name.removesuffix("_dolp_color_rgb.png"): p
            for p in Path(rgb_dolp_dir).glob("*_dolp_color_rgb.png")
        }
        nir_dolp = {
            p.name.removesuffix("_dolp_nir.png"): p
            for p in Path(nir_dolp_dir).glob("*_dolp_nir.png")
        }

        sources = {
            "RGB": rgb,
            "NIR": nir,
            "dense supervision": dense,
            "sparse evaluation GT": sparse,
            "RGB DoLP": rgb_dolp,
            "NIR DoLP": nir_dolp,
        }
        reference_keys = set(rgb)
        if not reference_keys:
            raise RuntimeError(f"No RGB inputs found in {self.rgb_dir}")
        problems = []
        extras_ignored = []
        for label, mapping in sources.items():
            missing = sorted(reference_keys - set(mapping))
            extra = sorted(set(mapping) - reference_keys)
            if missing:
                problems.append(
                    f"{label}: missing={missing[:5]} ({len(missing)})"
                )
            if extra:
                extras_ignored.append(
                    f"{label}: extra={extra[:5]} ({len(extra)})"
                )
        if problems:
            raise RuntimeError("Dataset prefixes are missing required fog samples:\n" + "\n".join(problems))
        if extras_ignored:
            print(
                "Ignoring files whose prefixes are not in the fog RGB set:\n"
                + "\n".join(extras_ignored),
                flush=True,
            )

        keys = sorted(reference_keys)
        self.samples = [(rgb[k], nir[k], dense[k], k) for k in keys]
        self.dolp_paths = [(rgb_dolp[k], nir_dolp[k]) for k in keys]
        self.sparse_gt_paths = [sparse[k] for k in keys]

    def __getitem__(self, index):
        # Parent loaders rectify RGB, NIR, dense supervision and both DoLP inputs.
        sample = super().__getitem__(index)
        key = self.samples[index][3]

        dense_path = self.samples[index][2]
        dense_raw = read_image(dense_path, cv2.IMREAD_UNCHANGED)
        if dense_raw.dtype != np.uint16 or dense_raw.shape != (512, 640):
            raise ValueError(
                f"Dense supervision must be uint16 640x512: {dense_path}; "
                f"got {dense_raw.dtype} {dense_raw.shape}"
            )

        sparse_path = self.sparse_gt_paths[index]
        sparse_raw = read_image(sparse_path, cv2.IMREAD_UNCHANGED)
        if sparse_raw.dtype != np.uint16 or sparse_raw.shape != (512, 640):
            raise ValueError(
                f"Sparse GT must be uint16 640x512: {sparse_path}; "
                f"got {sparse_raw.dtype} {sparse_raw.shape}"
            )
        sparse_m = sparse_raw.astype(np.float32) * DEPTH_METRES_PER_UINT16
        sparse_m[~np.isfinite(sparse_m)] = 0.0
        sparse_m[sparse_m <= 0.0] = 0.0

        # This GT is already in the rectified-left grid. Do not remap it again.
        sample["eval_sparse_depth_m"] = torch.from_numpy(sparse_m[None].copy()).float()
        sample["meta"]["eval_sparse_gt_path"] = str(sparse_path.resolve())
        sample["meta"]["sample_key"] = key
        return sample


class SparseGroundTruthMetrics:
    """Pixel-weighted metrics on nonzero sparse LiDAR ground-truth pixels."""

    def __init__(self, focal_px: float, baseline_m: float):
        self.focal_px = float(focal_px)
        self.baseline_m = float(baseline_m)
        self.depth_abs_sum = 0.0
        self.depth_sq_sum = 0.0
        self.depth_abs_rel_sum = 0.0
        self.disp_abs_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, outputs: dict, sparse_gt_depth_m: torch.Tensor) -> dict:
        gt = sparse_gt_depth_m.to(
            device=outputs["depth"].device, dtype=torch.float32, non_blocking=True
        )
        if gt.ndim == 3:
            gt = gt.unsqueeze(0)
        pred_depth = outputs["depth"].detach().float()[:, :1]
        pred_disp = outputs["disp"].detach().float()[:, :1]
        if pred_depth.shape != gt.shape or pred_disp.shape != gt.shape:
            raise ValueError(
                f"Sparse metric shape mismatch: depth={tuple(pred_depth.shape)}, "
                f"disp={tuple(pred_disp.shape)}, gt={tuple(gt.shape)}"
            )

        valid = (gt > 0.0) & torch.isfinite(gt)
        valid &= torch.isfinite(pred_depth) & torch.isfinite(pred_disp) & (pred_disp > 0.0)
        count = int(valid.sum().item())
        if not count:
            return {"depth_mae_m": None, "depth_rmse_m": None,
                    "depth_abs_rel": None, "disp_mae_px": None, "valid_pixels": 0}

        depth_error = pred_depth[valid] - gt[valid]
        gt_disp = (self.focal_px * self.baseline_m) / gt[valid]
        disp_error = pred_disp[valid] - gt_disp

        depth_abs = depth_error.abs().double()
        self.depth_abs_sum += depth_abs.sum().item()
        self.depth_sq_sum += depth_error.double().square().sum().item()
        self.depth_abs_rel_sum += (depth_abs / gt[valid].double()).sum().item()
        self.disp_abs_sum += disp_error.abs().double().sum().item()
        self.count += count
        return {
            "depth_mae_m": depth_abs.mean().item(),
            "depth_rmse_m": depth_error.double().square().mean().sqrt().item(),
            "depth_abs_rel": (depth_abs / gt[valid].double()).mean().item(),
            "disp_mae_px": disp_error.abs().double().mean().item(),
            "valid_pixels": count,
        }

    def summary(self) -> dict:
        if not self.count:
            return {"depth_mae_m": None, "depth_rmse_m": None,
                    "depth_abs_rel": None, "disp_mae_px": None, "valid_pixels": 0}
        return {
            "depth_mae_m": self.depth_abs_sum / self.count,
            "depth_rmse_m": (self.depth_sq_sum / self.count) ** 0.5,
            "depth_abs_rel": self.depth_abs_rel_sum / self.count,
            "disp_mae_px": self.disp_abs_sum / self.count,
            "valid_pixels": self.count,
        }


def metric_text(value) -> str:
    return "NA" if value is None else f"{value:.6f}"


def write_manifest(path: Path, dataset, indices, split: str, seed: int, val_fraction: float) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"# split={split} seed={seed} val_fraction={val_fraction} samples={len(indices)}\n"
        )
        handle.write(
            "# key\trgb\tnir\tdense_supervision\trgb_dolp\tnir_dolp\tsparse_eval_gt\n"
        )
        for index in indices:
            rgb, nir, dense, key = dataset.samples[index]
            rgb_dolp, nir_dolp = dataset.dolp_paths[index]
            sparse = dataset.sparse_gt_paths[index]
            handle.write(
                "\t".join(str(value.resolve()) if isinstance(value, Path) else str(value)
                           for value in (key, rgb, nir, dense, rgb_dolp, nir_dolp, sparse))
                + "\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--nir-dir", type=Path, default=DEFAULT_NIR)
    parser.add_argument("--depth-dir", type=Path, default=DEFAULT_DENSE_DEPTH,
                        help="dense uint16 x256 supervision directory")
    parser.add_argument("--sparse-gt-dir", type=Path, default=DEFAULT_SPARSE_GT,
                        help="rectified sparse uint16 x256 evaluation GT directory")
    parser.add_argument("--rgb-dolp-dir", type=Path, default=DEFAULT_RGB_DOLP)
    parser.add_argument("--nir-dolp-dir", type=Path, default=DEFAULT_NIR_DOLP)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "gatedrccbstereo.pt")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume only a checkpoint produced by this correct-scale trainer")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20,
                        help="target total epoch count when resuming")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="number of samples per optimizer step (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-train-depth", type=float, default=220.0,
                        help="dense supervision/model range; sparse metrics remain unfiltered")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--freeze-encoders", action="store_true")
    parser.add_argument("--checkpoint-encoders", action="store_true")
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or not 0 < args.val_fraction < 1:
        parser.error("Require epochs >= 1, batch-size >= 1, lr > 0 and 0 < val-fraction < 1")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    if args.from_scratch and args.freeze_encoders:
        parser.error("--from-scratch requires trainable encoders")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    calib = calibration()
    dataset = CorrectScalePIMDataset(
        args.rgb_dir,
        args.nir_dir,
        args.depth_dir,
        args.sparse_gt_dir,
        calib,
        args.rgb_dolp_dir,
        args.nir_dolp_dir,
        args.max_train_depth,
    )

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.limit is not None:
        indices = indices[: min(args.limit, len(indices))]
    if len(indices) < 2:
        raise RuntimeError("At least two matched samples are required")
    n_val = min(len(indices) - 1, max(1, int(len(indices) * args.val_fraction)))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output / "train.txt", dataset, train_idx, "train", args.seed, args.val_fraction)
    write_manifest(args.output / "val.txt", dataset, val_idx, "val", args.seed, args.val_fraction)
    print(f"Matched samples={len(dataset)}, train={len(train_idx)}, val={len(val_idx)}", flush=True)

    sample = dataset[train_idx[0]]
    print(
        "Data check: "
        + str({key: tuple(value.shape) for key, value in sample["inputs"].items()})
        + f", sparse_gt={tuple(sample['eval_sparse_depth_m'].shape)}, "
          f"sparse_valid={int((sample['eval_sparse_depth_m'] > 0).sum())}",
        flush=True,
    )
    if args.check_data_only:
        print("Data pairing, uint16 types and 1/256 scales validated; no training run.", flush=True)
        return

    resume_data = None
    if args.resume is not None:
        resume_data = torch.load(str(args.resume), map_location="cpu", weights_only=True)
        config = resume_data.get("run_config", {})
        if config.get("trainer_version") != "correct_scale_dense_train_sparse_gt_metrics_stn_v1":
            raise ValueError("Resume checkpoint is not from the RGB-to-pseudo-NIR STN trainer")
        if config.get("depth_scale_m_per_uint16") != DEPTH_METRES_PER_UINT16:
            raise ValueError("Resume checkpoint does not use the required 1/256 depth scale")
        if bool(config.get("from_scratch")) != bool(args.from_scratch):
            raise ValueError("--from-scratch must match the resumed run")
        if args.epochs <= int(resume_data.get("epoch", 0)):
            raise ValueError("--epochs is the target total and must exceed the resumed epoch")

    model, loaded = make_fusion_model(
        args.checkpoint,
        device,
        from_scratch=args.from_scratch,
        fusion_type="pim",
    )
    stn = CSSTereoSTN(in_shape=(512, 640), filt=True)
    stn.apply(cs_stn_init)
    if resume_data is not None:
        model.load_state_dict(resume_data["model"], strict=True)
        if "stn" not in resume_data:
            raise ValueError("Resume checkpoint is missing the trained RGB-to-NIR STN state")
        stn.load_state_dict(resume_data["stn"], strict=True)
    stn.to(device)
    dataset.nir_channels = model.backbone["gated"].in_channels
    if args.freeze_encoders:
        for encoders in (model.backbone, model.dolp_backbone):
            for parameter in encoders.parameters():
                parameter.requires_grad_(False)

    collate_samples = lambda samples: samples
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                              batch_size=args.batch_size, collate_fn=collate_samples,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx),
                            batch_size=args.batch_size, collate_fn=collate_samples,
                            shuffle=False, num_workers=0)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_parameters.extend(p for p in stn.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    start_epoch = 0
    history = []
    if resume_data is not None:
        optimizer.load_state_dict(resume_data["optimizer"])
        start_epoch = int(resume_data.get("epoch", 0))
        history = list(resume_data.get("history", []))

    criterion = BerHuSparseLoss().to(device)
    initialization = "random_all_dolp_wavelet_pim" if args.from_scratch else (
        "pretrained_stereo_encoders_random_rgb_to_nir_stn_and_wavelet_pim"
    )
    run_config = {
        "trainer_version": "correct_scale_dense_train_sparse_gt_metrics_stn_v1",
        "architecture": "rgb_to_pseudonir_cs_stn_nir_dolp_4encoders_wavelet_pim256_v1",
        "fusion_type": "pim",
        "rgb_to_pseudonir": "cs-stereo STN/EDNet: learned per-pixel RGB channel weights, summed to mono and scaled by NIR/RGB mean exposure ratio",
        "stn_source": str((CS_STEREO_ROOT / "stn.py").resolve()),
        "stn_pretrained_checkpoint": None,
        "from_scratch": args.from_scratch,
        "initialization": initialization,
        "source_checkpoint": None if args.from_scratch else str(args.checkpoint.resolve()),
        "depth_scale_m_per_uint16": DEPTH_METRES_PER_UINT16,
        "training_supervision": "dense_depth_u16_x256_rectified_by_dataset",
        "metric_ground_truth": "independent_sparse_lidar_u16_x256_already_rectified",
        "metric_valid_mask": "sparse_gt_depth_m > 0 and finite prediction",
        "sparse_metric_max_depth_filter": None,
        "max_train_depth": args.max_train_depth,
        "seed": args.seed,
        "limit": args.limit,
        "val_fraction": args.val_fraction,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "freeze_encoders": args.freeze_encoders,
        "checkpoint_encoders": args.checkpoint_encoders,
        "data_dirs": {
            "rgb": str(args.rgb_dir.resolve()),
            "nir": str(args.nir_dir.resolve()),
            "dense_supervision": str(args.depth_dir.resolve()),
            "sparse_evaluation_gt": str(args.sparse_gt_dir.resolve()),
            "rgb_dolp": str(args.rgb_dolp_dir.resolve()),
            "nir_dolp": str(args.nir_dolp_dir.resolve()),
        },
        "input_channels": {"rgb_raw_for_stn": 3, "rccb_pseudonir_replicated": 3,
                           "gated": dataset.nir_channels,
                           "rgb_dolp": 3, "nir_dolp": 1},
        "model_args": vars(model.args),
    }
    write_json(args.output / "run_config.json", run_config)
    write_json(args.output / "calibration.json", {
        "image_size_wh": [640, 512],
        "fx_rectified": calib["fx"],
        "baseline_m": calib["baseline"],
        "focal_times_baseline_px_m": calib["fx"] * calib["baseline"],
        "P1": calib["P1"].tolist(),
        "P2": calib["P2"].tolist(),
    })

    loss_log = args.output / "loss.txt"
    if start_epoch == 0:
        with loss_log.open("w", encoding="utf-8") as handle:
            handle.write(
                "# Loss uses dense x256 supervision; all metrics use independent sparse LiDAR GT only.\n"
            )
            handle.write(
                "epoch\ttrain_loss\tval_loss\t"
                "train_depth_mae_m\ttrain_depth_rmse_m\ttrain_depth_abs_rel\ttrain_disp_mae_px\ttrain_valid_pixels\t"
                "val_depth_mae_m\tval_depth_rmse_m\tval_depth_abs_rel\tval_disp_mae_px\tval_valid_pixels\n"
            )
    print(
        f"device={device}, loaded_baseline_tensors={loaded}, depth_scale=1/256, "
        f"sparse metrics unfiltered by distance",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs):
        phase_results = {}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            if phase == "train":
                set_training_mode(model)
                stn.train()
            else:
                model.eval()
                stn.eval()
            losses = []
            metrics = SparseGroundTruthMetrics(calib["fx"], calib["baseline"])
            total_steps = len(loader)
            grad_context = torch.enable_grad() if phase == "train" else torch.no_grad()
            with grad_context:
                for step, batch in enumerate(loader, start=1):
                    moved = [move_sample(sample, device) for sample in batch]
                    inputs = [item[0][0] for item in moved]
                    metas = [item[1][0] for item in moved]
                    pseudo_nir = transform_left_rgb_to_pseudo_nir(inputs, stn)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                    outputs = model(
                        inputs,
                        metas,
                        do_csm=False,
                        do_secondary=False,
                        hetero_encoders=True,
                        dolp_fusion=True,
                        iters=4,
                        iters_hd=2,
                        checkpoint_encoders=args.checkpoint_encoders,
                    )
                    loss, details = criterion(outputs)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite {phase} loss at epoch={epoch + 1}, step={step}: {details}"
                        )
                    sparse_gt = torch.stack(
                        [sample["eval_sparse_depth_m"] for sample in batch], dim=0
                    )
                    sample_metrics = metrics.update(outputs, sparse_gt)
                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            trainable_parameters, 1.0, error_if_nonfinite=True
                        )
                        optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    if step == 1 or step % max(1, args.log_interval) == 0 or step == total_steps:
                        pseudo_range = (
                            float(pseudo_nir.detach().min().cpu()),
                            float(pseudo_nir.detach().max().cpu()),
                        )
                        print(
                            f"epoch {epoch + 1}/{args.epochs} {phase} {step}/{total_steps}: "
                            f"loss={losses[-1]:.6f} avg={np.mean(losses):.6f} "
                            f"sparse_depth_mae_m={metric_text(sample_metrics['depth_mae_m'])} "
                            f"sparse_disp_mae_px={metric_text(sample_metrics['disp_mae_px'])} "
                            f"valid={sample_metrics['valid_pixels']} "
                            f"pseudo_nir_range={pseudo_range[0]:.3f}..{pseudo_range[1]:.3f}",
                            flush=True,
                        )
            phase_results[phase] = {"loss": float(np.mean(losses)), **metrics.summary()}

        row = {"epoch": epoch + 1}
        for phase in ("train", "val"):
            row[f"{phase}_loss"] = phase_results[phase]["loss"]
            for key in ("depth_mae_m", "depth_rmse_m", "depth_abs_rel",
                        "disp_mae_px", "valid_pixels"):
                row[f"{phase}_{key}"] = phase_results[phase][key]
        history.append(row)

        with loss_log.open("a", encoding="utf-8") as handle:
            values = [str(row["epoch"]), f"{row['train_loss']:.8f}", f"{row['val_loss']:.8f}"]
            for phase in ("train", "val"):
                values.extend([
                    metric_text(row[f"{phase}_depth_mae_m"]),
                    metric_text(row[f"{phase}_depth_rmse_m"]),
                    metric_text(row[f"{phase}_depth_abs_rel"]),
                    metric_text(row[f"{phase}_disp_mae_px"]),
                    str(row[f"{phase}_valid_pixels"]),
                ])
            handle.write("\t".join(values) + "\n")

        checkpoint = {
            "model": model.state_dict(),
            "stn": stn.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "history": history,
            "initialization": initialization,
            "run_config": run_config,
            "source_checkpoint": run_config["source_checkpoint"],
            "input_channels": run_config["input_channels"],
        }
        torch.save(checkpoint, args.output / "latest.pt")
        write_json(args.output / "history.json", {
            "history": history,
            "run_config": run_config,
            "device": str(device),
            "updated": datetime.now().astimezone().isoformat(),
        })
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
