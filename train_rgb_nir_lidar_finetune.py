"""Train RGB-left/NIR-right stereo with sparse LiDAR supervision.

The LiDAR PNGs are projected in the raw left-camera image coordinates and are
stored as uint16 millimetres.  This trainer rectifies RGB, NIR and the sparse
depth map with the calibration from ``sgbmS0.py`` and fine-tunes both encoders and the complete stereo/update/decoder trunk
from the supplied pretrained checkpoint. Same-side CSM fusion, secondary matching,
photometric loss, DOLP and AOLP are intentionally disabled.

Example (the defaults point at the paths supplied for this dataset)::

    D:\\anaconda\\envs\\pytorch\\python.exe train_rgb_nir_lidar_finetune.py \\
      --epochs 5 --output rgb_nir_lidar_finetune_run

The input images are paired by their timestamp stem, e.g.
``..._S0_rgb.png``, ``..._S0_nir.png`` and ``..._depth.png``.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT
DEFAULT_RGB = Path(r"I:\KEDADATA\data\fog\RGBS0")
DEFAULT_NIR = Path(r"I:\KEDADATA\data\fog\NIRS0")
DEFAULT_DEPTH = Path(r"I:\KEDADATA\data\fog\depth_new")


def read_image(path: Path, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def calibration(size=(640, 512), alpha=0.9):
    """Return raw cameras, rectification maps and rectified stereo geometry."""
    K1 = np.array(
        [[1265.5616099448073, 0.0, 492.5894472471048],
         [0.0, 1265.0089455071945, 257.24307296897143],
         [0.0, 0.0, 1.0]], dtype=np.float64)
    K2 = np.array(
        [[1277.6494552887766, 0.0, 474.185263116495],
         [0.0, 1278.5279034618504, 260.09879728682546],
         [0.0, 0.0, 1.0]], dtype=np.float64)
    d1 = np.array([[-0.02096660215918474, 0.2409083928738356,
                    -0.005298800804070935, 0.007648334809400866, 0.0]], dtype=np.float64)
    d2 = np.array([[-0.06681173268876799, 0.39362342592776267,
                    -0.007450119715294455, 0.0034144197720179475, 0.0]], dtype=np.float64)
    R = np.array(
        [[0.9998762017154619, 0.003542133347215963, 0.015330836063682721],
         [-0.003479358861003513, 0.9999854622282568, -0.004119392437720074],
         [-0.01534520462481131, 0.004065540983698188, 0.9998739900965185]], dtype=np.float64)
    T = np.array([-0.055078917279073276, 0.00011619425753116915,
                  0.0017826427597072247], dtype=np.float64)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, d1, K2, d2, size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
    )
    map1 = cv2.initUndistortRectifyMap(K1, d1, R1, P1, size, cv2.CV_32FC1)
    map2 = cv2.initUndistortRectifyMap(K2, d2, R2, P2, size, cv2.CV_32FC1)
    baseline = float(-P2[0, 3] / P2[0, 0])
    return {
        "K_left": K1, "K_right": K2, "dist_left": d1, "dist_right": d2,
        "R": R, "T": T, "R1": R1, "R2": R2, "P1": P1, "P2": P2,
        "Q": Q, "map_left": map1, "map_right": map2,
        "fx": float(P1[0, 0]), "baseline": baseline,
        "roi_left": roi1, "roi_right": roi2,
    }


class RGBNIRLiDARDataset(Dataset):
    def __init__(self, rgb_dir, nir_dir, depth_dir, calib, depth_scale=0.001,
                 max_depth=220.0):
        self.rgb_dir = Path(rgb_dir)
        self.nir_dir = Path(nir_dir)
        self.depth_dir = Path(depth_dir)
        self.calib = calib
        self.depth_scale = float(depth_scale)
        self.max_depth = float(max_depth)
        self.nir_channels = 5
        rgb = {p.name.replace("_S0_rgb.png", ""): p for p in self.rgb_dir.glob("*_S0_rgb.png")}
        nir = {p.name.replace("_S0_nir.png", ""): p for p in self.nir_dir.glob("*_S0_nir.png")}
        dep = {p.name.replace("_depth.png", ""): p for p in self.depth_dir.glob("*_depth.png")}
        keys = sorted(set(rgb) & set(nir) & set(dep))
        if not keys:
            raise RuntimeError("No timestamp-matched RGB/NIR/depth triplets were found.")
        self.samples = [(rgb[k], nir[k], dep[k], k) for k in keys]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rgb_p, nir_p, depth_p, key = self.samples[index]
        rgb = read_image(rgb_p, cv2.IMREAD_COLOR)
        nir = read_image(nir_p, cv2.IMREAD_UNCHANGED)
        depth = read_image(depth_p, cv2.IMREAD_UNCHANGED)
        if rgb.shape[:2] != (512, 640) or nir.shape[:2] != (512, 640) or depth.shape[:2] != (512, 640):
            raise ValueError(f"Expected 640x512 inputs, got {rgb.shape}, {nir.shape}, {depth.shape} for {key}")

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if nir.ndim == 3:
            nir = cv2.cvtColor(nir, cv2.COLOR_BGR2GRAY)
        # Rectify before adapting NIR to the checkpoint channel count.
        nir = cv2.remap(nir, *self.calib["map_right"], cv2.INTER_LINEAR)
        nir = np.repeat(nir[..., None], self.nir_channels, axis=2)
        depth_m = depth.astype(np.float32) * self.depth_scale
        # Zero is the sparse-invalid value.  Also reject NaN and nonphysical data.
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[(depth_m <= 0.0) | (depth_m > self.max_depth)] = 0.0

        # RGB/NIR/depth are all defined in raw camera coordinates.  Use nearest
        # interpolation for sparse depth so it is never averaged into a fake hit.
        rgb = cv2.remap(rgb, *self.calib["map_left"], cv2.INTER_LINEAR)
        depth_m = cv2.remap(depth_m, *self.calib["map_left"], cv2.INTER_NEAREST)
        depth_m[depth_m < 1e-3] = 0.0

        # The model consumes [2,C,H,W]: left/right slots.  For this transfer
        # experiment only RGB-left and NIR-right are used; duplicate each view
        # to satisfy the existing stereo input container.
        rgb_pair = torch.from_numpy(np.stack([rgb, rgb]).transpose(0, 3, 1, 2).copy()).float()
        nir_pair = torch.from_numpy(np.stack([nir, nir]).transpose(0, 3, 1, 2).copy()).float()
        # The model's modality depth tensor is [views,1,H,W]; it selects the
        # reference (left) view internally when constructing GT targets.
        gt = torch.from_numpy(np.stack([depth_m, depth_m])[:, None].copy()).float()
        return {
            "inputs": {
                "img_ref": rgb_pair,
                "img_comp": nir_pair,
                "gt_sparse_depth_rccb": gt,
            },
            "meta": {
                "augmentations": {
                    "ref_modality": "rccb",
                    "comp_modality": "gated",
                    "ref_side": "left",
                    "do_csm": False,
                },
                "view_names": ["rccb_left", "rccb_right", "gated_left", "gated_right"],
                "cam2img": [self.calib["P1"].tolist(), self.calib["P2"].tolist(),
                            self.calib["P1"].tolist(), self.calib["P2"].tolist()],
                "rccb_baseline": [self.calib["baseline"]],
                "sample_key": key,
            },
        }


def write_split_manifest(path, dataset, indices, split_name, seed, val_fraction, limit):
    """Persist the exact timestamped RGB/NIR/depth triplets assigned to a split."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as manifest:
        manifest.write(
            f"# split={split_name} seed={seed} val_fraction={val_fraction} "
            f"limit={limit if limit is not None else 'all'} samples={len(indices)}\n"
        )
        manifest.write("# columns: timestamp_prefix\trgb_path\tnir_path\tdepth_path\n")
        for index in indices:
            rgb_path, nir_path, depth_path, key = dataset.samples[index]
            manifest.write("\t".join((
                key,
                str(rgb_path.resolve()),
                str(nir_path.resolve()),
                str(depth_path.resolve()),
            )) + "\n")


class BerHuSparseLoss(nn.Module):
    def __init__(self, gamma=0.8, disp_weight=0.75):
        super().__init__()
        self.gamma = float(gamma)
        self.disp_weight = float(disp_weight)

    @staticmethod
    def berhu(pred, target, valid):
        pred, target = pred.float(), target.float()
        valid = valid.bool()
        if not valid.any():
            return pred.mean() * 0.0
        d = (pred - target).abs()
        c = 0.1 * d[valid].max().clamp(min=1e-6)
        per = torch.where(d <= c, d, (d * d + c * c) / (2.0 * c))
        return per[valid].mean()

    def forward(self, outputs):
        total = torch.zeros((), device=outputs["disp"].device)
        details = {}
        for key, scale in (("depth", 1.0), ("disp", self.disp_weight)):
            gt = outputs.get(f"gt_{key}")
            preds = outputs.get(f"{key}_preds")
            if gt is None or not preds:
                continue
            valid = outputs.get("gt_valid", gt > 0)
            weights = [self.gamma ** (len(preds) - i - 1) for i in range(len(preds))]
            value = torch.zeros((), device=total.device)
            for w, pred in zip(weights, preds):
                value = value + float(w) * self.berhu(pred[:, :1], gt[:, :1], valid[:, :1])
            details[f"loss/{key}"] = value.detach()
            total = total + float(scale) * value
        details["loss/total"] = total.detach()
        return total, details


class FinalOutputMAE:
    """Pixel-weighted MAE of final predictions on valid LiDAR targets only."""

    def __init__(self):
        self.sums = {"depth": 0.0, "disp": 0.0}
        self.counts = {"depth": 0, "disp": 0}

    @torch.no_grad()
    def update(self, outputs):
        sample = {}
        for key in self.sums:
            pred, gt = outputs[key].detach().float()[:, :1], outputs[f"gt_{key}"].detach().float()[:, :1]
            if pred.shape != gt.shape:
                raise ValueError(f"{key} prediction/target shape mismatch: {pred.shape}, {gt.shape}")
            valid = outputs.get("gt_valid", gt > 0).bool()[:, :1]
            valid = valid & torch.isfinite(gt) & (gt > 0)
            count = int(valid.sum().item())
            if not torch.isfinite(pred[valid]).all():
                raise FloatingPointError(f"Non-finite final {key} prediction at valid LiDAR pixels")
            error_sum = (pred[valid] - gt[valid]).abs().sum(dtype=torch.float64).item()
            self.sums[key] += error_sum
            self.counts[key] += count
            sample[key] = error_sum / count if count else None
        return sample

    def summary(self):
        return {key: self.sums[key] / self.counts[key] if self.counts[key] else None
                for key in self.sums}


def format_mae(value):
    return f"{value:.4f}" if value is not None else "NA"


def make_model(checkpoint, device):
    sys.path.insert(0, str(ROOT))
    from run_gatedrccbstereo_stereo import make_model as load_model
    model, loaded = load_model(
        Path(checkpoint), device, reinit_encoders=False
    )
    model.train()
    # The loader's model args are already configured as rccb + gated streams.
    return model, loaded


def move_sample(sample, device):
    inp = [{k: v.to(device, non_blocking=True) for k, v in sample["inputs"].items()}]
    meta = [sample["meta"]]
    return inp, meta


def freeze_batchnorm_statistics(module):
    """Avoid 1x1 PSP BatchNorm failures with sparse stereo batch size 1."""
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                           nn.SyncBatchNorm)):
        module.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--nir-dir", type=Path, default=DEFAULT_NIR)
    parser.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "gatedrccbstereo.pt",
                        help="Fully load both encoders and stereo/decoder weights; optimizer starts fresh")
    parser.add_argument("--output", type=Path, default=ROOT / "rgb_nir_lidar_finetune_run")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="depth PNG value to metres; default is uint16 millimetres")
    parser.add_argument("--max-depth", type=float, default=220.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None,
                        help="optionally train on only the first N shuffled samples (useful for smoke tests)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-interval", type=int, default=1,
                        help="print training progress every N samples; default: every sample")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--freeze-encoders", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or not 0 < args.val_fraction < 1:
        parser.error("Require epochs >= 1, lr > 0 and 0 < val-fraction < 1.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    calib = calibration()
    dataset = RGBNIRLiDARDataset(args.rgb_dir, args.nir_dir, args.depth_dir, calib,
                                 args.depth_scale, args.max_depth)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.limit is not None:
        if args.limit < 2:
            raise ValueError("--limit must be at least 2 so train and validation each have a sample")
        indices = indices[:min(args.limit, len(indices))]
    if len(indices) < 2:
        raise ValueError("At least two matched samples are needed for train/validation.")
    n_val = min(len(indices) - 1, max(1, int(len(indices) * args.val_fraction)))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    args.output.mkdir(parents=True, exist_ok=True)
    train_manifest = args.output / "train.txt"
    val_manifest = args.output / "val.txt"
    write_split_manifest(train_manifest, dataset, train_idx, "train", args.seed,
                         args.val_fraction, args.limit)
    write_split_manifest(val_manifest, dataset, val_idx, "val", args.seed,
                         args.val_fraction, args.limit)
    print(f"Saved fixed split manifests: train={train_manifest.resolve()} ({len(train_idx)}), "
          f"val={val_manifest.resolve()} ({len(val_idx)}); seed={args.seed}", flush=True)
    train_loader = DataLoader(train_set, batch_size=None, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=None, shuffle=False, num_workers=0)

    model, loaded = make_model(args.checkpoint, device)
    nir_channels = model.backbone["gated"].in_channels
    dataset.nir_channels = nir_channels
    if args.freeze_encoders:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    print(
        f"Full checkpoint loaded: {args.checkpoint.resolve()}; RGB=3, NIR={nir_channels}; "
        f"loaded tensors: {loaded}; encoders frozen: {args.freeze_encoders}",
        flush=True,
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = BerHuSparseLoss().to(device)
    loss_log = args.output / "loss.txt"
    with loss_log.open("a", encoding="utf-8") as log:
        log.write(f"\n# Run {datetime.now().astimezone().isoformat()} initialization=pretrained_full "
                  f"rgb_channels=3 nir_channels={nir_channels} checkpoint={args.checkpoint.resolve()} lr={args.lr} seed={args.seed} "
                  f"train_samples={len(train_set)} val_samples={len(val_set)}\n")
        log.write("# MAE: final output only, valid LiDAR pixels, epoch values weighted by pixel count\n")
        log.write("epoch\ttrain_loss\tval_loss\ttrain_depth_mae_m\ttrain_disp_mae_px\tval_depth_mae_m\tval_disp_mae_px\n")
    print(f"Epoch loss log: {loss_log.resolve()}", flush=True)
    write_json(args.output / "calibration.json", {
        "image_size_wh": [640, 512], "fx_rectified": calib["fx"],
        "baseline_m": calib["baseline"], "P1": calib["P1"].tolist(),
        "P2": calib["P2"].tolist(), "depth_scale_m_per_uint": args.depth_scale,
        "depth_is_projected_to": "left RGB camera, then remapped to rectified left grid",
        "rgb_nir_pairs": len(dataset),
        "input_channels": {"rccb": 3, "gated": nir_channels},
        "initialization": "pretrained_full",
        "checkpoint": str(args.checkpoint.resolve()),
    })

    history = []
    for epoch in range(args.epochs):
        model.train()
        model.apply(freeze_batchnorm_statistics)
        train_values = []
        train_mae = FinalOutputMAE()
        total_train = len(train_loader)
        print(f"[epoch {epoch + 1}/{args.epochs}] train: {total_train} samples", flush=True)
        for step, sample in enumerate(train_loader, start=1):
            inputs, metas = move_sample(sample, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, metas, do_csm=False, do_secondary=False,
                            hetero_encoders=True, iters=4, iters_hd=2)
            loss, details = criterion(outputs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {details}")
            sample_mae = train_mae.update(outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            train_values.append(loss_value)
            if step == 1 or step % max(1, args.log_interval) == 0 or step == total_train:
                depth_value = float(details.get("loss/depth", loss.detach()).cpu())
                disp_value = float(details.get("loss/disp", loss.detach()).cpu())
                running = float(np.mean(train_values))
                print(
                    f"  train {step:>4}/{total_train}: "
                    f"loss={loss_value:.4f} avg={running:.4f} "
                    f"depth={depth_value:.4f} disp={disp_value:.4f} "
                    f"depth_mae_m={format_mae(sample_mae['depth'])} "
                    f"disp_mae_px={format_mae(sample_mae['disp'])}",
                    flush=True,
                )

        model.eval()
        val_values = []
        val_mae = FinalOutputMAE()
        total_val = len(val_loader)
        print(f"[epoch {epoch + 1}/{args.epochs}] val: {total_val} samples", flush=True)
        with torch.no_grad():
            for step, sample in enumerate(val_loader, start=1):
                inputs, metas = move_sample(sample, device)
                outputs = model(inputs, metas, do_csm=False, do_secondary=False,
                                hetero_encoders=True, iters=4, iters_hd=2)
                loss, _ = criterion(outputs)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite validation loss at epoch {epoch + 1}, sample {step}")
                sample_mae = val_mae.update(outputs)
                loss_value = float(loss.detach().cpu())
                val_values.append(loss_value)
                if step == 1 or step % max(1, args.log_interval) == 0 or step == total_val:
                    print(
                        f"  val   {step:>4}/{total_val}: "
                        f"loss={loss_value:.4f} avg={np.mean(val_values):.4f} "
                        f"depth_mae_m={format_mae(sample_mae['depth'])} "
                        f"disp_mae_px={format_mae(sample_mae['disp'])}",
                        flush=True,
                    )
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(train_values)),
               "val_loss": float(np.mean(val_values))}
        for phase, metrics in (("train", train_mae), ("val", val_mae)):
            summary = metrics.summary()
            row[f"{phase}_depth_mae_m"] = summary["depth"]
            row[f"{phase}_disp_mae_px"] = summary["disp"]
            row[f"{phase}_depth_valid_pixels"] = metrics.counts["depth"]
            row[f"{phase}_disp_valid_pixels"] = metrics.counts["disp"]
        history.append(row)
        # Persist completed epochs before checkpoint serialization, so an
        # interruption in a later epoch does not discard previous loss records.
        with loss_log.open("a", encoding="utf-8") as log:
            log.write(f"{row['epoch']}\t{row['train_loss']:.8f}\t{row['val_loss']:.8f}\t"
                      + "\t".join(format_mae(row[key]) for key in (
                          "train_depth_mae_m", "train_disp_mae_px",
                          "val_depth_mae_m", "val_disp_mae_px")) + "\n")
        write_json(args.output / "history.json", {"history": history,
                   "initialization": "pretrained_full", "source_checkpoint": str(args.checkpoint.resolve()), "loaded_parameter_tensors": loaded,
                   "device": str(device), "csm": False, "secondary": False})
        torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(), "history": history,
                    "initialization": "pretrained_full", "source_checkpoint": str(args.checkpoint.resolve()),
                    "input_channels": {"rccb": 3, "gated": nir_channels}},
                   args.output / "latest.pt")
        print(row, flush=True)


if __name__ == "__main__":
    main()
