"""Fine-tune RGB-left/NIR-right stereo with independent DoLP encoders and iAFF.

Run with the same seed, limit, validation fraction and image checkpoint as the
baseline trainer. DoLP PNGs must be raw-camera, uint8 images: RGB DoLP has three
channels and NIR DoLP has one. AoLP is not used.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_rgb_nir_lidar_finetune import (
    ROOT, DEFAULT_RGB, DEFAULT_NIR, DEFAULT_DEPTH,
    RGBNIRLiDARDataset as BaselineDataset,
    BerHuSparseLoss, FinalOutputMAE, calibration, read_image, write_json,
    move_sample, freeze_batchnorm_statistics, format_mae,
    make_model as make_baseline_model,
)

DEFAULT_RGB_DOLP = Path(r"I:\KEDADATA\data\fog\RGB_color_polar_processed\RGBDOLP_color")
DEFAULT_NIR_DOLP = Path(r"I:\KEDADATA\data\fog\NIRuseI4\output_dolp")


class RGBNIRDoLPLiDARDataset(BaselineDataset):
    """Keep baseline sample indexing and add the two co-registered DoLP views."""

    def __init__(self, rgb_dir, nir_dir, depth_dir, calib, rgb_dolp_dir,
                 nir_dolp_dir, depth_scale=0.001, max_depth=220.0):
        super().__init__(rgb_dir, nir_dir, depth_dir, calib, depth_scale, max_depth)
        self.dolp_paths = []
        missing = []
        for _, _, _, key in self.samples:
            paths = (Path(rgb_dolp_dir) / f"{key}_dolp_color_rgb.png",
                     Path(nir_dolp_dir) / f"{key}_dolp_nir.png")
            missing.extend(str(path) for path in paths if not path.is_file())
            self.dolp_paths.append(paths)
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} DoLP files; refusing to change the baseline split. "
                f"Examples: {missing[:5]}"
            )

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        for path, channels, map_name, input_key in (
            (self.dolp_paths[index][0], 3, "map_left", "img_ref_dolp"),
            (self.dolp_paths[index][1], 1, "map_right", "img_comp_dolp"),
        ):
            image = read_image(path)
            expected_shape = (512, 640, 3) if channels == 3 else (512, 640)
            if image.dtype != np.uint8 or image.shape != expected_shape:
                raise ValueError(f"Expected uint8 {expected_shape} DoLP PNG: {path}; "
                                 f"got {image.dtype} {image.shape}")
            if channels == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.remap(image, *self.calib[map_name], cv2.INTER_LINEAR)
            if channels == 1:
                image = image[..., None]
            pair = np.stack([image, image]).transpose(0, 3, 1, 2).copy()
            sample["inputs"][input_key] = torch.from_numpy(pair).float()
        return sample


def write_split_manifest(path, dataset, indices, split_name, seed, val_fraction, limit,
                         auxiliary_input="dolp"):
    with Path(path).open("w", encoding="utf-8", newline="\n") as manifest:
        manifest.write(f"# split={split_name} seed={seed} val_fraction={val_fraction} "
                       f"limit={limit if limit is not None else 'all'} samples={len(indices)}\n")
        manifest.write("# columns: timestamp_prefix\trgb_path\tnir_path\tdepth_path"
                       f"\trgb_{auxiliary_input}_path\tnir_{auxiliary_input}_path\n")
        for index in indices:
            rgb, nir, depth, key = dataset.samples[index]
            auxiliary_paths = dataset.dolp_paths[index] if auxiliary_input == "dolp" else (rgb, nir)
            paths = (rgb, nir, depth, *auxiliary_paths)
            manifest.write("\t".join([key, *(str(p.resolve()) for p in paths)]) + "\n")


def make_model(checkpoint, device, fusion_checkpoint=ROOT / "gatedrccbstereo.pt"):
    # Build the baseline first, consuming exactly its normal initialization RNG.
    model, loaded = make_baseline_model(checkpoint, torch.device("cpu"))
    # Additional parameter initialization must not advance the baseline RNG stream.
    with torch.random.fork_rng(devices=[]):
        model.enable_dolp_fusion(rgb_channels=3, nir_channels=1)
    for modality in ("rccb", "gated"):
        state = model.backbone[modality].state_dict()
        if modality == "gated":
            stem = "mpvit.stem.0.conv.weight"
            # Sum preserves the original stem response to repeated gray input.
            state[stem] = state[stem].sum(dim=1, keepdim=True)
        model.dolp_backbone[modality].load_state_dict(state, strict=True)

    checkpoint_data = torch.load(str(fusion_checkpoint), map_location="cpu", weights_only=True)
    state = checkpoint_data.get("model", checkpoint_data)
    prefix = "csm.fusion."
    fusion_state = {key[len(prefix):]: value for key, value in state.items()
                    if key.startswith(prefix)}
    if not fusion_state:
        raise ValueError(f"No original csm.fusion iAFF weights in {fusion_checkpoint}; "
                         "use --fusion-checkpoint gatedrccbstereo.pt")
    # load_state_dict copies values; neither fusers nor encoders share storage.
    for fusion in (model.dolp_fusion_left, model.dolp_fusion_right):
        fusion.iaff.load_state_dict(fusion_state, strict=True)
    model.to(device).train()
    return model, loaded


def main(*, auxiliary_input="dolp", dataset_class=None):
    """Shared training loop; the intensity entry point supplies its own dataset."""
    if auxiliary_input not in ("dolp", "intensity"):
        raise ValueError(f"Unknown auxiliary input: {auxiliary_input}")
    use_dolp_data = auxiliary_input == "dolp"
    if not use_dolp_data and dataset_class is None:
        raise ValueError("The intensity entry point must supply its dataset class.")
    initialization = f"pretrained_full_{auxiliary_input}_residual_iaff"
    description = (__doc__ if use_dolp_data else
                   "Intensity control: four independent encoders and two iAFF fusers; "
                   "auxiliary inputs are RGB S0 (3 channels) and NIR S0 (1 channel). "
                   "No DoLP/AoLP images are read.")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB)
    parser.add_argument("--nir-dir", type=Path, default=DEFAULT_NIR)
    parser.add_argument("--depth-dir", type=Path, default=DEFAULT_DEPTH)
    if use_dolp_data:
        parser.add_argument("--rgb-dolp-dir", type=Path, default=DEFAULT_RGB_DOLP)
        parser.add_argument("--nir-dolp-dir", type=Path, default=DEFAULT_NIR_DOLP)
    parser.add_argument("--fusion-checkpoint", type=Path, default=ROOT / "gatedrccbstereo.pt",
                        help="Original checkpoint containing csm.fusion iAFF weights")
    parser.add_argument("--check-data-only", action="store_true",
                        help="Save split manifests and check one full sample; do not build/train model")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "gatedrccbstereo.pt",
                        help="Fully load both encoders and stereo/decoder weights; optimizer starts fresh")
    parser.add_argument("--output", type=Path,
                        default=ROOT / f"rgb_nir_lidar_{auxiliary_input}_finetune_depth_new")
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
    parser.add_argument("--checkpoint-encoders", action="store_true",
                        help="Recompute encoder activations in backward to reduce GPU memory; "
                             "preserves RNG state and uses frozen BatchNorm statistics")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or not 0 < args.val_fraction < 1:
        parser.error("Require epochs >= 1, lr > 0 and 0 < val-fraction < 1.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    calib = calibration()
    if use_dolp_data:
        dataset = RGBNIRDoLPLiDARDataset(
            args.rgb_dir, args.nir_dir, args.depth_dir, calib,
            args.rgb_dolp_dir, args.nir_dolp_dir, args.depth_scale, args.max_depth)
    else:
        dataset = dataset_class(args.rgb_dir, args.nir_dir, args.depth_dir, calib,
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
                         args.val_fraction, args.limit, auxiliary_input)
    write_split_manifest(val_manifest, dataset, val_idx, "val", args.seed,
                         args.val_fraction, args.limit, auxiliary_input)
    print(f"Saved fixed split manifests: train={train_manifest.resolve()} ({len(train_idx)}), "
          f"val={val_manifest.resolve()} ({len(val_idx)}); seed={args.seed}", flush=True)
    if args.check_data_only:
        sample = dataset[train_idx[0]]
        print({key: tuple(value.shape) for key, value in sample["inputs"].items()}, flush=True)
        print("Data pairing/split check complete; no model was trained.", flush=True)
        return
    train_loader = DataLoader(train_set, batch_size=None, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=None, shuffle=False, num_workers=0)

    model, loaded = make_model(args.checkpoint, device, args.fusion_checkpoint)
    nir_channels = model.backbone["gated"].in_channels
    dataset.nir_channels = nir_channels
    if args.freeze_encoders:
        for encoders in (model.backbone, model.dolp_backbone):
            for parameter in encoders.parameters():
                parameter.requires_grad_(False)
    print(
        f"Full checkpoint loaded: {args.checkpoint.resolve()}; RGB=3, NIR={nir_channels}; "
        f"auxiliary={auxiliary_input} (3/1), independent iAFF=128/128; loaded baseline tensors: {loaded}; "
        f"encoders frozen: {args.freeze_encoders}",
        flush=True,
    )
    run_config = {
        "seed": args.seed, "limit": args.limit, "val_fraction": args.val_fraction,
        "lr": args.lr, "epochs": args.epochs, "freeze_encoders": args.freeze_encoders,
        "checkpoint_encoders": args.checkpoint_encoders,
        "architecture": "rgb_nir_dolp_4encoders_residual_iaff128_v1",
        "model_args": vars(model.args),
        "dolp_fusion": True, "aolp": False, "shared_encoder_weights": False,
        "auxiliary_input": auxiliary_input, "uses_dolp_data": use_dolp_data,
        "input_channels": {"rccb": 3, "gated": nir_channels,
                           f"rgb_{auxiliary_input}": 3, f"nir_{auxiliary_input}": 1},
        "fusion_checkpoint": str(args.fusion_checkpoint.resolve()),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "data_dirs": {key: str(getattr(args, key).resolve()) for key in
                      (("rgb_dir", "nir_dir", "depth_dir", "rgb_dolp_dir", "nir_dolp_dir")
                       if use_dolp_data else ("rgb_dir", "nir_dir", "depth_dir"))},
        "train_indices": train_idx, "val_indices": val_idx,
        "depth_scale": args.depth_scale, "max_depth": args.max_depth,
    }
    write_json(args.output / "run_config.json", run_config)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = BerHuSparseLoss().to(device)
    loss_log = args.output / "loss.txt"
    with loss_log.open("a", encoding="utf-8") as log:
        log.write(f"\n# Run {datetime.now().astimezone().isoformat()} initialization={initialization} "
                  f"rgb_channels=3 nir_channels={nir_channels} checkpoint={args.checkpoint.resolve()} lr={args.lr} seed={args.seed} "
                  f"train_samples={len(train_set)} val_samples={len(val_set)} "
                  f"auxiliary_input={auxiliary_input} auxiliary_channels=3/1 "
                  f"fusion_checkpoint={args.fusion_checkpoint.resolve()}\n")
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
        "initialization": initialization, "run_config": run_config,
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
                            hetero_encoders=True, dolp_fusion=True, iters=4, iters_hd=2,
                            checkpoint_encoders=args.checkpoint_encoders)
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
                                hetero_encoders=True, dolp_fusion=True, iters=4, iters_hd=2,
                                checkpoint_encoders=args.checkpoint_encoders)
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
                   "initialization": initialization, "run_config": run_config, "source_checkpoint": str(args.checkpoint.resolve()), "loaded_parameter_tensors": loaded,
                   "device": str(device), "csm": False, "secondary": False})
        torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(), "history": history,
                    "initialization": initialization, "run_config": run_config, "source_checkpoint": str(args.checkpoint.resolve()),
                    "input_channels": {"rccb": 3, "gated": nir_channels}},
                   args.output / "latest.pt")
        print(row, flush=True)


if __name__ == "__main__":
    main()
