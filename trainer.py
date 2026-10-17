import os
import time
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List
from dtools import _bgr_to_rgb, _to_tb_chw, colorize_depth
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import Sampler
from tensorboardX import SummaryWriter

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from networks.gatedrccbstereo import GatedRCCBStereo
from networks.losses.losses import PhotometricLRConsistencyLoss
import hydra

# ----------------------------
# Utilities
# ----------------------------
def ddp_is_available() -> bool:
    return dist.is_available()

def ddp_is_initialized() -> bool:
    return ddp_is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def barrier():
    if ddp_is_initialized():
        dist.barrier()

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def move_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [move_to_device(v, device) for v in x]
        return type(x)(t)
    return x

def atomic_torch_save(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def cfg_get(cfg: DictConfig, key: str, default=None):
    # supports dot paths like "train.max_iters"
    val = OmegaConf.select(cfg, key, default=default)
    return val


# ----------------------------
# DDP setup
# ----------------------------
def setup_distributed() -> Tuple[torch.device, int, int, int]:
    """
    Assumes you launch with torchrun, which sets:
      - RANK, WORLD_SIZE, LOCAL_RANK
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        dist.init_process_group(backend=backend, init_method="env://")
        return device, rank, world_size, local_rank

    # Single process fallback
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    return device, 0, 1, 0


# ----------------------------
# Dummy loss (extend later)
# ----------------------------
class BerHuStereoLoss(nn.Module):
    def __init__(
        self,
        keys,                         # e.g. ["depth", "disp"]
        gamma: float = 0.8,
        res_weights=None,             # e.g. {"depth":1.0, "disp":0.5}
        depth0_weight_mode: str = "last",  # "last" => depth0 weighted like last iter
        strict: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.keys = list(keys)
        self.gamma = float(gamma)
        self.res_weights = res_weights or {}
        self.depth0_weight_mode = depth0_weight_mode
        self.strict = bool(strict)
        self.eps = float(eps)

    @staticmethod
    def _as_1ch(x: torch.Tensor) -> torch.Tensor:
        return x[:, :1] if (x.ndim == 4 and x.shape[1] > 1) else x

    def _pick_valid_mask(self, outputs: dict, key: str, gt: torch.Tensor) -> torch.Tensor:
        # NOTE: for sparse GT, gt_valid is expected to match gt resolution.
        m = outputs.get("gt_valid", None)
        if m is None:
            m = gt > 0
        return m.bool()

    def _pick_weight_map(self, outputs: dict, pred: torch.Tensor) -> torch.Tensor:
        """
        Returns weight map [B,1,H,W] aligned to pred resolution, or None if not available.
        Uses outputs["lidar_confidence"] if present.
        """
        w = outputs.get("lidar_confidence", None)
        if w is None or (not torch.is_tensor(w)):
            return None

        w = self._as_1ch(w).float()

        if w.shape[-2:] != pred.shape[-2:]:
            w = F.interpolate(w, size=pred.shape[-2:], mode="bilinear", align_corners=False)

        return w.clamp(min=0.0, max=1.0)

    def _berhu(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        pred = pred.float()
        gt = gt.float()
        valid = valid.bool()

        # broadcast valid to pred dims
        while valid.ndim < pred.ndim:
            valid = valid.unsqueeze(1)

        # broadcast/align weights if provided
        if weight is not None:
            weight = weight.float()
            while weight.ndim < pred.ndim:
                weight = weight.unsqueeze(1)
            if weight.shape[-2:] != pred.shape[-2:]:
                weight = F.interpolate(weight, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            weight = weight.clamp(min=0.0, max=1.0)

        diff = pred - gt
        l1 = diff.abs()

        if not valid.any():
            return pred.mean() * 0.0  # keep grads alive

        # berhu threshold from valid pixels (unweighted)
        l1v = l1[valid]
        c = 0.1 * l1v.max().clamp(min=self.eps)

        l2 = (diff * diff + c * c) / (2.0 * c)
        per = torch.where(l1 < c, l1, l2)

        if weight is None:
            return per[valid].mean()

        pv = per[valid]
        wv = weight[valid]
        denom = wv.sum().clamp(min=self.eps)
        return (pv * wv).sum() / denom

    def _iter_weights(self, n: int, key: str):
        return [self.gamma ** (n - i - 1) for i in range(n)]

    def _res_weight(self, key: str) -> float:
        if self.res_weights is None:
            return 1.0
        return float(self.res_weights.get(key, 1.0))

    def _maybe_use_hd_gt(self, outputs: dict, key: str, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor):
        """
        If pred resolution differs from gt resolution AND outputs contains HD GT for this key,
        switch to HD GT/valid (only if present). Otherwise keep original.
        Keeping it simple: only checks shape mismatch and presence of outputs[f"gt_{key}_hd"].
        """
        if gt is None or (not torch.is_tensor(gt)):
            return gt, valid

        if gt.shape[-2:] == pred.shape[-2:]:
            return gt, valid

        gt_hd_key = f"gt_{key}_hd"
        gt_hd = outputs.get(gt_hd_key, None)
        if gt_hd is None or (not torch.is_tensor(gt_hd)):
            return gt, valid

        gt_hd = self._as_1ch(gt_hd).to(device=pred.device, dtype=pred.dtype)

        # only use it if it actually matches pred resolution
        if gt_hd.shape[-2:] != pred.shape[-2:]:
            return gt, valid

        # valid mask for hd: prefer explicit if provided, else derive from sparse zeros
        valid_hd = outputs.get("gt_valid_hd", None)
        if valid_hd is None:
            valid_hd = gt_hd > 0
        else:
            valid_hd = valid_hd.to(device=pred.device).bool()
            # ensure shape match; if not, fall back to gt_hd>0
            if valid_hd.ndim == 4:
                valid_hd_ = self._as_1ch(valid_hd)
                if valid_hd_.shape[-2:] != pred.shape[-2:]:
                    valid_hd = (gt_hd > 0)
                else:
                    valid_hd = valid_hd_
            elif valid_hd.ndim == 3:
                if valid_hd.shape[-2:] != pred.shape[-2:]:
                    valid_hd = (gt_hd > 0)
            else:
                valid_hd = (gt_hd > 0)

        return gt_hd, valid_hd

    def forward(self, outputs, prev_total=None, losses=None, alpha=1.0):
        if not isinstance(outputs, dict):
            raise TypeError("BerHuStereoLoss expects outputs to be a dict.")

        device = next((v.device for v in outputs.values() if torch.is_tensor(v)), torch.device("cpu"))
        if prev_total is None:
            prev_total = torch.zeros((), device=device)

        if losses is None:
            losses = {}

        total = torch.zeros((), device=device)
        alpha = float(alpha)

        for key in self.keys:
            gt_key = f"gt_{key}"
            if gt_key not in outputs:
                if self.strict:
                    raise KeyError(f"Missing {gt_key} in outputs. Available keys: {list(outputs.keys())[:50]} ...")
                else:
                    continue

            gt = self._as_1ch(outputs[gt_key]).to(device)
            valid = self._pick_valid_mask(outputs, key, gt).to(device)

            pred_obj = outputs.get(f"{key}_preds", outputs.get(key, None))
            if pred_obj is None:
                if self.strict:
                    raise KeyError(f"Missing predictions for '{key}' (expected '{key}_preds' or '{key}')")
                else:
                    continue

            if torch.is_tensor(pred_obj):
                preds = [pred_obj]
            elif isinstance(pred_obj, (list, tuple)):
                preds = list(pred_obj)
            else:
                if self.strict:
                    raise TypeError(f"Predictions for '{key}' must be Tensor or list/tuple of Tensors.")
                else:
                    continue

            n = len(preds)
            iter_w = self._iter_weights(n, key)
            res_w = self._res_weight(key)

            key_total = torch.zeros((), device=device)

            # --- extra depth0 term (depth-only) -------------------------------
            if (
                key == "depth"
                and self.depth0_weight_mode == "last"
                and "depth0" in outputs
                and torch.is_tensor(outputs["depth0"])
            ):
                depth0 = self._as_1ch(outputs["depth0"]).to(device)
                wmap0 = self._pick_weight_map(outputs, depth0)

                # If depth0 is ever HD, we can also switch to gt_depth_hd; keep same rule.
                gt0, v0 = self._maybe_use_hd_gt(outputs, key="depth", pred=depth0, gt=gt, valid=valid)
                li0 = self._berhu(depth0, gt0, v0, weight=wmap0)

                w0 = 1.0 * res_w
                li0_w = li0 * w0

                key_total = key_total + li0_w
                total = total + li0_w
                losses["loss/depth0"] = li0_w.detach()

            # --- iterative predictions ---------------------------------------
            for i, pred in enumerate(preds):
                pred = self._as_1ch(pred).to(device)
                wmap = self._pick_weight_map(outputs, pred)

                # switch to HD GT only when needed and available
                gt_i, valid_i = self._maybe_use_hd_gt(outputs, key=key, pred=pred, gt=gt, valid=valid)

                w = float(iter_w[i]) * float(res_w)
                li = self._berhu(pred, gt_i, valid_i, weight=wmap)
                li_w = li * w

                key_total = key_total + li_w
                total = total + li_w

                losses[f"loss/{key}/i{i}"] = li_w.detach()

            losses[f"loss/{key}"] = key_total.detach()

        total = prev_total + alpha * total
        losses["loss/total"] = total.detach()
        return total, losses, outputs

# ----------------------------
# Trainer
# ----------------------------
@dataclass
class TrainState:
    iteration: int = 0
    epoch: int = 0




class Trainer:
    def __init__(self, cfg: DictConfig, model: nn.Module, dataloader, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.state = TrainState()

        self.model = model.to(device)

        # DDP wrap (only if initialized)
        self.use_ddp = ddp_is_initialized()
        if self.use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        self.dataloader = dataloader

        # Loss
        self.criterion = {}

        if self.cfg.train.loss.berhu:
            self.criterion["berhu"] = BerHuStereoLoss(
                keys=["depth", "disp"],
                gamma=0.8,
                res_weights={"depth": 1.0, "disp": 0.75},
            ).to(device)
        if self.cfg.train.loss.photo:
            self.criterion["photo"] = PhotometricLRConsistencyLoss().to(device)


        # Optimizer params
        base_lr = float(cfg_get(cfg, "optimizer.lr", 1e-5))
        weight_decay = float(cfg_get(cfg, "optimizer.weight_decay", 0.0))

        warmup_frac = float(cfg_get(cfg, "optimizer.warmup_frac", 0.02))
        const_frac = float(cfg_get(cfg, "optimizer.const_frac", 0.6))
        min_lr_rate = float(cfg_get(cfg, "optimizer.min_lr_rate", 0.05))
v
        # Iter / epoch limits
        n_total_epoch = int(cfg_get(cfg, "trainer.num_epochs", 1000))
        max_iters = int(cfg_get(cfg, "train.max_iters", 1000))

        # effective total optimizer steps = smaller of (max_iters, total_epoch_steps)
        total_epoch_steps = n_total_epoch * len(self.dataloader)
        self.total_steps = min(max_iters, total_epoch_steps)
        if is_main_process():
            print(f"Total Steps: {self.total_steps}")
        # ------------------------------------------------------

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
        )

        # ---- STEP-BASED SCHEDULER ----
        T = self.total_steps  # effective steps

        def lr_lambda(step_idx: int) -> float:
            """
            step_idx: 0-based optimizer step index (called by scheduler).
            Returns multiplicative factor for the base LR.
            """
            s = step_idx + 1  # make it 1-based like original warmup/decay formulas

            if s <= T * warmup_frac:
                return (1.0 - min_lr_rate) / (T * warmup_frac) * s + min_lr_rate
            elif s <= T * const_frac:
                return 1.0
            else:
                return ((min_lr_rate - 1.0) / ((1.0 - const_frac) * T)) * s + \
                    (1.0 - min_lr_rate * const_frac) / (1.0 - const_frac)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
        )

        # AMP
        self.use_amp = bool(cfg_get(cfg, "train.amp", False)) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Logging + ckpt intervals
        self.log_interval = int(cfg_get(cfg, "train.log_interval", 10))
        self.ckpt_interval = int(cfg_get(cfg, "train.ckpt_interval", 50))

        # Checkpoint dir
        out_dir = cfg_get(cfg, "train.output_dir", "outputs")
        self.ckpt_dir = Path(out_dir) / "checkpoints"

        self.maybe_resume()

        # TensorBoard stub (rank 0 only)
        self.tb_enabled = bool(cfg_get(cfg, "train.tensorboard.enable", False)) and is_main_process()
        self.tb_logdir = Path(cfg_get(cfg, "train.tensorboard.logdir", str(Path(out_dir) / "tb")))
        self.tb_image_interval = int(cfg_get(cfg, "train.tensorboard.image_interval", 200))
        self.tb_ref_layout = cfg_get(cfg, "train.tensorboard.ref_input_layout", {})  # dict

        self.tb_writer = None
        if self.tb_enabled:
            self.tb_logdir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(self.tb_logdir))

        # Timing
        self._t_last = time.time()

    def parse_batch(self, batch: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Tries to extract (inputs, metas) from batch with minimal assumptions.

        Expected (recommended) batch format:
            {
              "inputs": {...},   # your per-cam tensors, etc.
              "metas":  {...},   # intrinsics/extrinsics/augment details
              ...
            }

        If "inputs" is missing, we treat all keys except "metas" as inputs.
        """
        metas = {}
        if isinstance(batch, dict):
            metas = batch.get("data_samples", {})
            if "inputs" in batch:
                inputs = batch["inputs"]
            else:
                inputs = {k: v for k, v in batch.items() if k != "metas"}
            return inputs, metas

        # If your dataloader returns tuple/list, adapt here.
        # Example: (inputs, metas)
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]

        # Fallback: treat entire batch as inputs
        return batch, metas

    def train_step(self, batch: Any) -> Dict[str, float]:
        self.model.train()

        inputs, metas = self.parse_batch(batch)
        inputs = move_to_device(inputs, self.device)
        metas = move_to_device(metas, self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(inputs, metas)  # normal side

        if self.cfg.train.loss.photo:
            # complementary side: "other" of what the first pass used
            ref_side = outputs["chosen_ref_side"]
            other_side = "right" if ref_side == "left" else "left"

            # DDP-safe: get the real module
            model_core = self.model.module if hasattr(self.model, "module") else self.model

            # IMPORTANT: eval() so BN doesn't update running stats
            with torch.no_grad():  # stronger than no_grad
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs_other = model_core(inputs, metas, ref_side=other_side)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            total_loss = None
            losses = None
            if self.cfg.train.loss.berhu:
                total_loss, losses, outputs = self.criterion['berhu'](outputs, alpha=0.5,
                    prev_total=total_loss, losses=losses)
            if self.cfg.train.loss.photo:
                total_loss, losses, outputs = self.criterion['photo'](
                    outputs, outputs_other,
                    prev_total=total_loss, losses=losses, alpha=100.0
                )
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if hasattr(self, "scheduler") and self.scheduler is not None:
            if not hasattr(self, "total_steps"):
                self.scheduler.step()
            else:
                if self.state.iteration < self.total_steps:
                    self.scheduler.step()

        if isinstance(losses, dict):
            losses["lr"] = self.optimizer.param_groups[0]["lr"]

        return losses, outputs

    def log(self, losses, outputs):
        it = self.state.iteration
        now = time.time()
        dt = now - self._t_last
        self._t_last = now

        lr = losses.get("lr", 0.0)

        if isinstance(losses, dict):
            # expected from your BerHuStereoLoss: losses["loss/total"]
            total_loss = losses.get("loss/total", None)
            if torch.is_tensor(total_loss):
                total_loss_val = float(total_loss.detach().cpu())
            else:
                total_loss_val = float("nan")
        else:
            total_loss_val = float(losses.detach().cpu()) if torch.is_tensor(losses) else float("nan")

        msg = f"[rank {get_rank()}] iter={it:08d} loss={total_loss_val:.6f} lr={lr:.2e} dt={dt * 1000:.1f}ms"
        if self.device.type == "cuda":
            mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            msg += f" max_mem={mem:.1f}MB"
        if is_main_process():
            print(msg, flush=True)

        # -----------------------------
        # TensorBoard logging (main process only)
        # -----------------------------
        if self.tb_writer is None or (not is_main_process()):
            return

        # Scalars
        self.tb_writer.add_scalar("train/lr", lr, global_step=it)
        if isinstance(losses, dict):
            for k, v in losses.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    self.tb_writer.add_scalar(f"train/{k}", float(v.detach().cpu()), global_step=it)

        # Images (throttled)
        img_interval = getattr(self.cfg.train.tensorboard, "image_interval", 200) \
            if hasattr(self, "cfg") and hasattr(self.cfg.train, "tensorboard") else 200
        if (it % int(img_interval)) != 0:
            return

        # --- depth pred & GT (full-res only) ---
        if isinstance(outputs, dict):
            # final full-res depth
            pred_depth = outputs.get("depth", None)
            # init depth completion (full-res)
            pred0_depth = outputs.get("depth0", None)
            # full-res GT depth
            gt_depth = outputs.get("gt_depth", None)

            if (
                    pred_depth is not None and
                    gt_depth is not None and
                    torch.is_tensor(pred_depth) and
                    torch.is_tensor(gt_depth)
            ):
                # choose a reasonable max_depth for viz based on modality/layout
                modality = outputs.get("chosen_modality", "default")
                layout = None
                if hasattr(self, "cfg") and hasattr(self.cfg.train, "tensorboard"):
                    layout = self.cfg.train.tensorboard.ref_input_layout.get(
                        modality,
                        self.cfg.train.tensorboard.ref_input_layout.get("default", {})
                    )
                depth_max = float(layout.get("depth_max", 150.0)) if layout else 150.0

                # colorize predictions and GT
                pred_col = colorize_depth(pred_depth, max_depth=depth_max, radius=0, lidar=False)  # BGR uint8
                gt_col = colorize_depth(gt_depth, max_depth=depth_max, radius=3, lidar=True)  # BGR uint8 (sparse)

                pred_rgb = _bgr_to_rgb(pred_col)
                gt_rgb = _bgr_to_rgb(gt_col)

                self.tb_writer.add_image("viz/depth_pred", _to_tb_chw(pred_rgb), global_step=it)
                self.tb_writer.add_image("viz/depth_gt", _to_tb_chw(gt_rgb), global_step=it)

                # optional: log depth0 if present
                if pred0_depth is not None and torch.is_tensor(pred0_depth):
                    pred0_col = colorize_depth(pred0_depth, max_depth=depth_max, radius=0, lidar=False)
                    pred0_rgb = _bgr_to_rgb(pred0_col)
                    self.tb_writer.add_image("viz/depth0_pred", _to_tb_chw(pred0_rgb), global_step=it)

        # --- ref_input visualization with configurable layout ---
        ref_inp = outputs.get("ref_input", None) if isinstance(outputs, dict) else None
        ref_mod = outputs.get("ref_modality", outputs.get("chosen_modality", "default")) if isinstance(outputs,
                                                                                                       dict) else "default"

        if torch.is_tensor(ref_inp):
            # ref_inp expected BCHW; show() will take batch 0, but we want control over channels.
            x = ref_inp.detach()
            if x.is_cuda:
                x = x.cpu()
            x = x.float()[0]  # [C,H,W]

            # pick layout
            layout = None
            if hasattr(self, "cfg") and hasattr(self.cfg.train, "tensorboard"):
                layout = self.cfg.train.tensorboard.ref_input_layout.get(
                    ref_mod,
                    self.cfg.train.tensorboard.ref_input_layout.get("default", None)
                )

            if layout is None:
                layout = dict(
                    active_rgb=[0, 1, 2],
                    passive=[],
                    sparse_depth=min(3, x.shape[0] - 1),
                    depth_max=150.0,
                    radius=3,
                )

            H, W = x.shape[-2], x.shape[-1]

            # active RGB: assume normalized [-1,1] (like your forward). Convert to [0,1].
            rgb_idx = layout.get("active_rgb", [0, 1, 2])
            if len(rgb_idx) == 3 and max(rgb_idx) < x.shape[0]:
                rgb = x[rgb_idx]  # [3,H,W]
                rgb = (rgb + 1.0) * 0.5
                rgb = rgb.clamp(0.0, 1.0).numpy()
                rgb_hwc = np.transpose(rgb, (1, 2, 0))
                self.tb_writer.add_image("viz/ref_active_rgb", _to_tb_chw(rgb_hwc), global_step=it)

            # passive slices: plot each as grayscale
            passive = layout.get("passive", [])
            for j, ch in enumerate(passive):
                if ch < x.shape[0]:
                    img = x[ch].numpy()
                    self.tb_writer.add_image(f"viz/ref_passive_{j}", _to_tb_chw(img), global_step=it)

            # sparse depth channel: colorize with radius
            sd_ch = layout.get("sparse_depth", None)
            if sd_ch is not None and sd_ch < x.shape[0]:
                sd = x[sd_ch].numpy() * self.cfg.model.model_args.max_depth[outputs['dataset']][outputs['ref_modality']]
                depth_max = float(layout.get("depth_max", 150.0))
                radius = int(layout.get("radius", 3))
                sd_col = colorize_depth(sd, max_depth=depth_max, radius=radius, lidar=True)
                sd_rgb = _bgr_to_rgb(sd_col)
                self.tb_writer.add_image("viz/ref_sparse_depth", _to_tb_chw(sd_rgb), global_step=it)

        self.tb_writer.flush()

    def save_checkpoint(self):
        if not is_main_process():
            return

        it = self.state.iteration
        ckpt_path = self.ckpt_dir / f"iter_{it:08d}.pt"

        # unwrap model (DDP -> underlying module)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

        payload = {
            "iteration": it,
            "epoch": self.state.epoch,
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
            "scheduler": self.scheduler.state_dict(),
        }
        atomic_torch_save(payload, ckpt_path)

        # (optional) update "latest" pointer
        latest = self.ckpt_dir / "latest.pt"
        atomic_torch_save(payload, latest)

    def load_checkpoint(self, ckpt_path: str, strict: bool = True):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # model
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model

        model_to_load.load_state_dict(ckpt["model"], strict=strict)

        # optimizer
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            if strict:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            else:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                except ValueError as e:
                    print(f"[WARN] Could not load optimizer state_dict: {e}")
                    print("[WARN] Continuing with freshly initialized optimizer.")

        # scaler
        if self.use_amp and ckpt.get("scaler") is not None:
            if strict:
                self.scaler.load_state_dict(ckpt["scaler"])
            else:
                try:
                    self.scaler.load_state_dict(ckpt["scaler"])
                except Exception as e:
                    print(f"[WARN] Could not load scaler state_dict: {e}")
                    print("[WARN] Continuing with fresh scaler.")

        # scheduler
        if "scheduler" in ckpt and self.scheduler is not None:
            if strict:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            else:
                try:
                    self.scheduler.load_state_dict(ckpt["scheduler"])
                except Exception as e:
                    print(f"[WARN] Could not load scheduler state_dict: {e}")
                    print("[WARN] Continuing with fresh scheduler.")

        # restore state
        self.state.iteration = int(ckpt.get("iteration", 0))
        self.state.epoch = int(ckpt.get("epoch", 0))

    def maybe_resume(self):
        resume = bool(cfg_get(self.cfg, "train.resume", False))
        if not resume:
            return

        strict = bool(cfg_get(self.cfg, "train.resume_strict", True))
        path = str(cfg_get(self.cfg, "train.resume_path", "")).strip()

        # optional convenience: if resume_path empty, try latest.pt
        if not path:
            path = str(self.ckpt_dir / "latest.pt")

        if not Path(path).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")

        if is_main_process():
            print(f"[resume] Loading checkpoint: {path} (strict={strict})", flush=True)

        self.load_checkpoint(path, strict=strict)

    def train(self):
        # Works for:
        # - single dataset + DistributedSampler (dataloader.sampler)
        # - concat dataset + MultiDatasetBatchSampler (dataloader.batch_sampler)
        sampler = getattr(self.dataloader, "sampler", None)
        batch_sampler = getattr(self.dataloader, "batch_sampler", None)

        while self.state.iteration < self.total_steps:

            # 1) set epoch for whichever object controls shuffling
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(self.state.epoch)
            elif hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(self.state.epoch)

            # 2) normal loop
            for batch in self.dataloader:
                losses, outputs = self.train_step(batch)

                if (self.state.iteration % self.log_interval) == 0:
                    self.log(losses, outputs)

                if (self.state.iteration % self.ckpt_interval) == 0:
                    self.save_checkpoint()

                self.state.iteration += 1
                if self.state.iteration >= self.total_steps:
                    break

            self.state.epoch += 1

        barrier()
        if self.tb_writer is not None:
            self.tb_writer.close()



@torch.no_grad()
def _maybe_print_batch_structure(dataloader):
    # helpful once during wiring
    if is_main_process():
        for batch in dataloader:
            print("Example batch keys/type:", type(batch), getattr(batch, "keys", lambda: None)())
            break


def build_dataloader(cfg, dataset_or_datasets):
    """
    - If a single dataset: uses DistributedSampler under DDP (your original behavior).
    - If multiple datasets: uses ConcatDataset + MultiDatasetBatchSampler so that:
        * each batch is from exactly one dataset
        * dataset choice is randomized per batch
        * DDP-safe: each rank has the same #batches per epoch
        * Hydra-safe: instantiate() still resolves collate_fn, etc.
    """
    # normalize input
    datasets = list(dataset_or_datasets) if isinstance(dataset_or_datasets, (list, tuple)) else [dataset_or_datasets]

    # ---- single dataset path (your original logic) ----
    if len(datasets) == 1:
        dataset = datasets[0]
        if ddp_is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                drop_last=bool(getattr(cfg.dataloader, "drop_last", False)),
            )
            return instantiate(cfg.dataloader, dataset=dataset, sampler=sampler, shuffle=False)
        else:
            return instantiate(cfg.dataloader, dataset=dataset)

    # ---- multi dataset path ----
    concat = ConcatDataset(datasets)
    lengths = [len(d) for d in datasets]

    batch_size = int(cfg.dataloader.batch_size)
    drop_last = bool(getattr(cfg.dataloader, "drop_last", False))
    seed = int(getattr(cfg, "seed", 0))

    batch_sampler = MultiDatasetBatchSampler(
        lengths=lengths,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=True,
        seed=seed,
        num_replicas=get_world_size() if ddp_is_initialized() else 1,
        rank=get_rank() if ddp_is_initialized() else 0,
        # weights=[...]  # optional
    )

    # make a clean dataloader config for instantiate()
    # (remove args that conflict with batch_sampler)
    dl_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataloader, resolve=True))
    for k in ["batch_size", "shuffle", "sampler", "batch_sampler", "drop_last"]:
        if k in dl_cfg:
            dl_cfg.pop(k)

    return instantiate(dl_cfg, dataset=concat, batch_sampler=batch_sampler)

_DATASET_KEY_RE = re.compile(r"^dataset(\d+)?$")  # matches: dataset, dataset2, dataset10, ...

def instantiate_datasets_from_cfg(cfg):
    ds_items = []

    for k in cfg.keys():
        key = str(k)
        m = _DATASET_KEY_RE.match(key)
        if not m:
            continue

        ds_cfg = cfg.get(k)
        if ds_cfg is None:
            continue

        # dataset -> order 1, dataset2 -> order 2, ...
        order = int(m.group(1)) if m.group(1) is not None else 1
        ds_items.append((order, ds_cfg))

    ds_items.sort(key=lambda x: x[0])
    return [instantiate(ds_cfg) for _, ds_cfg in ds_items]


class MultiDatasetBatchSampler(Sampler):
    """
    Emits batches of indices for a ConcatDataset such that each batch is drawn
    from exactly ONE sub-dataset. Dataset choice is randomized per batch.

    Works in single-process and DDP (num_replicas/rank).
    """
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        weights=None,  # optional sampling weights per dataset
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        # offsets into ConcatDataset index space
        self.offsets = [0]
        for L in lengths[:-1]:
            self.offsets.append(self.offsets[-1] + L)

        if weights is not None:
            if len(weights) != len(lengths):
                raise ValueError("weights must have same length as lengths")
            w = torch.tensor(weights, dtype=torch.double)
            if (w < 0).any() or w.sum() <= 0:
                raise ValueError("weights must be non-negative and sum to > 0")
            self.weights = (w / w.sum()).tolist()
        else:
            self.weights = None

    def _distributed_indices(self, L: int, g: torch.Generator) -> List[int]:
        # base ordering
        if self.shuffle:
            idx = torch.randperm(L, generator=g)
        else:
            idx = torch.arange(L)

        if self.num_replicas == 1:
            return idx.tolist()

        if self.drop_last:
            # drop tail so divisible
            total_size = (L // self.num_replicas) * self.num_replicas
            idx = idx[:total_size]
        else:
            # pad so divisible
            num_samples = int(math.ceil(L / self.num_replicas))
            total_size = num_samples * self.num_replicas
            if total_size > L:
                idx = torch.cat([idx, idx[: (total_size - L)]])

        num_samples = len(idx) // self.num_replicas
        start = self.rank * num_samples
        end = start + num_samples
        return idx[start:end].tolist()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _per_dataset_batches(self, g: torch.Generator):
        """Create per-dataset shuffled+sharded batches (in concat index space)."""
        all_batches = []
        for di, L in enumerate(self.lengths):
            idx_list = self._distributed_indices(L, g)

            batches = []
            for start in range(0, len(idx_list), self.batch_size):
                b = idx_list[start: start + self.batch_size]
                if self.drop_last and len(b) < self.batch_size:
                    continue

                # map dataset-local indices -> concat-global indices
                off = self.offsets[di]
                batches.append([off + x for x in b])

            all_batches.append(batches)

        return all_batches

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        per_ds_batches = self._per_dataset_batches(g)

        # Build a list of dataset-ids, one per batch, then shuffle it
        # so all ranks iterate the same dataset order.
        ds_ids = []
        for di, batches in enumerate(per_ds_batches):
            ds_ids.extend([di] * len(batches))

        if len(ds_ids) == 0:
            return iter([])

        if self.weights is None:
            # uniform shuffle of batch order
            perm = torch.randperm(len(ds_ids), generator=g).tolist()
            ds_ids = [ds_ids[i] for i in perm]
        else:
            # If you want weighted mixing: we still must not exceed available batches.
            # We sample dataset ids repeatedly, skipping those that are exhausted.
            remaining = [len(b) for b in per_ds_batches]
            total = sum(remaining)
            sampled = []
            # convert to tensor once
            w = torch.tensor(self.weights, dtype=torch.double)
            while len(sampled) < total:
                di = int(torch.multinomial(w, 1, generator=g).item())
                if remaining[di] > 0:
                    sampled.append(di)
                    remaining[di] -= 1
            ds_ids = sampled

        # yield batches, popping from the corresponding dataset's batch list
        ptrs = [0] * len(per_ds_batches)
        for di in ds_ids:
            j = ptrs[di]
            ptrs[di] += 1
            yield per_ds_batches[di][j]

    def __len__(self) -> int:
        total = 0
        for L in self.lengths:
            if self.num_replicas == 1:
                num_samples = L
            else:
                num_samples = (L // self.num_replicas) if self.drop_last else int(math.ceil(L / self.num_replicas))

            if self.drop_last:
                total += num_samples // self.batch_size
            else:
                total += math.ceil(num_samples / self.batch_size) if num_samples > 0 else 0
        return total


@hydra.main(config_path="config", config_name="gatedrccbstereo", version_base=None)
def main(cfg: DictConfig):
    device, rank, world_size, local_rank = setup_distributed()

    base_seed = int(cfg_get(cfg, "seed", 0))
    seed_everything(base_seed + rank)

    # Build dataset + dataloader
    datasets = instantiate_datasets_from_cfg(cfg)
    dataloader = build_dataloader(cfg, datasets)

    # Create model (expects cfg.model.model_args, fallback to cfg.model_args)
    model_args = cfg_get(cfg, "model.model_args", None)
    if model_args is None:
        model_args = cfg_get(cfg, "model_args", None)
    if model_args is None:
        raise ValueError("Missing cfg.model.model_args (or cfg.model_args) for GatedRCCBStereo.")

    model = GatedRCCBStereo(model_args=model_args)
    if is_main_process():
        print("Using Model: GatedRCCBStereo")

    trainer = Trainer(cfg=cfg, model=model, dataloader=dataloader, device=device)

    trainer.train()

    if ddp_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

