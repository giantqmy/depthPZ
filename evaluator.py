# evaluator.py
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from networks.gatedrccbstereo import GatedRCCBStereo

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


def cfg_get(cfg: DictConfig, key: str, default=None):
    return OmegaConf.select(cfg, key, default=default)


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

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    return device, 0, 1, 0


def build_eval_dataloader(cfg: DictConfig, dataset):
    """
    Rebuilds the dataloader with a DistributedSampler if DDP is on.
    This assumes your cfg.dataloader accepts sampler=... and shuffle=...
    """
    if ddp_is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
            drop_last=False,
        )
        return instantiate(cfg.dataloader, dataset=dataset, sampler=sampler, shuffle=False)
    else:
        return instantiate(cfg.dataloader, dataset=dataset, shuffle=False)


# ----------------------------
# Depth metrics (LiDAR GT, center crop)
# ----------------------------
def _as_1hw(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts [B,1,H,W], [B,H,W], [1,H,W], [H,W] -> returns [B,H,W]
    """
    if x.ndim == 4:
        if x.shape[1] > 1:
            x = x[:, :1]
        return x[:, 0]
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return x.unsqueeze(0)
    raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")


def _infer_scale(h: int, w: int, base_h: int = 512, base_w: int = 1024) -> Tuple[float, float]:
    return (h / float(base_h), w / float(base_w))


def crop_eval_torch(img_bhw: torch.Tensor, *, rccb: bool) -> torch.Tensor:
    """
    Crop like your numpy version, but scaled to arbitrary resolutions using
    base full-res (H,W) = (512,1024) and crop (420,980).

    RCCB special crop:
      img[72*s : -20*s, : -44*s]
    """
    if img_bhw.ndim != 3:
        raise ValueError(f"crop_eval_torch expects [B,H,W], got {tuple(img_bhw.shape)}")

    B, H, W = img_bhw.shape
    sy, sx = _infer_scale(H, W, base_h=512, base_w=1024)

    if rccb:
        top = int(round(72 * sy))
        bottom = int(round(20 * sy))
        right = int(round(44 * sx))

        y0 = max(0, top)
        y1 = max(y0, H - max(0, bottom))
        x0 = 0
        x1 = max(x0, W - max(0, right))
        return img_bhw[:, y0:y1, x0:x1]

    crop_h = int(round(420 * sy))
    crop_w = int(round(980 * sx))

    crop_h = min(max(crop_h, 1), H)
    crop_w = min(max(crop_w, 1), W)

    y0 = (H - crop_h) // 2
    x0 = (W - crop_w) // 2
    y1 = y0 + crop_h
    x1 = x0 + crop_w
    return img_bhw[:, y0:y1, x0:x1]


def _pick_depth_pred(outputs: Dict[str, Any]) -> Optional[torch.Tensor]:
    x = outputs.get("depth", None)
    if torch.is_tensor(x):
        return x

    xp = outputs.get("depth_preds", None)
    if isinstance(xp, (list, tuple)) and len(xp) > 0 and torch.is_tensor(xp[-1]):
        return xp[-1]

    x0 = outputs.get("depth0", None)
    if torch.is_tensor(x0):
        return x0

    return None


def _pick_gt_depth(outputs: Dict[str, Any], pred: torch.Tensor) -> Optional[torch.Tensor]:
    gt = outputs.get("gt_depth", None)
    if not torch.is_tensor(gt):
        return None

    pred_bhw = _as_1hw(pred)
    ph, pw = pred_bhw.shape[-2], pred_bhw.shape[-1]

    gt_bhw = _as_1hw(gt)
    if gt_bhw.shape[-2:] == (ph, pw):
        return gt

    gt_hd = outputs.get("gt_depth_hd", None)
    if torch.is_tensor(gt_hd):
        gt_hd_bhw = _as_1hw(gt_hd)
        if gt_hd_bhw.shape[-2:] == (ph, pw):
            return gt_hd

    # nearest keeps sparsity / zeros
    gt_4d = gt_bhw.unsqueeze(1)
    gt_rs = F.interpolate(gt_4d, size=(ph, pw), mode="nearest")
    return gt_rs


def _infer_rccb(outputs: Dict[str, Any], default: bool = False) -> bool:
    for k in ("ref_modality", "chosen_modality"):
        s = outputs.get(k, None)
        if isinstance(s, str) and ("rccb" in s.lower()):
            return True
    return default


class DepthLidarMetrics:
    """
    OLD calc_metrics-style behavior (while keeping this evaluator's metric set):

    Changes vs previous implementation:
      1) Aggregation is PER-IMAGE (each sample contributes equally), not pixel-weighted.
      2) Applies clipping to BOTH pred and gt AFTER masking (like old calc_metrics).
      3) ARD is NOT multiplied by 100 (matches old calc_metrics convention).

    Metrics reported (same keys as before):
      mae, rmse, rmse_log, threshold1/2/3, silog, ard, sq_rel, irmse, n
    Here, `n` = number of images/samples that contributed (had any valid pixels).
    """

    def __init__(
        self,
        device: torch.device,
        rccb_default: bool = False,
        clip_min: float = 3.0,
        clip_max: float = 160.0,
    ):
        self.device = device
        self.rccb_default = rccb_default
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.reset()

    def reset(self):
        d = self.device
        self.n = torch.zeros((), device=d, dtype=torch.long)  # number of evaluated images

        # per-image metric sums (float64 for stability)
        self.sum_mae = torch.zeros((), device=d, dtype=torch.float64)
        self.sum_rmse = torch.zeros((), device=d, dtype=torch.float64)
        self.sum_rmse_log = torch.zeros((), device=d, dtype=torch.float64)

        self.sum_t1 = torch.zeros((), device=d, dtype=torch.float64)
        self.sum_t2 = torch.zeros((), device=d, dtype=torch.float64)
        self.sum_t3 = torch.zeros((), device=d, dtype=torch.float64)

        self.sum_silog = torch.zeros((), device=d, dtype=torch.float64)

        self.sum_ard = torch.zeros((), device=d, dtype=torch.float64)     # NOT *100
        self.sum_sq_rel = torch.zeros((), device=d, dtype=torch.float64)
        self.sum_irmse = torch.zeros((), device=d, dtype=torch.float64)

    @torch.no_grad()
    def update(self, outputs: Dict[str, Any]) -> None:
        if not isinstance(outputs, dict):
            return

        pred = _pick_depth_pred(outputs)
        if pred is None or (not torch.is_tensor(pred)):
            return

        gt = _pick_gt_depth(outputs, pred)
        if gt is None or (not torch.is_tensor(gt)):
            return

        pred_bhw = _as_1hw(pred).float()
        gt_bhw = _as_1hw(gt).float()

        rccb = _infer_rccb(outputs, default=self.rccb_default)
        pred_bhw = crop_eval_torch(pred_bhw, rccb=rccb)
        gt_bhw = crop_eval_torch(gt_bhw, rccb=rccb)

        # old calc_metrics valid mask: gt>0 and pred>0
        # keep finite checks (should be no-op for normal data)
        valid = torch.isfinite(gt_bhw) & torch.isfinite(pred_bhw)
        valid = valid & (gt_bhw > 0.0) & (pred_bhw > 0.0)

        if not torch.any(valid):
            return

        g = gt_bhw[valid].double()
        p = pred_bhw[valid].double()

        # (2) CLIP AFTER MASKING, to both pred and gt (old behavior)
        if self.clip_min is not None and self.clip_max is not None:
            p = torch.clamp(p, min=self.clip_min, max=self.clip_max)
            g = torch.clamp(g, min=self.clip_min, max=self.clip_max)

        # --- per-image metric computation (old behavior) ---
        diff = p - g
        abs_diff = diff.abs()
        sq_diff = diff * diff

        mae_i = abs_diff.mean()
        rmse_i = torch.sqrt(sq_diff.mean())

        log_p = p.log()
        log_g = g.log()
        d = log_p - log_g
        rmse_log_i = torch.sqrt((d * d).mean())

        max_ratio = torch.maximum(p / g, g / p)
        t1_i = (max_ratio < (1.25 ** 1)).double().mean() * 100.0
        t2_i = (max_ratio < (1.25 ** 2)).double().mean() * 100.0
        t3_i = (max_ratio < (1.25 ** 3)).double().mean() * 100.0

        # silog = 100 * sqrt(mean(d^2) - mean(d)^2) (old behavior)
        mean_d = d.mean()
        mean_d2 = (d * d).mean()
        var = torch.clamp(mean_d2 - mean_d * mean_d, min=0.0)
        silog_i = 100.0 * torch.sqrt(var)

        # (3) ARD (old convention): mean(|p-g|/g), NOT multiplied by 100
        ard_i = (abs_diff / g).mean()

        # keep existing extra metrics, but computed per-image then averaged per-image
        sq_rel_i = (sq_diff / g).mean()
        inv_err = (1.0 / p) - (1.0 / g)
        irmse_i = torch.sqrt((inv_err * inv_err).mean())

        # accumulate per-image
        self.n += 1
        self.sum_mae += mae_i
        self.sum_rmse += rmse_i
        self.sum_rmse_log += rmse_log_i
        self.sum_t1 += t1_i
        self.sum_t2 += t2_i
        self.sum_t3 += t3_i
        self.sum_silog += silog_i
        self.sum_ard += ard_i
        self.sum_sq_rel += sq_rel_i
        self.sum_irmse += irmse_i

    @staticmethod
    def _compute_from_accums(
        n: torch.Tensor,
        sum_mae: torch.Tensor,
        sum_rmse: torch.Tensor,
        sum_rmse_log: torch.Tensor,
        sum_t1: torch.Tensor,
        sum_t2: torch.Tensor,
        sum_t3: torch.Tensor,
        sum_silog: torch.Tensor,
        sum_ard: torch.Tensor,
        sum_sq_rel: torch.Tensor,
        sum_irmse: torch.Tensor,
    ) -> Dict[str, float]:
        n_int = int(n.item())
        if n_int <= 0:
            return {
                "mae": float("nan"),
                "rmse": float("nan"),
                "rmse_log": float("nan"),
                "threshold1": float("nan"),
                "threshold2": float("nan"),
                "threshold3": float("nan"),
                "silog": float("nan"),
                "ard": float("nan"),
                "sq_rel": float("nan"),
                "irmse": float("nan"),
                "n": 0,
            }

        n_f = float(n_int)
        return {
            "mae": float((sum_mae / n_f).item()),
            "rmse": float((sum_rmse / n_f).item()),
            "rmse_log": float((sum_rmse_log / n_f).item()),
            "threshold1": float((sum_t1 / n_f).item()),
            "threshold2": float((sum_t2 / n_f).item()),
            "threshold3": float((sum_t3 / n_f).item()),
            "silog": float((sum_silog / n_f).item()),
            "ard": float((sum_ard / n_f).item()),        # NOT *100
            "sq_rel": float((sum_sq_rel / n_f).item()),
            "irmse": float((sum_irmse / n_f).item()),
            "n": n_int,
        }

    def snapshot(self, reduce_ddp: bool) -> Dict[str, float]:
        """
        Returns metrics up to now.
        If reduce_ddp=True, does an all-reduce on CLONES (safe to call repeatedly).
        IMPORTANT: In DDP, all ranks must call this at the same time.
        """
        n = self.n.clone()

        sum_mae = self.sum_mae.clone()
        sum_rmse = self.sum_rmse.clone()
        sum_rmse_log = self.sum_rmse_log.clone()

        sum_t1 = self.sum_t1.clone()
        sum_t2 = self.sum_t2.clone()
        sum_t3 = self.sum_t3.clone()

        sum_silog = self.sum_silog.clone()

        sum_ard = self.sum_ard.clone()
        sum_sq_rel = self.sum_sq_rel.clone()
        sum_irmse = self.sum_irmse.clone()

        if reduce_ddp and ddp_is_initialized():
            dist.all_reduce(n, op=dist.ReduceOp.SUM)

            dist.all_reduce(sum_mae, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_rmse, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_rmse_log, op=dist.ReduceOp.SUM)

            dist.all_reduce(sum_t1, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_t2, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_t3, op=dist.ReduceOp.SUM)

            dist.all_reduce(sum_silog, op=dist.ReduceOp.SUM)

            dist.all_reduce(sum_ard, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_sq_rel, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_irmse, op=dist.ReduceOp.SUM)

        return self._compute_from_accums(
            n=n,
            sum_mae=sum_mae,
            sum_rmse=sum_rmse,
            sum_rmse_log=sum_rmse_log,
            sum_t1=sum_t1,
            sum_t2=sum_t2,
            sum_t3=sum_t3,
            sum_silog=sum_silog,
            sum_ard=sum_ard,
            sum_sq_rel=sum_sq_rel,
            sum_irmse=sum_irmse,
        )

    @staticmethod
    def format(metrics: Dict[str, float]) -> str:
        order = [
            "n",
            "mae",
            "rmse",
            "rmse_log",
            "threshold1",
            "threshold2",
            "threshold3",
            "silog",
            "ard",
            "sq_rel",
            "irmse",
        ]
        lines = []
        for k in order:
            v = metrics.get(k, float("nan"))
            if k == "n":
                lines.append(f"{k}: {int(v)}")
            elif "threshold" in k:
                lines.append(f"{k}: {v:.2f}%")
            elif k in ("ard", "silog"):
                lines.append(f"{k}: {v:.4f}")
            else:
                lines.append(f"{k}: {v:.6f}")
        return "\n".join(lines)



# ----------------------------
# Evaluator
# ----------------------------
class Evaluator:
    def __init__(self, cfg: DictConfig, model: nn.Module, dataloader, device: torch.device, *, rccb_default: bool):
        self.cfg = cfg
        self.device = device

        self.model = model.to(device)

        self.use_ddp = ddp_is_initialized()
        if self.use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        self.dataloader = dataloader
        self.metrics = DepthLidarMetrics(device=device, rccb_default=rccb_default)

        # train.* only
        self.use_amp = bool(cfg_get(cfg, "train.amp", False)) and device.type == "cuda"
        self.log_interval = int(cfg_get(cfg, "train.log_interval", 50))

        self.load_checkpoint_from_train_cfg()

    def parse_batch(self, batch: Any) -> Tuple[Any, Dict[str, Any]]:
        metas = {}
        if isinstance(batch, dict):
            metas = batch.get("data_samples", {})
            if "inputs" in batch:
                inputs = batch["inputs"]
            else:
                inputs = {k: v for k, v in batch.items() if k != "metas"}
            return inputs, metas

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]

        return batch, metas

    def load_checkpoint_from_train_cfg(self):
        """
        NO eval.* config.
        Uses cfg.train.resume_path, otherwise outputs/checkpoints/latest.pt
        """
        strict = bool(cfg_get(self.cfg, "train.resume_strict", True))

        path = str(cfg_get(self.cfg, "train.resume_path", "")).strip()
        if not path:
            out_dir = cfg_get(self.cfg, "train.output_dir", "outputs")
            path = str(Path(out_dir) / "checkpoints" / "latest.pt")

        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Set cfg.train.resume_path, or ensure latest.pt exists under "
                f"{Path(cfg_get(self.cfg, 'train.output_dir', 'outputs')) / 'checkpoints'}"
            )

        if is_main_process():
            print(f"[eval] Loading checkpoint: {ckpt_path} (strict={strict})", flush=True)

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model

        model_to_load.load_state_dict(ckpt["model"], strict=strict)

    @torch.no_grad()
    def eval_step(self, batch: Any) -> Dict[str, Any]:
        inputs, metas = self.parse_batch(batch)
        inputs = move_to_device(inputs, self.device)
        metas = move_to_device(metas, self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs, metas)
        return outputs

    def evaluate(self) -> Dict[str, float]:
        # self.model.eval()

        sampler = getattr(self.dataloader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(0)

        reduce_snapshots = ddp_is_initialized()
        t0 = time.time()

        pbar = tqdm(
            total=len(self.dataloader),
            disable=not is_main_process(),
            dynamic_ncols=True,
            desc="Evaluating",
        )

        try:
            for i, batch in enumerate(self.dataloader):
                outputs = self.eval_step(batch)

                self.metrics.update(outputs)

                if is_main_process():
                    pbar.update(1)

                # every 100 steps: print metrics so far (global if DDP)
                if ((i + 1) % 100) == 0:
                    snap = self.metrics.snapshot(reduce_ddp=reduce_snapshots)  # MUST be called on all ranks in DDP
                    if is_main_process():
                        dt = time.time() - t0
                        print(f"\n[eval @ step {i+1}] elapsed={dt:.1f}s")
                        print(self.metrics.format(snap), flush=True)

                        # nice quick glance in tqdm line too
                        pbar.set_postfix(
                            mae=f"{snap['mae']:.3f}" if snap["mae"] == snap["mae"] else "nan",
                            rmse=f"{snap['rmse']:.3f}" if snap["rmse"] == snap["rmse"] else "nan",
                            d1=f"{snap['threshold1']:.1f}%",
                        )

                # optional lightweight progress log interval (kept from before)
                if is_main_process() and (i % self.log_interval == 0):
                    dt_ms = (time.time() - t0) * 1000.0
                    # keep it short so tqdm doesn't get too spammy
                    # (metrics print happens every 100)
                    # print(f"[eval] iter={i:06d} dt_total={dt_ms:.1f}ms", flush=True)
        finally:
            if is_main_process():
                pbar.close()

        barrier()

        final = self.metrics.snapshot(reduce_ddp=reduce_snapshots)
        if is_main_process():
            print("\n===== Final Evaluation metrics (LiDAR + crop) =====")
            print(self.metrics.format(final))
            print("==================================================\n", flush=True)

        barrier()
        return final


# ----------------------------
# Main
# ----------------------------
@hydra.main(config_path="config", config_name="gatedrccbstereo", version_base=None)
def main(cfg: DictConfig):
    device, rank, world_size, local_rank = setup_distributed()

    base_seed = int(cfg_get(cfg, "seed", 0))
    seed_everything(base_seed + rank)

    cfg.dataset.test_mode = True
    cfg.dataloader.batch_size = 1
    dataset = instantiate(cfg.dataset)
    dataloader = build_eval_dataloader(cfg, dataset)

    model_args = cfg_get(cfg, "model.model_args", None)
    if model_args is None:
        model_args = cfg_get(cfg, "model_args", None)
    if model_args is None:
        raise ValueError("Missing cfg.model.model_args (or cfg.model_args).")

    model = GatedRCCBStereo(model_args=model_args)
    rccb_default = False
    if is_main_process():
        print("Using Model: GatedRCCBStereo")

    evaluator = Evaluator(cfg=cfg, model=model, dataloader=dataloader, device=device, rccb_default=rccb_default)
    evaluator.evaluate()

    if ddp_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
