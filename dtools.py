import numpy as np
import cv2

def show(x):
    """
    Convert a torch Tensor / numpy array / array-like to a numpy array suitable for display.

    Rules:
    - Always take batch index 0 if a batch dim exists (expects BCHW when 4D).
    - (B, 3, H, W)  -> (H, W, 3)
    - (1, 1, H, W)  -> (H, W)
    - (B, 5, H, W)  -> mean over the '5' (channel) -> (H, W)
    - More generally: for (B, C, H, W), C==1 -> (H,W), C==3 -> (H,W,3), else -> mean over C -> (H,W)
    """
    # --- to numpy ---
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.is_cuda:
                x = x.cpu()
            # bfloat16/float16 can be annoying downstream; keep as-is unless we have to reduce
            x = x.numpy()
    except Exception:
        pass

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    # --- peel leading dims down to <= 4 by taking index 0 (covers e.g., B,T,C,H,W -> T,C,H,W -> ...)
    while x.ndim > 4:
        x = x[0]

    # --- main handling ---
    if x.ndim == 4:
        # assume (B, C, H, W)
        x = x[0]
        c = x.shape[0]
        if c == 3:
            return np.transpose(x, (1, 2, 0))
        if c == 1:
            return x[0]
        return x.astype(np.float32, copy=False).mean(axis=0)

    if x.ndim == 3:
        # either (C,H,W) or (H,W,C); handle both
        if x.shape[0] in (1, 3) and x.shape[1] > 4 and x.shape[2] > 4:  # CHW-ish
            c = x.shape[0]
            if c == 3:
                return np.transpose(x, (1, 2, 0))
            if c == 1:
                return x[0]
            return x.astype(np.float32, copy=False).mean(axis=0)
        # otherwise assume HWC-ish
        if x.shape[2] == 1:
            return x[..., 0]
        return x

    # 2D or 1D (or scalar) -> just numpy
    return x

def _resolve_colormap(cmap):
    """Accept cv2 colormap int, or strings like 'jet', 'turbo', 'magma', etc."""
    if isinstance(cmap, int):
        return cmap
    if cmap is None:
        return cv2.COLORMAP_JET
    if isinstance(cmap, str):
        key = cmap.strip().upper()
        if not key.startswith("COLORMAP_"):
            key = "COLORMAP_" + key
        if hasattr(cv2, key):
            return getattr(cv2, key)
        raise ValueError(f"Unknown colormap string '{cmap}'. Try e.g. 'jet', 'turbo', 'viridis'.")
    raise TypeError("cmap must be an int (cv2 colormap) or a string like 'jet'.")

def colorize_depth(
    depth,
    cmap=cv2.COLORMAP_JET,
    *,
    lidar=False,
    radius=3,
    invert=False,
    min_depth=0.0,
    max_depth=150.0,
    eps=1e-6,
    no_color=False,
    auto_rescale_01=True,
):
    """
    Colorize a depth map to uint8 HxWx3 (BGR) using OpenCV colormaps (JET by default),
    OR return a 2D depth map if no_color=True.

    Semantics (matching your described behavior):
    - invert=True does a LINEAR flip after clipping:
        clipped = clip(depth, min_depth, max_depth)
        vals = max_depth - clipped
      (zeros/invalid stay zero and are rendered black)
    - invert=False is normal depth visualization.
    """
    # Ensure numpy HxW
    if not (isinstance(depth, np.ndarray) and depth.ndim == 2):
        depth = show(depth)
    if isinstance(depth, np.ndarray) and depth.ndim == 3 and depth.shape[-1] == 3:
        depth = depth.mean(axis=-1)
    if not (isinstance(depth, np.ndarray) and depth.ndim == 2):
        raise ValueError(f"Expected depth to be HxW after show(); got shape {getattr(depth, 'shape', None)}")

    depth = depth.astype(np.float32, copy=False)

    dmin = float(min_depth)
    dmax = float(max_depth)
    if dmax <= dmin:
        raise ValueError(f"max_depth ({max_depth}) must be > min_depth ({min_depth})")

    # Auto-rescale if it looks like [0,1] depth (robust to sparse maps by looking at positives)
    if auto_rescale_01:
        pos_vals = depth[depth > 0]
        if pos_vals.size > 0:
            mx = float(pos_vals.max())
            if mx <= 1.0 + 1e-3:
                depth = depth * float(dmax)

    # Sparse detection
    zeros = depth.size - np.count_nonzero(depth)
    is_sparse = lidar or (zeros > (depth.size / 2))

    # If sparse, enlarge points via circles (background stays 0)
    if is_sparse and radius and radius > 0:
        dense = np.zeros_like(depth, dtype=np.float32)
        ys, xs = np.nonzero(depth > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cv2.circle(dense, (int(x), int(y)), int(radius), float(depth[y, x]), thickness=-1)
        depth = dense

    # Mask invalid/zero depth (must remain black)
    zero_mask = depth <= 0
    pos = ~zero_mask

    # Clip in depth domain
    depth_clipped = depth.copy()
    if np.any(pos):
        depth_clipped[pos] = np.clip(depth_clipped[pos], dmin, dmax)

    # Invert (your way): linear flip after clipping
    if invert:
        vals = np.zeros_like(depth_clipped, dtype=np.float32)
        if np.any(pos):
            # optional: use dmax here to match your "max_depth - depth"
            vals[pos] = dmax - depth_clipped[pos]
        # Range becomes [0, dmax-dmin] for valid pixels
        vmin, vmax = 0.0, (dmax - dmin)
    else:
        vals = depth_clipped
        vmin, vmax = dmin, dmax

    if no_color:
        out = vals
        out[zero_mask] = 0
        return out

    # Normalize to uint8 [0,255]
    if vmax <= vmin + eps:
        norm = np.zeros_like(vals, dtype=np.uint8)
    else:
        norm_f = (vals - vmin) / (vmax - vmin)
        norm = np.clip(norm_f * 255.0, 0, 255).astype(np.uint8)

    cm = _resolve_colormap(cmap)
    colored = cv2.applyColorMap(norm, cm)  # HxWx3, BGR uint8
    colored[zero_mask] = 0
    return colored

def _norm01(x: np.ndarray, eps=1e-6) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def _to_tb_chw(img: np.ndarray) -> np.ndarray:
    """
    Accepts HxW or HxWx3, returns float32 CHW in [0,1] for TensorBoard.
    """
    if img.ndim == 2:
        img = _norm01(img)
        return img[None, ...]  # 1,H,W
    if img.ndim == 3 and img.shape[2] == 3:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        return np.transpose(img, (2, 0, 1))  # 3,H,W
    raise ValueError(f"Unsupported image shape for TB: {img.shape}")

def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        return img_bgr[..., ::-1]
    return img_bgr




