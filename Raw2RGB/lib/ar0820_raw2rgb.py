import os
import cv2
import numpy as np
from imageio import imread


def ar0820_linearize(data: np.ndarray) -> np.ndarray:
    if data.max() > 2**12:
        data = data >> 4

    knee_in = [
        0, 0x200, 0x400, 0x800, 0x820, 0x860, 0x8E0, 0x9E0, 0xBE0, 0xC20, 0xCA0, 0xDA0, 0xFA0
    ]
    knee_out = [
        0, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20
    ]

    decompanded = np.zeros_like(data, dtype=np.float32)
    for i in range(len(knee_in) - 1):
        mask = (knee_in[i] <= data) & (data < knee_in[i + 1])
        slope = (knee_out[i + 1] - knee_out[i]) / (knee_in[i + 1] - knee_in[i])
        decompanded[mask] = knee_out[i] + (data[mask] - knee_in[i]) * slope

    return decompanded


def ar0820_linearize_loader(
    path: str,
    whitelevel: int = (1 << 20) - 1,
    wb: np.ndarray = np.array([1.0, 1.0, 1.0]),
) -> np.ndarray:
    if os.path.splitext(path)[1].lower() == ".raw":
        img = np.fromfile(path, np.uint16)
        size = img.shape[0]
        if size % 3840 == 0:
            img = img.reshape(-1, 3840)
        elif size % 3848 == 0:
            img = img.reshape(-1, 3848)
        else:
            raise ValueError(f"Unknown raw shape from size={size} (expected width 3840 or 3848)")
    else:
        img = imread(path)

    dH2 = (img.shape[0] - 2160) // 2
    dH2 = dH2 + (dH2 & 1)
    img = img[dH2:-dH2]

    img = ar0820_linearize(img)

    wb = np.array(wb, dtype=np.float32)
    wb = wb / wb[1]

    img[0::2, 1::2] = img[0::2, 1::2] / wb[0]
    img[1::2, 0::2] = img[1::2, 0::2] / wb[2]

    img = np.clip(img / float(whitelevel), 0.0, 1.0)
    return img.astype(np.float32)


def norm(im: np.ndarray, d: int) -> np.ndarray:
    im = im.astype(np.float32)
    mn, mx = float(im.min()), float(im.max())
    if mx <= mn:
        return np.zeros_like(im, dtype=np.uint8 if d == 8 else np.uint16)
    out = (im - mn) / (mx - mn) * ((2**d) - 1)
    return out.astype(np.uint8 if d == 8 else np.uint16)


def contrast_stretch(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img[:, :, 1] = clahe.apply(img[:, :, 1])
    img[:, :, 2] = clahe.apply(img[:, :, 2])
    return img


def load_ar0820_image(filename: str) -> np.ndarray:
    img = ar0820_linearize_loader(filename)
    img = norm(img, 16)
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB)
    img = np.log(img.astype(np.float32) + 1.0)
    img = norm(img, 8)
    img = contrast_stretch(img)
    return img
