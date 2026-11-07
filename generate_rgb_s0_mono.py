"""Generate 8-bit monochrome S0 intensity images from decoded polarization views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(str(path))


def decoded_views_to_s0_u8(images: list[np.ndarray]) -> np.ndarray:
    if len(images) != 4:
        raise ValueError(f"Expected four polarization views, got {len(images)}")
    for image in images:
        if image.dtype != np.uint8 or image.shape != (512, 640, 3):
            raise ValueError(f"Expected decoded uint8 BGR 512x640x3, got {image.dtype} {image.shape}")

    gray_views = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) for image in images]
    # Physical S0 = 0.5 * (I0 + I45 + I90 + I135). Store S0/2 so the
    # decoded 8-bit intensity scale is retained without highlight clipping.
    return np.rint(np.mean(gray_views, axis=0)).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(r"I:\KEDADATA\data\fog\RGBuseI4"))
    parser.add_argument("--output", type=Path, default=Path(r"I:\KEDADATA\data\fog\rgbs0_mono"))
    args = parser.parse_args()

    angle_dirs = {angle: args.input / str(angle) for angle in (0, 45, 90, 135)}
    for directory in angle_dirs.values():
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    paths = sorted(angle_dirs[0].glob("*_0.png"))
    if not paths:
        raise FileNotFoundError(f"No decoded 0-degree PNG images found in {angle_dirs[0]}")
    args.output.mkdir(parents=True, exist_ok=True)

    saturated = 0
    pixels = 0
    output_min = 255
    output_max = 0
    for index, input_path in enumerate(paths, 1):
        key = input_path.name.removesuffix("_0.png")
        angle_paths = [angle_dirs[angle] / f"{key}_{angle}.png" for angle in (0, 45, 90, 135)]
        missing = [str(path) for path in angle_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing decoded polarization views: {missing}")
        s0 = decoded_views_to_s0_u8([read_image(path) for path in angle_paths])
        output_path = args.output / f"{key}_S0_rgb.png"
        write_image(output_path, s0)
        saturated += int(np.count_nonzero(s0 == 255))
        pixels += s0.size
        output_min = min(output_min, int(s0.min()))
        output_max = max(output_max, int(s0.max()))
        if index % 100 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)}", flush=True)

    metadata = {
        "input_directory": str(args.input.resolve()),
        "output_directory": str(args.output.resolve()),
        "image_count": len(paths),
        "input_format": "four Bayer-decoded uint8 BGR polarization views at 512x640",
        "polarization_layout": [[135, 0], [90, 45]],
        "physical_s0_formula": "0.5 * (I0 + I45 + I90 + I135)",
        "stored_intensity_formula": "S0 / 2 = (I0 + I45 + I90 + I135) / 4",
        "output_format": "uint8 single-channel 512x640 PNG",
        "output_conversion": "BGR-to-gray for each decoded angle, then rounded four-angle mean; no per-frame normalization",
        "output_range": [output_min, output_max],
        "saturated_pixel_fraction": saturated / pixels,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
