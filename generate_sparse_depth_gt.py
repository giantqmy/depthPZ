"""Project raw fog LiDAR points into a sparse left-camera depth ground truth.

No interpolation, dilation, densification, or hole filling is performed. If
multiple LiDAR points land on one pixel, the nearest positive camera-Z value is
kept.

The canonical output is float32 NumPy data in metres. Zero means invalid. A
uint16 PNG encoded as metres * 256 can also be written for compatibility, but
the float32 file should be used for final metric calculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_POINT_DIR = Path(r"I:\KEDADATA\data\fog\pcluse")
DEFAULT_IMAGE_DIR = Path(r"I:\KEDADATA\data\fog\RGBS0")
DEFAULT_OUTPUT_DIR = Path(r"I:\KEDADATA\data\fog\depth_gt_sparse")

IMAGE_SIZE = (640, 512)  # width, height

# Raw left RGB camera intrinsics from 深度图v2.py.
K_LEFT = np.array(
    [
        [1265.5616099448073, 0.0, 492.5894472471048],
        [0.0, 1265.0089455071945, 257.24307296897143],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DIST_LEFT = np.array(
    [[-0.02096660215918474, 0.2409083928738356,
      -0.005298800804070935, 0.007648334809400866, 0.0]],
    dtype=np.float64,
)

# LiDAR -> raw left RGB camera: p_camera = R @ p_lidar + t.
T_LIDAR_TO_LEFT = np.array(
    [
        [-0.0147331, -0.999842, -0.00999829, -0.0153519],
        [0.00641986, 0.00990464, -0.99993, 0.057766],
        [0.999871, -0.0147963, 0.00627293, 0.12535],
    ],
    dtype=np.float64,
)

# Right-camera parameters are needed only to reproduce the exact stereo-left
# rectification used by training and run_gatedrccbstereo_stereo.py.
K_RIGHT = np.array(
    [
        [1277.6494552887766, 0.0, 474.185263116495],
        [0.0, 1278.5279034618504, 260.09879728682546],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DIST_RIGHT = np.array(
    [[-0.06681173268876799, 0.39362342592776267,
      -0.007450119715294455, 0.0034144197720179475, 0.0]],
    dtype=np.float64,
)

R_LEFT_TO_RIGHT = np.array(
    [
        [0.9998762017154619, 0.003542133347215963, 0.015330836063682721],
        [-0.003479358861003513, 0.9999854622282568, -0.004119392437720074],
        [-0.01534520462481131, 0.004065540983698188, 0.9998739900965185],
    ],
    dtype=np.float64,
)

T_LEFT_TO_RIGHT = np.array(
    [-0.055078917279073276, 0.00011619425753116915, 0.0017826427597072247],
    dtype=np.float64,
)


def rectified_left_calibration(alpha: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    r_left, _, p_left, _, _, _, _ = cv2.stereoRectify(
        K_LEFT,
        DIST_LEFT,
        K_RIGHT,
        DIST_RIGHT,
        IMAGE_SIZE,
        R_LEFT_TO_RIGHT,
        T_LEFT_TO_RIGHT,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )
    return r_left, p_left


def read_points(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(f"Point file is not Nx4 float32: {path} ({values.size} values)")
    points = values.reshape(-1, 4)[:, :3]
    return points[np.isfinite(points).all(axis=1)]


def project_sparse_depth(
    points_lidar: np.ndarray,
    projection: str,
    r_left: np.ndarray,
    p_left: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    rotation = T_LIDAR_TO_LEFT[:, :3]
    translation = T_LIDAR_TO_LEFT[:, 3]
    points_camera = points_lidar.astype(np.float64) @ rotation.T + translation

    if projection == "rectified":
        points_camera = points_camera @ r_left.T
        camera_matrix = p_left[:, :3]
    else:
        camera_matrix = K_LEFT

    in_front = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 0.0)
    points_camera = points_camera[in_front]
    projected_count = int(points_camera.shape[0])

    projected = points_camera @ camera_matrix.T
    u_float = projected[:, 0] / projected[:, 2]
    v_float = projected[:, 1] / projected[:, 2]

    # Match 深度图v2.py/PyTorch long(): truncate toward zero.
    u = np.trunc(u_float).astype(np.int64)
    v = np.trunc(v_float).astype(np.int64)
    width, height = IMAGE_SIZE
    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[in_image], v[in_image]
    z = points_camera[in_image, 2].astype(np.float32)

    flat = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(flat, v * width + u, z)
    flat[~np.isfinite(flat)] = 0.0
    depth = flat.reshape(height, width)
    return depth, projected_count, int(z.size)


def save_depth(output_dir: Path, key: str, depth_m: np.ndarray, output_format: str) -> None:
    if output_format in ("npy", "both"):
        np.save(output_dir / f"{key}_depth_gt_sparse_m.npy", depth_m, allow_pickle=False)

    if output_format in ("png", "both"):
        # Compatibility copy only. Float32 NPY remains authoritative because PNG
        # quantizes to 1/256 m and saturates beyond 255.996 m.
        depth_u16 = np.clip(depth_m * 256.0, 0.0, 65535.0).astype(np.uint16)
        ok = cv2.imwrite(str(output_dir / f"{key}_depth_gt_sparse_u16.png"), depth_u16)
        if not ok:
            raise RuntimeError(f"Failed to write uint16 depth PNG for {key}")


def find_samples(point_dir: Path, image_dir: Path) -> list[tuple[str, Path, Path]]:
    images = {
        path.name.removesuffix("_S0_rgb.png"): path
        for path in image_dir.glob("*_S0_rgb.png")
    }
    points = {path.stem: path for path in point_dir.glob("*.bin")}
    missing_points = sorted(set(images) - set(points))
    missing_images = sorted(set(points) - set(images))
    if missing_points or missing_images:
        raise RuntimeError(
            "RGB/point-cloud prefixes do not match: "
            f"missing point clouds={missing_points[:5]} ({len(missing_points)}), "
            f"missing RGB images={missing_images[:5]} ({len(missing_images)})"
        )
    return [(key, images[key], points[key]) for key in sorted(images)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-dir", type=Path, default=DEFAULT_POINT_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--projection",
        choices=("rectified", "raw"),
        default="rectified",
        help="rectified aligns with model disparity output; raw aligns with the original RGB image",
    )
    parser.add_argument(
        "--format",
        choices=("npy", "png", "both"),
        default="both",
        help="NPY float32 metres is authoritative; PNG uses uint16 metres*256",
    )
    parser.add_argument("--limit", type=int, default=None, help="process only the first N samples")
    parser.add_argument("--check-only", action="store_true", help="validate and project one sample without writing")
    args = parser.parse_args()

    samples = find_samples(args.point_dir, args.image_dir)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        samples = samples[: args.limit]

    r_left, p_left = rectified_left_calibration()
    if not args.check_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    total_valid_pixels = 0
    global_min = np.inf
    global_max = 0.0
    for index, (key, image_path, point_path) in enumerate(samples, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None or (image.shape[1], image.shape[0]) != IMAGE_SIZE:
            raise ValueError(f"Expected a readable 640x512 RGB image: {image_path}")

        points = read_points(point_path)
        depth_m, in_front_count, in_image_count = project_sparse_depth(
            points, args.projection, r_left, p_left
        )
        valid = depth_m > 0.0
        valid_count = int(valid.sum())
        if valid_count:
            global_min = min(global_min, float(depth_m[valid].min()))
            global_max = max(global_max, float(depth_m[valid].max()))
        total_valid_pixels += valid_count

        if not args.check_only:
            save_depth(args.output_dir, key, depth_m, args.format)
        print(
            f"[{index}/{len(samples)}] {key}: points={len(points)}, front={in_front_count}, "
            f"inside={in_image_count}, valid_pixels={valid_count}",
            flush=True,
        )
        if args.check_only:
            break

    metadata = {
        "point_dir": str(args.point_dir.resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "projection": args.projection,
        "image_size_wh": list(IMAGE_SIZE),
        "canonical_format": "float32 NPY, metres, invalid=0",
        "png_compatibility_format": "uint16, stored_value=metres*256, invalid=0",
        "collision_rule": "nearest positive camera-Z per pixel",
        "densification": False,
        "samples": len(samples),
        "total_valid_pixels": total_valid_pixels,
        "valid_depth_min_m": None if not np.isfinite(global_min) else global_min,
        "valid_depth_max_m": None if global_max <= 0 else global_max,
        "K_left_raw": K_LEFT.tolist(),
        "T_lidar_to_left": T_LIDAR_TO_LEFT.tolist(),
        "R1_rectification": r_left.tolist(),
        "P1_rectified": p_left.tolist(),
    }
    if args.check_only:
        print(json.dumps(metadata, indent=2))
    else:
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Saved sparse ground truth to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
