import os
import cv2
import argparse

from .lib.ar0820_raw2rgb import load_ar0820_image


def generate_rgb_image(raw_path, camtype="ar0820"):
    if camtype != "ar0820":
        raise ValueError('Only camtype "ar0820" is supported.')
    return load_ar0820_image(raw_path)


def main():
    parser = argparse.ArgumentParser(description="Raw to RGB (ar0820 only)")
    parser.add_argument("-f", "--fpath", required=True, help="Path to raw image")
    parser.add_argument("-s", "--result_dir", required=True, help="Directory to store results")
    parser.add_argument("--ext", default="jpg", help="Output image extension (default: jpg)")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    save_name = f"{os.path.splitext(os.path.basename(args.fpath))[0]}.{args.ext}"

    bgr = generate_rgb_image(args.fpath, camtype="ar0820")
    cv2.imwrite(os.path.join(args.result_dir, save_name), bgr)


if __name__ == "__main__":
    main()
