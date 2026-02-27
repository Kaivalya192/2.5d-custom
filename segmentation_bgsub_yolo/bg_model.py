import argparse
import glob
import os

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def collect_images(input_dir: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="Build averaged background image from empty-bin frames.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    files = collect_images(args.input_dir)
    if not files:
        raise RuntimeError(f"No images found in: {args.input_dir}")

    acc = None
    n = 0
    for p in files:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        if acc is None:
            acc = np.zeros_like(img, dtype=np.float32)
        acc += img.astype(np.float32)
        n += 1

    if n == 0:
        raise RuntimeError("No valid images for averaging.")

    mean_img = np.clip(acc / float(n), 0, 255).astype(np.uint8)
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        ensure_dir(out_dir)
    cv2.imwrite(args.output_path, mean_img)
    np.save(os.path.splitext(args.output_path)[0] + ".npy", mean_img.astype(np.float32))
    print(f"[OK] Saved averaged background from {n} images: {args.output_path}")


if __name__ == "__main__":
    main()
