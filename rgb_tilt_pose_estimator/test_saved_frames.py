import argparse
import glob
import os

import cv2
import numpy as np

from estimator import EstimatorConfig, draw_pose_overlay, estimate_tilt_pose_from_mask


def synthetic_mask(w: int, h: int, angle_deg: float, ax1: int, ax2: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.ellipse(m, (cx, cy), (ax1, ax2), angle_deg, 0, 360, 255, -1)
    return m


def run_synthetic_tests(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cfg = EstimatorConfig(conf_thr=0.0, min_area=100)
    cases = [
        (0.0, 90, 90),
        (30.0, 120, 70),
        (75.0, 130, 60),
    ]
    for i, (ang, a, b) in enumerate(cases, start=1):
        m = synthetic_mask(480, 360, ang, a, b)
        est = estimate_tilt_pose_from_mask(m, confidence=0.99, cfg=cfg, K=None)
        rgb = np.zeros((360, 480, 3), dtype=np.uint8)
        rgb[:, :, 1] = m
        ov = draw_pose_overlay(rgb, est)
        cv2.imwrite(os.path.join(out_dir, f"synthetic_{i:02d}.png"), ov)
        print(f"[Synthetic {i}] valid={est.get('valid')} tilt={est.get('tilt_angle_deg', None)} dir={est.get('tilt_dir_deg', None)}")


def run_real_masks(mask_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cfg = EstimatorConfig(conf_thr=0.0, min_area=80)
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        files.extend(glob.glob(os.path.join(mask_dir, ext)))
    files.sort()
    for i, fp in enumerate(files, start=1):
        m = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 0).astype(np.uint8) * 255
        est = estimate_tilt_pose_from_mask(m, confidence=0.99, cfg=cfg, K=None)
        rgb = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 1] = m
        ov = draw_pose_overlay(rgb, est)
        cv2.imwrite(os.path.join(out_dir, f"realmask_{i:04d}.png"), ov)
    print(f"[Real masks] Processed {len(files)} files.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tests for RGB-only tilt estimator.")
    ap.add_argument("--out_dir", type=str, default="rgb_tilt_pose_estimator/test_outputs")
    ap.add_argument("--mask_dir", type=str, default="", help="optional binary-mask folder")
    args = ap.parse_args()

    run_synthetic_tests(args.out_dir)
    if args.mask_dir:
        run_real_masks(args.mask_dir, args.out_dir)


if __name__ == "__main__":
    main()
