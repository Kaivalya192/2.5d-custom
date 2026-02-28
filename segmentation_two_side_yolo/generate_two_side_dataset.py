import argparse
import glob
import os
import random
import sys
from typing import List, Tuple

import cv2
import numpy as np

# Allow running as a script from repo root on Windows.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from segmentation_bgsub_yolo.generate_dataset import (
    contour_to_yolo_polygon,
    ensure_dir,
    largest_component,
    save_overlay,
    segment_foreground,
    shift_contour,
)


def collect_images(input_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
    files.sort()
    return files


def pick_split(val_ratio: float, test_ratio: float) -> str:
    v = float(np.clip(val_ratio, 0.0, 0.9))
    t = float(np.clip(test_ratio, 0.0, 0.9))
    if v + t > 0.95:
        s = v + t
        v = v / s * 0.95
        t = t / s * 0.95
    r = random.random()
    if r < t:
        return "test"
    if r < (t + v):
        return "val"
    return "train"


def write_data_yaml(path: str, names: List[str]) -> None:
    txt = (
        f"path: {os.path.abspath(os.path.dirname(path))}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
    )
    for i, n in enumerate(names):
        txt += f"  {i}: {n}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def load_roi_config(path: str, img_w: int, img_h: int) -> dict:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open ROI yaml: {path}")
    mode_node = fs.getNode("mode")
    mode = mode_node.string() if not mode_node.empty() else "rect"
    x = int(fs.getNode("x").real())
    y = int(fs.getNode("y").real())
    w = int(fs.getNode("w").real())
    h = int(fs.getNode("h").real())
    pts_node = fs.getNode("points")
    pts = None
    if not pts_node.empty():
        m = pts_node.mat()
        if m is not None:
            pts = np.array(m, dtype=np.int32).reshape(-1, 2)
    fs.release()

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("ROI is invalid/outside image.")

    if pts is None or pts.shape[0] < 3:
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
        mode = "rect"
    else:
        pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
        x0 = int(np.min(pts[:, 0]))
        y0 = int(np.min(pts[:, 1]))
        x1 = int(np.max(pts[:, 0])) + 1
        y1 = int(np.max(pts[:, 1])) + 1

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return {
        "mode": mode,
        "bbox": (x0, y0, x1 - x0, y1 - y0),
        "points": pts,
        "mask": mask,
    }


def process_one(
    img_path: str,
    class_id: int,
    class_name: str,
    bg: np.ndarray,
    out_root: str,
    roi_cfg: dict | None,
    args,
    counters: dict,
) -> None:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    if img.shape[:2] != bg.shape[:2]:
        img = cv2.resize(img, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask, poly = segment_foreground(
        img=img,
        bg=bg,
        diff_thresh=args.diff_thresh,
        channel_thresh=args.channel_thresh,
        shadow_tol=args.shadow_tol,
        open_ksize=args.open_ksize,
        close_ksize=args.close_ksize,
        median_ksize=args.median_ksize,
        min_area=args.min_area,
        polygon_epsilon_ratio=args.polygon_epsilon_ratio,
        strict_mode=bool(args.strict_mode),
        strict_white_thresh=int(args.strict_white_thresh),
        strict_luma_delta=int(args.strict_luma_delta),
        strict_diff_thresh=int(args.strict_diff_thresh),
        strict_min_area=int(args.strict_min_area),
    )

    reason = ""
    if np.count_nonzero(mask) == 0:
        reason = "no_mask_after_seg"

    if roi_cfg is not None:
        roi_mask = roi_cfg["mask"]
        mask = cv2.bitwise_and(mask, roi_mask)
        mask = largest_component(mask, min_area=args.min_area)
        if np.count_nonzero(mask) > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                eps = float(args.polygon_epsilon_ratio) * float(cv2.arcLength(c, True))
                poly = cv2.approxPolyDP(c, eps, True)
            else:
                poly = np.empty((0, 1, 2), dtype=np.int32)
        else:
            poly = np.empty((0, 1, 2), dtype=np.int32)
            reason = "no_mask_after_roi"
            # ROI-local fallback with looser thresholds for low-contrast frames.
            bx, by, bw, bh = roi_cfg["bbox"]
            x0 = max(0, bx)
            y0 = max(0, by)
            x1 = min(img.shape[1], bx + bw)
            y1 = min(img.shape[0], by + bh)
            if x1 > x0 and y1 > y0:
                img_roi = img[y0:y1, x0:x1]
                bg_roi = bg[y0:y1, x0:x1]
                loose_diff = max(8, int(args.diff_thresh) - 8)
                loose_ch = max(6, int(args.channel_thresh) - 6)
                loose_min_area = max(120, int(args.min_area) // 3)
                m2, p2 = segment_foreground(
                    img=img_roi,
                    bg=bg_roi,
                    diff_thresh=loose_diff,
                    channel_thresh=loose_ch,
                    shadow_tol=max(8, int(args.shadow_tol)),
                    open_ksize=args.open_ksize,
                    close_ksize=args.close_ksize,
                    median_ksize=args.median_ksize,
                    min_area=loose_min_area,
                    polygon_epsilon_ratio=args.polygon_epsilon_ratio,
                    strict_mode=True,
                    strict_white_thresh=int(args.strict_white_thresh),
                    strict_luma_delta=max(4, int(args.strict_luma_delta) - 3),
                    strict_diff_thresh=max(4, int(args.strict_diff_thresh) - 3),
                    strict_min_area=max(60, int(args.strict_min_area)),
                )
                if np.count_nonzero(m2) > 0 and p2 is not None and p2.size > 0:
                    full = np.zeros_like(mask)
                    full[y0:y1, x0:x1] = m2
                    full = cv2.bitwise_and(full, roi_mask)
                    full = largest_component(full, min_area=loose_min_area)
                    if np.count_nonzero(full) > 0:
                        contours, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            eps = float(args.polygon_epsilon_ratio) * float(cv2.arcLength(c, True))
                            poly = cv2.approxPolyDP(c, eps, True)
                            if poly.shape[0] < 3:
                                poly = c
                            if poly is not None and poly.size > 0 and poly.shape[0] >= 3:
                                mask = full
                                reason = ""

    if poly is None or poly.size == 0 or np.count_nonzero(mask) == 0:
        if not reason:
            reason = "no_poly"
        counters["empty"] += 1
        counters[reason] = counters.get(reason, 0) + 1
        if bool(args.save_reject_overlay):
            rej_dir = os.path.join(out_root, "reject_overlays")
            ensure_dir(rej_dir)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            base_name = f"{class_name}_{stem}"
            rej = img.copy()
            if np.count_nonzero(mask) > 0:
                cmask = np.zeros_like(img)
                cmask[:, :, 2] = mask
                rej = cv2.addWeighted(rej, 1.0, cmask, 0.35, 0)
            cv2.putText(rej, f"REJECT: {reason}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(rej_dir, f"{base_name}.png"), rej)
        if args.skip_empty:
            return

    split = pick_split(args.val_ratio, args.test_ratio)
    img_out = os.path.join(out_root, "images", split)
    lbl_out = os.path.join(out_root, "labels", split)
    ensure_dir(img_out)
    ensure_dir(lbl_out)

    stem = os.path.splitext(os.path.basename(img_path))[0]
    base_name = f"{class_name}_{stem}"
    out_img_path = os.path.join(img_out, f"{base_name}.png")
    out_lbl_path = os.path.join(lbl_out, f"{base_name}.txt")

    out_img = img
    out_poly = poly
    if bool(args.crop_to_roi) and roi_cfg is not None:
        x, y, w, h = roi_cfg["bbox"]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(img.shape[1], x + w)
        y1 = min(img.shape[0], y + h)
        if x1 > x0 and y1 > y0:
            out_img = img[y0:y1, x0:x1]
            if poly is not None and poly.size > 0:
                out_poly = shift_contour(poly, x0, y0)
            else:
                out_poly = np.empty((0, 1, 2), dtype=np.int32)

    cv2.imwrite(out_img_path, out_img)

    with open(out_lbl_path, "w", encoding="utf-8") as f:
        if out_poly is not None and out_poly.size > 0:
            hh, ww = out_img.shape[:2]
            vals = contour_to_yolo_polygon(out_poly, ww, hh)
            if len(vals) >= 6:
                f.write(str(int(class_id)) + " " + " ".join(f"{v:.6f}" for v in vals) + "\n")

    if bool(args.save_overlay):
        ov_dir = os.path.join(out_root, "overlays", split)
        ensure_dir(ov_dir)
        if bool(args.crop_to_roi) and roi_cfg is not None:
            crop_mask = np.zeros(out_img.shape[:2], dtype=np.uint8)
            if out_poly is not None and out_poly.size > 0:
                cv2.drawContours(crop_mask, [out_poly], -1, 255, thickness=cv2.FILLED)
            save_overlay(out_img, crop_mask, out_poly, os.path.join(ov_dir, f"{base_name}.png"))
        else:
            save_overlay(img, mask, poly if poly is not None and poly.size > 0 else None, os.path.join(ov_dir, f"{base_name}.png"))

    counters["kept"] += 1


def main():
    ap = argparse.ArgumentParser(description="Build 2-class (front/back) YOLOv8-seg dataset from captured images.")
    ap.add_argument("--background", type=str, required=True)
    ap.add_argument("--front_dir", type=str, required=True)
    ap.add_argument("--back_dir", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, default="segmentation_two_side_yolo/dataset")
    ap.add_argument("--roi_yaml", type=str, default="")
    ap.add_argument("--crop_to_roi", action="store_true")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--diff_thresh", type=int, default=28)
    ap.add_argument("--channel_thresh", type=int, default=18)
    ap.add_argument("--shadow_tol", type=int, default=14)
    ap.add_argument("--open_ksize", type=int, default=3)
    ap.add_argument("--close_ksize", type=int, default=7)
    ap.add_argument("--median_ksize", type=int, default=5)
    ap.add_argument("--min_area", type=int, default=900)
    ap.add_argument("--polygon_epsilon_ratio", type=float, default=0.002)
    ap.add_argument("--strict_mode", action="store_true")
    ap.add_argument("--strict_white_thresh", type=int, default=165)
    ap.add_argument("--strict_luma_delta", type=int, default=10)
    ap.add_argument("--strict_diff_thresh", type=int, default=10)
    ap.add_argument("--strict_min_area", type=int, default=120)
    ap.add_argument("--skip_empty", action="store_true")
    ap.add_argument("--save_overlay", action="store_true")
    ap.add_argument("--save_reject_overlay", action="store_true")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    bg = cv2.imread(args.background, cv2.IMREAD_COLOR)
    if bg is None:
        raise RuntimeError(f"Cannot read background: {args.background}")

    roi_cfg = None
    if args.roi_yaml:
        roi_cfg = load_roi_config(args.roi_yaml, bg.shape[1], bg.shape[0])
        if not args.crop_to_roi:
            args.crop_to_roi = True
        bx, by, bw, bh = roi_cfg["bbox"]
        print(f"[Info] ROI: mode={roi_cfg['mode']} x={bx} y={by} w={bw} h={bh} crop={args.crop_to_roi}")

    ensure_dir(os.path.join(args.dataset_root, "images", "train"))
    ensure_dir(os.path.join(args.dataset_root, "images", "val"))
    ensure_dir(os.path.join(args.dataset_root, "images", "test"))
    ensure_dir(os.path.join(args.dataset_root, "labels", "train"))
    ensure_dir(os.path.join(args.dataset_root, "labels", "val"))
    ensure_dir(os.path.join(args.dataset_root, "labels", "test"))

    front_files = collect_images(args.front_dir)
    back_files = collect_images(args.back_dir)
    if not front_files:
        raise RuntimeError(f"No images in front_dir: {args.front_dir}")
    if not back_files:
        raise RuntimeError(f"No images in back_dir: {args.back_dir}")

    counters = {"kept": 0, "empty": 0}
    for p in front_files:
        process_one(
            img_path=p,
            class_id=0,
            class_name="front",
            bg=bg,
            out_root=args.dataset_root,
            roi_cfg=roi_cfg,
            args=args,
            counters=counters,
        )
    for p in back_files:
        process_one(
            img_path=p,
            class_id=1,
            class_name="back",
            bg=bg,
            out_root=args.dataset_root,
            roi_cfg=roi_cfg,
            args=args,
            counters=counters,
        )

    write_data_yaml(os.path.join(args.dataset_root, "data.yaml"), names=["front", "back"])
    print(f"[Done] kept={counters['kept']} empty={counters['empty']}")
    extra = {k: v for k, v in counters.items() if k not in ("kept", "empty")}
    if extra:
        details = ", ".join(f"{k}={v}" for k, v in sorted(extra.items()))
        print(f"[Done] empty_reasons: {details}")
    print(f"[Done] dataset: {args.dataset_root}")
    print(
        "Train command:\n"
        f"yolo task=segment mode=train model=yolov8n-seg.pt data={os.path.join(args.dataset_root, 'data.yaml')} imgsz=640 epochs=120 batch=8 workers=0"
    )


if __name__ == "__main__":
    main()
