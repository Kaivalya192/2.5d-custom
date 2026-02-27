import argparse
import glob
import os
from typing import List, Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def collect_images(input_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
    files.sort()
    return files


def odd(v: int) -> int:
    if v <= 0:
        return 0
    return v if (v % 2 == 1) else (v + 1)


def largest_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if int(stats[idx, cv2.CC_STAT_AREA]) < int(min_area):
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    out[labels == idx] = 255
    return out


def contour_to_yolo_polygon(contour: np.ndarray, w: int, h: int) -> List[float]:
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0] / float(w), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(h), 0.0, 1.0)
    return pts.reshape(-1).tolist()


def shift_contour(contour: np.ndarray, dx: int, dy: int) -> np.ndarray:
    c = contour.copy().astype(np.int32)
    c[:, 0, 0] = c[:, 0, 0] - int(dx)
    c[:, 0, 1] = c[:, 0, 1] - int(dy)
    return c


def segment_foreground(
    img: np.ndarray,
    bg: np.ndarray,
    diff_thresh: int,
    channel_thresh: int,
    shadow_tol: int,
    open_ksize: int,
    close_ksize: int,
    median_ksize: int,
    min_area: int,
    polygon_epsilon_ratio: float,
    strict_mode: bool = False,
    strict_white_thresh: int = 165,
    strict_luma_delta: int = 10,
    strict_diff_thresh: int = 10,
    strict_min_area: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    # RGB absolute difference.
    diff = cv2.absdiff(img, bg)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ch_max = np.max(diff, axis=2)

    # Shadow rejection: likely shadow if current pixel is much darker than background.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gray_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.int16)
    not_shadow = gray_img >= (gray_bg - int(shadow_tol))

    mask = (diff_gray >= int(diff_thresh)) & (ch_max >= int(channel_thresh)) & not_shadow
    mask = (mask.astype(np.uint8) * 255)

    k_open = int(max(0, open_ksize))
    k_close = int(max(0, close_ksize))
    k_med = odd(int(max(0, median_ksize)))

    if k_open > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
    if k_close > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
    if k_med >= 3:
        mask = cv2.medianBlur(mask, k_med)

    mask = largest_component(mask, min_area=min_area)

    # Strict fallback for bright white object on dark bin/background.
    if strict_mode and np.count_nonzero(mask) == 0:
        bright = gray_img >= int(strict_white_thresh)
        luma_up = (gray_img - gray_bg) >= int(strict_luma_delta)
        dmask = diff_gray >= int(strict_diff_thresh)
        mask_strict = (bright & (luma_up | dmask)).astype(np.uint8) * 255

        if k_close > 0:
            kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
            mask_strict = cv2.morphologyEx(mask_strict, cv2.MORPH_CLOSE, kc)
        if k_open > 0:
            ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
            mask_strict = cv2.morphologyEx(mask_strict, cv2.MORPH_OPEN, ko)
        if k_med >= 3:
            mask_strict = cv2.medianBlur(mask_strict, k_med)

        mask = largest_component(mask_strict, min_area=max(5, int(strict_min_area)))
    if np.count_nonzero(mask) == 0:
        return mask, np.empty((0, 1, 2), dtype=np.int32)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask), np.empty((0, 1, 2), dtype=np.int32)
    c = max(contours, key=cv2.contourArea)
    eps = float(polygon_epsilon_ratio) * float(cv2.arcLength(c, True))
    approx = cv2.approxPolyDP(c, eps, True)
    if approx.shape[0] < 3:
        approx = c
    if approx.shape[0] < 3:
        return np.zeros_like(mask), np.empty((0, 1, 2), dtype=np.int32)
    return mask, approx


def load_roi_yaml(path: str):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open ROI yaml: {path}")
    x = int(fs.getNode("x").real())
    y = int(fs.getNode("y").real())
    w = int(fs.getNode("w").real())
    h = int(fs.getNode("h").real())
    fs.release()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Invalid ROI in yaml: {path}")
    return x, y, w, h


def save_overlay(img: np.ndarray, mask: np.ndarray, contour: np.ndarray, save_path: str) -> None:
    overlay = img.copy()
    color_mask = np.zeros_like(img)
    color_mask[:, :, 1] = mask
    overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.35, 0)
    if contour is not None and contour.size > 0:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 2)
    cv2.imwrite(save_path, overlay)


def write_data_yaml(path: str, class_name: str) -> None:
    txt = (
        f"path: {os.path.abspath(os.path.dirname(path))}\n"
        "train: images/train\n"
        "val: images/train\n"
        f"names:\n  0: {class_name}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


def main():
    parser = argparse.ArgumentParser(description="Generate YOLOv8 segmentation dataset using background subtraction.")
    parser.add_argument("--background", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--roi_yaml", type=str, default="", help="optional ROI yaml from select_roi.py")
    parser.add_argument("--crop_to_roi", action="store_true",
                        help="when ROI is provided, crop saved image/label domain to ROI")
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--class_name", type=str, default="part")

    parser.add_argument("--diff_thresh", type=int, default=28)
    parser.add_argument("--channel_thresh", type=int, default=18)
    parser.add_argument("--shadow_tol", type=int, default=14)
    parser.add_argument("--open_ksize", type=int, default=3)
    parser.add_argument("--close_ksize", type=int, default=7)
    parser.add_argument("--median_ksize", type=int, default=5)
    parser.add_argument("--min_area", type=int, default=1200)
    parser.add_argument("--polygon_epsilon_ratio", type=float, default=0.002)
    parser.add_argument("--strict_mode", action="store_true",
                        help="fallback segmentation for white object on dark background")
    parser.add_argument("--strict_white_thresh", type=int, default=165)
    parser.add_argument("--strict_luma_delta", type=int, default=10)
    parser.add_argument("--strict_diff_thresh", type=int, default=10)
    parser.add_argument("--strict_min_area", type=int, default=120)

    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--skip_empty", action="store_true", help="skip images when no valid mask found")
    args = parser.parse_args()

    bg = cv2.imread(args.background, cv2.IMREAD_COLOR)
    if bg is None:
        raise RuntimeError(f"Cannot read background image: {args.background}")

    files = collect_images(args.input_dir)
    if not files:
        raise RuntimeError(f"No input images found in {args.input_dir}")

    roi = None
    if args.roi_yaml:
        roi = load_roi_yaml(args.roi_yaml)
        print(f"[Info] Using ROI from yaml: {args.roi_yaml} -> x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
        if not args.crop_to_roi:
            args.crop_to_roi = True
            print("[Info] ROI provided: enabling --crop_to_roi automatically")

    img_out = os.path.join(args.dataset_root, "images", "train")
    lbl_out = os.path.join(args.dataset_root, "labels", "train")
    ovl_out = os.path.join(args.dataset_root, "overlays")
    ensure_dir(img_out)
    ensure_dir(lbl_out)
    if args.save_overlay:
        ensure_dir(ovl_out)

    kept = 0
    empty = 0

    for i, path in enumerate(files, start=1):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
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

        if roi is not None:
            x, y, w, h = roi
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(mask.shape[1], x + w)
            y1 = min(mask.shape[0], y + h)
            roi_mask = np.zeros_like(mask)
            if x1 > x0 and y1 > y0:
                roi_mask[y0:y1, x0:x1] = 255
            mask = cv2.bitwise_and(mask, roi_mask)
            mask = largest_component(mask, min_area=args.min_area)
            if np.count_nonzero(mask) > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(contours, key=cv2.contourArea)
                eps = float(args.polygon_epsilon_ratio) * float(cv2.arcLength(c, True))
                poly = cv2.approxPolyDP(c, eps, True)
                if poly.shape[0] < 3:
                    poly = c
            else:
                poly = np.empty((0, 1, 2), dtype=np.int32)

            # ROI-local strict fallback for small/low-contrast objects.
            if np.count_nonzero(mask) == 0 and bool(args.strict_mode) and x1 > x0 and y1 > y0:
                img_roi = img[y0:y1, x0:x1]
                bg_roi = bg[y0:y1, x0:x1]
                mask_roi, poly_roi = segment_foreground(
                    img=img_roi,
                    bg=bg_roi,
                    diff_thresh=args.diff_thresh,
                    channel_thresh=args.channel_thresh,
                    shadow_tol=args.shadow_tol,
                    open_ksize=args.open_ksize,
                    close_ksize=args.close_ksize,
                    median_ksize=args.median_ksize,
                    min_area=max(1, int(args.strict_min_area)),
                    polygon_epsilon_ratio=args.polygon_epsilon_ratio,
                    strict_mode=True,
                    strict_white_thresh=int(args.strict_white_thresh),
                    strict_luma_delta=int(args.strict_luma_delta),
                    strict_diff_thresh=int(args.strict_diff_thresh),
                    strict_min_area=int(args.strict_min_area),
                )
                if np.count_nonzero(mask_roi) > 0 and poly_roi is not None and poly_roi.size > 0:
                    mask = np.zeros_like(mask)
                    mask[y0:y1, x0:x1] = mask_roi
                    poly = poly_roi.copy().astype(np.int32)
                    poly[:, 0, 0] += x0
                    poly[:, 0, 1] += y0

        stem = os.path.splitext(os.path.basename(path))[0]
        img_path = os.path.join(img_out, f"{stem}.png")
        lbl_path = os.path.join(lbl_out, f"{stem}.txt")

        out_img = img
        out_poly = poly
        if roi is not None and bool(args.crop_to_roi):
            x, y, w, h = roi
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(img.shape[1], x + w)
            y1 = min(img.shape[0], y + h)
            if x1 > x0 and y1 > y0:
                out_img = img[y0:y1, x0:x1].copy()
                if out_poly is not None and out_poly.size > 0:
                    out_poly = shift_contour(out_poly, x0, y0)
            else:
                out_img = img

        has_poly = out_poly is not None and out_poly.size > 0 and out_poly.shape[0] >= 3
        if not has_poly:
            empty += 1
            if args.skip_empty:
                continue
            cv2.imwrite(img_path, out_img)
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("")
            if args.save_overlay:
                if out_img is img:
                    save_overlay(img, mask, None, os.path.join(ovl_out, f"{stem}.png"))
                else:
                    save_overlay(out_img, np.zeros(out_img.shape[:2], dtype=np.uint8), None, os.path.join(ovl_out, f"{stem}.png"))
            continue

        points = contour_to_yolo_polygon(out_poly, w=out_img.shape[1], h=out_img.shape[0])
        if len(points) < 6:
            empty += 1
            continue

        line = f"{int(args.class_id)} " + " ".join(f"{v:.6f}" for v in points)
        cv2.imwrite(img_path, out_img)
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(line + "\n")

        if args.save_overlay:
            if out_img is img:
                save_overlay(img, mask, out_poly, os.path.join(ovl_out, f"{stem}.png"))
            else:
                crop_mask = np.zeros(out_img.shape[:2], dtype=np.uint8)
                cv2.drawContours(crop_mask, [out_poly], -1, 255, thickness=cv2.FILLED)
                save_overlay(out_img, crop_mask, out_poly, os.path.join(ovl_out, f"{stem}.png"))

        if args.show:
            disp = out_img.copy()
            cv2.drawContours(disp, [out_poly], -1, (0, 255, 255), 2)
            cv2.imshow("Mask QA", disp)
            if out_img is img:
                cv2.imshow("Mask Binary", mask)
            else:
                crop_mask = np.zeros(out_img.shape[:2], dtype=np.uint8)
                cv2.drawContours(crop_mask, [out_poly], -1, 255, thickness=cv2.FILLED)
                cv2.imshow("Mask Binary", crop_mask)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

        kept += 1
        if i % 20 == 0:
            print(f"[Progress] {i}/{len(files)} processed | kept={kept} empty={empty}")

    data_yaml_path = os.path.join(args.dataset_root, "data.yaml")
    write_data_yaml(data_yaml_path, args.class_name)

    cv2.destroyAllWindows()
    print(f"\n[Done] processed={len(files)} kept={kept} empty={empty}")
    print(f"[Dataset] images: {img_out}")
    print(f"[Dataset] labels: {lbl_out}")
    if args.save_overlay:
        print(f"[Dataset] overlays: {ovl_out}")
    print(f"[Dataset] data.yaml: {data_yaml_path}")
    print(
        "Train command:\n"
        f"yolo task=segment mode=train model=yolov8n-seg.pt data={data_yaml_path} imgsz=640 epochs=100 batch=16"
    )


if __name__ == "__main__":
    main()
