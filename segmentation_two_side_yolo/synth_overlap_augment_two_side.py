import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class ObjectPatch:
    image: np.ndarray
    alpha: np.ndarray
    class_id: int


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def collect_images(input_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
    files.sort()
    return files


def parse_yolo_seg_line(line: str) -> Tuple[int, np.ndarray]:
    vals = line.strip().split()
    if len(vals) < 7:
        raise ValueError("Invalid segmentation line")
    cid = int(float(vals[0]))
    pts = np.array([float(v) for v in vals[1:]], dtype=np.float32).reshape(-1, 2)
    return cid, pts


def contour_to_yolo(contour: np.ndarray, w: int, h: int) -> List[float]:
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0] / float(w), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(h), 0.0, 1.0)
    return pts.reshape(-1).tolist()


def load_roi_yaml(path: str, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open ROI yaml: {path}")
    x = int(fs.getNode("x").real())
    y = int(fs.getNode("y").real())
    w = int(fs.getNode("w").real())
    h = int(fs.getNode("h").real())
    fs.release()
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("ROI outside image")
    return x0, y0, x1, y1


def write_data_yaml(path: str) -> None:
    txt = (
        f"path: {os.path.abspath(os.path.dirname(path))}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: front\n"
        "  1: back\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)


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


def random_jitter(patch: np.ndarray, jitter: float) -> np.ndarray:
    if jitter <= 0:
        return patch
    gain = 1.0 + random.uniform(-jitter, jitter)
    bias = random.uniform(-15.0 * jitter, 15.0 * jitter)
    out = patch.astype(np.float32) * gain + bias
    return np.clip(out, 0, 255).astype(np.uint8)


def warp_patch_to_canvas(
    patch_img: np.ndarray,
    patch_alpha: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    center_x: int,
    center_y: int,
    angle_deg: float,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ph, pw = patch_img.shape[:2]
    cx = (pw - 1) * 0.5
    cy = (ph - 1) * 0.5
    m = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    m[0, 2] += float(center_x - cx)
    m[1, 2] += float(center_y - cy)
    w_img = cv2.warpAffine(
        patch_img,
        m,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    w_alpha = cv2.warpAffine(
        patch_alpha,
        m,
        (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return w_img, w_alpha


def load_patches_two_class(images_dirs: List[str], labels_dirs: List[str]) -> Dict[int, List[ObjectPatch]]:
    out: Dict[int, List[ObjectPatch]] = {0: [], 1: []}
    for images_dir, labels_dir in zip(images_dirs, labels_dirs):
        image_paths = collect_images(images_dir)
        for ip in image_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(labels_dir, f"{stem}.txt")
            if not os.path.exists(lp):
                continue
            img = cv2.imread(ip, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            with open(lp, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            for ln in lines:
                try:
                    cid, pts_n = parse_yolo_seg_line(ln)
                except Exception:
                    continue
                if cid not in (0, 1):
                    continue
                pts = np.zeros_like(pts_n, dtype=np.int32)
                pts[:, 0] = np.clip((pts_n[:, 0] * w).round().astype(np.int32), 0, w - 1)
                pts[:, 1] = np.clip((pts_n[:, 1] * h).round().astype(np.int32), 0, h - 1)
                if pts.shape[0] < 3:
                    continue
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
                ys, xs = np.where(mask > 0)
                if xs.size == 0:
                    continue
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                crop_img = img[y0:y1, x0:x1].copy()
                crop_alpha = mask[y0:y1, x0:x1].copy()
                out[cid].append(ObjectPatch(crop_img, crop_alpha, cid))
    return out


def main():
    ap = argparse.ArgumentParser(description="Two-class synthetic overlap augmentation for front/back YOLO segmentation.")
    ap.add_argument("--src_dataset_root", type=str, default="segmentation_two_side_yolo/dataset")
    ap.add_argument("--background", type=str, required=True)
    ap.add_argument("--out_dataset_root", type=str, default="segmentation_two_side_yolo/dataset_aug")
    ap.add_argument("--num_images_per_class", type=int, default=400, help="400 front-dominant + 400 back-dominant")
    ap.add_argument("--min_instances", type=int, default=2)
    ap.add_argument("--max_instances", type=int, default=24)
    ap.add_argument("--dominant_class_ratio", type=float, default=0.65, help="min ratio of dominant class per image")
    ap.add_argument("--min_scale", type=float, default=0.85)
    ap.add_argument("--max_scale", type=float, default=1.2)
    ap.add_argument("--max_rotation", type=float, default=180.0)
    ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--min_visible_area", type=int, default=80)
    ap.add_argument("--min_visible_ratio", type=float, default=0.15)
    ap.add_argument("--min_success_ratio", type=float, default=0.45)
    ap.add_argument("--max_tries_per_image", type=int, default=12)
    ap.add_argument("--copy_original", action="store_true")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_overlays", action="store_true")
    ap.add_argument("--roi_yaml", type=str, default="segmentation_two_side_yolo/config/roi.yaml")
    ap.add_argument("--placement_margin", type=int, default=4)
    ap.add_argument(
        "--src_splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to extract object patches from (default: train,val,test).",
    )
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    split_names = [s.strip() for s in str(args.src_splits).split(",") if s.strip()]
    if not split_names:
        split_names = ["train", "val", "test"]
    src_images_dirs = [os.path.join(args.src_dataset_root, "images", s) for s in split_names]
    src_labels_dirs = [os.path.join(args.src_dataset_root, "labels", s) for s in split_names]
    src_images = os.path.join(args.src_dataset_root, "images", "train")
    src_labels = os.path.join(args.src_dataset_root, "labels", "train")
    out_images_train = os.path.join(args.out_dataset_root, "images", "train")
    out_labels_train = os.path.join(args.out_dataset_root, "labels", "train")
    out_images_val = os.path.join(args.out_dataset_root, "images", "val")
    out_labels_val = os.path.join(args.out_dataset_root, "labels", "val")
    out_images_test = os.path.join(args.out_dataset_root, "images", "test")
    out_labels_test = os.path.join(args.out_dataset_root, "labels", "test")
    out_overlays = os.path.join(args.out_dataset_root, "overlays")
    ensure_dir(out_images_train)
    ensure_dir(out_labels_train)
    ensure_dir(out_images_val)
    ensure_dir(out_labels_val)
    ensure_dir(out_images_test)
    ensure_dir(out_labels_test)
    if args.save_overlays:
        ensure_dir(out_overlays)

    bg = cv2.imread(args.background, cv2.IMREAD_COLOR)
    if bg is None:
        raise RuntimeError(f"Cannot read background image: {args.background}")
    h, w = bg.shape[:2]
    if args.max_instances < args.min_instances:
        args.max_instances = args.min_instances
    if args.min_instances < 1:
        args.min_instances = 1

    if args.roi_yaml:
        rx0, ry0, rx1, ry1 = load_roi_yaml(args.roi_yaml, w, h)
    else:
        rx0, ry0, rx1, ry1 = 0, 0, w, h
    print(f"[Info] Placement ROI: x={rx0}:{rx1} y={ry0}:{ry1}")

    patches_by_class = load_patches_two_class(src_images_dirs, src_labels_dirs)
    if len(patches_by_class[0]) == 0 or len(patches_by_class[1]) == 0:
        raise RuntimeError("Need both class patches in selected source splits.")
    print(
        f"[Info] loaded patches from splits {split_names} -> "
        f"front={len(patches_by_class[0])} back={len(patches_by_class[1])}"
    )

    if args.copy_original:
        for ip in collect_images(src_images):
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(src_labels, f"{stem}.txt")
            if not os.path.exists(lp):
                continue
            img0 = cv2.imread(ip, cv2.IMREAD_COLOR)
            if img0 is None:
                continue
            split = pick_split(float(args.val_ratio), float(args.test_ratio))
            if split == "val":
                out_images, out_labels = out_images_val, out_labels_val
            elif split == "test":
                out_images, out_labels = out_images_test, out_labels_test
            else:
                out_images, out_labels = out_images_train, out_labels_train
            if args.roi_yaml:
                img_crop = img0[ry0:ry1, rx0:rx1]
                lines_out = []
                with open(lp, "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        cid, pts_n = parse_yolo_seg_line(ln)
                        pts = np.zeros_like(pts_n, dtype=np.int32)
                        pts[:, 0] = np.clip((pts_n[:, 0] * img0.shape[1]).round().astype(np.int32), 0, img0.shape[1] - 1)
                        pts[:, 1] = np.clip((pts_n[:, 1] * img0.shape[0]).round().astype(np.int32), 0, img0.shape[0] - 1)
                        mask = np.zeros((img0.shape[0], img0.shape[1]), dtype=np.uint8)
                        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
                        roi_mask = mask[ry0:ry1, rx0:rx1]
                        if np.count_nonzero(roi_mask) < int(args.min_visible_area):
                            continue
                        cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not cnts:
                            continue
                        c = max(cnts, key=cv2.contourArea)
                        if cv2.contourArea(c) < float(args.min_visible_area):
                            continue
                        ypts = contour_to_yolo(c, img_crop.shape[1], img_crop.shape[0])
                        if len(ypts) >= 6:
                            lines_out.append(f"{int(cid)} " + " ".join(f"{v:.6f}" for v in ypts))
                cv2.imwrite(os.path.join(out_images, f"{stem}.png"), img_crop)
                with open(os.path.join(out_labels, f"{stem}.txt"), "w", encoding="utf-8") as f:
                    if lines_out:
                        f.write("\n".join(lines_out) + "\n")
            else:
                cv2.imwrite(os.path.join(out_images, f"{stem}.png"), img0)
                with open(lp, "r", encoding="utf-8") as f:
                    txt = f.read()
                with open(os.path.join(out_labels, f"{stem}.txt"), "w", encoding="utf-8") as f:
                    f.write(txt)

    dominant_sequence = [0] * int(args.num_images_per_class) + [1] * int(args.num_images_per_class)
    random.shuffle(dominant_sequence)
    total = len(dominant_sequence)

    for i, dom_cls in enumerate(dominant_sequence, start=1):
        final_canvas = None
        final_overlay = None
        final_labels = None
        final_cls = None
        for _ in range(int(args.max_tries_per_image)):
            canvas = bg.copy()
            inst_map = np.zeros((h, w), dtype=np.int32)
            inst_cls: Dict[int, int] = {}
            orig_area: Dict[int, int] = {}
            n_inst = random.randint(int(args.min_instances), int(args.max_instances))
            n_dom = max(1, int(round(float(n_inst) * float(np.clip(args.dominant_class_ratio, 0.05, 0.95)))))
            n_other = n_inst - n_dom
            choose_cls = [dom_cls] * n_dom + [1 - dom_cls] * n_other
            random.shuffle(choose_cls)
            placed_centers: List[Tuple[int, int]] = []
            inst_id = 0
            for cid in choose_cls:
                plist = patches_by_class[int(cid)]
                if not plist:
                    continue
                p = random.choice(plist)
                patch_img = random_jitter(p.image, float(args.jitter))
                patch_alpha = p.alpha
                scale = random.uniform(float(args.min_scale), float(args.max_scale))
                angle = random.uniform(-float(args.max_rotation), float(args.max_rotation))
                overlap_bias = (len(placed_centers) > 0) and (random.random() < 0.70)
                if overlap_bias:
                    bx, by = random.choice(placed_centers)
                    cx = int(np.clip(bx + random.randint(-70, 70), rx0 + args.placement_margin, rx1 - 1 - args.placement_margin))
                    cy = int(np.clip(by + random.randint(-70, 70), ry0 + args.placement_margin, ry1 - 1 - args.placement_margin))
                else:
                    cx = random.randint(rx0 + args.placement_margin, max(rx0 + args.placement_margin, rx1 - 1 - args.placement_margin))
                    cy = random.randint(ry0 + args.placement_margin, max(ry0 + args.placement_margin, ry1 - 1 - args.placement_margin))
                w_img, w_alpha = warp_patch_to_canvas(
                    patch_img, patch_alpha, w, h, cx, cy, angle, scale
                )
                m = w_alpha > 0
                if rx0 > 0 or ry0 > 0 or rx1 < w or ry1 < h:
                    roi_mask = np.zeros((h, w), dtype=bool)
                    roi_mask[ry0:ry1, rx0:rx1] = True
                    m &= roi_mask
                area = int(np.count_nonzero(m))
                if area < int(args.min_visible_area):
                    continue
                inst_id += 1
                orig_area[inst_id] = area
                inst_cls[inst_id] = int(cid)
                placed_centers.append((cx, cy))
                canvas[m] = w_img[m]
                inst_map[m] = inst_id

            label_polys: List[np.ndarray] = []
            label_cls: List[int] = []
            overlay = canvas.copy()
            for k, area0 in orig_area.items():
                vis = (inst_map == k).astype(np.uint8) * 255
                vis_area = int(np.count_nonzero(vis))
                if vis_area < int(args.min_visible_area):
                    continue
                if vis_area / max(1.0, float(area0)) < float(args.min_visible_ratio):
                    continue
                contours, _ = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) < float(args.min_visible_area):
                    continue
                eps = 0.002 * cv2.arcLength(c, True)
                poly = cv2.approxPolyDP(c, eps, True)
                if poly.shape[0] < 3:
                    poly = c
                if poly.shape[0] < 3:
                    continue
                ypts = contour_to_yolo(poly, w=w, h=h)
                if len(ypts) < 6:
                    continue
                label_polys.append(poly.astype(np.int32))
                label_cls.append(int(inst_cls[k]))
                if args.save_overlays:
                    col = (0, 255, 255) if int(inst_cls[k]) == 0 else (255, 180, 0)
                    cv2.drawContours(overlay, [poly], -1, col, 2)

            needed = max(1, int(np.ceil(float(max(1, len(orig_area))) * float(np.clip(args.min_success_ratio, 0.0, 1.0)))))
            if len(label_polys) >= needed:
                final_canvas, final_overlay, final_labels, final_cls = canvas, overlay, label_polys, label_cls
                break
            if final_labels is None or len(label_polys) > len(final_labels):
                final_canvas, final_overlay, final_labels, final_cls = canvas, overlay, label_polys, label_cls

        if not final_labels:
            continue

        stem = f"synth2_{i:06d}"
        split = pick_split(float(args.val_ratio), float(args.test_ratio))
        if split == "val":
            out_images, out_labels = out_images_val, out_labels_val
        elif split == "test":
            out_images, out_labels = out_images_test, out_labels_test
        else:
            out_images, out_labels = out_images_train, out_labels_train

        save_img = final_canvas
        save_polys = final_labels
        save_cls = final_cls
        if args.roi_yaml:
            save_img = final_canvas[ry0:ry1, rx0:rx1]
            keep_polys: List[np.ndarray] = []
            keep_cls: List[int] = []
            for poly, cid in zip(final_labels, final_cls):
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)
                roi_mask = mask[ry0:ry1, rx0:rx1]
                if np.count_nonzero(roi_mask) < int(args.min_visible_area):
                    continue
                contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) < float(args.min_visible_area):
                    continue
                eps = 0.002 * cv2.arcLength(c, True)
                c = cv2.approxPolyDP(c, eps, True)
                if c.shape[0] < 3:
                    continue
                keep_polys.append(c)
                keep_cls.append(int(cid))
            save_polys = keep_polys
            save_cls = keep_cls

        lines_out = []
        for poly, cid in zip(save_polys, save_cls):
            ypts = contour_to_yolo(poly.reshape(-1, 2), save_img.shape[1], save_img.shape[0])
            if len(ypts) >= 6:
                lines_out.append(f"{int(cid)} " + " ".join(f"{v:.6f}" for v in ypts))
        if len(lines_out) == 0:
            continue

        cv2.imwrite(os.path.join(out_images, f"{stem}.png"), save_img)
        with open(os.path.join(out_labels, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out) + "\n")
        if args.save_overlays:
            ov = final_overlay if not args.roi_yaml else final_overlay[ry0:ry1, rx0:rx1]
            cv2.imwrite(os.path.join(out_overlays, f"{stem}.png"), ov)

        if i % 25 == 0:
            print(f"[Progress] {i}/{total}")

    write_data_yaml(os.path.join(args.out_dataset_root, "data.yaml"))
    print(f"[Done] Augmented two-class dataset created at: {args.out_dataset_root}")
    print(
        "Train command:\n"
        f"yolo task=segment mode=train model=yolov8n-seg.pt data={os.path.join(args.out_dataset_root, 'data.yaml')} imgsz=640 epochs=120 batch=8 workers=0"
    )


if __name__ == "__main__":
    main()
