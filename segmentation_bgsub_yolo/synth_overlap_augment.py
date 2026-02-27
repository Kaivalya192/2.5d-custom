import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class ObjectPatch:
    image: np.ndarray  # HxWx3
    alpha: np.ndarray  # HxW uint8


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def collect_images(input_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
    files.sort()
    return files


def parse_yolo_seg_line(line: str) -> Tuple[int, np.ndarray]:
    vals = line.strip().split()
    if len(vals) < 7:
        raise ValueError("Invalid segmentation line.")
    cls_id = int(float(vals[0]))
    pts = np.array([float(v) for v in vals[1:]], dtype=np.float32).reshape(-1, 2)
    return cls_id, pts


def load_patches(images_dir: str, labels_dir: str, class_id: int) -> List[ObjectPatch]:
    image_paths = collect_images(images_dir)
    patches: List[ObjectPatch] = []
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
            if cid != int(class_id):
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
            patches.append(ObjectPatch(crop_img, crop_alpha))
    return patches


def contour_to_yolo(contour: np.ndarray, w: int, h: int) -> List[float]:
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0] / float(w), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(h), 0.0, 1.0)
    return pts.reshape(-1).tolist()


def yolo_line_to_contour(line: str, img_w: int, img_h: int) -> Tuple[int, np.ndarray]:
    vals = line.strip().split()
    cls_id = int(float(vals[0]))
    pts_n = np.array([float(v) for v in vals[1:]], dtype=np.float32).reshape(-1, 2)
    pts = np.zeros_like(pts_n, dtype=np.int32)
    pts[:, 0] = np.clip((pts_n[:, 0] * img_w).round().astype(np.int32), 0, img_w - 1)
    pts[:, 1] = np.clip((pts_n[:, 1] * img_h).round().astype(np.int32), 0, img_h - 1)
    return cls_id, pts.reshape(-1, 1, 2)


def crop_polys_to_roi(
    polys: List[np.ndarray],
    roi_xyxy: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    min_area: int,
) -> List[np.ndarray]:
    x0, y0, x1, y1 = roi_xyxy
    rw, rh = int(x1 - x0), int(y1 - y0)
    cropped_polys: List[np.ndarray] = []
    for poly in polys:
        if poly is None or len(poly) < 3:
            continue
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        roi_mask = mask[y0:y1, x0:x1]
        if np.count_nonzero(roi_mask) < int(min_area):
            continue
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < float(min_area):
            continue
        eps = 0.002 * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True)
        if c.shape[0] < 3:
            continue
        c = c.reshape(-1, 2).astype(np.int32)
        c[:, 0] = np.clip(c[:, 0], 0, rw - 1)
        c[:, 1] = np.clip(c[:, 1], 0, rh - 1)
        cropped_polys.append(c.reshape(-1, 1, 2))
    return cropped_polys


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
        patch_img, m, (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    w_alpha = cv2.warpAffine(
        patch_alpha, m, (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return w_img, w_alpha


def write_data_yaml(dataset_root: str, class_name: str) -> None:
    data_yaml = os.path.join(dataset_root, "data.yaml")
    txt = (
        f"path: {os.path.abspath(dataset_root)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n  0: {class_name}\n"
    )
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(txt)


def sample_instance_count(
    min_instances: int,
    max_instances: int,
    count_mode: str,
    dense_probability: float,
) -> int:
    min_i = int(min_instances)
    max_i = int(max_instances)
    if max_i <= min_i:
        return min_i

    if count_mode == "uniform":
        return random.randint(min_i, max_i)

    if count_mode == "dense":
        lo = max(min_i, int(round(0.65 * max_i)))
        return random.randint(lo, max_i)

    if count_mode == "sparse":
        hi = max(min_i, int(round(0.35 * max_i)))
        return random.randint(min_i, hi)

    # "mixed": sparse + dense mixture for realistic bin fill levels
    p_dense = float(np.clip(dense_probability, 0.0, 1.0))
    if random.random() < p_dense:
        lo = max(min_i, int(round(0.65 * max_i)))
        return random.randint(lo, max_i)
    hi = max(min_i, int(round(0.35 * max_i)))
    return random.randint(min_i, hi)


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


def load_roi_yaml(path: str, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
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
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("ROI is outside image bounds.")
    return x0, y0, x1, y1


def main():
    parser = argparse.ArgumentParser(description="Synthetic overlap augmentation for YOLOv8 segmentation.")
    parser.add_argument("--src_dataset_root", type=str, required=True)
    parser.add_argument("--background", type=str, required=True)
    parser.add_argument("--out_dataset_root", type=str, default="segmentation_bgsub_yolo/dataset_aug")
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--class_name", type=str, default="part")
    parser.add_argument("--num_images", type=int, default=300)
    parser.add_argument("--min_instances", type=int, default=2)
    parser.add_argument("--max_instances", type=int, default=30)
    parser.add_argument("--count_mode", type=str, default="mixed",
                        choices=["mixed", "uniform", "dense", "sparse"])
    parser.add_argument("--dense_probability", type=float, default=0.6,
                        help="used by mixed mode: probability of dense scenes")
    parser.add_argument("--min_success_ratio", type=float, default=0.45,
                        help="minimum visible-instance ratio required per synthesized image")
    parser.add_argument("--min_scale", type=float, default=0.85)
    parser.add_argument("--max_scale", type=float, default=1.20)
    parser.add_argument("--max_rotation", type=float, default=180.0)
    parser.add_argument("--jitter", type=float, default=0.15, help="intensity jitter fraction")
    parser.add_argument("--min_visible_area", type=int, default=80)
    parser.add_argument("--min_visible_ratio", type=float, default=0.15)
    parser.add_argument("--copy_original", action="store_true", help="copy original dataset into output too")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="fraction of samples assigned to val split")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="fraction of samples assigned to test split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_overlays", action="store_true")
    parser.add_argument("--roi_yaml", type=str, default="", help="place objects inside this ROI only")
    parser.add_argument("--placement_margin", type=int, default=4)
    parser.add_argument("--max_tries_per_image", type=int, default=12,
                        help="retry synth image until min visible instances are reached")
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

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

    patches = load_patches(src_images, src_labels, args.class_id)
    if len(patches) == 0:
        raise RuntimeError("No valid patches loaded from source dataset.")
    print(f"[Info] Loaded {len(patches)} object patches from source labels.")

    if args.copy_original:
        for ip in collect_images(src_images):
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(src_labels, f"{stem}.txt")
            if not os.path.exists(lp):
                continue
            split = pick_split(float(args.val_ratio), float(args.test_ratio))
            if split == "val":
                out_images, out_labels = out_images_val, out_labels_val
            elif split == "test":
                out_images, out_labels = out_images_test, out_labels_test
            else:
                out_images, out_labels = out_images_train, out_labels_train
            with open(lp, "r", encoding="utf-8") as f:
                txt = f.read()
            if args.roi_yaml:
                img0 = cv2.imread(ip, cv2.IMREAD_COLOR)
                if img0 is None:
                    continue
                img_crop = img0[ry0:ry1, rx0:rx1]
                lines_out = []
                for ln in txt.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    cid, poly = yolo_line_to_contour(ln, img0.shape[1], img0.shape[0])
                    if cid != int(args.class_id):
                        continue
                    cps = crop_polys_to_roi([poly], (rx0, ry0, rx1, ry1), img0.shape[1], img0.shape[0], int(args.min_visible_area))
                    for cp in cps:
                        ypts = contour_to_yolo(cp.reshape(-1, 2), img_crop.shape[1], img_crop.shape[0])
                        if len(ypts) >= 6:
                            lines_out.append(f"{int(args.class_id)} " + " ".join(f"{v:.6f}" for v in ypts))
                cv2.imwrite(os.path.join(out_images, f"{stem}.png"), img_crop)
                with open(os.path.join(out_labels, f"{stem}.txt"), "w", encoding="utf-8") as f:
                    if lines_out:
                        f.write("\n".join(lines_out) + "\n")
            else:
                with open(os.path.join(out_labels, f"{stem}.txt"), "w", encoding="utf-8") as f:
                    f.write(txt)

    for i in range(int(args.num_images)):
        final_canvas = None
        final_overlay = None
        final_labels = None

        for _try in range(int(args.max_tries_per_image)):
            canvas = bg.copy()
            inst_map = np.zeros((h, w), dtype=np.int32)
            orig_area = {}
            n_inst = sample_instance_count(
                int(args.min_instances),
                int(args.max_instances),
                str(args.count_mode),
                float(args.dense_probability),
            )

            placed_centers: List[Tuple[int, int]] = []
            for k in range(1, n_inst + 1):
                p = random.choice(patches)
                patch_img = random_jitter(p.image, float(args.jitter))
                patch_alpha = p.alpha
                scale = random.uniform(float(args.min_scale), float(args.max_scale))
                angle = random.uniform(-float(args.max_rotation), float(args.max_rotation))

                # Encourage overlap: usually sample near existing centers, otherwise random in ROI.
                overlap_bias = (len(placed_centers) > 0) and (random.random() < 0.70)
                if overlap_bias:
                    bx, by = random.choice(placed_centers)
                    cx = int(np.clip(bx + random.randint(-70, 70), rx0 + args.placement_margin, rx1 - 1 - args.placement_margin))
                    cy = int(np.clip(by + random.randint(-70, 70), ry0 + args.placement_margin, ry1 - 1 - args.placement_margin))
                else:
                    cx = random.randint(rx0 + args.placement_margin, max(rx0 + args.placement_margin, rx1 - 1 - args.placement_margin))
                    cy = random.randint(ry0 + args.placement_margin, max(ry0 + args.placement_margin, ry1 - 1 - args.placement_margin))

                w_img, w_alpha = warp_patch_to_canvas(
                    patch_img=patch_img,
                    patch_alpha=patch_alpha,
                    canvas_w=w,
                    canvas_h=h,
                    center_x=cx,
                    center_y=cy,
                    angle_deg=angle,
                    scale=scale,
                )
                m = w_alpha > 0
                if rx0 > 0 or ry0 > 0 or rx1 < w or ry1 < h:
                    roi_mask = np.zeros((h, w), dtype=bool)
                    roi_mask[ry0:ry1, rx0:rx1] = True
                    m &= roi_mask
                area = int(np.count_nonzero(m))
                if area < int(args.min_visible_area):
                    continue
                orig_area[k] = area
                placed_centers.append((cx, cy))
                canvas[m] = w_img[m]
                inst_map[m] = k

            label_polys: List[np.ndarray] = []
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
                yolo_pts = contour_to_yolo(poly, w=w, h=h)
                if len(yolo_pts) < 6:
                    continue
                label_polys.append(poly.astype(np.int32))
                if args.save_overlays:
                    cv2.drawContours(overlay, [poly], -1, (0, 255, 255), 2)

            needed = max(1, int(np.ceil(float(n_inst) * float(np.clip(args.min_success_ratio, 0.0, 1.0)))))
            if len(label_polys) >= needed:
                final_canvas, final_overlay, final_labels = canvas, overlay, label_polys
                break
            if final_labels is None or len(label_polys) > len(final_labels):
                final_canvas, final_overlay, final_labels = canvas, overlay, label_polys

        if not final_labels:
            continue

        stem = f"synth_{i + 1:06d}"
        split = pick_split(float(args.val_ratio), float(args.test_ratio))
        if split == "val":
            out_images, out_labels = out_images_val, out_labels_val
        elif split == "test":
            out_images, out_labels = out_images_test, out_labels_test
        else:
            out_images, out_labels = out_images_train, out_labels_train
        save_img = final_canvas
        save_polys = final_labels
        if args.roi_yaml:
            save_img = final_canvas[ry0:ry1, rx0:rx1]
            save_polys = crop_polys_to_roi(final_labels, (rx0, ry0, rx1, ry1), w, h, int(args.min_visible_area))
        lines_out = []
        for p in save_polys:
            ypts = contour_to_yolo(p.reshape(-1, 2), save_img.shape[1], save_img.shape[0])
            if len(ypts) >= 6:
                lines_out.append(f"{int(args.class_id)} " + " ".join(f"{v:.6f}" for v in ypts))
        if len(lines_out) == 0:
            continue
        cv2.imwrite(os.path.join(out_images, f"{stem}.png"), save_img)
        with open(os.path.join(out_labels, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out) + "\n")
        if args.save_overlays:
            ov = final_overlay
            if args.roi_yaml:
                ov = ov[ry0:ry1, rx0:rx1]
            cv2.imwrite(os.path.join(out_overlays, f"{stem}.png"), ov)

        if (i + 1) % 25 == 0:
            print(f"[Progress] {i + 1}/{args.num_images}")

    write_data_yaml(args.out_dataset_root, args.class_name)
    print(f"[Done] Augmented dataset created at: {args.out_dataset_root}")
    print("Train command:")
    print(
        f"yolo task=segment mode=train model=yolov8n-seg.pt "
        f"data={os.path.join(args.out_dataset_root, 'data.yaml')} imgsz=640 epochs=100 batch=8 workers=0"
    )


if __name__ == "__main__":
    main()
