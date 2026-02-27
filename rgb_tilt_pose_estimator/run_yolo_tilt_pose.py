import argparse
import glob
import os
from typing import Generator, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from estimator import EstimatorConfig, draw_pose_overlay, estimate_tilt_pose_from_mask


def image_stream(path: str) -> Generator[np.ndarray, None, None]:
    if os.path.isdir(path):
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            files.extend(glob.glob(os.path.join(path, ext)))
        files.sort()
        for fp in files:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if img is not None:
                yield img
    else:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
            cap.release()
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Cannot open source: {path}")
            yield img


def make_mask_from_polygon(poly_xy: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 255)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="RGB-only tilt/yaw(mod symmetry) estimator using YOLOv8 masks.")
    ap.add_argument("--model", type=str, required=True, help="YOLOv8 segmentation model path (e.g., runs/.../best.pt)")
    ap.add_argument("--source", type=str, required=True, help="image, folder, or video path")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max_det", type=int, default=100)
    ap.add_argument("--symmetry_mod_deg", type=float, default=180.0)
    ap.add_argument("--conf_thr", type=float, default=0.55)
    ap.add_argument("--min_area", type=int, default=700)
    ap.add_argument("--border_margin_px", type=int, default=8)
    ap.add_argument("--solidity_thr", type=float, default=0.90)
    ap.add_argument("--completeness_thr", type=float, default=0.55)
    ap.add_argument("--save_dir", type=str, default="")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cfg = EstimatorConfig(
        conf_thr=float(args.conf_thr),
        min_area=int(args.min_area),
        border_margin_px=int(args.border_margin_px),
        solidity_thr=float(args.solidity_thr),
        completeness_thr=float(args.completeness_thr),
        symmetry_mod_deg=float(args.symmetry_mod_deg),
    )

    model = YOLO(args.model)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    idx = 0
    for frame in image_stream(args.source):
        idx += 1
        res = model.predict(
            source=frame,
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=int(args.imgsz),
            max_det=int(args.max_det),
            device=args.device,
            verbose=False,
        )[0]

        out = frame.copy()
        best_text_y = 24
        if res.masks is not None and res.boxes is not None:
            polys = res.masks.xy
            confs = res.boxes.conf.cpu().numpy()
            for i, poly in enumerate(polys):
                if poly is None or len(poly) < 3:
                    continue
                mask = make_mask_from_polygon(poly, frame.shape[0], frame.shape[1])
                est = estimate_tilt_pose_from_mask(mask, float(confs[i]), cfg=cfg, K=None)
                ov = draw_pose_overlay(frame, est)
                # Blend each instance overlay progressively.
                out = cv2.addWeighted(out, 0.62, ov, 0.38, 0)
                if est.get("valid", False):
                    cx, cy = est["pose2d"]
                    t = est["tilt_angle_deg"]
                    d = est["tilt_dir_deg"]
                    y = est["yaw_mod_deg"]
                    txt = f"#{i+1} ({cx:.1f},{cy:.1f}) yawM={y:.1f} tilt={t:.1f} dir={d:.1f}"
                    cv2.putText(out, txt, (10, best_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                    best_text_y += 22

        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir, f"pose_{idx:06d}.png"), out)
        if args.show:
            cv2.imshow("RGB Tilt Pose", out)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
