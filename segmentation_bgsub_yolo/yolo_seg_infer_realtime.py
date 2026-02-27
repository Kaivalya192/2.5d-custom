import argparse
import glob
import os
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


def random_color(seed: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed)
    c = rng.integers(60, 255, size=3, dtype=np.uint8)
    return int(c[0]), int(c[1]), int(c[2])


def draw_instances(
    frame_bgr: np.ndarray,
    polygons: List[np.ndarray],
    labels: List[str],
    alphas: float = 0.35,
) -> np.ndarray:
    out = frame_bgr.copy()
    overlay = frame_bgr.copy()
    for i, (poly, txt) in enumerate(zip(polygons, labels), start=1):
        color = random_color(i * 9973)
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(out, [poly], True, color, 2, cv2.LINE_AA)
        x, y = int(poly[:, 0].min()), int(poly[:, 1].min()) - 6
        y = max(12, y)
        cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, alphas, out, 1.0 - alphas, 0)
    return out


def collect_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files.sort()
    return files


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
        raise RuntimeError("Invalid ROI or ROI outside image bounds.")
    return x0, y0, x1, y1


def frame_source_realsense(width: int, height: int, fps: int) -> Generator[np.ndarray, None, None]:
    if rs is None:
        raise RuntimeError("pyrealsense2 is not available in this environment.")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipe.start(cfg)
    try:
        for _ in range(15):
            _ = pipe.wait_for_frames(5000)
        while True:
            frames = pipe.wait_for_frames(5000)
            color = frames.get_color_frame()
            if not color:
                continue
            yield np.asanyarray(color.get_data())
    finally:
        pipe.stop()


def frame_source_webcam(index: int) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {index}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def frame_source_video(path: str) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def frame_source_images(folder: str) -> Generator[np.ndarray, None, None]:
    files = collect_images(folder)
    if not files:
        raise RuntimeError(f"No images found in: {folder}")
    for p in files:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        yield img


def run(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    print(f"[Info] Model loaded: {args.model}")

    if args.source == "rs":
        frames = frame_source_realsense(args.width, args.height, args.fps)
    elif args.source == "webcam":
        frames = frame_source_webcam(args.webcam_index)
    elif args.source == "video":
        frames = frame_source_video(args.video)
    else:
        frames = frame_source_images(args.image_dir)

    writer: Optional[cv2.VideoWriter] = None
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None

    for frame in frames:
        if args.roi_yaml:
            if roi_xyxy is None:
                roi_xyxy = load_roi_yaml(args.roi_yaml, frame.shape[1], frame.shape[0])
                print(f"[Info] ROI loaded: x={roi_xyxy[0]}:{roi_xyxy[2]} y={roi_xyxy[1]}:{roi_xyxy[3]}")
            x0, y0, x1, y1 = roi_xyxy
            infer_frame = frame[y0:y1, x0:x1]
        else:
            x0, y0 = 0, 0
            infer_frame = frame

        result = model.predict(
            source=infer_frame,
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=int(args.imgsz),
            device=args.device,
            max_det=int(args.max_det),
            verbose=False,
        )[0]

        polygons: List[np.ndarray] = []
        labels: List[str] = []

        if result.masks is not None and result.boxes is not None:
            xy_list = result.masks.xy
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.zeros((len(xy_list),), dtype=np.float32)
            clss = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros((len(xy_list),), dtype=np.int32)
            for i, xy in enumerate(xy_list):
                if xy is None or len(xy) < 3:
                    continue
                poly = np.round(xy).astype(np.int32)
                if args.roi_yaml:
                    poly[:, 0] += int(x0)
                    poly[:, 1] += int(y0)
                polygons.append(poly)
                cname = result.names.get(int(clss[i]), str(int(clss[i])))
                labels.append(f"#{len(polygons)} {cname} {float(confs[i]):.2f}")

        out = draw_instances(frame, polygons, labels, alphas=float(args.alpha))
        if args.roi_yaml and roi_xyxy is not None and args.show_roi:
            cv2.rectangle(out, (roi_xyxy[0], roi_xyxy[1]), (roi_xyxy[2], roi_xyxy[3]), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(
            out,
            f"Instances: {len(polygons)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if args.save_video:
            if writer is None:
                h, w = out.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save_video, fourcc, float(args.out_fps), (w, h))
                print(f"[Info] Writing video: {args.save_video}")
            writer.write(out)

        cv2.imshow(args.window, out)
        key = cv2.waitKey(1 if args.source in ("rs", "webcam", "video") else 0) & 0xFF
        if key in (ord("q"), 27):
            break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time YOLOv8 segmentation instance annotation.")
    parser.add_argument("--model", type=str, default="runs/segment/train/weights/best.pt")
    parser.add_argument("--source", type=str, default="rs", choices=["rs", "webcam", "video", "images"])
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--webcam_index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max_det", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--window", type=str, default="YOLOv8 Seg Inference")
    parser.add_argument("--save_video", type=str, default="")
    parser.add_argument("--out_fps", type=float, default=20.0)
    parser.add_argument("--roi_yaml", type=str, default="", help="crop frame to ROI before YOLO inference")
    parser.add_argument("--show_roi", action="store_true", help="draw ROI rectangle on output")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
