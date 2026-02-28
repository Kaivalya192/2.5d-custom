import argparse
import os
import sys

from ultralytics import YOLO

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rgb_tilt_pose_estimator.apps._run_loop import run_pose_loop
from rgb_tilt_pose_estimator.sources.source_images import source_images
from rgb_tilt_pose_estimator.sources.source_rs import source_realsense
from rgb_tilt_pose_estimator.sources.source_video import source_video
from rgb_tilt_pose_estimator.sources.source_webcam import source_webcam


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robust RGB-only roll/pitch estimator from YOLOv8 segmentation masks (RS/Webcam/Video/Images)."
    )
    p.add_argument("--model", type=str, default="runs/segment/Imported/best.pt")
    p.add_argument("--source", type=str, default="rs", choices=["rs", "webcam", "video", "images"])
    p.add_argument("--video", type=str, default="")
    p.add_argument("--image_dir", type=str, default="")
    p.add_argument("--webcam_index", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--roi_yaml", type=str, default="segmentation_bgsub_yolo/config/roi.yaml")
    p.add_argument("--show_roi", action="store_true")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max_det", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--overlap_suppress_ratio", type=float, default=0.35)
    p.add_argument("--overlay_alpha", type=float, default=0.30)
    p.add_argument("--track_alpha", type=float, default=0.40)
    p.add_argument("--track_max_dist_px", type=float, default=75.0)
    p.add_argument("--track_max_missed", type=int, default=10)
    p.add_argument("--track_min_hits", type=int, default=2)
    p.add_argument("--show_invalid", action="store_true")
    p.add_argument("--save_video", type=str, default="")
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--out_fps", type=float, default=20.0)
    p.add_argument("--window", type=str, default="YOLO Tilt Roll/Pitch (RS)")
    p.add_argument("--exposure", type=float, default=140.0)
    p.add_argument("--gain", type=float, default=16.0)
    p.add_argument("--white_balance", type=float, default=4500.0)
    p.add_argument("--auto_exposure", action="store_true")
    p.add_argument("--auto_white_balance", action="store_true")
    p.add_argument("--conf_thr", type=float, default=0.55)
    p.add_argument("--min_area", type=int, default=700)
    p.add_argument("--border_margin_px", type=int, default=8)
    p.add_argument("--solidity_thr", type=float, default=0.90)
    p.add_argument("--completeness_thr", type=float, default=0.62)
    p.add_argument("--min_axis_ratio", type=float, default=0.18)
    p.add_argument("--max_axis_ratio", type=float, default=1.00)
    p.add_argument("--min_major_px", type=float, default=18.0)
    p.add_argument("--max_ellipse_residual", type=float, default=0.36)
    p.add_argument("--min_ellipse_iou", type=float, default=0.58)
    p.add_argument("--max_occlusion_score", type=float, default=0.48)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "video" and not args.video:
        raise RuntimeError("--video is required when --source video")
    if args.source == "images" and not args.image_dir:
        raise RuntimeError("--image_dir is required when --source images")
    if args.roi_yaml and not os.path.exists(args.roi_yaml):
        raise RuntimeError(f"ROI file not found: {args.roi_yaml}")

    model = YOLO(args.model)
    print(f"[Info] model: {args.model}")
    if args.source == "rs":
        print(
            f"[Info] RealSense color config: {args.width}x{args.height}@{args.fps} "
            f"exp={args.exposure} gain={args.gain} wb={args.white_balance} "
            f"auto_exp={args.auto_exposure} auto_wb={args.auto_white_balance}"
        )

    if args.source == "rs":
        frames = source_realsense(
            width=args.width,
            height=args.height,
            fps=args.fps,
            exposure=args.exposure,
            gain=args.gain,
            white_balance=args.white_balance,
            auto_exposure=args.auto_exposure,
            auto_white_balance=args.auto_white_balance,
        )
    elif args.source == "webcam":
        frames = source_webcam(args.webcam_index)
    elif args.source == "video":
        frames = source_video(args.video)
    else:
        frames = source_images(args.image_dir)

    run_pose_loop(args, model, frames)


if __name__ == "__main__":
    main()

