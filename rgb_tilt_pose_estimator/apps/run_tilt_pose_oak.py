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
from rgb_tilt_pose_estimator.sources.source_oak import source_oak


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robust RGB-only roll/pitch estimator from YOLOv8 segmentation masks (OAK-D)."
    )
    p.add_argument("--model", type=str, default="runs/segment/Imported/best.pt")
    p.add_argument("--source", type=str, default="oak", choices=["oak"])
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--oak_socket", type=str, default="CAM_A", help="OAK color socket: CAM_A/RGB.")
    p.add_argument("--oak_sensor_resolution", type=str, default="1080p", choices=["720p", "1080p", "4k", "12mp"])
    p.add_argument("--oak_warmup_frames", type=int, default=15)
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
    p.add_argument("--window", type=str, default="YOLO Tilt Roll/Pitch (OAK)")
    p.add_argument("--auto_exposure", action="store_true")
    p.add_argument("--auto_white_balance", action="store_true")
    p.add_argument("--exposure_us", type=float, default=10000.0, help="OAK manual exposure in microseconds.")
    p.add_argument("--iso", type=float, default=800.0, help="OAK manual ISO value.")
    p.add_argument("--white_balance", type=float, default=4500.0)
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
    if args.roi_yaml and not os.path.exists(args.roi_yaml):
        raise RuntimeError(f"ROI file not found: {args.roi_yaml}")

    model = YOLO(args.model)
    print(f"[Info] model: {args.model}")
    print(
        f"[Info] OAK color config: {args.width}x{args.height}@{args.fps} "
        f"socket={args.oak_socket} res={args.oak_sensor_resolution} "
        f"auto_exp={args.auto_exposure} exp_us={args.exposure_us} iso={args.iso} "
        f"auto_wb={args.auto_white_balance} wb={args.white_balance}"
    )

    frames = source_oak(
        width=args.width,
        height=args.height,
        fps=float(args.fps),
        socket=args.oak_socket,
        sensor_resolution=args.oak_sensor_resolution,
        warmup_frames=int(args.oak_warmup_frames),
        auto_exposure=bool(args.auto_exposure),
        exposure_us=float(args.exposure_us),
        iso=float(args.iso),
        auto_white_balance=bool(args.auto_white_balance),
        white_balance=float(args.white_balance),
    )

    run_pose_loop(args, model, frames)


if __name__ == "__main__":
    main()

