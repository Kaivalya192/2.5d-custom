import argparse
import os
import time
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def configure_color_sensor(device, exposure: float, gain: float, white_balance: float) -> None:
    sensors = device.query_sensors()
    color_sensor = None
    for s in sensors:
        name = s.get_info(rs.camera_info.name).lower()
        if "rgb" in name or "color" in name:
            color_sensor = s
            break
    if color_sensor is None:
        raise RuntimeError("Color sensor not found on device.")
    if color_sensor.supports(rs.option.enable_auto_exposure):
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    if color_sensor.supports(rs.option.exposure):
        color_sensor.set_option(rs.option.exposure, float(exposure))
    if color_sensor.supports(rs.option.gain):
        color_sensor.set_option(rs.option.gain, float(gain))
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
    if color_sensor.supports(rs.option.white_balance):
        color_sensor.set_option(rs.option.white_balance, float(white_balance))


def create_pipeline(width: int, height: int, fps: int):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipe.start(cfg)
    return pipe, profile


def get_color_frame(pipe: rs.pipeline, timeout_ms: int = 5000) -> Optional[np.ndarray]:
    frames = pipe.wait_for_frames(timeout_ms)
    color = frames.get_color_frame()
    if not color:
        return None
    return np.asanyarray(color.get_data())


def capture_background(
    pipe: rs.pipeline,
    out_dir: str,
    num_frames: int,
    warmup_frames: int,
    save_individual: bool,
) -> None:
    ensure_dir(out_dir)
    for _ in range(max(0, warmup_frames)):
        _ = get_color_frame(pipe)

    acc = None
    captured = 0
    while captured < num_frames:
        frame = get_color_frame(pipe)
        if frame is None:
            continue
        frame_f = frame.astype(np.float32)
        if acc is None:
            acc = np.zeros_like(frame_f)
        acc += frame_f
        captured += 1
        if save_individual:
            p = os.path.join(out_dir, f"background_{captured:06d}.png")
            cv2.imwrite(p, frame)
        disp = frame.copy()
        cv2.putText(
            disp,
            f"Background capture: {captured}/{num_frames}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Two-Side Capture", disp)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

    if captured == 0:
        raise RuntimeError("No background frames captured.")
    mean_img = np.clip(acc / float(captured), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "background_mean.png"), mean_img)
    np.save(os.path.join(out_dir, "background_mean.npy"), mean_img.astype(np.float32))
    print(f"[OK] Saved background mean from {captured} frames to {out_dir}")


def capture_objects(
    pipe: rs.pipeline,
    out_dir: str,
    prefix: str,
    num_frames: int,
    warmup_frames: int,
) -> None:
    ensure_dir(out_dir)
    for _ in range(max(0, warmup_frames)):
        _ = get_color_frame(pipe)

    saved = 0
    print("[Info] Press 's' to save frame, 'q' to quit.")
    while saved < num_frames:
        frame = get_color_frame(pipe)
        if frame is None:
            continue
        disp = frame.copy()
        cv2.putText(
            disp,
            f"{prefix} capture: {saved}/{num_frames}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Two-Side Capture", disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            saved += 1
            path = os.path.join(out_dir, f"{prefix}_{saved:06d}.png")
            cv2.imwrite(path, frame)
            print(f"[Saved] {path}")
    print(f"[OK] Saved {saved} {prefix} images to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture background/front/back images for two-side YOLO segmentation.")
    sub = parser.add_subparsers(dest="mode", required=True)

    for mode in ("background", "front", "back"):
        sp = sub.add_parser(mode)
        sp.add_argument("--out_dir", type=str, required=True)
        sp.add_argument("--num_frames", type=int, default=30 if mode == "background" else 20)
        sp.add_argument("--width", type=int, default=1280)
        sp.add_argument("--height", type=int, default=720)
        sp.add_argument("--fps", type=int, default=30)
        sp.add_argument("--warmup_frames", type=int, default=20)
        sp.add_argument("--exposure", type=float, default=140.0)
        sp.add_argument("--gain", type=float, default=16.0)
        sp.add_argument("--white_balance", type=float, default=4500.0)
    sub.choices["background"].add_argument("--save_individual", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pipe = None
    try:
        pipe, profile = create_pipeline(args.width, args.height, args.fps)
        configure_color_sensor(profile.get_device(), args.exposure, args.gain, args.white_balance)
        time.sleep(0.3)
        if args.mode == "background":
            capture_background(
                pipe=pipe,
                out_dir=args.out_dir,
                num_frames=args.num_frames,
                warmup_frames=args.warmup_frames,
                save_individual=bool(args.save_individual),
            )
        elif args.mode == "front":
            capture_objects(pipe=pipe, out_dir=args.out_dir, prefix="front", num_frames=args.num_frames, warmup_frames=args.warmup_frames)
        else:
            capture_objects(pipe=pipe, out_dir=args.out_dir, prefix="back", num_frames=args.num_frames, warmup_frames=args.warmup_frames)
    finally:
        if pipe is not None:
            pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

