import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from estimator import EstimatorConfig, estimate_tilt_pose_from_mask
from run_yolo_tilt_pose import (
    PoseTracker,
    load_roi_yaml,
    make_mask_from_polygon,
    normal_to_pose,
    suppress_overlaps,
)

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


def _configure_realsense_color_sensor(
    profile,
    exposure: float,
    gain: float,
    white_balance: float,
    auto_exposure: bool,
    auto_white_balance: bool,
) -> None:
    dev = profile.get_device()
    sensors = dev.query_sensors()
    color_sensor = None
    for s in sensors:
        name = s.get_info(rs.camera_info.name).lower()
        if "rgb" in name or "color" in name:
            color_sensor = s
            break
    if color_sensor is None:
        return
    if color_sensor.supports(rs.option.enable_auto_exposure):
        color_sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
    if color_sensor.supports(rs.option.exposure):
        color_sensor.set_option(rs.option.exposure, float(exposure))
    if color_sensor.supports(rs.option.gain):
        color_sensor.set_option(rs.option.gain, float(gain))
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0 if auto_white_balance else 0.0)
    if color_sensor.supports(rs.option.white_balance):
        color_sensor.set_option(rs.option.white_balance, float(white_balance))


def _depth_at_pixel_m(
    depth_u16: np.ndarray,
    u: float,
    v: float,
    depth_scale: float,
    window_radius: int = 2,
) -> Optional[float]:
    h, w = depth_u16.shape[:2]
    ui = int(round(u))
    vi = int(round(v))
    if ui < 0 or ui >= w or vi < 0 or vi >= h:
        return None
    r = max(0, int(window_radius))
    x0 = max(0, ui - r)
    y0 = max(0, vi - r)
    x1 = min(w, ui + r + 1)
    y1 = min(h, vi + r + 1)
    patch = depth_u16[y0:y1, x0:x1].astype(np.float32)
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    return float(np.median(patch) * depth_scale)


def _estimate_rim_depth_m(
    det_result: Dict,
    depth_u16: np.ndarray,
    depth_scale: float,
    x_off: int,
    y_off: int,
    sample_win: int,
) -> Optional[float]:
    vals: List[float] = []
    ellipse = det_result.get("ellipse", None)
    if ellipse is not None:
        (ecx, ecy), (d1, d2), ang = ellipse
        a = max(float(d1) * 0.5, 2.0)
        b = max(float(d2) * 0.5, 2.0)
        th = np.deg2rad(float(ang))
        ct, st = np.cos(th), np.sin(th)
        ts = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False, dtype=np.float64)
        for t in ts:
            ex = a * np.cos(t)
            ey = b * np.sin(t)
            u = float(ecx + ex * ct - ey * st + x_off)
            v = float(ecy + ex * st + ey * ct + y_off)
            z = _depth_at_pixel_m(depth_u16, u, v, depth_scale, window_radius=sample_win)
            if z is not None:
                vals.append(float(z))
    if not vals:
        contour = det_result.get("contour", None)
        if contour is not None and len(contour) >= 8:
            pts = contour.reshape(-1, 2).astype(np.float32)
            if pts.shape[0] > 64:
                idx = np.linspace(0, pts.shape[0] - 1, 64, dtype=np.int32)
                pts = pts[idx]
            for p in pts:
                u = float(p[0] + x_off)
                v = float(p[1] + y_off)
                z = _depth_at_pixel_m(depth_u16, u, v, depth_scale, window_radius=sample_win)
                if z is not None:
                    vals.append(float(z))
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _choose_axis_sign_from_depth(
    base_normal: np.ndarray,
    det_result: Dict,
    depth_u16: np.ndarray,
    depth_scale: float,
    x_off: int,
    y_off: int,
    depth_win: int,
    depth_diff_m: float,
    xy_min: float,
    vertical_z_thr: float,
) -> Tuple[float, str]:
    n = base_normal.astype(np.float64).copy()
    pose2d = det_result.get("pose2d", None)
    if pose2d is None:
        return 1.0, "no_pose2d"
    uc = float(pose2d[0] + x_off)
    vc = float(pose2d[1] + y_off)

    zc = _depth_at_pixel_m(depth_u16, uc, vc, depth_scale, window_radius=depth_win)
    zr = _estimate_rim_depth_m(
        det_result=det_result,
        depth_u16=depth_u16,
        depth_scale=depth_scale,
        x_off=int(x_off),
        y_off=int(y_off),
        sample_win=int(depth_win),
    )
    center_rim_sign: Optional[float] = None
    if zc is not None and zr is not None and abs(float(zc - zr)) >= float(depth_diff_m):
        # center deeper than rim => hollow/open side faces camera => axis should point toward camera (negative z).
        want_neg_z = float(zc) > float(zr)
        if want_neg_z:
            center_rim_sign = (-1.0 if float(n[2]) > 0.0 else 1.0)
        else:
            center_rim_sign = (1.0 if float(n[2]) > 0.0 else -1.0)

    # Near-vertical caps: center-rim is typically more reliable than endpoint depth.
    if center_rim_sign is not None and abs(float(n[2])) >= float(vertical_z_thr):
        return float(center_rim_sign), "center_rim_vertical"

    nxy = np.asarray([n[0], n[1]], dtype=np.float64)
    nxy_norm = float(np.linalg.norm(nxy))
    if nxy_norm >= float(xy_min):
        dxy = nxy / nxy_norm
        major_px = float(det_result.get("metrics", {}).get("major_px", 0.0))
        step_px = float(np.clip(0.35 * max(major_px, 8.0), 6.0, 50.0))
        zp = _depth_at_pixel_m(
            depth_u16,
            uc + dxy[0] * step_px,
            vc + dxy[1] * step_px,
            depth_scale,
            window_radius=depth_win,
        )
        zn = _depth_at_pixel_m(
            depth_u16,
            uc - dxy[0] * step_px,
            vc - dxy[1] * step_px,
            depth_scale,
            window_radius=depth_win,
        )
        if zp is not None and zn is not None and abs(float(zp - zn)) >= float(depth_diff_m):
            # Assume open side appears deeper.
            return (1.0 if float(zp) > float(zn) else -1.0), "end_depth"

    if center_rim_sign is not None:
        return float(center_rim_sign), "center_rim_fallback"

    return 1.0, "default"


def _estimate_plane_up_from_roi(
    depth_u16: np.ndarray,
    depth_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    roi_xyxy: Tuple[int, int, int, int],
    object_mask_full: np.ndarray,
    stride: int,
    z_min: float,
    z_max: float,
    min_pts: int = 250,
) -> Optional[np.ndarray]:
    x0, y0, x1, y1 = [int(v) for v in roi_xyxy]
    if x1 <= x0 or y1 <= y0:
        return None
    s = max(1, int(stride))
    us = np.arange(x0, x1, s, dtype=np.int32)
    vs = np.arange(y0, y1, s, dtype=np.int32)
    if us.size == 0 or vs.size == 0:
        return None
    uu, vv = np.meshgrid(us, vs)
    u = uu.reshape(-1)
    v = vv.reshape(-1)
    keep_bg = object_mask_full[v, u] == 0
    if not np.any(keep_bg):
        return None
    u = u[keep_bg]
    v = v[keep_bg]
    z_raw = depth_u16[v, u].astype(np.float32)
    z_m = z_raw * float(depth_scale)
    valid = (z_m > 0.0) & (z_m >= float(z_min)) & (z_m <= float(z_max))
    if int(np.count_nonzero(valid)) < int(min_pts):
        return None
    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z_m[valid].astype(np.float32)
    x = (u - float(cx)) * z / float(fx)
    y = (v - float(cy)) * z / float(fy)
    pts = np.stack([x, y, z], axis=1).astype(np.float64)
    if pts.shape[0] < int(min_pts):
        return None

    c = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    n = vh[-1].astype(np.float64)
    # Orient first normal toward camera (camera at origin).
    if float(np.dot(n, -c)) < 0.0:
        n *= -1.0
    d = np.abs((pts - c) @ n)
    inliers = d <= 0.012  # 12 mm tolerance for bin plane noise.
    if int(np.count_nonzero(inliers)) >= int(min_pts):
        pts_i = pts[inliers]
        c = np.mean(pts_i, axis=0)
        _, _, vh = np.linalg.svd(pts_i - c, full_matrices=False)
        n = vh[-1].astype(np.float64)
        if float(np.dot(n, -c)) < 0.0:
            n *= -1.0
    # Up direction is opposite of camera-facing plane normal.
    up = -n
    up_n = float(np.linalg.norm(up))
    if up_n <= 1e-8:
        return None
    return (up / up_n).astype(np.float32)


def _pixel_to_point(u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (float(u) - cx) * z_m / fx
    y = (float(v) - cy) * z_m / fy
    z = z_m
    return np.array([x, y, z], dtype=np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v.astype(np.float64)
    return (v / n).astype(np.float64)


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = _unit(normal)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ex = _unit(np.cross(ref, n))
    ey = _unit(np.cross(n, ex))
    return ex, ey


def _transform_from_center_normal(center: np.ndarray, normal: np.ndarray) -> np.ndarray:
    z_axis = _unit(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = ref - np.dot(ref, z_axis) * z_axis
    if float(np.linalg.norm(x_axis)) < 1e-6:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = ref - np.dot(ref, z_axis) * z_axis
    x_axis = _unit(x_axis)
    y_axis = _unit(np.cross(z_axis, x_axis))
    x_axis = _unit(np.cross(y_axis, z_axis))
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    t[:3, 3] = center.astype(np.float64)
    return t


def _color_from_track_id(tid: int) -> np.ndarray:
    hue = float((int(tid) * 57) % 360) / 360.0
    s = 0.95
    v = 1.0
    h6 = hue * 6.0
    i = int(np.floor(h6)) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0:
        rgb = np.array([v, t, p], dtype=np.float64)
    elif i == 1:
        rgb = np.array([q, v, p], dtype=np.float64)
    elif i == 2:
        rgb = np.array([p, v, t], dtype=np.float64)
    elif i == 3:
        rgb = np.array([p, q, v], dtype=np.float64)
    elif i == 4:
        rgb = np.array([t, p, v], dtype=np.float64)
    else:
        rgb = np.array([v, p, q], dtype=np.float64)
    return rgb


def _make_normal_ray(center: np.ndarray, normal: np.ndarray, length_m: float, color_rgb: np.ndarray):
    c0 = center.astype(np.float64)
    c1 = (center + _unit(normal) * float(length_m)).astype(np.float64)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.stack([c0, c1], axis=0))
    ls.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(color_rgb.reshape(1, 3).astype(np.float64))
    return ls


def _make_pose_ring(center: np.ndarray, normal: np.ndarray, radius_m: float, seg: int, color_rgb: np.ndarray):
    ex, ey = _plane_basis(normal)
    seg = max(24, int(seg))
    a = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False, dtype=np.float64)
    ring = (
        center.reshape(1, 3)
        + float(radius_m) * np.cos(a)[:, None] * ex.reshape(1, 3)
        + float(radius_m) * np.sin(a)[:, None] * ey.reshape(1, 3)
    )
    lines = [[i, (i + 1) % seg] for i in range(seg)]
    cols = np.tile(color_rgb.reshape(1, 3), (seg, 1)).astype(np.float64)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(ring.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls


def _build_point_cloud_from_depth(
    depth_u16: np.ndarray,
    color_bgr: np.ndarray,
    depth_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int,
    z_min: float,
    z_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stride = max(1, int(stride))
    h, w = depth_u16.shape[:2]
    us = np.arange(0, w, stride, dtype=np.float32)
    vs = np.arange(0, h, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    u = uu.reshape(-1)
    v = vv.reshape(-1)

    z_raw = depth_u16[0:h:stride, 0:w:stride].reshape(-1).astype(np.float32)
    z_m = z_raw * float(depth_scale)
    valid = (z_m > 0.0) & (z_m >= float(z_min)) & (z_m <= float(z_max))
    if np.count_nonzero(valid) < 10:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    u = u[valid]
    v = v[valid]
    z = z_m[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    # Convert BGR->RGB for Open3D color convention.
    rgb = color_bgr[:, :, ::-1]
    ui = np.clip(np.rint(u).astype(np.int32), 0, w - 1)
    vi = np.clip(np.rint(v).astype(np.int32), 0, h - 1)
    cols = rgb[vi, ui, :].astype(np.float32) / 255.0
    return pts, cols, ui, vi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO mask + RGB-D point cloud visualization with 3D normal vectors in Open3D."
    )
    p.add_argument("--model", type=str, default="runs/segment/Imported/best.pt")
    p.add_argument("--roi_yaml", type=str, default="segmentation_bgsub_yolo/config/roi.yaml")
    p.add_argument("--show_roi", action="store_true")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max_det", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--exposure", type=float, default=140.0)
    p.add_argument("--gain", type=float, default=16.0)
    p.add_argument("--white_balance", type=float, default=4500.0)
    p.add_argument("--auto_exposure", action="store_true")
    p.add_argument("--auto_white_balance", action="store_true")
    p.add_argument("--pc_stride", type=int, default=2)
    p.add_argument("--z_min", type=float, default=0.10)
    p.add_argument("--z_max", type=float, default=1.00)
    p.add_argument("--vector_len_m", type=float, default=0.04)
    p.add_argument("--center_depth_win", type=int, default=2)
    p.add_argument("--flip_normal", action="store_true")
    p.add_argument("--only_best", action="store_true")
    p.add_argument("--show_2d", dest="show_2d", action="store_true")
    p.add_argument("--no_show_2d", dest="show_2d", action="store_false")
    p.set_defaults(show_2d=True)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--overlap_suppress_ratio", type=float, default=0.35)
    p.add_argument("--track_alpha", type=float, default=0.40)
    p.add_argument("--track_max_dist_px", type=float, default=75.0)
    p.add_argument("--track_max_missed", type=int, default=10)
    p.add_argument("--track_min_hits", type=int, default=2)
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
    p.add_argument("--pc_bg_dim", type=float, default=0.15, help="background point cloud dim factor [0..1]")
    p.add_argument("--pc_axis_size", type=float, default=0.012)
    p.add_argument("--pc_ring_segments", type=int, default=64)
    p.add_argument("--pc_center_radius", type=float, default=0.003)
    p.add_argument(
        "--pc_ring_radius_m",
        type=float,
        default=0.01465,
        help="fallback ring radius in meters when depth-size estimate is unstable",
    )
    p.add_argument("--axis_from_depth", dest="axis_from_depth", action="store_true")
    p.add_argument("--no_axis_from_depth", dest="axis_from_depth", action="store_false")
    p.set_defaults(axis_from_depth=True)
    p.add_argument("--axis_depth_diff_mm", type=float, default=1.2)
    p.add_argument("--axis_xy_min", type=float, default=0.12)
    p.add_argument("--axis_vertical_z_thr", type=float, default=0.86)
    p.add_argument("--axis_up_axis_ratio_thr", type=float, default=0.78)
    p.add_argument("--plane_stride", type=int, default=6)
    p.add_argument("--axis_up_margin", type=float, default=0.02)
    p.add_argument(
        "--axis_sign_mode",
        type=str,
        default="auto",
        choices=["auto", "toward_camera", "away_camera", "up_from_plane"],
        help="final axis sign policy after depth cue; auto prefers plane-up for near-circular top views",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if o3d is None:
        raise RuntimeError("open3d is not available. Install open3d in your environment.")
    if rs is None:
        raise RuntimeError("pyrealsense2 is not available. Install pyrealsense2 in your environment.")
    if args.roi_yaml and not os.path.exists(args.roi_yaml):
        raise RuntimeError(f"ROI file not found: {args.roi_yaml}")

    model = YOLO(args.model)
    print(f"[Info] model: {args.model}")
    print(
        f"[Info] stream color+depth={args.width}x{args.height}@{args.fps} "
        f"exp={args.exposure} gain={args.gain} wb={args.white_balance} "
        f"auto_exp={args.auto_exposure} auto_wb={args.auto_white_balance}"
    )

    cfg = EstimatorConfig(
        conf_thr=float(args.conf_thr),
        min_area=int(args.min_area),
        border_margin_px=int(args.border_margin_px),
        solidity_thr=float(args.solidity_thr),
        completeness_thr=float(args.completeness_thr),
        min_axis_ratio=float(args.min_axis_ratio),
        max_axis_ratio=float(args.max_axis_ratio),
        min_major_px=float(args.min_major_px),
        max_ellipse_residual=float(args.max_ellipse_residual),
        min_ellipse_iou=float(args.min_ellipse_iou),
        max_occlusion_score=float(args.max_occlusion_score),
    )
    tracker = PoseTracker(
        alpha=float(args.track_alpha),
        max_dist_px=float(args.track_max_dist_px),
        max_missed=int(args.track_max_missed),
        min_hits=int(args.track_min_hits),
    )

    pipe = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    rs_cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipe.start(rs_cfg)
    _configure_realsense_color_sensor(
        profile,
        exposure=float(args.exposure),
        gain=float(args.gain),
        white_balance=float(args.white_balance),
        auto_exposure=bool(args.auto_exposure),
        auto_white_balance=bool(args.auto_white_balance),
    )
    align = rs.align(rs.stream.color)

    color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_prof.get_intrinsics()
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"[Info] intr fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f} depth_scale={depth_scale:.6f}")

    vis = o3d.visualization.Visualizer()
    vis.create_window("YOLO Tilt Vector in PointCloud", width=1280, height=780, visible=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
    vis.add_geometry(pcd)

    pose_geoms: List[object] = []

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0.0, 0.0, 0.0])
    vis.add_geometry(coord)
    vis.get_render_option().point_size = 2.0

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    roi_xyxy: Optional[Tuple[int, int, int, int]] = None
    first_view = True
    frame_idx = 0
    try:
        while True:
            frame_idx += 1
            frames = pipe.wait_for_frames(5000)
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_u16 = np.asanyarray(depth_frame.get_data())
            h, w = color_bgr.shape[:2]

            if args.roi_yaml:
                if roi_xyxy is None:
                    roi_xyxy = load_roi_yaml(args.roi_yaml, w, h)
                    print(f"[Info] ROI loaded: x={roi_xyxy[0]}:{roi_xyxy[2]} y={roi_xyxy[1]}:{roi_xyxy[3]}")
                x0, y0, x1, y1 = roi_xyxy
            else:
                x0, y0, x1, y1 = 0, 0, w, h
            roi = color_bgr[y0:y1, x0:x1]

            pred = model.predict(
                source=roi,
                conf=float(args.conf),
                iou=float(args.iou),
                imgsz=int(args.imgsz),
                max_det=int(args.max_det),
                device=args.device,
                verbose=False,
            )[0]

            raw_candidates: List[Dict] = []
            if pred.masks is not None and pred.boxes is not None and len(pred.masks.xy) > 0:
                polys = pred.masks.xy
                confs = pred.boxes.conf.cpu().numpy() if pred.boxes.conf is not None else np.zeros((len(polys),), dtype=np.float32)
                for i, poly in enumerate(polys):
                    if poly is None or len(poly) < 3:
                        continue
                    mask = make_mask_from_polygon(poly, roi.shape[0], roi.shape[1])
                    area = int(np.count_nonzero(mask))
                    if area <= 0:
                        continue
                    raw_candidates.append(
                        {
                            "poly": np.round(poly).astype(np.int32),
                            "mask": mask,
                            "area": area,
                            "confidence": float(confs[i]),
                        }
                    )

            candidates = suppress_overlaps(raw_candidates, ratio_thr=float(args.overlap_suppress_ratio))
            object_mask_full = np.zeros((h, w), dtype=np.uint8)
            if len(candidates) > 0:
                roi_slice = object_mask_full[y0:y1, x0:x1]
                for c in candidates:
                    roi_slice[c["mask"] > 0] = 255
            plane_up = _estimate_plane_up_from_roi(
                depth_u16=depth_u16,
                depth_scale=depth_scale,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                roi_xyxy=(x0, y0, x1, y1),
                object_mask_full=object_mask_full,
                stride=int(args.plane_stride),
                z_min=float(args.z_min),
                z_max=float(args.z_max),
            )
            valid_dets: List[Dict] = []
            for c in candidates:
                est = estimate_tilt_pose_from_mask(c["mask"], c["confidence"], cfg=cfg, K=None)
                if not est.get("valid", False):
                    continue
                valid_dets.append(
                    {
                        "center": np.asarray(est["pose2d"], dtype=np.float32),
                        "normal": np.asarray(est["normal_vec"], dtype=np.float32),
                        "confidence": float(c["confidence"]),
                        "metrics": dict(est.get("metrics", {})),
                        "result": est,
                    }
                )

            det_track_ids = tracker.update(valid_dets)

            # 2D debug overlay.
            out = color_bgr.copy()
            if args.show_roi and roi_xyxy is not None:
                cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 1, cv2.LINE_AA)

            vec_records: List[Tuple[float, int, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, str]] = []
            stable_instances: List[Dict] = []
            for det, tid in zip(valid_dets, det_track_ids):
                tr = tracker.tracks.get(tid)
                if tr is None or tr.missed > 0:
                    continue
                if tr.hits < tracker.min_hits:
                    continue
                u = float(tr.center[0] + x0)
                v = float(tr.center[1] + y0)
                z_m = _depth_at_pixel_m(
                    depth_u16=depth_u16,
                    u=u,
                    v=v,
                    depth_scale=depth_scale,
                    window_radius=int(args.center_depth_win),
                )
                if z_m is None or z_m < float(args.z_min) or z_m > float(args.z_max):
                    continue

                p0 = _pixel_to_point(u, v, z_m, fx, fy, cx, cy)
                n = tr.normal.astype(np.float32).copy()
                n_norm = float(np.linalg.norm(n))
                if n_norm <= 1e-8:
                    continue
                n /= n_norm
                if bool(args.axis_from_depth):
                    sign_scale, sign_reason = _choose_axis_sign_from_depth(
                        base_normal=n,
                        det_result=det["result"],
                        depth_u16=depth_u16,
                        depth_scale=depth_scale,
                        x_off=int(x0),
                        y_off=int(y0),
                        depth_win=int(args.center_depth_win),
                        depth_diff_m=float(args.axis_depth_diff_mm) * 1e-3,
                        xy_min=float(args.axis_xy_min),
                        vertical_z_thr=float(args.axis_vertical_z_thr),
                    )
                    n *= float(sign_scale)
                else:
                    sign_reason = "axis_from_depth_off"
                sign_mode = str(args.axis_sign_mode)
                axis_ratio = float(det["metrics"].get("axis_ratio", 0.0))
                if sign_mode == "toward_camera" and float(n[2]) > 0.0:
                    n *= -1.0
                    sign_reason = f"{sign_reason}+prior_cam"
                elif sign_mode == "away_camera" and float(n[2]) < 0.0:
                    n *= -1.0
                    sign_reason = f"{sign_reason}+prior_away"
                else:
                    use_plane = False
                    if plane_up is not None:
                        if sign_mode == "up_from_plane":
                            use_plane = True
                        elif sign_mode == "auto":
                            # For near-circular top views prefer bin-plane up direction.
                            if axis_ratio >= float(args.axis_up_axis_ratio_thr):
                                use_plane = True
                    if use_plane:
                        dcur = float(np.dot(n, plane_up))
                        if dcur < -float(args.axis_up_margin):
                            n *= -1.0
                            sign_reason = f"{sign_reason}+plane_up"
                        else:
                            sign_reason = f"{sign_reason}+plane_keep"
                if args.flip_normal:
                    n *= -1.0
                    sign_reason = f"{sign_reason}+manual_flip"
                p1 = p0 + n * float(args.vector_len_m)
                major_px = float(det["metrics"].get("major_px", 0.0))
                radius_from_depth = 0.5 * major_px * z_m / max(1e-6, 0.5 * (fx + fy))
                if (not np.isfinite(radius_from_depth)) or radius_from_depth < 0.003 or radius_from_depth > 0.04:
                    radius_from_depth = float(args.pc_ring_radius_m)

                roll, pitch, tilt, _ = normal_to_pose(n)
                txt = f"#{tid} r={roll:+.1f} p={pitch:+.1f} t={tilt:.1f}"
                if bool(args.debug):
                    txt = f"{txt} [{sign_reason}]"
                color_rgb = _color_from_track_id(tid)
                color = (int(round(color_rgb[2] * 255.0)), int(round(color_rgb[1] * 255.0)), int(round(color_rgb[0] * 255.0)))
                stable_instances.append(
                    {
                        "tid": tid,
                        "confidence": float(tr.confidence),
                        "mask_roi": det["result"]["mask"].copy(),
                        "color_bgr": color,
                    }
                )
                vec_records.append((tr.confidence, tid, p0, p1, n.copy(), float(radius_from_depth), color_rgb.copy(), txt))

            if args.only_best and len(vec_records) > 1:
                vec_records = [max(vec_records, key=lambda z: z[0])]

            # Update Open3D point cloud.
            pts, cols, ui, vi = _build_point_cloud_from_depth(
                depth_u16=depth_u16,
                color_bgr=color_bgr,
                depth_scale=depth_scale,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                stride=int(args.pc_stride),
                z_min=float(args.z_min),
                z_max=float(args.z_max),
            )
            if pts.shape[0] > 0:
                id_map = np.zeros((h, w), dtype=np.int32)
                # Highest-confidence instance wins if masks overlap.
                stable_sorted = sorted(stable_instances, key=lambda z: z["confidence"], reverse=True)
                id_to_rgb: Dict[int, np.ndarray] = {}
                for inst_idx, inst in enumerate(stable_sorted, start=1):
                    mr = inst["mask_roi"]
                    if mr is None or mr.size == 0:
                        continue
                    roi_slice = id_map[y0:y1, x0:x1]
                    roi_slice[mr > 0] = int(inst_idx)
                    c = inst["color_bgr"]
                    id_to_rgb[int(inst_idx)] = np.array(
                        [float(c[2]) / 255.0, float(c[1]) / 255.0, float(c[0]) / 255.0], dtype=np.float32
                    )

                if np.max(id_map) > 0 and ui.size > 0:
                    point_ids = id_map[vi, ui]
                    cols_annot = cols.copy()
                    bg_dim = float(np.clip(args.pc_bg_dim, 0.0, 1.0))
                    cols_annot[point_ids == 0] *= bg_dim
                    for inst_id, rgb_col in id_to_rgb.items():
                        cols_annot[point_ids == inst_id] = rgb_col
                    cols = cols_annot

                pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
                vis.update_geometry(pcd)

            for g in pose_geoms:
                vis.remove_geometry(g, reset_bounding_box=False)
            pose_geoms = []
            for _, tid, p0, _, n, ring_r, color_rgb, _ in vec_records:
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.pc_axis_size))
                axis.transform(_transform_from_center_normal(p0.astype(np.float64), n.astype(np.float64)))
                ring = _make_pose_ring(
                    center=p0.astype(np.float64),
                    normal=n.astype(np.float64),
                    radius_m=float(ring_r),
                    seg=int(args.pc_ring_segments),
                    color_rgb=color_rgb.astype(np.float64),
                )
                ray = _make_normal_ray(
                    center=p0.astype(np.float64),
                    normal=n.astype(np.float64),
                    length_m=float(args.vector_len_m),
                    color_rgb=color_rgb.astype(np.float64),
                )
                center_dot = o3d.geometry.TriangleMesh.create_sphere(radius=float(args.pc_center_radius))
                center_dot.paint_uniform_color(color_rgb.astype(np.float64).tolist())
                center_dot.translate(p0.astype(np.float64), relative=False)
                vis.add_geometry(axis, reset_bounding_box=False)
                vis.add_geometry(ring, reset_bounding_box=False)
                vis.add_geometry(ray, reset_bounding_box=False)
                vis.add_geometry(center_dot, reset_bounding_box=False)
                pose_geoms.extend([axis, ring, ray, center_dot])

            if first_view and pts.shape[0] > 100:
                vis.reset_view_point(True)
                vc = vis.get_view_control()
                look = np.median(pts, axis=0)
                vc.set_lookat([float(look[0]), float(look[1]), float(look[2])])
                vc.set_front([0.0, 0.0, -1.0])
                vc.set_up([0.0, -1.0, 0.0])
                vc.set_zoom(0.70)
                first_view = False

            # 2D annotations for convenience.
            for _, tid, p0, p1, _, _, color_rgb, txt in vec_records:
                color = (int(round(color_rgb[2] * 255.0)), int(round(color_rgb[1] * 255.0)), int(round(color_rgb[0] * 255.0)))
                # Project with pinhole for display consistency.
                u0 = int(round((p0[0] * fx / max(p0[2], 1e-8)) + cx))
                v0 = int(round((p0[1] * fy / max(p0[2], 1e-8)) + cy))
                u1 = int(round((p1[0] * fx / max(p1[2], 1e-8)) + cx))
                v1 = int(round((p1[1] * fy / max(p1[2], 1e-8)) + cy))
                cv2.circle(out, (u0, v0), 4, (0, 255, 0), -1)
                cv2.arrowedLine(out, (u0, v0), (u1, v1), (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.16)
                cv2.putText(out, txt, (max(10, u0 - 20), max(20, v0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
                if args.debug:
                    tr = tracker.tracks.get(tid)
                    if tr is not None:
                        m = tr.metrics
                        dbg = (
                            f"sol={m.get('solidity', 0):.2f} comp={m.get('completeness', 0):.2f} "
                            f"iouE={m.get('ellipse_iou', 0):.2f} occ={m.get('occlusion_score', 1):.2f}"
                        )
                        cv2.putText(out, dbg, (max(10, u0 - 20), min(h - 10, max(20, v0 + 12))), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

            stable_tracks = sum(1 for t in tracker.tracks.values() if t.missed == 0 and t.hits >= tracker.min_hits)
            cv2.putText(
                out,
                f"frame={frame_idx} det={len(raw_candidates)} keep={len(candidates)} stable={stable_tracks} vectors={len(vec_records)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if args.debug:
                if plane_up is None:
                    ptxt = "plane_up: none"
                else:
                    ptxt = f"plane_up=({plane_up[0]:+.2f},{plane_up[1]:+.2f},{plane_up[2]:+.2f}) mode={args.axis_sign_mode}"
                cv2.putText(out, ptxt, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

            vis.poll_events()
            vis.update_renderer()

            if args.save_dir:
                cv2.imwrite(os.path.join(args.save_dir, f"o3d_pose_{frame_idx:06d}.png"), out)
            if args.show_2d:
                cv2.imshow("YOLO Tilt 2D Overlay", out)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
            else:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()


if __name__ == "__main__":
    main()
