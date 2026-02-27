import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


@dataclass
class EllipseCandidate:
    cx: float
    cy: float
    w: float
    h: float
    angle_deg: float
    score: float
    support: float
    arc_cov: float
    fit_residual: float
    completion: float
    area_ratio: float
    solidity: float
    inlier_count: int
    unique_ratio: float
    is_open_arc: bool
    contour: np.ndarray


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
        raise RuntimeError("Invalid ROI bounds.")
    return x0, y0, x1, y1


def ellipse_support_from_dist(
    dist_map: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
    samples: int,
    max_dist_px: float,
) -> float:
    n = max(36, int(samples))
    a = max(1e-3, 0.5 * float(w))
    b = max(1e-3, 0.5 * float(h))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    ang = np.deg2rad(float(angle_deg))
    ca = float(np.cos(ang))
    sa = float(np.sin(ang))
    x = a * np.cos(t)
    y = b * np.sin(t)
    u = float(cx) + x * ca - y * sa
    v = float(cy) + x * sa + y * ca
    h_img, w_img = dist_map.shape[:2]
    m = (u >= 0.0) & (u <= float(w_img - 1)) & (v >= 0.0) & (v <= float(h_img - 1))
    if int(np.count_nonzero(m)) < 16:
        return 0.0
    ui = np.clip(np.rint(u[m]).astype(np.int32), 0, w_img - 1)
    vi = np.clip(np.rint(v[m]).astype(np.int32), 0, h_img - 1)
    d = dist_map[vi, ui]
    return float(np.count_nonzero(d <= float(max_dist_px))) / float(max(1, d.size))


def contour_arc_coverage(cnt: np.ndarray, cx: float, cy: float, w: float, h: float, angle_deg: float, bins: int) -> float:
    pts = cnt.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] < 8:
        return 0.0
    a = max(1e-3, 0.5 * float(w))
    b = max(1e-3, 0.5 * float(h))
    ang = np.deg2rad(float(angle_deg))
    ca = float(np.cos(ang))
    sa = float(np.sin(ang))

    x = pts[:, 0] - float(cx)
    y = pts[:, 1] - float(cy)
    xr = x * ca + y * sa
    yr = -x * sa + y * ca
    # Normalize to ellipse coordinates so contour points map to ellipse angle.
    th = np.mod(np.arctan2(yr / b, xr / a), 2.0 * np.pi)
    bcount = max(12, int(bins))
    hist, _ = np.histogram(th, bins=bcount, range=(0.0, 2.0 * np.pi))
    return float(np.count_nonzero(hist > 0)) / float(bcount)


def contour_ellipse_residual(cnt: np.ndarray, cx: float, cy: float, w: float, h: float, angle_deg: float) -> float:
    pts = cnt.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] < 8:
        return 1.0
    a = max(1e-3, 0.5 * float(w))
    b = max(1e-3, 0.5 * float(h))
    ang = np.deg2rad(float(angle_deg))
    ca = float(np.cos(ang))
    sa = float(np.sin(ang))
    x = pts[:, 0] - float(cx)
    y = pts[:, 1] - float(cy)
    xr = x * ca + y * sa
    yr = -x * sa + y * ca
    val = (xr * xr) / (a * a) + (yr * yr) / (b * b)
    return float(np.mean(np.abs(val - 1.0)))


def ellipse_polar_value(pts_xy: np.ndarray, cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    a = max(1e-3, 0.5 * float(w))
    b = max(1e-3, 0.5 * float(h))
    ang = np.deg2rad(float(angle_deg))
    ca = float(np.cos(ang))
    sa = float(np.sin(ang))
    x = pts_xy[:, 0] - float(cx)
    y = pts_xy[:, 1] - float(cy)
    xr = x * ca + y * sa
    yr = -x * sa + y * ca
    return np.sqrt((xr * xr) / (a * a) + (yr * yr) / (b * b))


def ellipse_inlier_refit(
    edge_pts: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
    max_dist_px: float,
    iters: int,
) -> Tuple[float, float, float, float, float, np.ndarray]:
    if edge_pts.shape[0] < 16:
        return cx, cy, w, h, angle_deg, np.empty((0, 2), dtype=np.float32)
    cur = (float(cx), float(cy), float(w), float(h), float(angle_deg))
    inliers = np.empty((0, 2), dtype=np.float32)
    for _ in range(max(1, int(iters))):
        p = ellipse_polar_value(edge_pts, cur[0], cur[1], cur[2], cur[3], cur[4])
        r = 0.25 * (cur[2] + cur[3])
        d = np.abs(p - 1.0) * max(1.0, r)
        m = d <= float(max_dist_px)
        if int(np.count_nonzero(m)) < 20:
            break
        inliers = edge_pts[m].astype(np.float32)
        if inliers.shape[0] < 20:
            break
        fit_cnt = inliers.reshape(-1, 1, 2)
        try:
            (ncx, ncy), (nw, nh), nang = cv2.fitEllipseAMS(fit_cnt) if hasattr(cv2, "fitEllipseAMS") else cv2.fitEllipse(fit_cnt)
            cur = (float(ncx), float(ncy), float(nw), float(nh), float(nang))
        except cv2.error:
            break
    return cur[0], cur[1], cur[2], cur[3], cur[4], inliers


def ellipse_inlier_indices(
    edge_pts: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
    max_dist_px: float,
) -> np.ndarray:
    if edge_pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    p = ellipse_polar_value(edge_pts, cx, cy, w, h, angle_deg)
    r = 0.25 * (float(w) + float(h))
    d = np.abs(p - 1.0) * max(1.0, r)
    return np.where(d <= float(max_dist_px))[0].astype(np.int32)


def edge_completion_from_inliers(
    inliers: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
    bins: int,
) -> float:
    if inliers.shape[0] < 16:
        return 0.0
    a = max(1e-3, 0.5 * float(w))
    b = max(1e-3, 0.5 * float(h))
    ang = np.deg2rad(float(angle_deg))
    ca = float(np.cos(ang))
    sa = float(np.sin(ang))
    x = inliers[:, 0].astype(np.float64) - float(cx)
    y = inliers[:, 1].astype(np.float64) - float(cy)
    xr = x * ca + y * sa
    yr = -x * sa + y * ca
    th = np.mod(np.arctan2(yr / b, xr / a), 2.0 * np.pi)
    bcount = max(16, int(bins))
    hist, _ = np.histogram(th, bins=bcount, range=(0.0, 2.0 * np.pi))
    return float(np.count_nonzero(hist > 0)) / float(bcount)


def dedup_candidates(cands: List[EllipseCandidate], center_tol_px: float, radius_tol_px: float) -> List[EllipseCandidate]:
    out: List[EllipseCandidate] = []
    for c in sorted(cands, key=lambda z: z.score, reverse=True):
        rc = 0.25 * (c.w + c.h)
        dup = False
        for k in out:
            rk = 0.25 * (k.w + k.h)
            d = float(np.hypot(c.cx - k.cx, c.cy - k.cy))
            if d <= float(center_tol_px) and abs(rc - rk) <= float(radius_tol_px):
                dup = True
                break
        if not dup:
            out.append(c)
    return out


def angle_mod_180_diff(a_deg: float, b_deg: float) -> float:
    d = abs(float(a_deg) - float(b_deg)) % 180.0
    return min(d, 180.0 - d)


def suppress_concentric(cands: List[EllipseCandidate], center_px: float, angle_deg: float, radius_ratio_max: float) -> List[EllipseCandidate]:
    if not cands:
        return []
    kept: List[EllipseCandidate] = []
    for c in sorted(cands, key=lambda z: (z.score, z.completion, z.w + z.h), reverse=True):
        rc = 0.25 * (c.w + c.h)
        drop = False
        for k in kept:
            rk = 0.25 * (k.w + k.h)
            d = float(np.hypot(c.cx - k.cx, c.cy - k.cy))
            if d > float(center_px):
                continue
            if angle_mod_180_diff(c.angle_deg, k.angle_deg) > float(angle_deg):
                continue
            rr = max(rc, rk) / max(1e-6, min(rc, rk))
            if rr <= float(radius_ratio_max):
                drop = True
                break
        if not drop:
            kept.append(c)
    return kept


def greedy_unique_support(
    cands: List[EllipseCandidate],
    edge_pts: np.ndarray,
    max_dist_px: float,
    min_unique_edge_pts: int,
    min_unique_ratio: float,
    max_keep: int,
) -> List[EllipseCandidate]:
    if not cands:
        return []
    taken = np.zeros((edge_pts.shape[0],), dtype=bool)
    out: List[EllipseCandidate] = []
    for c in sorted(cands, key=lambda z: (z.score, z.completion, z.inlier_count), reverse=True):
        idx = ellipse_inlier_indices(
            edge_pts=edge_pts,
            cx=c.cx,
            cy=c.cy,
            w=c.w,
            h=c.h,
            angle_deg=c.angle_deg,
            max_dist_px=max_dist_px,
        )
        if idx.size == 0:
            continue
        unique_idx = idx[~taken[idx]]
        unique_ratio = float(unique_idx.size) / float(max(1, idx.size))
        if int(unique_idx.size) < int(min_unique_edge_pts):
            continue
        if unique_ratio < float(min_unique_ratio):
            continue
        c.unique_ratio = float(unique_ratio)
        out.append(c)
        taken[unique_idx] = True
        if len(out) >= int(max_keep):
            break
    return out


def detect_ellipses(gray: np.ndarray, edges: np.ndarray, args: argparse.Namespace) -> List[EllipseCandidate]:
    cands: List[EllipseCandidate] = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ys, xs = np.where(edges > 0)
    edge_pts = np.stack([xs, ys], axis=1).astype(np.float64) if xs.size > 0 else np.zeros((0, 2), dtype=np.float64)
    inv = cv2.bitwise_not(edges)
    dist_map = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    for cnt in contours:
        if cnt is None or len(cnt) < int(args.min_contour_points):
            continue
        per_open = float(cv2.arcLength(cnt, False))
        if per_open < float(args.min_open_arc_len_px):
            continue
        p0 = cnt[0, 0].astype(np.float64)
        p1 = cnt[-1, 0].astype(np.float64)
        end_gap = float(np.linalg.norm(p0 - p1))
        is_open = bool(end_gap >= float(args.open_arc_end_gap_px))
        if len(cnt) < 5:
            continue
        try:
            if bool(args.use_ams) and hasattr(cv2, "fitEllipseAMS"):
                (cx, cy), (w, h), ang = cv2.fitEllipseAMS(cnt)
            else:
                (cx, cy), (w, h), ang = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        major = max(float(w), float(h))
        minor = min(float(w), float(h))
        if major < float(args.min_diameter_px) or major > float(args.max_diameter_px):
            continue
        if (minor / max(major, 1e-6)) < float(args.min_aspect):
            continue

        # Geometry refinement: collect edge inliers around fitted ellipse and refit.
        cx, cy, w, h, ang, inliers = ellipse_inlier_refit(
            edge_pts=edge_pts,
            cx=float(cx),
            cy=float(cy),
            w=float(w),
            h=float(h),
            angle_deg=float(ang),
            max_dist_px=float(args.refine_band_px),
            iters=int(args.refine_iters),
        )
        major = max(float(w), float(h))
        minor = min(float(w), float(h))
        if major < float(args.min_diameter_px) or major > float(args.max_diameter_px):
            continue
        if (minor / max(major, 1e-6)) < float(args.min_aspect):
            continue

        support = ellipse_support_from_dist(
            dist_map=dist_map,
            cx=float(cx),
            cy=float(cy),
            w=float(w),
            h=float(h),
            angle_deg=float(ang),
            samples=int(args.edge_support_samples),
            max_dist_px=float(args.edge_support_dist_px),
        )
        arc_cov = contour_arc_coverage(
            cnt=cnt,
            cx=float(cx),
            cy=float(cy),
            w=float(w),
            h=float(h),
            angle_deg=float(ang),
            bins=int(args.arc_bins),
        )
        fit_residual = contour_ellipse_residual(
            cnt=cnt,
            cx=float(cx),
            cy=float(cy),
            w=float(w),
            h=float(h),
            angle_deg=float(ang),
        )

        c_area = float(max(0.0, cv2.contourArea(cnt)))
        e_area = float(np.pi * 0.25 * float(w) * float(h))
        area_ratio = float(c_area / max(e_area, 1e-6))
        hull = cv2.convexHull(cnt)
        h_area = float(max(1e-6, cv2.contourArea(hull)))
        solidity = float(c_area / h_area)
        completion = edge_completion_from_inliers(
            inliers=inliers,
            cx=float(cx),
            cy=float(cy),
            w=float(w),
            h=float(h),
            angle_deg=float(ang),
            bins=int(args.completion_bins),
        )

        # Reclassify arc type by geometric completion; avoids mislabeling merged arcs as full.
        is_open = bool(is_open or (completion < float(args.full_completion_min)))

        min_sup = float(args.open_arc_min_support) if is_open else float(args.full_min_support)
        min_arc = float(args.open_arc_min_arc) if is_open else float(args.full_min_arc)
        min_comp = float(args.open_completion_min) if is_open else float(args.full_completion_min)
        if support < min_sup or arc_cov < min_arc:
            continue
        if completion < min_comp:
            continue
        max_res = float(args.open_max_fit_residual) if is_open else float(args.full_max_fit_residual)
        if fit_residual > max_res:
            continue
        if not is_open:
            if area_ratio < float(args.full_min_area_ratio):
                continue
            if solidity < float(args.full_min_solidity):
                continue

        len_score = float(np.clip(per_open / max(1.0, 2.0 * np.pi * (0.25 * (w + h))), 0.0, 1.0))
        aspect = float(minor / max(major, 1e-6))
        fit_score = float(np.exp(-3.0 * max(0.0, fit_residual)))
        score = 0.30 * float(support) + 0.22 * float(arc_cov) + 0.20 * float(completion) + 0.10 * float(len_score) + 0.08 * float(aspect) + 0.10 * float(fit_score)
        inlier_count = int(inliers.shape[0])
        cands.append(
            EllipseCandidate(
                cx=float(cx),
                cy=float(cy),
                w=float(w),
                h=float(h),
                angle_deg=float(ang),
                score=float(score),
                support=float(support),
                arc_cov=float(arc_cov),
                fit_residual=float(fit_residual),
                completion=float(completion),
                area_ratio=float(area_ratio),
                solidity=float(solidity),
                inlier_count=int(inlier_count),
                unique_ratio=0.0,
                is_open_arc=is_open,
                contour=cnt.reshape(-1, 2).astype(np.float32),
            )
        )

    cands = dedup_candidates(
        cands=cands,
        center_tol_px=float(args.dedup_center_px),
        radius_tol_px=float(args.dedup_radius_px),
    )
    if bool(args.concentric_suppress):
        cands = suppress_concentric(
            cands=cands,
            center_px=float(args.concentric_center_px),
            angle_deg=float(args.concentric_angle_deg),
            radius_ratio_max=float(args.concentric_radius_ratio_max),
        )
    if bool(args.greedy_unique):
        cands = greedy_unique_support(
            cands=cands,
            edge_pts=edge_pts,
            max_dist_px=float(args.refine_band_px),
            min_unique_edge_pts=int(args.min_unique_edge_pts),
            min_unique_ratio=float(args.min_unique_ratio),
            max_keep=int(args.max_keep),
        )
    if len(cands) > int(args.max_keep):
        cands = cands[: int(args.max_keep)]
    return cands


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-object robust ellipse + half-ellipse detector (debug stage).")
    p.add_argument("--source", type=str, default="rs", choices=["rs", "webcam"])
    p.add_argument("--webcam_index", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--roi_yaml", type=str, default="segmentation_bgsub_yolo/config/roi.yaml")
    p.add_argument("--show_roi", action="store_true")
    p.add_argument("--exposure", type=float, default=140.0)
    p.add_argument("--gain", type=float, default=16.0)
    p.add_argument("--white_balance", type=float, default=4500.0)
    p.add_argument("--auto_exposure", action="store_true")
    p.add_argument("--auto_white_balance", action="store_true")
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--blur_ksize", type=int, default=5)
    p.add_argument("--canny_low", type=int, default=16)
    p.add_argument("--canny_high", type=int, default=26)
    p.add_argument("--morph_close", type=int, default=3)
    p.add_argument("--min_contour_points", type=int, default=24)
    p.add_argument("--min_open_arc_len_px", type=float, default=30.0)
    p.add_argument("--open_arc_end_gap_px", type=float, default=4.0)
    p.add_argument("--min_diameter_px", type=float, default=16.0)
    p.add_argument("--max_diameter_px", type=float, default=260.0)
    p.add_argument("--min_aspect", type=float, default=0.20)
    p.add_argument("--edge_support_samples", type=int, default=96)
    p.add_argument("--edge_support_dist_px", type=float, default=1.8)
    p.add_argument("--refine_band_px", type=float, default=2.2)
    p.add_argument("--refine_iters", type=int, default=2)
    p.add_argument("--arc_bins", type=int, default=48)
    p.add_argument("--completion_bins", type=int, default=64)
    p.add_argument("--full_min_support", type=float, default=0.24)
    p.add_argument("--full_min_arc", type=float, default=0.40)
    p.add_argument("--open_arc_min_support", type=float, default=0.17)
    p.add_argument("--open_arc_min_arc", type=float, default=0.18)
    p.add_argument("--full_completion_min", type=float, default=0.64)
    p.add_argument("--open_completion_min", type=float, default=0.18)
    p.add_argument("--full_max_fit_residual", type=float, default=0.18)
    p.add_argument("--open_max_fit_residual", type=float, default=0.28)
    p.add_argument("--full_min_area_ratio", type=float, default=0.60)
    p.add_argument("--full_min_solidity", type=float, default=0.86)
    p.add_argument("--dedup_center_px", type=float, default=10.0)
    p.add_argument("--dedup_radius_px", type=float, default=4.0)
    p.add_argument("--concentric_suppress", action="store_true", default=True)
    p.add_argument("--concentric_center_px", type=float, default=14.0)
    p.add_argument("--concentric_angle_deg", type=float, default=20.0)
    p.add_argument("--concentric_radius_ratio_max", type=float, default=2.2)
    p.add_argument("--greedy_unique", action="store_true", default=True)
    p.add_argument("--min_unique_edge_pts", type=int, default=36)
    p.add_argument("--min_unique_ratio", type=float, default=0.24)
    p.add_argument("--max_keep", type=int, default=6)
    p.add_argument("--use_ams", action="store_true")
    p.add_argument("--tweak_canny", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "rs" and rs is None:
        raise RuntimeError("pyrealsense2 not available.")

    cap = None
    pipe = None
    align = None

    if args.source == "rs":
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, int(args.width), int(args.height), rs.format.bgr8, int(args.fps))
        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)
        # Fixed camera settings for stable edges.
        dev = profile.get_device()
        color_sensor = None
        for s in dev.query_sensors():
            name = s.get_info(rs.camera_info.name).lower()
            if "rgb" in name or "color" in name:
                color_sensor = s
                break
        if color_sensor is not None:
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1.0 if bool(args.auto_exposure) else 0.0)
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, float(args.exposure))
            if color_sensor.supports(rs.option.gain):
                color_sensor.set_option(rs.option.gain, float(args.gain))
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0 if bool(args.auto_white_balance) else 0.0)
            if color_sensor.supports(rs.option.white_balance):
                color_sensor.set_option(rs.option.white_balance, float(args.white_balance))
    else:
        cap = cv2.VideoCapture(int(args.webcam_index))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {args.webcam_index}")

    if bool(args.tweak_canny):
        def _noop(_v: int) -> None:
            return
        cv2.namedWindow("Ellipse Canny Tuner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ellipse Canny Tuner", 420, 120)
        cv2.createTrackbar("low", "Ellipse Canny Tuner", int(args.canny_low), 255, _noop)
        cv2.createTrackbar("high", "Ellipse Canny Tuner", int(args.canny_high), 255, _noop)

    roi_xyxy: Optional[Tuple[int, int, int, int]] = None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if bool(args.clahe) else None
    print("[Keys] q / ESC to quit")
    try:
        while True:
            if args.source == "rs":
                fr = align.process(pipe.wait_for_frames(5000))
                color = fr.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data())
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            h, w = frame.shape[:2]
            if args.roi_yaml:
                if roi_xyxy is None:
                    roi_xyxy = load_roi_yaml(args.roi_yaml, w, h)
                    print(f"[Info] ROI: x={roi_xyxy[0]}:{roi_xyxy[2]} y={roi_xyxy[1]}:{roi_xyxy[3]}")
                x0, y0, x1, y1 = roi_xyxy
            else:
                x0, y0, x1, y1 = 0, 0, w, h
            roi = frame[y0:y1, x0:x1]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if clahe is not None:
                gray = clahe.apply(gray)
            bk = max(3, int(args.blur_ksize))
            if bk % 2 == 0:
                bk += 1
            gray = cv2.GaussianBlur(gray, (bk, bk), 0)

            if bool(args.tweak_canny):
                args.canny_low = int(cv2.getTrackbarPos("low", "Ellipse Canny Tuner"))
                args.canny_high = int(cv2.getTrackbarPos("high", "Ellipse Canny Tuner"))
                if int(args.canny_high) <= int(args.canny_low):
                    args.canny_high = min(255, int(args.canny_low) + 1)
                    cv2.setTrackbarPos("high", "Ellipse Canny Tuner", int(args.canny_high))

            edges = cv2.Canny(gray, int(args.canny_low), int(args.canny_high))
            if int(args.morph_close) > 1:
                k = int(args.morph_close)
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ker)

            cands = detect_ellipses(gray=gray, edges=edges, args=args)

            out = frame.copy()
            if args.show_roi and roi_xyxy is not None:
                cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 1, cv2.LINE_AA)

            for i, c in enumerate(cands, start=1):
                col = (0, 255, 0) if not c.is_open_arc else (0, 210, 255)
                cv2.ellipse(
                    out,
                    (int(round(c.cx + x0)), int(round(c.cy + y0))),
                    (int(round(c.w * 0.5)), int(round(c.h * 0.5))),
                    float(c.angle_deg),
                    0,
                    360,
                    col,
                    2,
                    cv2.LINE_AA,
                )
                txt = f"#{i} {'open' if c.is_open_arc else 'full'} s={c.score:.2f} sup={c.support:.2f} arc={c.arc_cov:.2f}"
                txt2 = f"cmp={c.completion:.2f} res={c.fit_residual:.2f} ar={c.area_ratio:.2f} so={c.solidity:.2f} uq={c.unique_ratio:.2f}"
                cv2.putText(
                    out,
                    txt,
                    (int(round(c.cx + x0)) + 6, int(round(c.cy + y0)) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    col,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    out,
                    txt2,
                    (int(round(c.cx + x0)) + 6, int(round(c.cy + y0)) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    col,
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                out,
                f"ellipses={len(cands)} canny={int(args.canny_low)}/{int(args.canny_high)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Single Object Ellipse Debug", out)
            cv2.imshow("Single Object Edges", edges)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
    finally:
        if pipe is not None:
            pipe.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
