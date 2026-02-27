import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


@dataclass
class CircleCandidate:
    center_u: float
    center_v: float
    radius_px: float
    score_2d: float
    source: str
    contour: Optional[np.ndarray] = None
    edge_support: float = 0.0
    grad_consistency: float = 0.0
    grad_support: float = 0.0


@dataclass
class PoseEstimate:
    center_3d: np.ndarray
    normal_3d: np.ndarray
    radius_m: float
    diameter_mm: float
    diameter_error_mm: float
    side_label: str
    side_score: float
    inner_median_offset_mm: float
    outer_median_offset_mm: float
    rim_depth_step_mm: float
    rim_valid_ratio: float
    center_luma: float
    outer_luma: float
    luma_contrast: float
    quality: float
    plane_rmse_mm: float
    circle_rmse_mm: float
    boundary_points: int
    coverage: float
    transform: np.ndarray
    candidate: CircleCandidate


@dataclass
class TrackState:
    center_3d: Optional[np.ndarray] = None
    normal_3d: Optional[np.ndarray] = None
    radius_m: Optional[float] = None
    last_label: str = "unknown"
    last_uv: Optional[Tuple[float, float]] = None
    last_radius_px: Optional[float] = None
    missed_frames: int = 0
    last_estimate: Optional[PoseEstimate] = None
    pending_estimate: Optional[PoseEstimate] = None
    confirm_count: int = 0
    confirmed: bool = False
    circle_uv: Optional[Tuple[float, float]] = None
    circle_radius_px: Optional[float] = None
    circle_missed_frames: int = 0


def normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v.copy()
    return v / n


def _noop_trackbar(_value: int):
    return


def make_filters(enable: bool):
    if not enable:
        return None
    spat = rs.spatial_filter()
    temp = rs.temporal_filter()
    hole = rs.hole_filling_filter()
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20)
    temp.set_option(rs.option.filter_smooth_alpha, 0.4)
    temp.set_option(rs.option.filter_smooth_delta, 20)
    return [spat, temp, hole]


def start_rs_pipeline_with_fallback(
    pipe: rs.pipeline,
    req_w: int,
    req_h: int,
    req_fps: int,
    allow_fallback: bool,
):
    candidates = [(int(req_w), int(req_h), int(req_fps))]
    if allow_fallback:
        for mode in [(1280, 720, int(req_fps)), (848, 480, int(req_fps)), (640, 480, int(req_fps))]:
            if mode not in candidates:
                candidates.append(mode)

    tried = []
    last_err = None
    for w, h, fps in candidates:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        try:
            profile = pipe.start(cfg)
            return profile, w, h, fps
        except Exception as exc:
            tried.append(f"{w}x{h}@{fps}")
            last_err = exc

    raise RuntimeError(f"Failed to start RealSense mode(s): {', '.join(tried)} | last error: {last_err}")


def rs_cloud_from_aligned_depth(
    depth_u16: np.ndarray,
    rgb: np.ndarray,
    depth_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int,
    z_min: float,
    z_max: float,
    flip_y: bool,
):
    h, w = depth_u16.shape
    stride = max(1, int(stride))
    z = depth_u16[0:h:stride, 0:w:stride].reshape(-1).astype(np.float32) * float(depth_scale)
    valid = (z > 0.0) & (z >= float(z_min)) & (z <= float(z_max))
    if np.count_nonzero(valid) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    us = np.arange(0, w, stride, dtype=np.float32)
    vs = np.arange(0, h, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    u = uu.reshape(-1)[valid]
    v = vv.reshape(-1)[valid]
    z = z[valid]

    x = (u - float(cx)) * z / float(fx)
    y = (v - float(cy)) * z / float(fy)
    if flip_y:
        y = -y
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    ui = np.clip(u.astype(np.int32), 0, rgb.shape[1] - 1)
    vi = np.clip(v.astype(np.int32), 0, rgb.shape[0] - 1)
    cols = rgb[vi, ui, :].astype(np.float32) / 255.0
    return pts, cols


def get_search_roi(
    image_shape: Tuple[int, int],
    prev_center_uv: Optional[Tuple[float, float]],
    prev_radius_px: Optional[float],
    roi_scale: float,
    roi_min_px: int,
):
    h, w = image_shape
    if prev_center_uv is None or prev_radius_px is None:
        return 0, 0, w, h

    half = int(max(float(roi_min_px), float(prev_radius_px) * float(roi_scale)))
    cx = int(round(prev_center_uv[0]))
    cy = int(round(prev_center_uv[1]))
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0, 0, w, h
    return x0, y0, x1, y1


def get_center_roi(image_shape: Tuple[int, int], roi_ratio: float):
    h, w = image_shape
    r = float(np.clip(float(roi_ratio), 0.05, 1.0))
    rw = max(8, int(round(w * r)))
    rh = max(8, int(round(h * r)))
    x0 = (w - rw) // 2
    y0 = (h - rh) // 2
    x1 = x0 + rw
    y1 = y0 + rh
    return x0, y0, x1, y1


def make_track_candidate(track: TrackState) -> Optional[CircleCandidate]:
    if not track.confirmed or track.last_uv is None or track.last_radius_px is None:
        return None
    return CircleCandidate(
        center_u=float(track.last_uv[0]),
        center_v=float(track.last_uv[1]),
        radius_px=float(track.last_radius_px),
        score_2d=2.0,
        source="track",
        contour=None,
        edge_support=1.0,
        grad_consistency=1.0,
        grad_support=1.0,
    )


def make_circle_track_candidate(track: TrackState) -> Optional[CircleCandidate]:
    if track.circle_uv is None or track.circle_radius_px is None:
        return None
    return CircleCandidate(
        center_u=float(track.circle_uv[0]),
        center_v=float(track.circle_uv[1]),
        radius_px=float(track.circle_radius_px),
        score_2d=2.0,
        source="circle_track",
        contour=None,
        edge_support=1.0,
        grad_consistency=1.0,
        grad_support=1.0,
    )


def circle_size_prior_score(
    candidate: CircleCandidate,
    depth_u16: np.ndarray,
    depth_scale: float,
    fx: float,
    target_diameter_mm: float,
    depth_patch: int,
):
    z = depth_at_pixel_m(
        depth_u16=depth_u16,
        u=float(candidate.center_u),
        v=float(candidate.center_v),
        depth_scale=float(depth_scale),
        patch=int(depth_patch),
    )
    if not np.isfinite(z) or z <= 0.0:
        return 0.0, float("nan"), float("nan")
    r_pred = float(fx * (float(target_diameter_mm) * 0.001 * 0.5) / z)
    rel_err = float(abs(float(candidate.radius_px) - r_pred) / max(r_pred, 1e-6))
    return float(np.clip(1.0 - rel_err, -1.0, 1.0)), rel_err, r_pred


def choose_best_circle_candidate(
    candidates: List[CircleCandidate],
    track: TrackState,
    gray_u8: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale: float,
    fx: float,
    args,
):
    if len(candidates) == 0:
        return None, float("-inf"), float("nan"), float("nan"), float("nan"), float("nan"), False, 0

    def run_selection(enforce_photo_gate: bool):
        best = None
        best_score = float("-inf")
        best_rel_err = float("nan")
        best_r_pred = float("nan")
        best_center_luma = float("nan")
        best_luma_contrast = float("nan")
        photo_pass_count = 0

        anchor_uv = track.circle_uv if track.circle_uv is not None else track.last_uv
        for c in candidates[: int(args.max_candidates)]:
            s = float(c.score_2d)
            center_luma = float("nan")
            luma_contrast = float("nan")

            if bool(args.circle_use_photo_prior):
                center_luma, _, luma_contrast, inner_std, _ = photometric_circle_metrics(
                    gray_u8=gray_u8,
                    candidate=c,
                    inner_ratio=float(args.photo_inner_ratio),
                    outer_inner_ratio=float(args.photo_outer_inner_ratio),
                    outer_outer_ratio=float(args.photo_outer_outer_ratio),
                )
                if not np.isfinite(center_luma):
                    continue
                if enforce_photo_gate:
                    if center_luma < float(args.circle_min_center_luma):
                        continue
                    if luma_contrast < float(args.circle_min_luma_contrast):
                        continue
                    if inner_std > float(args.circle_max_center_luma_std):
                        continue
                photo_pass_count += 1

                bright_score = np.clip(
                    (center_luma - float(args.circle_min_center_luma)) / 80.0,
                    -1.0,
                    1.0,
                )
                contrast_score = np.clip(
                    (luma_contrast - float(args.circle_min_luma_contrast)) / 25.0,
                    -1.0,
                    1.0,
                )
                tex_score = np.clip(
                    (float(args.circle_max_center_luma_std) - float(inner_std))
                    / max(float(args.circle_max_center_luma_std), 1.0),
                    -1.0,
                    1.0,
                )
                s += float(args.circle_track_photo_weight) * float(
                    0.45 * bright_score + 0.45 * contrast_score + 0.10 * tex_score
                )

            if anchor_uv is not None:
                d = float(np.hypot(c.center_u - anchor_uv[0], c.center_v - anchor_uv[1]))
                prox = float(np.clip(1.0 - d / max(1.0, float(args.circle_track_jump_px)), -1.0, 1.0))
                s += float(args.circle_track_prox_weight) * prox

            rel_err = float("nan")
            r_pred = float("nan")
            if bool(args.use_circle_depth_size_prior):
                size_s, rel_err, r_pred = circle_size_prior_score(
                    candidate=c,
                    depth_u16=depth_u16,
                    depth_scale=float(depth_scale),
                    fx=float(fx),
                    target_diameter_mm=float(args.target_diameter_mm),
                    depth_patch=int(args.depth_patch),
                )
                if np.isfinite(rel_err) and rel_err > float(args.circle_size_prior_hard):
                    continue
                s += float(args.circle_track_size_weight) * size_s

            if s > best_score:
                best = c
                best_score = float(s)
                best_rel_err = rel_err
                best_r_pred = r_pred
                best_center_luma = center_luma
                best_luma_contrast = luma_contrast

        return (
            best,
            best_score,
            best_rel_err,
            best_r_pred,
            best_center_luma,
            best_luma_contrast,
            photo_pass_count,
        )

    use_relaxed = False
    (
        best,
        best_score,
        best_rel_err,
        best_r_pred,
        best_center_luma,
        best_luma_contrast,
        photo_pass_count,
    ) = run_selection(enforce_photo_gate=bool(args.circle_photo_hard_gate) and bool(args.circle_use_photo_prior))

    if (
        best is None
        and bool(args.circle_use_photo_prior)
        and bool(args.circle_photo_fallback_relaxed)
        and bool(args.circle_photo_hard_gate)
    ):
        (
            best,
            best_score,
            best_rel_err,
            best_r_pred,
            best_center_luma,
            best_luma_contrast,
            photo_pass_count,
        ) = run_selection(enforce_photo_gate=False)
        use_relaxed = True

    return (
        best,
        best_score,
        best_rel_err,
        best_r_pred,
        best_center_luma,
        best_luma_contrast,
        use_relaxed,
        photo_pass_count,
    )


def update_circle_track(
    track: TrackState,
    raw_best: Optional[CircleCandidate],
    args,
) -> Optional[CircleCandidate]:
    accepted = raw_best
    if accepted is not None and track.circle_uv is not None:
        jump = float(np.hypot(accepted.center_u - track.circle_uv[0], accepted.center_v - track.circle_uv[1]))
        if jump > float(args.circle_track_jump_px):
            accepted = None

    if accepted is not None:
        if track.circle_uv is None or track.circle_radius_px is None:
            new_u = float(accepted.center_u)
            new_v = float(accepted.center_v)
            new_r = float(accepted.radius_px)
        else:
            a = float(np.clip(float(args.circle_track_alpha), 0.0, 0.98))
            new_u = float(a * track.circle_uv[0] + (1.0 - a) * float(accepted.center_u))
            new_v = float(a * track.circle_uv[1] + (1.0 - a) * float(accepted.center_v))
            new_r = float(a * track.circle_radius_px + (1.0 - a) * float(accepted.radius_px))

        track.circle_uv = (new_u, new_v)
        track.circle_radius_px = new_r
        track.circle_missed_frames = 0
        return CircleCandidate(
            center_u=new_u,
            center_v=new_v,
            radius_px=new_r,
            score_2d=float(accepted.score_2d),
            source="circle_track",
            contour=None,
            edge_support=float(accepted.edge_support),
            grad_consistency=float(accepted.grad_consistency),
            grad_support=float(accepted.grad_support),
        )

    track.circle_missed_frames += 1
    if track.circle_uv is not None and track.circle_radius_px is not None:
        if track.circle_missed_frames <= int(args.circle_track_max_missed):
            return make_circle_track_candidate(track)

    track.circle_uv = None
    track.circle_radius_px = None
    return None


def debug_reasons_line(stats: dict, top_k: int = 4) -> str:
    if not stats:
        return ""
    items = [(k, int(v)) for k, v in stats.items() if k != "pass" and int(v) > 0]
    if not items:
        return "reasons=none"
    items.sort(key=lambda kv: kv[1], reverse=True)
    top = items[: max(1, int(top_k))]
    return "reasons=" + ",".join([f"{k}:{v}" for k, v in top])


def init_canny_sliders(window_name: str, low: float, high: float):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 460, 120)
    low_i = int(np.clip(int(round(float(low))), 0, 255))
    high_i = int(np.clip(int(round(float(high))), 0, 255))
    cv2.createTrackbar("canny_low", window_name, low_i, 255, _noop_trackbar)
    cv2.createTrackbar("canny_high", window_name, max(low_i + 1, high_i), 255, _noop_trackbar)


def read_canny_sliders(window_name: str, prev_low: float, prev_high: float) -> Tuple[float, float]:
    try:
        low = int(cv2.getTrackbarPos("canny_low", window_name))
        high = int(cv2.getTrackbarPos("canny_high", window_name))
    except cv2.error:
        return float(prev_low), float(prev_high)

    low = int(np.clip(low, 0, 254))
    high = int(np.clip(high, low + 1, 255))
    cv2.setTrackbarPos("canny_high", window_name, high)
    return float(low), float(high)


def deduplicate_candidates(
    candidates: List[CircleCandidate],
    merge_center_px: float,
    merge_radius_px: float,
    max_candidates: int,
) -> List[CircleCandidate]:
    ordered = sorted(candidates, key=lambda c: float(c.score_2d), reverse=True)
    kept: List[CircleCandidate] = []
    for c in ordered:
        duplicate = False
        for k in kept:
            d = float(np.hypot(c.center_u - k.center_u, c.center_v - k.center_v))
            if d <= float(merge_center_px) and abs(c.radius_px - k.radius_px) <= float(merge_radius_px):
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(c)
        if len(kept) >= int(max_candidates):
            break
    return kept


def filter_edge_components(edges: np.ndarray, min_pixels: int, max_pixels: int) -> np.ndarray:
    if edges.size == 0:
        return edges
    bw = (edges > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(edges, dtype=np.uint8)
    min_px = max(1, int(min_pixels))
    max_px = int(max_pixels)
    for idx in range(1, int(n_labels)):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        if max_px > 0 and area > max_px:
            continue
        out[labels == idx] = 255
    return out


def keep_inner_concentric_candidates(
    candidates: List[CircleCandidate],
    center_tol_px: float,
    radius_gap_px: float,
    max_candidates: int,
) -> List[CircleCandidate]:
    if len(candidates) <= 1:
        return candidates

    kept: List[CircleCandidate] = []
    for c in sorted(candidates, key=lambda x: float(x.radius_px)):
        merged = False
        for i, k in enumerate(kept):
            d = float(np.hypot(c.center_u - k.center_u, c.center_v - k.center_v))
            if d > float(center_tol_px):
                continue
            # Same center neighborhood: keep the smaller radius (inner ring).
            if c.radius_px + float(radius_gap_px) < k.radius_px:
                kept[i] = c
            merged = True
            break
        if not merged:
            kept.append(c)

    kept = sorted(kept, key=lambda x: float(x.score_2d), reverse=True)
    return kept[: int(max_candidates)]


def ring_signal_metrics(
    candidate: CircleCandidate,
    edges: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    samples: int,
    band_px: float,
    grad_mag_min: float,
    grad_align_abs_min: float,
):
    h, w = edges.shape
    n = max(24, int(samples))
    b = max(1, int(round(float(band_px))))

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    uu = float(candidate.center_u) + float(candidate.radius_px) * np.cos(angles)
    vv = float(candidate.center_v) + float(candidate.radius_px) * np.sin(angles)

    valid = 0
    edge_hits = 0
    strong_grad = 0
    radial_grad = 0

    for u, v, c, s in zip(uu, vv, np.cos(angles), np.sin(angles)):
        ui = int(round(float(u)))
        vi = int(round(float(v)))
        if ui < 0 or ui >= w or vi < 0 or vi >= h:
            continue
        valid += 1

        x0 = max(0, ui - b)
        y0 = max(0, vi - b)
        x1 = min(w, ui + b + 1)
        y1 = min(h, vi + b + 1)
        if np.any(edges[y0:y1, x0:x1] > 0):
            edge_hits += 1

        gx = float(grad_x[vi, ui])
        gy = float(grad_y[vi, ui])
        gm = float(np.hypot(gx, gy))
        if gm >= float(grad_mag_min):
            strong_grad += 1
            # For a circular edge, local image gradient should align with radial direction.
            align = abs((gx * float(c) + gy * float(s)) / max(gm, 1e-6))
            if align >= float(grad_align_abs_min):
                radial_grad += 1

    if valid == 0:
        return 0.0, 0.0, 0.0
    edge_support = float(edge_hits) / float(valid)
    grad_support = float(strong_grad) / float(valid)
    grad_consistency = float(radial_grad) / float(max(1, strong_grad))
    return edge_support, grad_consistency, grad_support


def detect_circle_candidates(gray: np.ndarray, args) -> Tuple[List[CircleCandidate], np.ndarray]:
    blur_k = max(3, int(args.blur_ksize))
    if blur_k % 2 == 0:
        blur_k += 1

    proc = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    if args.equalize:
        proc = cv2.equalizeHist(proc)

    edges_raw = cv2.Canny(proc, int(args.canny_low), int(args.canny_high))
    if int(args.morph_close) > 1:
        k = int(args.morph_close)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges_raw = cv2.morphologyEx(edges_raw, cv2.MORPH_CLOSE, kernel)
    edges = edges_raw
    if bool(args.edge_cc_filter):
        edges = filter_edge_components(
            edges_raw,
            min_pixels=int(args.edge_cc_min_pixels),
            max_pixels=int(args.edge_cc_max_pixels),
        )
    grad_x = cv2.Sobel(proc, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(proc, cv2.CV_32F, 0, 1, ksize=3)

    def extract_candidates(edge_img: np.ndarray) -> List[CircleCandidate]:
        candidates_local: List[CircleCandidate] = []

        contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cnt is None or len(cnt) < 20:
                continue
            area = float(cv2.contourArea(cnt))
            if area < float(args.contour_min_area_px):
                continue
            per = float(cv2.arcLength(cnt, True))
            if per <= 1e-6:
                continue
            circularity = float((4.0 * np.pi * area) / (per * per))
            if circularity < float(args.contour_min_circularity):
                continue

            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (w, h), _ = ellipse
                major = max(float(w), float(h))
                minor = min(float(w), float(h))
                aspect = minor / max(major, 1e-6)
                radius_px = 0.25 * (major + minor)
            else:
                (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)
                aspect = 1.0

            if aspect < float(args.contour_min_aspect):
                continue
            if radius_px < float(args.min_radius_px) or radius_px > float(args.max_radius_px):
                continue
            fill_ratio = float(area / max(np.pi * radius_px * radius_px, 1e-6))
            if fill_ratio < float(args.contour_min_fill_ratio):
                continue

            candidate = CircleCandidate(
                center_u=float(cx),
                center_v=float(cy),
                radius_px=float(radius_px),
                score_2d=0.0,
                source="contour",
                contour=cnt.reshape(-1, 2).astype(np.float32),
            )
            edge_support, grad_consistency, grad_support = ring_signal_metrics(
                candidate,
                edge_img,
                grad_x,
                grad_y,
                samples=int(args.ring_eval_samples),
                band_px=float(args.ring_edge_band_px),
                grad_mag_min=float(args.grad_mag_min),
                grad_align_abs_min=float(args.grad_align_abs_min),
            )
            if edge_support < float(args.min_edge_support):
                continue
            if grad_support < float(args.min_grad_support):
                continue
            if grad_consistency < float(args.min_grad_consistency):
                continue

            score = (
                0.40 * circularity
                + 0.20 * aspect
                + 0.15 * np.clip(fill_ratio, 0.0, 1.0)
                + 0.15 * edge_support
                + 0.10 * grad_consistency
            )
            candidate.score_2d = float(score)
            candidate.edge_support = float(edge_support)
            candidate.grad_consistency = float(grad_consistency)
            candidate.grad_support = float(grad_support)
            candidates_local.append(candidate)

        if args.use_hough:
            circles = cv2.HoughCircles(
                proc,
                cv2.HOUGH_GRADIENT,
                dp=float(args.hough_dp),
                minDist=float(args.hough_min_dist_px),
                param1=float(args.hough_param1),
                param2=float(args.hough_param2),
                minRadius=int(args.min_radius_px),
                maxRadius=int(args.max_radius_px),
            )
            if circles is not None:
                for c in circles[0]:
                    candidate = CircleCandidate(
                        center_u=float(c[0]),
                        center_v=float(c[1]),
                        radius_px=float(c[2]),
                        score_2d=0.0,
                        source="hough",
                        contour=None,
                    )
                    edge_support, grad_consistency, grad_support = ring_signal_metrics(
                        candidate,
                        edge_img,
                        grad_x,
                        grad_y,
                        samples=int(args.ring_eval_samples),
                        band_px=float(args.ring_edge_band_px),
                        grad_mag_min=float(args.grad_mag_min),
                        grad_align_abs_min=float(args.grad_align_abs_min),
                    )
                    if edge_support < float(args.min_edge_support):
                        continue
                    if grad_support < float(args.min_grad_support):
                        continue
                    if grad_consistency < float(args.min_grad_consistency):
                        continue
                    candidate.score_2d = float(0.55 + 0.25 * edge_support + 0.20 * grad_consistency)
                    candidate.edge_support = float(edge_support)
                    candidate.grad_consistency = float(grad_consistency)
                    candidate.grad_support = float(grad_support)
                    candidates_local.append(candidate)

        candidates_local = deduplicate_candidates(
            candidates_local,
            merge_center_px=float(args.candidate_merge_center_px),
            merge_radius_px=float(args.candidate_merge_radius_px),
            max_candidates=int(args.max_candidates),
        )
        if bool(args.prefer_inner_circle):
            candidates_local = keep_inner_concentric_candidates(
                candidates_local,
                center_tol_px=float(args.inner_center_tol_px),
                radius_gap_px=float(args.inner_radius_gap_px),
                max_candidates=int(args.max_candidates),
            )
        return candidates_local

    candidates = extract_candidates(edges)
    if (
        bool(args.edge_cc_filter)
        and bool(args.edge_cc_fallback)
        and (
            len(candidates) < int(args.edge_cc_fallback_min_candidates)
            or int(np.count_nonzero(edges)) < int(args.edge_cc_fallback_min_pixels)
        )
    ):
        raw_candidates = extract_candidates(edges_raw)
        if len(raw_candidates) > len(candidates):
            return raw_candidates, edges_raw
    return candidates, edges


def depth_at_pixel_m(depth_u16: np.ndarray, u: float, v: float, depth_scale: float, patch: int) -> float:
    h, w = depth_u16.shape
    ui = int(round(float(u)))
    vi = int(round(float(v)))
    if ui < 0 or ui >= w or vi < 0 or vi >= h:
        return float("nan")

    p = max(0, int(patch))
    x0 = max(0, ui - p)
    y0 = max(0, vi - p)
    x1 = min(w, ui + p + 1)
    y1 = min(h, vi + p + 1)
    patch_vals = depth_u16[y0:y1, x0:x1].reshape(-1)
    patch_vals = patch_vals[patch_vals > 0]
    if patch_vals.size == 0:
        return float("nan")
    return float(np.median(patch_vals) * float(depth_scale))


def sample_boundary_uv(candidate: CircleCandidate, sample_count: int, contour_band_ratio: float) -> np.ndarray:
    n_samples = max(16, int(sample_count))

    if candidate.contour is not None and candidate.contour.shape[0] >= 12:
        contour = candidate.contour.copy()
        d = np.hypot(contour[:, 0] - float(candidate.center_u), contour[:, 1] - float(candidate.center_v))
        band = np.abs(d - float(candidate.radius_px)) <= max(1.5, float(candidate.radius_px) * float(contour_band_ratio))
        if np.count_nonzero(band) >= max(12, n_samples // 3):
            contour = contour[band]

        if contour.shape[0] > n_samples:
            idx = np.linspace(0, contour.shape[0] - 1, n_samples).astype(np.int32)
            contour = contour[idx]
        return contour.astype(np.float32)

    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False, dtype=np.float32)
    u = float(candidate.center_u) + float(candidate.radius_px) * np.cos(angles)
    v = float(candidate.center_v) + float(candidate.radius_px) * np.sin(angles)
    return np.stack([u, v], axis=1).astype(np.float32)


def deproject_uvz_to_points(uv: np.ndarray, z_m: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (uv[:, 0] - float(cx)) * z_m / float(fx)
    y = (uv[:, 1] - float(cy)) * z_m / float(fy)
    return np.stack([x, y, z_m], axis=1).astype(np.float64)


def pixels_to_3d_boundary_points(
    depth_u16: np.ndarray,
    boundary_uv: np.ndarray,
    depth_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    patch: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if boundary_uv.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    valid_uv = []
    z_vals = []
    for uv in boundary_uv:
        z = depth_at_pixel_m(depth_u16, float(uv[0]), float(uv[1]), depth_scale, patch)
        if np.isfinite(z) and z > 0.0:
            valid_uv.append([float(uv[0]), float(uv[1])])
            z_vals.append(float(z))

    if len(valid_uv) < 3:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    valid_uv = np.array(valid_uv, dtype=np.float64)
    z_vals = np.array(z_vals, dtype=np.float64)
    pts = deproject_uvz_to_points(valid_uv, z_vals, fx, fy, cx, cy)
    return pts, valid_uv


def rim_depth_step_metrics(
    depth_u16: np.ndarray,
    candidate: CircleCandidate,
    depth_scale: float,
    depth_patch: int,
    rim_offset_px: float,
    rim_eval_samples: int,
    rim_step_clip_mm: float,
):
    n = max(24, int(rim_eval_samples))
    dr = max(1.0, float(rim_offset_px))
    clip_m = max(0.001, float(rim_step_clip_mm) / 1000.0)

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    diffs = []
    valid = 0
    for a in angles:
        ca = float(np.cos(a))
        sa = float(np.sin(a))
        u_in = float(candidate.center_u) + (float(candidate.radius_px) - dr) * ca
        v_in = float(candidate.center_v) + (float(candidate.radius_px) - dr) * sa
        u_out = float(candidate.center_u) + (float(candidate.radius_px) + dr) * ca
        v_out = float(candidate.center_v) + (float(candidate.radius_px) + dr) * sa

        z_in = depth_at_pixel_m(depth_u16, u_in, v_in, depth_scale, depth_patch)
        z_out = depth_at_pixel_m(depth_u16, u_out, v_out, depth_scale, depth_patch)
        if not (np.isfinite(z_in) and np.isfinite(z_out) and z_in > 0.0 and z_out > 0.0):
            continue
        d = float(z_out - z_in)
        if abs(d) <= clip_m:
            diffs.append(d)
        valid += 1

    if valid == 0 or len(diffs) == 0:
        return 0.0, 0.0
    diffs = np.asarray(diffs, dtype=np.float64)
    step_mm = float(np.median(np.abs(diffs)) * 1000.0)
    valid_ratio = float(valid) / float(n)
    return step_mm, valid_ratio


def photometric_circle_metrics(
    gray_u8: np.ndarray,
    candidate: CircleCandidate,
    inner_ratio: float,
    outer_inner_ratio: float,
    outer_outer_ratio: float,
):
    h, w = gray_u8.shape
    r = max(2.0, float(candidate.radius_px))
    x0 = max(0, int(np.floor(candidate.center_u - outer_outer_ratio * r)))
    y0 = max(0, int(np.floor(candidate.center_v - outer_outer_ratio * r)))
    x1 = min(w, int(np.ceil(candidate.center_u + outer_outer_ratio * r)) + 1)
    y1 = min(h, int(np.ceil(candidate.center_v + outer_outer_ratio * r)) + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return float("nan"), float("nan"), float("nan"), 0.0, 0.0

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx.astype(np.float64) - float(candidate.center_u)
    dy = yy.astype(np.float64) - float(candidate.center_v)
    rr = np.hypot(dx, dy)

    inner_mask = rr <= (float(inner_ratio) * r)
    outer_mask = (rr >= float(outer_inner_ratio) * r) & (rr <= float(outer_outer_ratio) * r)
    n_inner = int(np.count_nonzero(inner_mask))
    n_outer = int(np.count_nonzero(outer_mask))
    if n_inner < 20 or n_outer < 20:
        return float("nan"), float("nan"), float("nan"), 0.0, 0.0

    patch = gray_u8[y0:y1, x0:x1].astype(np.float32)
    center_vals = patch[inner_mask]
    outer_vals = patch[outer_mask]
    center_luma = float(np.mean(center_vals))
    outer_luma = float(np.mean(outer_vals))
    contrast = float(center_luma - outer_luma)
    inner_std = float(np.std(center_vals))
    outer_std = float(np.std(outer_vals))
    return center_luma, outer_luma, contrast, inner_std, outer_std


def fit_plane_ransac(
    pts: np.ndarray,
    thresh_m: float,
    ransac_iters: int,
    min_inliers: int,
    rng: np.random.Generator,
):
    n = int(pts.shape[0])
    if n < 3:
        return None

    best_count = 0
    best_mask = None
    iters = max(10, int(ransac_iters))

    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        nn = float(np.linalg.norm(normal))
        if nn < 1e-10:
            continue
        normal = normal / nn
        dist = np.abs((pts - p0) @ normal)
        mask = dist <= float(thresh_m)
        cnt = int(np.count_nonzero(mask))
        if cnt > best_count:
            best_count = cnt
            best_mask = mask

    if best_mask is None or best_count < int(min_inliers):
        return None

    inliers = pts[best_mask]
    centroid = np.mean(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = normalize_vec(vh[-1])

    all_dist = np.abs((pts - centroid) @ normal)
    refined_mask = all_dist <= float(thresh_m)
    if int(np.count_nonzero(refined_mask)) >= int(min_inliers):
        inliers = pts[refined_mask]
        centroid = np.mean(inliers, axis=0)
        _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
        normal = normalize_vec(vh[-1])
        residuals = np.abs((inliers - centroid) @ normal)
        rmse = float(np.sqrt(np.mean(residuals * residuals)))
        return centroid, normal, refined_mask, rmse

    residuals = np.abs((inliers - centroid) @ normal)
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    return centroid, normal, best_mask, rmse


def build_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = normalize_vec(normal)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(n, ref))) > 0.90:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ex = normalize_vec(np.cross(ref, n))
    ey = normalize_vec(np.cross(n, ex))
    return ex, ey


def fit_circle_kasa(xy: np.ndarray):
    if xy.shape[0] < 3:
        return None
    x = xy[:, 0]
    y = xy[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = float(cx * cx + cy * cy + c)
    if not np.isfinite(r2) or r2 <= 0.0:
        return None
    return float(cx), float(cy), float(np.sqrt(r2))


def fit_circle_trimmed(xy: np.ndarray, trim_m: float, min_inliers: int, max_iters: int = 4):
    if xy.shape[0] < 3:
        return None
    mask = np.ones(xy.shape[0], dtype=bool)
    trim = max(1e-5, float(trim_m))

    for _ in range(max(1, int(max_iters))):
        fit = fit_circle_kasa(xy[mask])
        if fit is None:
            return None
        cx, cy, r = fit
        resid = np.abs(np.linalg.norm(xy - np.array([cx, cy]), axis=1) - r)
        new_mask = resid <= trim
        if int(np.count_nonzero(new_mask)) < int(min_inliers):
            return None
        if np.array_equal(mask, new_mask):
            mask = new_mask
            break
        mask = new_mask

    fit = fit_circle_kasa(xy[mask])
    if fit is None:
        return None
    cx, cy, r = fit
    resid = np.abs(np.linalg.norm(xy - np.array([cx, cy]), axis=1) - r)
    mask = resid <= trim
    if int(np.count_nonzero(mask)) < int(min_inliers):
        return None
    rmse = float(np.sqrt(np.mean((resid[mask] ** 2))))
    return float(cx), float(cy), float(r), mask, rmse


def compute_angular_coverage(xy: np.ndarray, cx: float, cy: float, bins: int) -> float:
    if xy.shape[0] < 3:
        return 0.0
    a = np.mod(np.arctan2(xy[:, 1] - float(cy), xy[:, 0] - float(cx)), 2.0 * np.pi)
    b = max(8, int(bins))
    hist, _ = np.histogram(a, bins=b, range=(0.0, 2.0 * np.pi))
    covered = np.count_nonzero(hist > 0)
    return float(covered) / float(b)


def orient_normal_toward_camera(normal: np.ndarray, point: np.ndarray) -> np.ndarray:
    n = normalize_vec(normal)
    if float(np.dot(n, point)) > 0.0:
        n = -n
    return n


def inner_offsets_from_depth(
    depth_u16: np.ndarray,
    candidate: CircleCandidate,
    center_3d: np.ndarray,
    normal_3d: np.ndarray,
    radius_m: float,
    depth_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    args,
) -> np.ndarray:
    h, w = depth_u16.shape
    half = int(max(6.0, float(candidate.radius_px) * float(args.inner_bbox_scale)))
    cu = int(round(float(candidate.center_u)))
    cv = int(round(float(candidate.center_v)))
    x0 = max(0, cu - half)
    y0 = max(0, cv - half)
    x1 = min(w, cu + half + 1)
    y1 = min(h, cv + half + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((0,), dtype=np.float64)

    sub = depth_u16[y0:y1, x0:x1].astype(np.float32)
    valid = sub > 0
    if int(np.count_nonzero(valid)) < int(args.min_inner_points):
        return np.zeros((0,), dtype=np.float64)

    ys, xs = np.where(valid)
    u = xs.astype(np.float64) + float(x0)
    v = ys.astype(np.float64) + float(y0)
    z = sub[valid].astype(np.float64) * float(depth_scale)

    z_win = float(args.inner_z_window_mm) / 1000.0
    keep_z = np.abs(z - float(center_3d[2])) <= z_win
    if int(np.count_nonzero(keep_z)) < int(args.min_inner_points):
        return np.zeros((0,), dtype=np.float64)

    u = u[keep_z]
    v = v[keep_z]
    z = z[keep_z]
    pts = deproject_uvz_to_points(np.stack([u, v], axis=1), z, fx, fy, cx, cy)

    vec = pts - center_3d.reshape(1, 3)
    plane_d = vec @ normal_3d
    radial_vec = vec - np.outer(plane_d, normal_3d)
    rho = np.linalg.norm(radial_vec, axis=1)
    inner = rho <= float(radius_m) * float(args.inner_radius_ratio)
    if int(np.count_nonzero(inner)) < int(args.min_inner_points):
        return np.zeros((0,), dtype=np.float64)

    offsets = plane_d[inner]
    clip = float(args.inner_offset_clip_mm) / 1000.0
    offsets = offsets[np.abs(offsets) <= clip]
    return offsets.astype(np.float64)


def outer_offsets_from_depth(
    depth_u16: np.ndarray,
    candidate: CircleCandidate,
    center_3d: np.ndarray,
    normal_3d: np.ndarray,
    radius_m: float,
    depth_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    args,
) -> np.ndarray:
    h, w = depth_u16.shape
    half = int(max(8.0, float(candidate.radius_px) * float(args.outer_bbox_scale)))
    cu = int(round(float(candidate.center_u)))
    cv = int(round(float(candidate.center_v)))
    x0 = max(0, cu - half)
    y0 = max(0, cv - half)
    x1 = min(w, cu + half + 1)
    y1 = min(h, cv + half + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return np.zeros((0,), dtype=np.float64)

    sub = depth_u16[y0:y1, x0:x1].astype(np.float32)
    valid = sub > 0
    if int(np.count_nonzero(valid)) < int(args.min_outer_points):
        return np.zeros((0,), dtype=np.float64)

    ys, xs = np.where(valid)
    u = xs.astype(np.float64) + float(x0)
    v = ys.astype(np.float64) + float(y0)
    z = sub[valid].astype(np.float64) * float(depth_scale)

    z_win = float(args.outer_z_window_mm) / 1000.0
    keep_z = np.abs(z - float(center_3d[2])) <= z_win
    if int(np.count_nonzero(keep_z)) < int(args.min_outer_points):
        return np.zeros((0,), dtype=np.float64)

    u = u[keep_z]
    v = v[keep_z]
    z = z[keep_z]
    pts = deproject_uvz_to_points(np.stack([u, v], axis=1), z, fx, fy, cx, cy)

    vec = pts - center_3d.reshape(1, 3)
    plane_d = vec @ normal_3d
    radial_vec = vec - np.outer(plane_d, normal_3d)
    rho = np.linalg.norm(radial_vec, axis=1)
    outer = (rho >= float(radius_m) * float(args.outer_inner_ratio)) & (
        rho <= float(radius_m) * float(args.outer_outer_ratio)
    )
    if int(np.count_nonzero(outer)) < int(args.min_outer_points):
        return np.zeros((0,), dtype=np.float64)

    offsets = plane_d[outer]
    clip = float(args.outer_offset_clip_mm) / 1000.0
    offsets = offsets[np.abs(offsets) <= clip]
    return offsets.astype(np.float64)


def classify_side(
    inner_offsets_m: np.ndarray,
    prev_label: str,
    hollow_threshold_mm: float,
    hysteresis_mm: float,
    min_points: int,
):
    if inner_offsets_m.size < int(min_points):
        return "unknown", 0.0, float("nan")

    median_offset = float(np.median(inner_offsets_m))
    thresh = float(hollow_threshold_mm) / 1000.0
    hyst = float(hysteresis_mm) / 1000.0

    if prev_label == "upside_down":
        decision = -(thresh - hyst)
    elif prev_label == "flat_up":
        decision = -(thresh + hyst)
    else:
        decision = -thresh

    is_hollow_up = median_offset < decision
    label = "upside_down" if is_hollow_up else "flat_up"
    margin = abs(median_offset - decision)
    score = float(np.clip(margin / max(thresh, 1e-6), 0.0, 1.0))
    return label, score, median_offset * 1000.0


def transform_from_center_normal(center_3d: np.ndarray, normal_3d: np.ndarray) -> np.ndarray:
    z_axis = normalize_vec(normal_3d)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = ref - np.dot(ref, z_axis) * z_axis
    if float(np.linalg.norm(x_axis)) < 1e-6:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = ref - np.dot(ref, z_axis) * z_axis
    x_axis = normalize_vec(x_axis)
    y_axis = normalize_vec(np.cross(z_axis, x_axis))
    x_axis = normalize_vec(np.cross(y_axis, z_axis))

    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    t[:3, 3] = center_3d
    return t


def quality_score(
    diameter_error_mm: float,
    diameter_tol_mm: float,
    plane_rmse_mm: float,
    circle_rmse_mm: float,
    coverage: float,
    boundary_points: int,
    expected_points: int,
    side_label: str,
) -> float:
    tol = max(0.1, float(diameter_tol_mm))
    diam_score = float(np.exp(-((float(diameter_error_mm) / tol) ** 2)))
    plane_score = float(np.exp(-((float(plane_rmse_mm) / 2.0) ** 2)))
    circle_score = float(np.exp(-((float(circle_rmse_mm) / 2.0) ** 2)))
    coverage_score = float(np.clip((float(coverage) - 0.4) / 0.6, 0.0, 1.0))
    support_score = float(np.clip(float(boundary_points) / max(1.0, float(expected_points) * 0.7), 0.0, 1.0))
    side_boost = 1.0 if side_label != "unknown" else 0.7
    return side_boost * (
        0.30 * diam_score
        + 0.20 * plane_score
        + 0.20 * circle_score
        + 0.15 * coverage_score
        + 0.15 * support_score
    )


def evaluate_candidate(
    candidate: CircleCandidate,
    gray_u8: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    args,
    rng: np.random.Generator,
    prev_label: str,
    debug_stats: Optional[dict] = None,
) -> Optional[PoseEstimate]:
    def fail(reason: str):
        if debug_stats is not None:
            debug_stats[reason] = int(debug_stats.get(reason, 0)) + 1
        return None

    boundary_uv = sample_boundary_uv(
        candidate,
        sample_count=int(args.boundary_samples),
        contour_band_ratio=float(args.contour_band_ratio),
    )
    boundary_pts, _ = pixels_to_3d_boundary_points(
        depth_u16,
        boundary_uv,
        depth_scale,
        fx,
        fy,
        cx,
        cy,
        patch=int(args.depth_patch),
    )
    if boundary_pts.shape[0] < int(args.min_boundary_points):
        return fail("boundary_pts")

    plane = fit_plane_ransac(
        boundary_pts,
        thresh_m=float(args.plane_inlier_mm) / 1000.0,
        ransac_iters=int(args.plane_ransac_iters),
        min_inliers=int(args.min_boundary_points),
        rng=rng,
    )
    if plane is None:
        return fail("plane")
    plane_center, plane_normal, plane_mask, plane_rmse_m = plane
    boundary_inliers = boundary_pts[plane_mask]
    if boundary_inliers.shape[0] < int(args.min_boundary_points):
        return fail("plane_inliers")

    ex, ey = build_plane_basis(plane_normal)
    rel = boundary_inliers - plane_center.reshape(1, 3)
    xy = np.stack([rel @ ex, rel @ ey], axis=1)
    circle_fit = fit_circle_trimmed(
        xy,
        trim_m=float(args.circle_inlier_mm) / 1000.0,
        min_inliers=int(args.min_circle_points),
        max_iters=4,
    )
    if circle_fit is None:
        return fail("circle_fit")
    cx2, cy2, radius_m, circle_mask, circle_rmse_m = circle_fit
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        return fail("radius_invalid")

    coverage = compute_angular_coverage(xy[circle_mask], cx2, cy2, bins=int(args.coverage_bins))
    if coverage < float(args.min_coverage):
        return fail("coverage")

    center_3d = plane_center + cx2 * ex + cy2 * ey
    normal_3d = orient_normal_toward_camera(plane_normal, center_3d)

    diameter_mm = float(radius_m * 2000.0)
    diameter_error_mm = float(diameter_mm - float(args.target_diameter_mm))
    if abs(diameter_error_mm) > float(args.diameter_tol_mm):
        return fail("diameter")

    rim_step_mm, rim_valid_ratio = rim_depth_step_metrics(
        depth_u16=depth_u16,
        candidate=candidate,
        depth_scale=depth_scale,
        depth_patch=int(args.depth_patch),
        rim_offset_px=float(args.rim_offset_px),
        rim_eval_samples=int(args.rim_eval_samples),
        rim_step_clip_mm=float(args.rim_step_clip_mm),
    )
    if rim_valid_ratio < float(args.min_rim_valid_ratio):
        return fail("rim_valid")
    if rim_step_mm < float(args.min_rim_depth_step_mm):
        return fail("rim_step")

    center_luma = float("nan")
    outer_luma = float("nan")
    luma_contrast = float("nan")
    if bool(args.use_white_prior):
        center_luma, outer_luma, luma_contrast, inner_std, _ = photometric_circle_metrics(
            gray_u8=gray_u8,
            candidate=candidate,
            inner_ratio=float(args.photo_inner_ratio),
            outer_inner_ratio=float(args.photo_outer_inner_ratio),
            outer_outer_ratio=float(args.photo_outer_outer_ratio),
        )
        if not np.isfinite(center_luma):
            return fail("photo_invalid")
        if center_luma < float(args.min_center_luma):
            return fail("photo_dark")
        if luma_contrast < float(args.min_luma_contrast):
            return fail("photo_contrast")
        if inner_std > float(args.max_center_luma_std):
            return fail("photo_std")

    inner_offsets = inner_offsets_from_depth(
        depth_u16,
        candidate,
        center_3d,
        normal_3d,
        radius_m,
        depth_scale,
        fx,
        fy,
        cx,
        cy,
        args,
    )
    outer_offsets = outer_offsets_from_depth(
        depth_u16,
        candidate,
        center_3d,
        normal_3d,
        radius_m,
        depth_scale,
        fx,
        fy,
        cx,
        cy,
        args,
    )
    outer_med_mm = float("nan")
    if outer_offsets.size >= int(args.min_outer_points):
        outer_med_mm = float(np.median(outer_offsets) * 1000.0)
    if bool(args.enforce_outer_step):
        if outer_offsets.size < int(args.min_outer_points):
            return fail("outer_pts")
        if outer_med_mm > -float(args.outer_step_min_mm):
            return fail("outer_step")

    side_label, side_score, inner_med_mm = classify_side(
        inner_offsets,
        prev_label=prev_label,
        hollow_threshold_mm=float(args.hollow_threshold_mm),
        hysteresis_mm=float(args.side_hysteresis_mm),
        min_points=int(args.min_inner_points),
    )

    transform = transform_from_center_normal(center_3d, normal_3d)
    q = quality_score(
        diameter_error_mm=diameter_error_mm,
        diameter_tol_mm=float(args.diameter_tol_mm),
        plane_rmse_mm=float(plane_rmse_m * 1000.0),
        circle_rmse_mm=float(circle_rmse_m * 1000.0),
        coverage=coverage,
        boundary_points=int(boundary_inliers.shape[0]),
        expected_points=int(args.boundary_samples),
        side_label=side_label,
    )
    if debug_stats is not None:
        debug_stats["pass"] = int(debug_stats.get("pass", 0)) + 1

    return PoseEstimate(
        center_3d=center_3d.astype(np.float64),
        normal_3d=normal_3d.astype(np.float64),
        radius_m=float(radius_m),
        diameter_mm=diameter_mm,
        diameter_error_mm=diameter_error_mm,
        side_label=side_label,
        side_score=float(side_score),
        inner_median_offset_mm=float(inner_med_mm),
        outer_median_offset_mm=float(outer_med_mm),
        rim_depth_step_mm=float(rim_step_mm),
        rim_valid_ratio=float(rim_valid_ratio),
        center_luma=float(center_luma),
        outer_luma=float(outer_luma),
        luma_contrast=float(luma_contrast),
        quality=float(q),
        plane_rmse_mm=float(plane_rmse_m * 1000.0),
        circle_rmse_mm=float(circle_rmse_m * 1000.0),
        boundary_points=int(boundary_inliers.shape[0]),
        coverage=float(coverage),
        transform=transform,
        candidate=candidate,
    )


def apply_temporal_smoothing(est: PoseEstimate, state: TrackState, alpha_old: float, target_diam_mm: float) -> PoseEstimate:
    a = float(np.clip(alpha_old, 0.0, 0.98))
    if state.center_3d is None or state.normal_3d is None or state.radius_m is None:
        return est

    est.center_3d = a * state.center_3d + (1.0 - a) * est.center_3d
    est.normal_3d = normalize_vec(a * state.normal_3d + (1.0 - a) * est.normal_3d)
    est.radius_m = float(a * state.radius_m + (1.0 - a) * est.radius_m)
    est.diameter_mm = float(est.radius_m * 2000.0)
    est.diameter_error_mm = float(est.diameter_mm - float(target_diam_mm))
    est.transform = transform_from_center_normal(est.center_3d, est.normal_3d)
    return est


def update_track_state_from_estimate(state: TrackState, est: PoseEstimate):
    state.center_3d = est.center_3d.copy()
    state.normal_3d = est.normal_3d.copy()
    state.radius_m = float(est.radius_m)
    if est.side_label != "unknown":
        state.last_label = est.side_label
    state.last_uv = (float(est.candidate.center_u), float(est.candidate.center_v))
    state.last_radius_px = float(est.candidate.radius_px)
    state.missed_frames = 0
    state.last_estimate = est
    state.pending_estimate = est
    state.confirm_count = max(1, int(state.confirm_count))
    state.confirmed = True


def make_pose_lineset(
    center_3d: np.ndarray,
    normal_3d: np.ndarray,
    radius_m: float,
    color_rgb: Tuple[float, float, float],
    segments: int,
    normal_len_m: float,
) -> o3d.geometry.LineSet:
    seg = max(20, int(segments))
    ex, ey = build_plane_basis(normal_3d)
    angles = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False, dtype=np.float64)
    ring = (
        center_3d.reshape(1, 3)
        + float(radius_m) * np.cos(angles)[:, None] * ex.reshape(1, 3)
        + float(radius_m) * np.sin(angles)[:, None] * ey.reshape(1, 3)
    )
    tip = center_3d + normalize_vec(normal_3d) * float(normal_len_m)
    pts = np.vstack([ring, center_3d.reshape(1, 3), tip.reshape(1, 3)])
    lines = [[i, (i + 1) % seg] for i in range(seg)]
    lines.append([seg, seg + 1])
    cols = [list(color_rgb) for _ in range(seg)] + [[1.0, 1.0, 1.0]]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.array(cols, dtype=np.float64))
    return ls


def pose_color_for_open3d(side_label: str):
    if side_label == "upside_down":
        return 1.0, 0.0, 0.0
    if side_label == "flat_up":
        return 0.0, 1.0, 0.0
    return 1.0, 1.0, 0.0


def pose_color_for_cv(side_label: str):
    if side_label == "upside_down":
        return (0, 0, 255)
    if side_label == "flat_up":
        return (0, 255, 0)
    return (0, 255, 255)


def update_pose_geometries(
    vis: o3d.visualization.Visualizer,
    old_geoms: List[o3d.geometry.Geometry],
    est: Optional[PoseEstimate],
    axis_size: float,
    ring_segments: int,
    normal_len_m: float,
) -> List[o3d.geometry.Geometry]:
    for g in old_geoms:
        vis.remove_geometry(g, reset_bounding_box=False)

    if est is None:
        return []

    color = pose_color_for_open3d(est.side_label)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axis_size))
    axis.transform(est.transform)
    ring = make_pose_lineset(
        est.center_3d,
        est.normal_3d,
        est.radius_m,
        color_rgb=color,
        segments=int(ring_segments),
        normal_len_m=float(normal_len_m),
    )
    vis.add_geometry(axis, reset_bounding_box=False)
    vis.add_geometry(ring, reset_bounding_box=False)
    return [axis, ring]


def project_point(point_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    z = float(point_3d[2])
    if z <= 1e-8:
        return None
    u = float(fx * (point_3d[0] / z) + cx)
    v = float(fy * (point_3d[1] / z) + cy)
    return int(round(u)), int(round(v))


def draw_overlay(
    bgr: np.ndarray,
    est: Optional[PoseEstimate],
    candidates: List[CircleCandidate],
    stable_circle: Optional[CircleCandidate],
    roi_rect: Tuple[int, int, int, int],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    args,
):
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = roi_rect

    if bool(args.canny_sliders):
        cv2.putText(
            bgr,
            f"Canny low/high: {int(args.canny_low)}/{int(args.canny_high)}",
            (14, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    if args.show_roi and not (x0 == 0 and y0 == 0 and x1 == w and y1 == h):
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (255, 180, 0), 1)

    if args.debug_candidates:
        max_dbg = max(1, int(args.debug_max_circles))
        for c in candidates[:max_dbg]:
            col = (200, 120, 0) if c.source == "contour" else (255, 220, 0)
            cv2.circle(
                bgr,
                (int(round(c.center_u)), int(round(c.center_v))),
                int(round(c.radius_px)),
                col,
                1,
                cv2.LINE_AA,
            )

    if bool(args.show_stable_circle) and stable_circle is not None:
        su = int(round(stable_circle.center_u))
        sv = int(round(stable_circle.center_v))
        sr = int(round(stable_circle.radius_px))
        cv2.circle(bgr, (su, sv), max(1, sr), (255, 170, 0), 2, cv2.LINE_AA)
        cv2.circle(bgr, (su, sv), 2, (255, 170, 0), -1, cv2.LINE_AA)
        cv2.putText(
            bgr,
            "stable circle",
            (su + 8, max(18, sv - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 170, 0),
            1,
            cv2.LINE_AA,
        )

    if est is not None:
        color = pose_color_for_cv(est.side_label)
        cu = int(round(est.candidate.center_u))
        cv = int(round(est.candidate.center_v))
        cr = int(round(est.candidate.radius_px))
        cv2.circle(bgr, (cu, cv), max(1, cr), color, 2, cv2.LINE_AA)
        cv2.circle(bgr, (cu, cv), 3, color, -1, cv2.LINE_AA)

        if est.candidate.contour is not None and est.candidate.contour.shape[0] >= 12:
            contour = est.candidate.contour.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(bgr, [contour], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

        p0 = project_point(est.center_3d, fx, fy, cx, cy)
        p1 = project_point(est.center_3d + est.normal_3d * float(args.normal_len_m), fx, fy, cx, cy)
        if p0 is not None and p1 is not None:
            cv2.arrowedLine(bgr, p0, p1, color, 2, tipLength=0.25)

        line1 = (
            f"side={est.side_label} "
            f"diam={est.diameter_mm:.2f}mm "
            f"err={est.diameter_error_mm:+.2f}mm "
            f"Q={est.quality:.2f}"
        )
        line2 = (
            f"plane={est.plane_rmse_mm:.2f}mm "
            f"circle={est.circle_rmse_mm:.2f}mm "
            f"coverage={est.coverage:.2f} "
            f"inner={est.inner_median_offset_mm:+.2f}mm "
            f"outer={est.outer_median_offset_mm:+.2f}mm "
            f"rim={est.rim_depth_step_mm:.2f}mm "
            f"L={est.center_luma:.0f}/{est.outer_luma:.0f} dL={est.luma_contrast:+.1f}"
        )
        line3 = (
            f"boundary_pts={est.boundary_points} source={est.candidate.source} "
            f"edge={est.candidate.edge_support:.2f} grad={est.candidate.grad_consistency:.2f}"
        )
        cv2.putText(bgr, line1, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        cv2.putText(bgr, line2, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, line3, (14, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(
            bgr,
            "No valid circle pose (metric or depth checks failed)",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )


def parse_args():
    ap = argparse.ArgumentParser(
        description="RGB-first circle-based 6D pose for bottle-cap-like cylinder using RealSense depth validation"
    )

    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no_res_fallback", action="store_true")

    ap.add_argument("--z_min", type=float, default=0.10)
    ap.add_argument("--z_max", type=float, default=1.00)
    ap.add_argument("--cloud_stride", type=int, default=2)
    ap.add_argument("--flip_y", action="store_true")
    ap.add_argument("--min_cloud_points", type=int, default=500)

    ap.add_argument("--filters", dest="filters", action="store_true")
    ap.add_argument("--no_filters", dest="filters", action="store_false")
    ap.set_defaults(filters=True)
    ap.add_argument("--emitter", type=int, default=1)
    ap.add_argument("--laser_power", type=float, default=None)

    ap.add_argument("--target_diameter_mm", type=float, default=29.3)
    ap.add_argument("--diameter_tol_mm", type=float, default=1.0)

    ap.add_argument("--min_radius_px", type=float, default=8.0)
    ap.add_argument("--max_radius_px", type=float, default=120.0)
    ap.add_argument("--blur_ksize", type=int, default=5)
    ap.add_argument("--equalize", dest="equalize", action="store_true")
    ap.add_argument("--no_equalize", dest="equalize", action="store_false")
    ap.set_defaults(equalize=False)
    ap.add_argument("--canny_low", type=float, default=60.0)
    ap.add_argument("--canny_high", type=float, default=180.0)
    ap.add_argument("--morph_close", type=int, default=3)
    ap.add_argument("--edge_cc_filter", dest="edge_cc_filter", action="store_true")
    ap.add_argument("--no_edge_cc_filter", dest="edge_cc_filter", action="store_false")
    ap.set_defaults(edge_cc_filter=True)
    ap.add_argument("--edge_cc_min_pixels", type=int, default=18)
    ap.add_argument("--edge_cc_max_pixels", type=int, default=0)
    ap.add_argument("--edge_cc_fallback", dest="edge_cc_fallback", action="store_true")
    ap.add_argument("--no_edge_cc_fallback", dest="edge_cc_fallback", action="store_false")
    ap.set_defaults(edge_cc_fallback=True)
    ap.add_argument("--edge_cc_fallback_min_candidates", type=int, default=1)
    ap.add_argument("--edge_cc_fallback_min_pixels", type=int, default=220)
    ap.add_argument("--contour_min_area_px", type=float, default=100.0)
    ap.add_argument("--contour_min_circularity", type=float, default=0.45)
    ap.add_argument("--contour_min_aspect", type=float, default=0.35)
    ap.add_argument("--contour_min_fill_ratio", type=float, default=0.45)
    ap.add_argument("--contour_band_ratio", type=float, default=0.35)
    ap.add_argument("--ring_eval_samples", type=int, default=96)
    ap.add_argument("--ring_edge_band_px", type=float, default=2.0)
    ap.add_argument("--min_edge_support", type=float, default=0.45)
    ap.add_argument("--grad_mag_min", type=float, default=10.0)
    ap.add_argument("--grad_align_abs_min", type=float, default=0.55)
    ap.add_argument("--min_grad_support", type=float, default=0.20)
    ap.add_argument("--min_grad_consistency", type=float, default=0.45)

    ap.add_argument("--use_hough", dest="use_hough", action="store_true")
    ap.add_argument("--no_hough", dest="use_hough", action="store_false")
    ap.set_defaults(use_hough=True)
    ap.add_argument("--hough_dp", type=float, default=1.2)
    ap.add_argument("--hough_min_dist_px", type=float, default=20.0)
    ap.add_argument("--hough_param1", type=float, default=120.0)
    ap.add_argument("--hough_param2", type=float, default=20.0)

    ap.add_argument("--max_candidates", type=int, default=10)
    ap.add_argument("--candidate_merge_center_px", type=float, default=14.0)
    ap.add_argument("--candidate_merge_radius_px", type=float, default=8.0)
    ap.add_argument("--prefer_inner_circle", dest="prefer_inner_circle", action="store_true")
    ap.add_argument("--no_prefer_inner_circle", dest="prefer_inner_circle", action="store_false")
    ap.set_defaults(prefer_inner_circle=True)
    ap.add_argument("--inner_center_tol_px", type=float, default=16.0)
    ap.add_argument("--inner_radius_gap_px", type=float, default=1.2)
    ap.add_argument("--boundary_samples", type=int, default=220)
    ap.add_argument("--depth_patch", type=int, default=2)
    ap.add_argument("--min_boundary_points", type=int, default=45)
    ap.add_argument("--rim_eval_samples", type=int, default=96)
    ap.add_argument("--rim_offset_px", type=float, default=3.0)
    ap.add_argument("--min_rim_valid_ratio", type=float, default=0.55)
    ap.add_argument("--min_rim_depth_step_mm", type=float, default=0.9)
    ap.add_argument("--rim_step_clip_mm", type=float, default=30.0)
    ap.add_argument("--use_white_prior", dest="use_white_prior", action="store_true")
    ap.add_argument("--no_white_prior", dest="use_white_prior", action="store_false")
    ap.set_defaults(use_white_prior=True)
    ap.add_argument("--photo_inner_ratio", type=float, default=0.52)
    ap.add_argument("--photo_outer_inner_ratio", type=float, default=1.12)
    ap.add_argument("--photo_outer_outer_ratio", type=float, default=1.65)
    ap.add_argument("--min_center_luma", type=float, default=115.0)
    ap.add_argument("--min_luma_contrast", type=float, default=10.0)
    ap.add_argument("--max_center_luma_std", type=float, default=42.0)

    ap.add_argument("--plane_ransac_iters", type=int, default=180)
    ap.add_argument("--plane_inlier_mm", type=float, default=1.8)
    ap.add_argument("--circle_inlier_mm", type=float, default=1.6)
    ap.add_argument("--min_circle_points", type=int, default=35)
    ap.add_argument("--coverage_bins", type=int, default=36)
    ap.add_argument("--min_coverage", type=float, default=0.55)

    ap.add_argument("--inner_radius_ratio", type=float, default=0.45)
    ap.add_argument("--inner_bbox_scale", type=float, default=1.25)
    ap.add_argument("--inner_z_window_mm", type=float, default=25.0)
    ap.add_argument("--inner_offset_clip_mm", type=float, default=20.0)
    ap.add_argument("--min_inner_points", type=int, default=80)
    ap.add_argument("--hollow_threshold_mm", type=float, default=1.2)
    ap.add_argument("--side_hysteresis_mm", type=float, default=0.25)
    ap.add_argument("--enforce_outer_step", dest="enforce_outer_step", action="store_true")
    ap.add_argument("--no_enforce_outer_step", dest="enforce_outer_step", action="store_false")
    ap.set_defaults(enforce_outer_step=True)
    ap.add_argument("--outer_inner_ratio", type=float, default=1.08)
    ap.add_argument("--outer_outer_ratio", type=float, default=1.45)
    ap.add_argument("--outer_bbox_scale", type=float, default=1.8)
    ap.add_argument("--outer_z_window_mm", type=float, default=60.0)
    ap.add_argument("--outer_offset_clip_mm", type=float, default=60.0)
    ap.add_argument("--min_outer_points", type=int, default=80)
    ap.add_argument("--outer_step_min_mm", type=float, default=0.8)

    ap.add_argument("--pose_alpha", type=float, default=0.45)
    ap.add_argument("--max_missed_frames", type=int, default=45)
    ap.add_argument("--confirm_frames", type=int, default=1)
    ap.add_argument("--confirm_center_tol_px", type=float, default=20.0)
    ap.add_argument("--confirm_radius_tol_px", type=float, default=8.0)
    ap.add_argument("--max_jump_px", type=float, default=80.0)
    ap.add_argument("--use_track_hypothesis", dest="use_track_hypothesis", action="store_true")
    ap.add_argument("--no_track_hypothesis", dest="use_track_hypothesis", action="store_false")
    ap.set_defaults(use_track_hypothesis=True)
    ap.add_argument("--track_center_z_tol_mm", type=float, default=25.0)
    ap.add_argument("--circle_track_alpha", type=float, default=0.70)
    ap.add_argument("--circle_track_jump_px", type=float, default=85.0)
    ap.add_argument("--circle_track_max_missed", type=int, default=30)
    ap.add_argument("--circle_track_prox_weight", type=float, default=0.55)
    ap.add_argument("--circle_use_photo_prior", dest="circle_use_photo_prior", action="store_true")
    ap.add_argument("--no_circle_use_photo_prior", dest="circle_use_photo_prior", action="store_false")
    ap.set_defaults(circle_use_photo_prior=True)
    ap.add_argument("--circle_photo_hard_gate", dest="circle_photo_hard_gate", action="store_true")
    ap.add_argument("--no_circle_photo_hard_gate", dest="circle_photo_hard_gate", action="store_false")
    ap.set_defaults(circle_photo_hard_gate=False)
    ap.add_argument("--circle_photo_fallback_relaxed", dest="circle_photo_fallback_relaxed", action="store_true")
    ap.add_argument("--no_circle_photo_fallback_relaxed", dest="circle_photo_fallback_relaxed", action="store_false")
    ap.set_defaults(circle_photo_fallback_relaxed=True)
    ap.add_argument("--circle_min_center_luma", type=float, default=115.0)
    ap.add_argument("--circle_min_luma_contrast", type=float, default=8.0)
    ap.add_argument("--circle_max_center_luma_std", type=float, default=42.0)
    ap.add_argument("--circle_track_photo_weight", type=float, default=1.10)
    ap.add_argument("--use_circle_depth_size_prior", dest="use_circle_depth_size_prior", action="store_true")
    ap.add_argument("--no_circle_depth_size_prior", dest="use_circle_depth_size_prior", action="store_false")
    ap.set_defaults(use_circle_depth_size_prior=True)
    ap.add_argument("--circle_size_prior_hard", type=float, default=0.70)
    ap.add_argument("--circle_track_size_weight", type=float, default=0.75)
    ap.add_argument("--show_stable_circle", dest="show_stable_circle", action="store_true")
    ap.add_argument("--no_show_stable_circle", dest="show_stable_circle", action="store_false")
    ap.set_defaults(show_stable_circle=True)
    ap.add_argument("--show_unconfirmed", dest="show_unconfirmed", action="store_true")
    ap.add_argument("--no_show_unconfirmed", dest="show_unconfirmed", action="store_false")
    ap.set_defaults(show_unconfirmed=True)
    ap.add_argument("--center_roi_ratio", type=float, default=0.50)
    ap.add_argument("--roi_scale", type=float, default=3.0)
    ap.add_argument("--roi_min_px", type=int, default=80)
    ap.add_argument("--keep_last_pose", dest="keep_last_pose", action="store_true")
    ap.add_argument("--no_keep_last_pose", dest="keep_last_pose", action="store_false")
    ap.set_defaults(keep_last_pose=True)

    ap.add_argument("--point_size", type=float, default=2.5)
    ap.add_argument("--axis_size", type=float, default=0.02)
    ap.add_argument("--ring_segments", type=int, default=72)
    ap.add_argument("--normal_len_m", type=float, default=0.02)

    ap.add_argument("--show_roi", action="store_true")
    ap.add_argument("--debug_candidates", action="store_true")
    ap.add_argument("--debug_max_circles", type=int, default=3)
    ap.add_argument("--debug_edges", action="store_true")
    ap.add_argument("--canny_sliders", action="store_true")
    ap.add_argument("--log_interval", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    pipe = rs.pipeline()
    profile, run_w, run_h, run_fps = start_rs_pipeline_with_fallback(
        pipe,
        req_w=int(args.w),
        req_h=int(args.h),
        req_fps=int(args.fps),
        allow_fallback=(not bool(args.no_res_fallback)),
    )
    print(f"[Info] stream mode depth+color={run_w}x{run_h}@{run_fps}")

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"[Info] depth_scale={depth_scale:.8f} m/unit")

    try:
        depth_sensor.set_option(rs.option.emitter_enabled, float(args.emitter))
    except Exception:
        pass
    if args.laser_power is not None:
        try:
            depth_sensor.set_option(rs.option.laser_power, float(args.laser_power))
        except Exception as exc:
            print(f"[WARN] laser_power not set: {exc}")

    align = rs.align(rs.stream.color)
    filters = make_filters(bool(args.filters))

    init_frames = align.process(pipe.wait_for_frames(5000))
    color0 = init_frames.get_color_frame()
    if not color0:
        raise RuntimeError("Color frame unavailable.")
    cprof = color0.get_profile().as_video_stream_profile()
    intr = cprof.get_intrinsics()
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)
    print(f"[Info] intr fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")
    print(
        "[Info] color code: red=upside_down (hollow side up), green=flat_up (closed flat side up), yellow=uncertain"
    )
    print(
        f"[Info] fixed center ROI ratio={float(args.center_roi_ratio):.2f} "
        f"(search area={float(args.center_roi_ratio) * 100.0:.0f}% x {float(args.center_roi_ratio) * 100.0:.0f}%) "
        f"confirm_frames={int(args.confirm_frames)} "
        f"rim_step>={float(args.min_rim_depth_step_mm):.2f}mm "
        f"max_missed={int(args.max_missed_frames)} "
        f"track_hyp={'on' if bool(args.use_track_hypothesis) else 'off'} "
        f"jump<={float(args.max_jump_px):.0f}px pose_alpha={float(args.pose_alpha):.2f} "
        f"white_prior={'on' if bool(args.use_white_prior) else 'off'} "
        f"luma>={float(args.min_center_luma):.0f} dL>={float(args.min_luma_contrast):.1f} "
        f"circle_track={'on' if bool(args.show_stable_circle) else 'off'} "
        f"cjump<={float(args.circle_track_jump_px):.0f}px calpha={float(args.circle_track_alpha):.2f} "
        f"cPhoto={'on' if bool(args.circle_use_photo_prior) else 'off'} "
        f"cPhotoHard={'on' if bool(args.circle_photo_hard_gate) else 'off'} "
        f"cL>={float(args.circle_min_center_luma):.0f} cdL>={float(args.circle_min_luma_contrast):.1f} "
        f"edgeCC={'on' if bool(args.edge_cc_filter) else 'off'} "
        f"edgeCCfb={'on' if bool(args.edge_cc_fallback) else 'off'} "
        f"innerPref={'on' if bool(args.prefer_inner_circle) else 'off'} "
        f"eq={'on' if bool(args.equalize) else 'off'}"
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window("RGB Circle 6D Pose (RealSense + Open3D)", 1400, 850, visible=True)
    render_opt = vis.get_render_option()
    render_opt.point_size = float(args.point_size)

    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    vis.add_geometry(world_axis)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
    vis.add_geometry(pcd)

    pose_geoms: List[o3d.geometry.Geometry] = []
    track = TrackState()

    first_view = True
    last_log_t = 0.0
    slider_window = "Canny Sliders"
    canny_low_rt = float(args.canny_low)
    canny_high_rt = float(args.canny_high)
    last_slider_log_t = 0.0
    if bool(args.canny_sliders):
        init_canny_sliders(slider_window, canny_low_rt, canny_high_rt)
        print("[Info] canny sliders enabled: adjust 'canny_low' and 'canny_high' live")
    print("[Keys] q / ESC to quit")

    try:
        while True:
            frames = pipe.wait_for_frames(5000)
            frames = align.process(frames)
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            if filters is not None:
                df = depth
                for f in filters:
                    df = f.process(df)
                depth = df.as_depth_frame()

            rgb = np.asanyarray(color.get_data())
            depth_u16 = np.asanyarray(depth.get_data())

            if bool(args.canny_sliders):
                canny_low_rt, canny_high_rt = read_canny_sliders(slider_window, canny_low_rt, canny_high_rt)
                args.canny_low = float(canny_low_rt)
                args.canny_high = float(canny_high_rt)
                now_s = time.time()
                if now_s - last_slider_log_t > 1.0:
                    print(f"[Canny] low={args.canny_low:.0f} high={args.canny_high:.0f}")
                    last_slider_log_t = now_s

            cloud_pts, cloud_cols = rs_cloud_from_aligned_depth(
                depth_u16=depth_u16,
                rgb=rgb,
                depth_scale=depth_scale,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                stride=int(args.cloud_stride),
                z_min=float(args.z_min),
                z_max=float(args.z_max),
                flip_y=bool(args.flip_y),
            )
            if cloud_pts.shape[0] >= int(args.min_cloud_points):
                pcd.points = o3d.utility.Vector3dVector(cloud_pts.astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(cloud_cols.astype(np.float64))
                vis.update_geometry(pcd)
                if first_view:
                    vis.reset_view_point(True)
                    first_view = False

            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            roi_rect = get_center_roi(gray.shape, float(args.center_roi_ratio))
            x0, y0, x1, y1 = roi_rect
            gray_roi = gray[y0:y1, x0:x1]
            candidates, edges = detect_circle_candidates(gray_roi, args)

            for c in candidates:
                c.center_u += float(x0)
                c.center_v += float(y0)
                if c.contour is not None:
                    c.contour[:, 0] += float(x0)
                    c.contour[:, 1] += float(y0)

            anchor_uv = track.circle_uv if track.circle_uv is not None else track.last_uv
            if anchor_uv is not None:
                for c in candidates:
                    d = float(np.hypot(c.center_u - anchor_uv[0], c.center_v - anchor_uv[1]))
                    scale = max(1.0, float(c.radius_px) * 3.0)
                    c.score_2d += float(np.clip(1.0 - d / scale, 0.0, 1.0)) * 0.25
            candidates = sorted(candidates, key=lambda x: float(x.score_2d), reverse=True)

            debug_stats = {}
            (
                raw_circle_best,
                circle_best_score,
                circle_rel_err,
                circle_r_pred,
                circle_center_luma,
                circle_luma_contrast,
                circle_relaxed_mode,
                circle_photo_pass_count,
            ) = choose_best_circle_candidate(
                candidates=candidates,
                track=track,
                gray_u8=gray,
                depth_u16=depth_u16,
                depth_scale=float(depth_scale),
                fx=float(fx),
                args=args,
            )
            stable_circle = update_circle_track(track, raw_circle_best, args)
            if raw_circle_best is None:
                debug_stats["circle_pick_none"] = int(debug_stats.get("circle_pick_none", 0)) + 1
            if circle_relaxed_mode:
                debug_stats["circle_photo_relaxed"] = int(debug_stats.get("circle_photo_relaxed", 0)) + 1

            eval_candidates = []
            if stable_circle is not None:
                eval_candidates.append(stable_circle)
            for c in candidates[: int(args.max_candidates)]:
                if stable_circle is not None:
                    d = float(np.hypot(c.center_u - stable_circle.center_u, c.center_v - stable_circle.center_v))
                    if d <= float(args.candidate_merge_center_px) and abs(c.radius_px - stable_circle.radius_px) <= float(
                        args.candidate_merge_radius_px
                    ):
                        continue
                eval_candidates.append(c)
            if bool(args.use_track_hypothesis):
                tc = make_track_candidate(track)
                if tc is not None:
                    zc = depth_at_pixel_m(
                        depth_u16,
                        tc.center_u,
                        tc.center_v,
                        depth_scale=float(depth_scale),
                        patch=int(args.depth_patch),
                    )
                    if np.isfinite(zc) and track.center_3d is not None:
                        dz_mm = abs(float(zc) - float(track.center_3d[2])) * 1000.0
                        if dz_mm <= float(args.track_center_z_tol_mm):
                            eval_candidates.insert(0, tc)
                        else:
                            debug_stats["track_z_gate"] = int(debug_stats.get("track_z_gate", 0)) + 1
                    else:
                        debug_stats["track_z_invalid"] = int(debug_stats.get("track_z_invalid", 0)) + 1

            best_est = None
            best_q = -1.0
            for c in eval_candidates:
                if (
                    c.source not in ("track", "circle_track")
                    and track.confirmed
                    and track.last_uv is not None
                    and track.missed_frames <= int(args.max_missed_frames)
                ):
                    jump = float(np.hypot(c.center_u - track.last_uv[0], c.center_v - track.last_uv[1]))
                    if jump > float(args.max_jump_px):
                        debug_stats["jump_gate"] = int(debug_stats.get("jump_gate", 0)) + 1
                        continue
                est = evaluate_candidate(
                    candidate=c,
                    gray_u8=gray,
                    depth_u16=depth_u16,
                    depth_scale=depth_scale,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    args=args,
                    rng=rng,
                    prev_label=track.last_label,
                    debug_stats=debug_stats,
                )
                if est is None:
                    continue
                candidate_q = float(est.quality)
                if c.source in ("track", "circle_track"):
                    candidate_q += 0.03
                if candidate_q > best_q:
                    best_q = float(candidate_q)
                    best_est = est

            if best_est is not None:
                track.missed_frames = 0
                best_est = apply_temporal_smoothing(
                    best_est,
                    track,
                    alpha_old=float(args.pose_alpha),
                    target_diam_mm=float(args.target_diameter_mm),
                )
                if track.pending_estimate is None:
                    track.pending_estimate = best_est
                    track.confirm_count = 1
                else:
                    d = float(
                        np.hypot(
                            best_est.candidate.center_u - track.pending_estimate.candidate.center_u,
                            best_est.candidate.center_v - track.pending_estimate.candidate.center_v,
                        )
                    )
                    dr = float(abs(best_est.candidate.radius_px - track.pending_estimate.candidate.radius_px))
                    if d <= float(args.confirm_center_tol_px) and dr <= float(args.confirm_radius_tol_px):
                        track.confirm_count += 1
                    else:
                        track.confirm_count = 1
                    track.pending_estimate = best_est

                if track.confirm_count >= int(args.confirm_frames):
                    track.confirmed = True
                    update_track_state_from_estimate(track, best_est)
                    display_est = best_est
                else:
                    if track.confirmed and bool(args.keep_last_pose) and track.last_estimate is not None:
                        display_est = track.last_estimate
                    else:
                        display_est = best_est if bool(args.show_unconfirmed) else None
            else:
                track.pending_estimate = None
                if track.confirmed:
                    track.confirm_count = int(args.confirm_frames)
                else:
                    track.confirm_count = 0
                track.missed_frames += 1
                if track.missed_frames > int(args.max_missed_frames):
                    track.last_uv = None
                    track.last_radius_px = None
                    track.confirmed = False
                    display_est = None
                else:
                    if track.confirmed and bool(args.keep_last_pose):
                        display_est = track.last_estimate
                    else:
                        display_est = None

            pose_geoms = update_pose_geometries(
                vis=vis,
                old_geoms=pose_geoms,
                est=display_est,
                axis_size=float(args.axis_size),
                ring_segments=int(args.ring_segments),
                normal_len_m=float(args.normal_len_m),
            )

            is_open = vis.poll_events()
            vis.update_renderer()
            if not is_open:
                break

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            draw_overlay(
                bgr=bgr,
                est=display_est,
                candidates=candidates[: int(args.max_candidates)],
                stable_circle=stable_circle,
                roi_rect=roi_rect,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                args=args,
            )
            cv2.imshow("RGB Circle Pose", bgr)
            if bool(args.debug_edges):
                cv2.imshow("Circle Edges", edges)

            now = time.time()
            if now - last_log_t >= float(args.log_interval):
                reason_txt = debug_reasons_line(debug_stats, top_k=5)
                if stable_circle is None:
                    circle_txt = f"circle=none cmiss={track.circle_missed_frames}"
                else:
                    circle_txt = (
                        f"circle=({stable_circle.center_u:.1f},{stable_circle.center_v:.1f},r={stable_circle.radius_px:.1f}) "
                        f"cmiss={track.circle_missed_frames} cscore={circle_best_score:.2f} "
                        f"rpred={circle_r_pred:.1f} rerr={circle_rel_err:.2f} "
                        f"cL={circle_center_luma:.0f} cdL={circle_luma_contrast:+.1f} "
                        f"cPass={int(circle_photo_pass_count)} cMode={'relaxed' if circle_relaxed_mode else 'normal'}"
                    )
                if display_est is None:
                    print(
                        f"[Pose] none | candidates={len(candidates)} | missed={track.missed_frames} "
                        f"| confirm={track.confirm_count}/{int(args.confirm_frames)} "
                        f"| cloud_pts={cloud_pts.shape[0]} | {circle_txt} | {reason_txt}"
                    )
                else:
                    t = display_est.center_3d
                    print(
                        f"[Pose] side={display_est.side_label:11s} q={display_est.quality:.2f} "
                        f"diam={display_est.diameter_mm:.2f}mm err={display_est.diameter_error_mm:+.2f}mm "
                        f"inner={display_est.inner_median_offset_mm:+.2f}mm outer={display_est.outer_median_offset_mm:+.2f}mm "
                        f"rim={display_est.rim_depth_step_mm:.2f}mm rv={display_est.rim_valid_ratio:.2f} "
                        f"L={display_est.center_luma:.0f}/{display_est.outer_luma:.0f} dL={display_est.luma_contrast:+.1f} "
                        f"t=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) "
                        f"cov={display_est.coverage:.2f} pts={display_est.boundary_points} "
                        f"edge={display_est.candidate.edge_support:.2f} grad={display_est.candidate.grad_consistency:.2f} "
                        f"confirm={track.confirm_count}/{int(args.confirm_frames)} | {circle_txt} | {reason_txt}"
                    )
                last_log_t = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()


if __name__ == "__main__":
    main()
