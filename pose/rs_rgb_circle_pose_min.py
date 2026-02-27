import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

try:
    import open3d as o3d
except Exception:
    o3d = None


@dataclass
class EllipseCandidate:
    cx: float
    cy: float
    w: float
    h: float
    angle_deg: float
    score2d: float
    source: str = "mix"
    edge_support: float = 0.0
    arc_coverage: float = 0.0
    center_depth_m: float = float("nan")
    boundary_center_dz_m: float = 0.0
    contour: Optional[np.ndarray] = None


@dataclass
class PoseResult:
    center_3d: np.ndarray
    normal_3d: np.ndarray
    radius_m: float
    diameter_mm: float
    diameter_err_mm: float
    diameter_circle_mm: float
    diameter_mode: str
    plane_rmse_mm: float
    circle_rmse_mm: float
    coverage: float
    quality: float
    cand: EllipseCandidate


@dataclass
class PoseTrack:
    tid: int
    cand: EllipseCandidate
    center_3d: np.ndarray
    normal_3d: np.ndarray
    radius_m: float
    diameter_mm: float
    diameter_circle_mm: float
    diameter_mode: str
    quality: float
    coverage: float
    hits: int = 1
    missed: int = 0


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v.copy()
    return v / n


def depth_at_m(depth_u16: np.ndarray, u: float, v: float, depth_scale: float, patch: int) -> float:
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
    vals = depth_u16[y0:y1, x0:x1].reshape(-1)
    vals = vals[vals > 0]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals) * float(depth_scale))


def deproject(uv: np.ndarray, z: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    x = (uv[:, 0] - cx) * z / fx
    y = (uv[:, 1] - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float64)


def sample_ellipse_boundary(c: EllipseCandidate, n: int) -> np.ndarray:
    n = max(24, int(n))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    a = max(1e-3, float(c.w) * 0.5)
    b = max(1e-3, float(c.h) * 0.5)
    ang = np.deg2rad(float(c.angle_deg))
    ca = float(np.cos(ang))
    sa = float(np.sin(ang))
    x = a * np.cos(t)
    y = b * np.sin(t)
    u = float(c.cx) + x * ca - y * sa
    v = float(c.cy) + x * sa + y * ca
    return np.stack([u, v], axis=1).astype(np.float64)


def sample_candidate_boundary(c: EllipseCandidate, n: int) -> np.ndarray:
    n = max(24, int(n))
    if c.contour is None or int(c.contour.shape[0]) < 8:
        return sample_ellipse_boundary(c, n)
    pts = c.contour.reshape(-1, 2).astype(np.float64)
    m = int(pts.shape[0])
    if m <= n:
        return pts
    idx = np.linspace(0, m - 1, n, dtype=np.int32)
    return pts[idx]


def fit_plane(pts: np.ndarray, inlier_mm: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if pts.shape[0] < 6:
        return None
    center = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    normal = unit(vh[-1])
    d = np.abs((pts - center) @ normal)
    thr = float(inlier_mm) / 1000.0
    inliers = d <= thr
    if int(np.count_nonzero(inliers)) < 6:
        return None
    p = pts[inliers]
    center = np.mean(p, axis=0)
    _, _, vh = np.linalg.svd(p - center, full_matrices=False)
    normal = unit(vh[-1])
    d2 = np.abs((p - center) @ normal)
    rmse = float(np.sqrt(np.mean(d2 * d2)))
    return center, normal, inliers, rmse


def plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = unit(normal)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ex = unit(np.cross(ref, n))
    ey = unit(np.cross(n, ex))
    return ex, ey


def fit_circle_kasa(xy: np.ndarray) -> Optional[Tuple[float, float, float]]:
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
    if r2 <= 0.0 or not np.isfinite(r2):
        return None
    return float(cx), float(cy), float(np.sqrt(r2))


def fit_circle_trimmed(xy: np.ndarray, inlier_mm: float, min_pts: int) -> Optional[Tuple[float, float, float, np.ndarray, float]]:
    fit = fit_circle_kasa(xy)
    if fit is None:
        return None
    cx, cy, r = fit
    resid = np.abs(np.linalg.norm(xy - np.array([cx, cy]), axis=1) - r)
    thr = float(inlier_mm) / 1000.0
    inliers = resid <= thr
    if int(np.count_nonzero(inliers)) < int(min_pts):
        return None
    fit2 = fit_circle_kasa(xy[inliers])
    if fit2 is None:
        return None
    cx, cy, r = fit2
    resid = np.abs(np.linalg.norm(xy - np.array([cx, cy]), axis=1) - r)
    inliers = resid <= thr
    if int(np.count_nonzero(inliers)) < int(min_pts):
        return None
    rmse = float(np.sqrt(np.mean((resid[inliers] ** 2))))
    return float(cx), float(cy), float(r), inliers, rmse


def coverage_ratio(xy: np.ndarray, cx: float, cy: float, bins: int = 36) -> float:
    if xy.shape[0] < 6:
        return 0.0
    a = np.mod(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx), 2.0 * np.pi)
    h, _ = np.histogram(a, bins=max(12, int(bins)), range=(0.0, 2.0 * np.pi))
    return float(np.count_nonzero(h > 0)) / float(h.size)


def ring_edge_support(edges: np.ndarray, cx: float, cy: float, r: float, samples: int, band: int) -> float:
    h, w = edges.shape
    n = max(24, int(samples))
    b = max(1, int(band))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    hit = 0
    valid = 0
    for a in t:
        u = int(round(float(cx + r * np.cos(a))))
        v = int(round(float(cy + r * np.sin(a))))
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
        valid += 1
        x0 = max(0, u - b)
        y0 = max(0, v - b)
        x1 = min(w, u + b + 1)
        y1 = min(h, v + b + 1)
        if np.any(edges[y0:y1, x0:x1] > 0):
            hit += 1
    if valid == 0:
        return 0.0
    return float(hit) / float(valid)


def ellipse_edge_support_from_dist(
    dist_map: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
    samples: int,
    max_dist_px: float,
    arc_bins: int,
) -> Tuple[float, float, float]:
    img_h, img_w = dist_map.shape[:2]
    n = max(36, int(samples))
    bins = max(12, int(arc_bins))
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
    m = (u >= 0.0) & (u <= float(img_w - 1)) & (v >= 0.0) & (v <= float(img_h - 1))
    if int(np.count_nonzero(m)) < 16:
        return 0.0, 0.0, 1e9
    ui = np.clip(np.rint(u[m]).astype(np.int32), 0, img_w - 1)
    vi = np.clip(np.rint(v[m]).astype(np.int32), 0, img_h - 1)
    d = dist_map[vi, ui]
    hits = d <= float(max_dist_px)
    support = float(np.count_nonzero(hits)) / float(d.size)
    if int(np.count_nonzero(hits)) == 0:
        return support, 0.0, float(np.mean(d))
    t_hit = t[m][hits]
    bidx = np.mod(np.floor((t_hit / (2.0 * np.pi)) * float(bins)).astype(np.int32), bins)
    occ = np.zeros((bins,), dtype=np.uint8)
    occ[bidx] = 1
    arc = float(np.count_nonzero(occ)) / float(bins)
    return support, arc, float(np.mean(d))


def ellipse_geom_ok(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    min_diam_px: float,
    max_diam_px: float,
    min_aspect: float,
    border_px: float,
) -> bool:
    major = max(float(w), float(h))
    minor = min(float(w), float(h))
    if major < float(min_diam_px) or major > float(max_diam_px):
        return False
    if minor / max(major, 1e-6) < float(min_aspect):
        return False
    b = float(border_px)
    if float(cx) < b or float(cx) > (float(img_w - 1) - b) or float(cy) < b or float(cy) > (float(img_h - 1) - b):
        return False
    return True


def depth_size_prior_score(
    cxp: float,
    cyp: float,
    major_px: float,
    depth_u16: Optional[np.ndarray],
    depth_scale: float,
    fx: Optional[float],
    fy: Optional[float],
    args,
) -> float:
    if depth_u16 is None or fx is None or fy is None or (not bool(args.use_depth_size_prior)):
        return 0.5
    d = depth_at_m(depth_u16, float(cxp), float(cyp), float(depth_scale), int(args.size_prior_depth_patch))
    if (not np.isfinite(d)) or d <= 1e-6:
        return 0.5
    d_m = float(args.target_diameter_mm) / 1000.0
    f = 0.5 * (float(fx) + float(fy))
    pred_px = float(d_m * f / d)
    err = abs(float(major_px) - pred_px)
    if err > float(args.size_prior_tol_px):
        return -1.0
    sig = max(1e-6, float(args.size_prior_sigma_px))
    return float(np.exp(-((err / sig) ** 2)))


def boundary_center_depth_delta(
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
    depth_u16: Optional[np.ndarray],
    depth_scale: float,
    patch: int,
    samples: int,
) -> Tuple[bool, float, float]:
    if depth_u16 is None:
        return True, float("nan"), 0.0
    dc = depth_at_m(depth_u16, float(cx), float(cy), float(depth_scale), int(patch))
    if not np.isfinite(dc) or dc <= 1e-6:
        return True, float("nan"), 0.0
    n = max(24, int(samples))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    a = max(1e-3, 0.5 * float(w))
    b = max(1e-3, 0.5 * float(h))
    ang = np.deg2rad(float(angle_deg))
    ca = float(np.cos(ang))
    sa = float(np.sin(ang))
    x = a * np.cos(t)
    y = b * np.sin(t)
    uu = float(cx) + x * ca - y * sa
    vv = float(cy) + x * sa + y * ca
    dz_vals: List[float] = []
    for u, v in zip(uu, vv):
        db = depth_at_m(depth_u16, float(u), float(v), float(depth_scale), int(patch))
        if np.isfinite(db) and db > 1e-6:
            dz_vals.append(float(db - dc))
    if len(dz_vals) < max(10, n // 4):
        return True, float(dc), 0.0
    dz_med = float(np.median(np.asarray(dz_vals, dtype=np.float64)))
    return True, float(dc), float(dz_med)


def deduplicate_candidates(
    cands: List[EllipseCandidate],
    center_tol_px: float,
    radius_tol_px: float,
    center_frac: float,
    radius_frac: float,
    aspect_tol: float,
    max_keep: int,
) -> List[EllipseCandidate]:
    out: List[EllipseCandidate] = []
    ordered = sorted(cands, key=lambda c: float(c.score2d), reverse=True)
    for c in ordered:
        rc = 0.25 * (float(c.w) + float(c.h))
        dup = False
        for k in out:
            rk = 0.25 * (float(k.w) + float(k.h))
            d = float(np.hypot(float(c.cx) - float(k.cx), float(c.cy) - float(k.cy)))
            ct = max(float(center_tol_px), float(center_frac) * min(rc, rk))
            rt = max(float(radius_tol_px), float(radius_frac) * min(rc, rk))
            ac = min(float(c.w), float(c.h)) / max(float(c.w), float(c.h), 1e-6)
            ak = min(float(k.w), float(k.h)) / max(float(k.w), float(k.h), 1e-6)
            same_shape = abs(ac - ak) <= float(aspect_tol)
            if d <= ct and abs(rc - rk) <= rt and same_shape:
                dup = True
                break
        if dup:
            continue
        out.append(c)
        if len(out) >= int(max_keep):
            break
    return out


def merge_close_tracks(tracks: List[PoseTrack], merge_px: float, merge_3d_m: float) -> List[PoseTrack]:
    if len(tracks) < 2:
        return tracks
    keep: List[PoseTrack] = []
    order = sorted(tracks, key=lambda t: (int(t.hits), float(t.quality)), reverse=True)
    for t in order:
        dup = False
        for k in keep:
            d2 = float(np.hypot(float(t.cand.cx) - float(k.cand.cx), float(t.cand.cy) - float(k.cand.cy)))
            d3 = float(np.linalg.norm(t.center_3d - k.center_3d))
            if d2 <= float(merge_px) and d3 <= float(merge_3d_m):
                dup = True
                break
        if not dup:
            keep.append(t)
    keep.sort(key=lambda t: int(t.tid))
    return keep


def detect_ellipse_candidates(
    gray: np.ndarray,
    args,
    depth_u16: Optional[np.ndarray] = None,
    depth_scale: float = 1.0,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
) -> Tuple[List[EllipseCandidate], np.ndarray]:
    img_h, img_w = gray.shape[:2]
    blur_k = max(3, int(args.blur_ksize))
    if blur_k % 2 == 0:
        blur_k += 1
    g = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    edges = cv2.Canny(g, int(args.canny_low), int(args.canny_high))
    if int(args.morph_close) > 1:
        k = int(args.morph_close)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ker)

    inv = cv2.bitwise_not(edges)
    dist_map = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    out: List[EllipseCandidate] = []

    def add_candidate(
        cx: float,
        cy: float,
        ew: float,
        eh: float,
        angle: float,
        source: str,
        min_support: float,
        min_arc: float,
        base_score: float,
        contour: Optional[np.ndarray] = None,
    ) -> None:
        if not ellipse_geom_ok(
            cx=float(cx),
            cy=float(cy),
            w=float(ew),
            h=float(eh),
            img_w=int(img_w),
            img_h=int(img_h),
            min_diam_px=float(args.min_diameter_px),
            max_diam_px=float(args.max_diameter_px),
            min_aspect=float(args.min_aspect),
            border_px=float(args.candidate_border_margin_px),
        ):
            return
        support, arc_cov, mean_dist = ellipse_edge_support_from_dist(
            dist_map=dist_map,
            cx=float(cx),
            cy=float(cy),
            w=float(ew),
            h=float(eh),
            angle_deg=float(angle),
            samples=int(args.edge_support_samples),
            max_dist_px=float(args.edge_support_dist_px),
            arc_bins=int(args.edge_arc_bins),
        )
        if support < float(min_support) or arc_cov < float(min_arc):
            return
        _, center_depth_m, dz_med = boundary_center_depth_delta(
            cx=float(cx),
            cy=float(cy),
            w=float(ew),
            h=float(eh),
            angle_deg=float(angle),
            depth_u16=depth_u16,
            depth_scale=float(depth_scale),
            patch=int(args.depth_gate_patch),
            samples=int(args.depth_gate_samples),
        )
        if bool(args.reject_lower_edge) and np.isfinite(center_depth_m):
            if float(dz_med) > (float(args.max_boundary_center_dz_mm) / 1000.0):
                return
        major = max(float(ew), float(eh))
        minor = min(float(ew), float(eh))
        aspect = minor / max(major, 1e-6)
        size_prior = depth_size_prior_score(
            cxp=float(cx),
            cyp=float(cy),
            major_px=major,
            depth_u16=depth_u16,
            depth_scale=float(depth_scale),
            fx=fx,
            fy=fy,
            args=args,
        )
        if size_prior < 0.0:
            return
        score = (
            float(base_score)
            + 0.42 * float(support)
            + 0.28 * float(arc_cov)
            + 0.15 * float(aspect)
            + 0.10 * float(size_prior)
            + 0.05 * float(np.exp(-max(0.0, mean_dist - 0.5)))
            + 0.06 * float(np.exp(-max(0.0, dz_med * 1000.0 / max(1e-6, float(args.max_boundary_center_dz_mm)))))
        )
        out.append(
            EllipseCandidate(
                cx=float(cx),
                cy=float(cy),
                w=float(ew),
                h=float(eh),
                angle_deg=float(angle),
                score2d=float(score),
                source=str(source),
                edge_support=float(support),
                arc_coverage=float(arc_cov),
                center_depth_m=float(center_depth_m),
                boundary_center_dz_m=float(dz_med),
                contour=contour,
            )
        )

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cnt is None or len(cnt) < int(args.min_contour_points):
            continue
        if len(cnt) < 5:
            continue
        p0 = cnt[0, 0].astype(np.float64)
        p1 = cnt[-1, 0].astype(np.float64)
        end_gap = float(np.linalg.norm(p0 - p1))
        per_open = float(cv2.arcLength(cnt, False))
        per_closed = float(cv2.arcLength(cnt, True))
        area = float(abs(cv2.contourArea(cnt)))
        circularity = 0.0
        if per_closed > 1e-6:
            circularity = float((4.0 * np.pi * area) / (per_closed * per_closed))
        # Open arc contours are expected with overlap/occlusion; closed contours use classic gating.
        is_open_arc = (end_gap > float(args.open_arc_end_gap_px)) or (per_open > 1e-6 and end_gap > 0.08 * per_open)
        if is_open_arc:
            if per_open < float(args.min_open_arc_length_px):
                continue
        else:
            if area < float(args.min_area_px):
                continue
            if circularity < float(args.min_circularity):
                continue
        try:
            (cx, cy), (ew, eh), angle = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        add_candidate(
            cx=float(cx),
            cy=float(cy),
            ew=float(ew),
            eh=float(eh),
            angle=float(angle),
            source="contour",
            min_support=float(args.open_arc_min_edge_support) if is_open_arc else float(args.contour_min_edge_support),
            min_arc=float(args.open_arc_min_arc) if is_open_arc else float(args.contour_min_arc),
            base_score=float(0.20 + 0.12 * circularity + (0.14 if is_open_arc else 0.0)),
            contour=cnt.reshape(-1, 2).astype(np.float32),
        )

    if bool(args.use_hough):
        circles = cv2.HoughCircles(
            g,
            cv2.HOUGH_GRADIENT,
            dp=float(args.hough_dp),
            minDist=float(args.hough_min_dist_px),
            param1=float(args.hough_param1),
            param2=float(args.hough_param2),
            minRadius=int(args.min_diameter_px * 0.5),
            maxRadius=int(args.max_diameter_px * 0.5),
        )
        if circles is not None:
            for x, y, r in circles[0]:
                add_candidate(
                    cx=float(x),
                    cy=float(y),
                    ew=float(2.0 * r),
                    eh=float(2.0 * r),
                    angle=0.0,
                    source="hough",
                    min_support=float(args.hough_min_edge_support),
                    min_arc=float(args.hough_min_arc),
                    base_score=0.14,
                    contour=None,
                )

    if bool(args.use_ransac):
        ys, xs = np.where(edges > 0)
        n_edge = int(xs.size)
        if n_edge >= max(5, int(args.ransac_min_edge_points)):
            pts_xy = np.column_stack([xs, ys]).astype(np.float32)
            max_pts = max(64, int(args.ransac_max_edge_points))
            rng = np.random.default_rng(None if int(args.ransac_seed) < 0 else int(args.ransac_seed))
            if pts_xy.shape[0] > max_pts:
                sel = rng.choice(pts_xy.shape[0], size=max_pts, replace=False)
                pts_xy = pts_xy[sel]
            n = pts_xy.shape[0]
            iters = max(0, int(args.ransac_iters))
            for _ in range(iters):
                if n < 5:
                    break
                idx = rng.choice(n, size=5, replace=False)
                smp = pts_xy[idx].reshape(-1, 1, 2)
                try:
                    (cx, cy), (ew, eh), angle = cv2.fitEllipse(smp)
                except cv2.error:
                    continue
                add_candidate(
                    cx=float(cx),
                    cy=float(cy),
                    ew=float(ew),
                    eh=float(eh),
                    angle=float(angle),
                    source="ransac",
                    min_support=float(args.ransac_min_edge_support),
                    min_arc=float(args.ransac_min_arc),
                    base_score=0.24,
                    contour=None,
                )

    out = deduplicate_candidates(
        out,
        center_tol_px=float(args.merge_center_px),
        radius_tol_px=float(args.merge_radius_px),
        center_frac=float(args.merge_center_frac),
        radius_frac=float(args.merge_radius_frac),
        aspect_tol=float(args.merge_aspect_tol),
        max_keep=int(args.max_candidates),
    )
    return out, edges


def estimate_pose_from_candidate(
    cand: EllipseCandidate,
    depth_u16: np.ndarray,
    depth_scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    args,
) -> Optional[PoseResult]:
    uv = sample_candidate_boundary(cand, int(args.boundary_samples))
    z = []
    keep_uv = []
    for p in uv:
        d = depth_at_m(depth_u16, float(p[0]), float(p[1]), depth_scale, int(args.depth_patch))
        if np.isfinite(d) and d >= float(args.z_min) and d <= float(args.z_max):
            keep_uv.append([float(p[0]), float(p[1])])
            z.append(float(d))
    if len(keep_uv) < int(args.min_boundary_points):
        return None

    uvv = np.asarray(keep_uv, dtype=np.float64)
    zv = np.asarray(z, dtype=np.float64)
    pts = deproject(uvv, zv, fx, fy, cx, cy)

    plane = fit_plane(pts, inlier_mm=float(args.plane_inlier_mm))
    if plane is None:
        return None
    pcenter, pnormal, inliers, prmse = plane
    pin = pts[inliers]
    if pin.shape[0] < int(args.min_boundary_points):
        return None

    ex, ey = plane_basis(pnormal)
    rel = pin - pcenter.reshape(1, 3)
    xy = np.stack([rel @ ex, rel @ ey], axis=1)
    cfit = fit_circle_trimmed(xy, inlier_mm=float(args.circle_inlier_mm), min_pts=int(args.min_circle_points))
    if cfit is None:
        return None
    ccx, ccy, rr, cinliers, crmse = cfit

    cov = coverage_ratio(xy[cinliers], ccx, ccy, bins=int(args.coverage_bins))
    if cov < float(args.min_coverage):
        return None

    center3d = pcenter + ccx * ex + ccy * ey
    normal3d = unit(pnormal)
    if float(np.dot(normal3d, center3d)) > 0.0:
        normal3d = -normal3d

    # Circle-fit diameter for reference.
    diam_circle_mm = float(2.0 * rr * 1000.0)
    # If projected shape is ellipse (tilted/foreshortened), use farthest boundary radius.
    radial = np.linalg.norm(xy[cinliers] - np.array([ccx, ccy], dtype=np.float64), axis=1)
    if radial.size == 0:
        return None
    farthest_r = float(np.max(radial))
    aspect = min(float(cand.w), float(cand.h)) / max(float(cand.w), float(cand.h), 1e-6)
    is_ellipse = aspect < float(args.farthest_aspect_thresh)
    if is_ellipse:
        max_boost_m = max(0.0, float(args.max_farthest_boost_mm)) / 1000.0
        r_use = min(farthest_r, float(rr) + max_boost_m)
    else:
        r_use = float(rr)
    dmode = "farthest" if is_ellipse else "circle_fit"
    diam_mm = float(2.0 * r_use * 1000.0)
    err_mm = float(diam_mm - float(args.target_diameter_mm))
    if abs(err_mm) > float(args.diameter_tol_mm):
        return None

    e = abs(err_mm) / max(float(args.diameter_tol_mm), 1e-6)
    q = 0.55 * float(np.exp(-(e * e))) + 0.25 * float(cov) + 0.20 * float(np.exp(-((crmse * 1000.0 / 2.0) ** 2)))
    return PoseResult(
        center_3d=center3d.astype(np.float64),
        normal_3d=normal3d.astype(np.float64),
        radius_m=float(r_use),
        diameter_mm=diam_mm,
        diameter_err_mm=err_mm,
        diameter_circle_mm=diam_circle_mm,
        diameter_mode=dmode,
        plane_rmse_mm=float(prmse * 1000.0),
        circle_rmse_mm=float(crmse * 1000.0),
        coverage=float(cov),
        quality=float(q),
        cand=cand,
    )


def project(pt: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> Optional[Tuple[int, int]]:
    z = float(pt[2])
    if z <= 1e-6:
        return None
    u = int(round(float(fx * pt[0] / z + cx)))
    v = int(round(float(fy * pt[1] / z + cy)))
    return u, v


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
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth_u16.shape
    s = max(1, int(stride))
    z = depth_u16[0:h:s, 0:w:s].reshape(-1).astype(np.float32) * float(depth_scale)
    valid = (z > 0.0) & (z >= float(z_min)) & (z <= float(z_max))
    if int(np.count_nonzero(valid)) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    us = np.arange(0, w, s, dtype=np.float32)
    vs = np.arange(0, h, s, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    u = uu.reshape(-1)[valid]
    v = vv.reshape(-1)[valid]
    z = z[valid]

    x = (u - float(cx)) * z / float(fx)
    y = (v - float(cy)) * z / float(fy)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    ui = np.clip(u.astype(np.int32), 0, rgb.shape[1] - 1)
    vi = np.clip(v.astype(np.int32), 0, rgb.shape[0] - 1)
    cols = rgb[vi, ui, :].astype(np.float32) / 255.0
    return pts, cols


def rs_cloud_from_sdk(
    depth_frame,
    color_frame,
    rgb: np.ndarray,
    pc_calc,
    stride: int,
    z_min: float,
    z_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if depth_frame is None or color_frame is None or pc_calc is None:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    pc_calc.map_to(color_frame)
    pts_rs = pc_calc.calculate(depth_frame)

    w = int(depth_frame.get_width())
    h = int(depth_frame.get_height())
    if w <= 0 or h <= 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    verts = np.asarray(pts_rs.get_vertices(), dtype=np.float32).view(np.float32).reshape(-1, 3)
    tex = np.asarray(pts_rs.get_texture_coordinates(), dtype=np.float32).view(np.float32).reshape(-1, 2)
    if verts.shape[0] != w * h or tex.shape[0] != w * h:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    s = max(1, int(stride))
    verts = verts.reshape(h, w, 3)[0:h:s, 0:w:s, :].reshape(-1, 3)
    tex = tex.reshape(h, w, 2)[0:h:s, 0:w:s, :].reshape(-1, 2)

    z = verts[:, 2]
    valid = np.isfinite(z) & (z > 0.0) & (z >= float(z_min)) & (z <= float(z_max))
    if int(np.count_nonzero(valid)) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    pts = verts[valid].astype(np.float32)
    uv = tex[valid]
    ui = np.clip(np.rint(uv[:, 0] * float(rgb.shape[1] - 1)).astype(np.int32), 0, rgb.shape[1] - 1)
    vi = np.clip(np.rint(uv[:, 1] * float(rgb.shape[0] - 1)).astype(np.int32), 0, rgb.shape[0] - 1)
    cols = rgb[vi, ui, :].astype(np.float32) / 255.0
    return pts, cols


def transform_from_center_normal(center: np.ndarray, normal: np.ndarray) -> np.ndarray:
    z_axis = unit(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = ref - np.dot(ref, z_axis) * z_axis
    if float(np.linalg.norm(x_axis)) < 1e-6:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = ref - np.dot(ref, z_axis) * z_axis
    x_axis = unit(x_axis)
    y_axis = unit(np.cross(z_axis, x_axis))
    x_axis = unit(np.cross(y_axis, z_axis))
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    t[:3, 3] = center
    return t


def color_from_track_id(tid: int) -> np.ndarray:
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


def make_normal_ray(center: np.ndarray, normal: np.ndarray, length_m: float, color: np.ndarray):
    c0 = center.astype(np.float64)
    c1 = (center + unit(normal) * float(length_m)).astype(np.float64)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.stack([c0, c1], axis=0))
    ls.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    col = color.reshape(1, 3).astype(np.float64)
    ls.colors = o3d.utility.Vector3dVector(col)
    return ls


def make_pose_ring(center: np.ndarray, normal: np.ndarray, radius_m: float, seg: int = 64, color: Optional[np.ndarray] = None):
    n = unit(normal)
    ex, ey = plane_basis(n)
    seg = max(24, int(seg))
    a = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False, dtype=np.float64)
    ring = (
        center.reshape(1, 3)
        + float(radius_m) * np.cos(a)[:, None] * ex.reshape(1, 3)
        + float(radius_m) * np.sin(a)[:, None] * ey.reshape(1, 3)
    )
    pts = ring.astype(np.float64)
    lines = [[i, (i + 1) % seg] for i in range(seg)]
    if color is None:
        color = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cols = [color.astype(np.float64).tolist() for _ in range(seg)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=np.float64))
    return ls


def pose_to_track(p: PoseResult, tid: int) -> PoseTrack:
    return PoseTrack(
        tid=int(tid),
        cand=p.cand,
        center_3d=p.center_3d.copy(),
        normal_3d=unit(p.normal_3d.copy()),
        radius_m=float(p.radius_m),
        diameter_mm=float(p.diameter_mm),
        diameter_circle_mm=float(p.diameter_circle_mm),
        diameter_mode=str(p.diameter_mode),
        quality=float(p.quality),
        coverage=float(p.coverage),
        hits=1,
        missed=0,
    )


def update_pose_tracks(
    tracks: List[PoseTrack],
    detections: List[PoseResult],
    next_id: int,
    match_px: float,
    alpha: float,
    max_missed: int,
) -> Tuple[List[PoseTrack], int]:
    if len(tracks) == 0 and len(detections) == 0:
        return tracks, int(next_id)

    a = float(np.clip(alpha, 0.01, 0.99))
    used = [False] * len(detections)

    for t in tracks:
        best_j = -1
        best_d = float("inf")
        for j, d in enumerate(detections):
            if used[j]:
                continue
            dv = float(np.hypot(t.cand.cx - d.cand.cx, t.cand.cy - d.cand.cy))
            if dv < best_d:
                best_d = dv
                best_j = j
        if best_j >= 0 and best_d <= float(match_px):
            d = detections[best_j]
            used[best_j] = True
            t.cand = d.cand
            t.center_3d = (1.0 - a) * t.center_3d + a * d.center_3d
            t.normal_3d = unit((1.0 - a) * t.normal_3d + a * d.normal_3d)
            t.radius_m = (1.0 - a) * float(t.radius_m) + a * float(d.radius_m)
            t.diameter_mm = (1.0 - a) * float(t.diameter_mm) + a * float(d.diameter_mm)
            t.diameter_circle_mm = (1.0 - a) * float(t.diameter_circle_mm) + a * float(d.diameter_circle_mm)
            t.diameter_mode = str(d.diameter_mode)
            t.quality = (1.0 - a) * float(t.quality) + a * float(d.quality)
            t.coverage = (1.0 - a) * float(t.coverage) + a * float(d.coverage)
            t.hits += 1
            t.missed = 0
        else:
            t.missed += 1

    tracks = [t for t in tracks if int(t.missed) <= int(max_missed)]

    for j, d in enumerate(detections):
        if used[j]:
            continue
        tracks.append(pose_to_track(d, next_id))
        next_id += 1

    tracks.sort(key=lambda t: int(t.tid))
    return tracks, int(next_id)


def parse_args():
    ap = argparse.ArgumentParser(description="Minimal RGB+Depth ellipse/circle pose for 29.3 mm caps")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--z_min", type=float, default=0.15)
    ap.add_argument("--z_max", type=float, default=1.20)

    ap.add_argument("--target_diameter_mm", type=float, default=29.3)
    ap.add_argument("--diameter_tol_mm", type=float, default=4.0)

    # Fixed canny defaults requested.
    ap.add_argument("--canny_low", type=float, default=16.0)
    ap.add_argument("--canny_high", type=float, default=26.0)
    ap.add_argument("--tweak_canny", action="store_true", help="Enable realtime Canny low/high sliders in OpenCV")
    ap.add_argument("--canny_slider_max", type=int, default=255)
    ap.add_argument("--blur_ksize", type=int, default=5)
    ap.add_argument("--morph_close", type=int, default=3)

    ap.add_argument("--min_contour_points", type=int, default=18)
    ap.add_argument("--min_area_px", type=float, default=40.0)
    ap.add_argument("--min_circularity", type=float, default=0.12)
    ap.add_argument("--min_aspect", type=float, default=0.30)
    ap.add_argument("--min_diameter_px", type=float, default=12.0)
    ap.add_argument("--max_diameter_px", type=float, default=220.0)
    ap.add_argument("--candidate_border_margin_px", type=int, default=14)
    ap.add_argument("--max_candidates", type=int, default=60)
    ap.add_argument("--edge_support_samples", type=int, default=96)
    ap.add_argument("--edge_support_dist_px", type=float, default=1.8)
    ap.add_argument("--edge_arc_bins", type=int, default=48)
    ap.add_argument("--contour_min_edge_support", type=float, default=0.20)
    ap.add_argument("--contour_min_arc", type=float, default=0.18)
    ap.add_argument("--open_arc_end_gap_px", type=float, default=8.0)
    ap.add_argument("--min_open_arc_length_px", type=float, default=26.0)
    ap.add_argument("--open_arc_min_edge_support", type=float, default=0.12)
    ap.add_argument("--open_arc_min_arc", type=float, default=0.10)
    ap.add_argument("--reject_lower_edge", dest="reject_lower_edge", action="store_true")
    ap.add_argument("--no_reject_lower_edge", dest="reject_lower_edge", action="store_false")
    ap.set_defaults(reject_lower_edge=True)
    ap.add_argument("--max_boundary_center_dz_mm", type=float, default=8.0)
    ap.add_argument("--depth_gate_patch", type=int, default=2)
    ap.add_argument("--depth_gate_samples", type=int, default=72)
    ap.add_argument("--use_hough", dest="use_hough", action="store_true")
    ap.add_argument("--no_hough", dest="use_hough", action="store_false")
    ap.set_defaults(use_hough=True)
    ap.add_argument("--hough_dp", type=float, default=1.2)
    ap.add_argument("--hough_min_dist_px", type=float, default=20.0)
    ap.add_argument("--hough_param1", type=float, default=80.0)
    ap.add_argument("--hough_param2", type=float, default=18.0)
    ap.add_argument("--hough_support_samples", type=int, default=60)
    ap.add_argument("--hough_support_band", type=int, default=2)
    ap.add_argument("--hough_min_edge_support", type=float, default=0.30)
    ap.add_argument("--hough_min_arc", type=float, default=0.18)
    ap.add_argument("--use_ransac", dest="use_ransac", action="store_true")
    ap.add_argument("--no_ransac", dest="use_ransac", action="store_false")
    ap.set_defaults(use_ransac=True)
    ap.add_argument("--ransac_iters", type=int, default=380)
    ap.add_argument("--ransac_min_edge_points", type=int, default=120)
    ap.add_argument("--ransac_max_edge_points", type=int, default=3200)
    ap.add_argument("--ransac_min_edge_support", type=float, default=0.16)
    ap.add_argument("--ransac_min_arc", type=float, default=0.14)
    ap.add_argument("--ransac_seed", type=int, default=-1)
    ap.add_argument("--use_depth_size_prior", dest="use_depth_size_prior", action="store_true")
    ap.add_argument("--no_depth_size_prior", dest="use_depth_size_prior", action="store_false")
    ap.set_defaults(use_depth_size_prior=True)
    ap.add_argument("--size_prior_depth_patch", type=int, default=2)
    ap.add_argument("--size_prior_tol_px", type=float, default=14.0)
    ap.add_argument("--size_prior_sigma_px", type=float, default=7.0)
    ap.add_argument("--merge_center_px", type=float, default=10.0)
    ap.add_argument("--merge_radius_px", type=float, default=4.0)
    ap.add_argument("--merge_center_frac", type=float, default=0.45)
    ap.add_argument("--merge_radius_frac", type=float, default=0.30)
    ap.add_argument("--merge_aspect_tol", type=float, default=0.22)

    ap.add_argument("--boundary_samples", type=int, default=120)
    ap.add_argument("--depth_patch", type=int, default=2)
    ap.add_argument("--min_boundary_points", type=int, default=28)
    ap.add_argument("--plane_inlier_mm", type=float, default=2.0)
    ap.add_argument("--circle_inlier_mm", type=float, default=2.0)
    ap.add_argument("--min_circle_points", type=int, default=22)
    ap.add_argument("--farthest_aspect_thresh", type=float, default=0.92)
    ap.add_argument("--max_farthest_boost_mm", type=float, default=1.2)
    ap.add_argument("--coverage_bins", type=int, default=36)
    ap.add_argument("--min_coverage", type=float, default=0.22)
    ap.add_argument("--track_match_px", type=float, default=70.0)
    ap.add_argument("--track_alpha", type=float, default=0.25)
    ap.add_argument("--track_min_hits", type=int, default=2)
    ap.add_argument("--track_max_missed", type=int, default=16)
    ap.add_argument("--stable_max_missed", type=int, default=3)
    ap.add_argument("--track_merge_px", type=float, default=16.0)
    ap.add_argument("--track_merge_3d_m", type=float, default=0.015)
    ap.add_argument("--show_candidates", action="store_true")
    ap.add_argument("--show_candidate_labels", action="store_true")
    ap.add_argument("--vis_pc", action="store_true")
    ap.add_argument(
        "--pc_source",
        type=str,
        default="custom",
        choices=["custom", "realsense"],
        help="Point cloud source for Open3D view: custom deprojection or RealSense SDK pointcloud",
    )
    ap.add_argument("--pc_stride", type=int, default=2)
    ap.add_argument("--pc_point_size", type=float, default=2.0)
    ap.add_argument("--pc_axis_size", type=float, default=0.02)
    ap.add_argument("--pc_ring_segments", type=int, default=64)
    ap.add_argument("--pc_normal_len", type=float, default=0.025)
    ap.add_argument("--pc_center_radius", type=float, default=0.004)

    ap.add_argument("--debug_edges", action="store_true")
    ap.add_argument("--log_interval", type=float, default=0.5)
    return ap.parse_args()


def main():
    args = parse_args()
    if bool(args.vis_pc) and o3d is None:
        print("[Warn] open3d not available; disabling --vis_pc")
        args.vis_pc = False

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, int(args.w), int(args.h), rs.format.z16, int(args.fps))
    cfg.enable_stream(rs.stream.color, int(args.w), int(args.h), rs.format.rgb8, int(args.fps))
    profile = pipe.start(cfg)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    align = rs.align(rs.stream.color)

    first = align.process(pipe.wait_for_frames(5000))
    color0 = first.get_color_frame()
    if not color0:
        raise RuntimeError("No color frame.")
    intr = color0.get_profile().as_video_stream_profile().get_intrinsics()
    fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

    print(f"[Info] fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f} depth_scale={depth_scale:.6f}")
    print(f"[Info] canny fixed at low/high = {int(args.canny_low)}/{int(args.canny_high)}")
    print(
        f"[Info] detector: contour+{('hough' if bool(args.use_hough) else 'no_hough')}+"
        f"{('ransac' if bool(args.use_ransac) else 'no_ransac')} "
        f"edgeBand<={float(args.edge_support_dist_px):.1f}px arcBins={int(args.edge_arc_bins)} "
        f"sizePrior={'on' if bool(args.use_depth_size_prior) else 'off'}"
    )
    print(
        f"[Info] hough: minDist={float(args.hough_min_dist_px):.1f} param2={float(args.hough_param2):.1f} "
        f"sup>={float(args.hough_min_edge_support):.2f} arc>={float(args.hough_min_arc):.2f} | "
        f"ransac: iters={int(args.ransac_iters)} sup>={float(args.ransac_min_edge_support):.2f} "
        f"arc>={float(args.ransac_min_arc):.2f}"
    )
    print(
        f"[Info] open-arc: endGap>={float(args.open_arc_end_gap_px):.1f}px arcLen>={float(args.min_open_arc_length_px):.1f}px "
        f"sup>={float(args.open_arc_min_edge_support):.2f} arc>={float(args.open_arc_min_arc):.2f} | "
        f"poseMinPts={int(args.min_boundary_points)}/{int(args.min_circle_points)} cov>={float(args.min_coverage):.2f}"
    )
    print(
        f"[Info] lower-edge reject: {'on' if bool(args.reject_lower_edge) else 'off'} "
        f"dz<= {float(args.max_boundary_center_dz_mm):.1f}mm (boundary-center median)"
    )
    print(
        f"[Info] diameter gate: {float(args.target_diameter_mm):.2f} +- {float(args.diameter_tol_mm):.2f} mm | "
        f"stable tracking: hits>={int(args.track_min_hits)} match<={float(args.track_match_px):.1f}px "
        f"alpha={float(args.track_alpha):.2f} max_missed={int(args.track_max_missed)} stable_hold={int(args.stable_max_missed)} "
        f"merge<={float(args.track_merge_px):.1f}px/{float(args.track_merge_3d_m):.3f}m"
    )
    print("[Info] edges view: always on (Minimal Edges window)")
    if bool(args.tweak_canny):
        print(f"[Info] realtime canny sliders: low/high in [0..{int(args.canny_slider_max)}]")
    if bool(args.vis_pc):
        print(
            f"[Info] point-cloud vis enabled src={str(args.pc_source)} (stride={int(args.pc_stride)}) "
            f"annot ring+axis+ray+center (ray={float(args.pc_normal_len):.3f}m center_r={float(args.pc_center_radius):.3f}m)"
        )
        print("[Info] point-cloud clip: z=0.10..1.00 m (fixed)")
    print("[Keys] q / ESC to quit")

    tracks: List[PoseTrack] = []
    next_track_id = 1
    last_log = 0.0
    pc_vis = None
    pc_geom = None
    pc_calc = None
    pose_geoms: List[object] = []
    first_pc_view = True
    pc_z_min = 0.10
    pc_z_max = 1.00
    if bool(args.vis_pc):
        if str(args.pc_source) == "realsense":
            pc_calc = rs.pointcloud()
        pc_vis = o3d.visualization.Visualizer()
        pc_vis.create_window("Minimal PointCloud Pose", 1200, 780, visible=True)
        opt = pc_vis.get_render_option()
        opt.point_size = float(args.pc_point_size)
        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        pc_vis.add_geometry(world)
        pc_geom = o3d.geometry.PointCloud()
        pc_geom.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
        pc_geom.colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
        pc_vis.add_geometry(pc_geom)

    if bool(args.tweak_canny):
        def _noop(_v: int) -> None:
            return

        smax = max(1, int(args.canny_slider_max))
        init_low = int(np.clip(int(round(float(args.canny_low))), 0, smax))
        init_high = int(np.clip(int(round(float(args.canny_high))), 0, smax))
        if init_high <= init_low:
            init_high = min(smax, init_low + 1)
        cv2.namedWindow("Canny Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Canny Controls", 420, 120)
        cv2.createTrackbar("low", "Canny Controls", init_low, smax, _noop)
        cv2.createTrackbar("high", "Canny Controls", init_high, smax, _noop)

    try:
        while True:
            fr = align.process(pipe.wait_for_frames(5000))
            depth = fr.get_depth_frame()
            color = fr.get_color_frame()
            if not depth or not color:
                continue

            depth_u16 = np.asanyarray(depth.get_data())
            rgb = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            if bool(args.tweak_canny):
                low_v = int(cv2.getTrackbarPos("low", "Canny Controls"))
                high_v = int(cv2.getTrackbarPos("high", "Canny Controls"))
                if high_v <= low_v:
                    high_v = min(int(args.canny_slider_max), low_v + 1)
                    cv2.setTrackbarPos("high", "Canny Controls", high_v)
                args.canny_low = float(low_v)
                args.canny_high = float(high_v)

            cands, edges = detect_ellipse_candidates(
                gray,
                args,
                depth_u16=depth_u16,
                depth_scale=depth_scale,
                fx=fx,
                fy=fy,
            )
            poses: List[PoseResult] = []
            for c in cands:
                p = estimate_pose_from_candidate(
                    c,
                    depth_u16=depth_u16,
                    depth_scale=depth_scale,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    args=args,
                )
                if p is not None:
                    poses.append(p)
            n_contour = int(sum(1 for c in cands if c.source == "contour"))
            n_hough = int(sum(1 for c in cands if c.source == "hough"))
            n_ransac = int(sum(1 for c in cands if c.source == "ransac"))
            tracks, next_track_id = update_pose_tracks(
                tracks=tracks,
                detections=poses,
                next_id=next_track_id,
                match_px=float(args.track_match_px),
                alpha=float(args.track_alpha),
                max_missed=int(args.track_max_missed),
            )
            tracks = merge_close_tracks(
                tracks,
                merge_px=float(args.track_merge_px),
                merge_3d_m=float(args.track_merge_3d_m),
            )
            stable = [
                t
                for t in tracks
                if int(t.hits) >= int(args.track_min_hits) and int(t.missed) <= int(args.stable_max_missed)
            ]
            warmup = [
                t
                for t in tracks
                if int(t.hits) < int(args.track_min_hits) and int(t.missed) <= int(args.stable_max_missed)
            ]

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # Optional raw 2D detections.
            if bool(args.show_candidates):
                for c in cands:
                    col = (255, 200, 0)
                    if c.source == "hough":
                        col = (255, 120, 0)
                    elif c.source == "ransac":
                        col = (220, 80, 255)
                    cv2.ellipse(
                        bgr,
                        (int(round(c.cx)), int(round(c.cy))),
                        (int(round(c.w * 0.5)), int(round(c.h * 0.5))),
                        float(c.angle_deg),
                        0,
                        360,
                        col,
                        1,
                        cv2.LINE_AA,
                    )
                    if bool(args.show_candidate_labels):
                        dz_mm = float(c.boundary_center_dz_m) * 1000.0
                        cv2.putText(
                            bgr,
                            f"{c.source[0]} s={c.edge_support:.2f} a={c.arc_coverage:.2f} dz={dz_mm:+.1f}",
                            (int(round(c.cx)) + 4, int(round(c.cy)) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.36,
                            col,
                            1,
                            cv2.LINE_AA,
                        )

            # Warm-up tracks: visible but not yet stable.
            for t in warmup:
                cv2.ellipse(
                    bgr,
                    (int(round(t.cand.cx)), int(round(t.cand.cy))),
                    (int(round(t.cand.w * 0.5)), int(round(t.cand.h * 0.5))),
                    float(t.cand.angle_deg),
                    0,
                    360,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Stable: show pose for all circles that pass 29.3 +- tol mm.
            for t in stable:
                col = (0, 255, 0) if int(t.missed) == 0 else (0, 185, 0)
                cv2.ellipse(
                    bgr,
                    (int(round(t.cand.cx)), int(round(t.cand.cy))),
                    (int(round(t.cand.w * 0.5)), int(round(t.cand.h * 0.5))),
                    float(t.cand.angle_deg),
                    0,
                    360,
                    col,
                    2 if int(t.missed) == 0 else 1,
                    cv2.LINE_AA,
                )
                p0 = project(t.center_3d, fx, fy, cx, cy)
                p1 = project(t.center_3d + t.normal_3d * 0.02, fx, fy, cx, cy)
                if p0 is not None and p1 is not None:
                    cv2.arrowedLine(bgr, p0, p1, col, 2, tipLength=0.3)
                    cv2.putText(
                        bgr,
                        f"#{int(t.tid)} {t.diameter_mm:.2f}mm m{int(t.missed)}",
                        (p0[0] + 6, p0[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        col,
                        1,
                        cv2.LINE_AA,
                    )

            if len(stable) == 0:
                cv2.putText(
                    bgr,
                    f"No stable pose yet (need {float(args.target_diameter_mm):.1f}+-{float(args.diameter_tol_mm):.1f}mm and warm-up)",
                    (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    bgr,
                    f"Stable poses: {len(stable)}",
                    (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                bgr,
                f"Canny {int(args.canny_low)}/{int(args.canny_high)} detected2D={len(cands)} "
                f"(c/h/r={n_contour}/{n_hough}/{n_ransac}) valid3D={len(poses)} "
                f"stable={len(stable)} warmup={len(warmup)}",
                (14, bgr.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Minimal Circle Pose", bgr)
            cv2.imshow("Minimal Edges", edges)

            if pc_vis is not None and pc_geom is not None:
                if str(args.pc_source) == "realsense":
                    pts, cols = rs_cloud_from_sdk(
                        depth_frame=depth,
                        color_frame=color,
                        rgb=rgb,
                        pc_calc=pc_calc,
                        stride=int(args.pc_stride),
                        z_min=float(pc_z_min),
                        z_max=float(pc_z_max),
                    )
                else:
                    pts, cols = rs_cloud_from_aligned_depth(
                        depth_u16=depth_u16,
                        rgb=rgb,
                        depth_scale=depth_scale,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        stride=int(args.pc_stride),
                        z_min=float(pc_z_min),
                        z_max=float(pc_z_max),
                    )
                if pts.shape[0] > 0:
                    pc_geom.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                    pc_geom.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
                    pc_vis.update_geometry(pc_geom)
                    if first_pc_view:
                        vc = pc_vis.get_view_control()
                        z_med = float(np.median(pts[:, 2])) if pts.shape[0] > 0 else 0.45
                        vc.set_lookat([0.0, 0.0, max(0.20, min(0.90, z_med))])
                        # Camera-side view: look along +Z of camera coordinates.
                        vc.set_front([0.0, 0.0, -1.0])
                        vc.set_up([0.0, -1.0, 0.0])
                        vc.set_zoom(0.70)
                        first_pc_view = False

                # Remove old pose geoms
                for g in pose_geoms:
                    pc_vis.remove_geometry(g, reset_bounding_box=False)
                pose_geoms = []

                for t in stable:
                    col = color_from_track_id(int(t.tid))
                    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.pc_axis_size))
                    axis.transform(transform_from_center_normal(t.center_3d, t.normal_3d))
                    ring = make_pose_ring(
                        center=t.center_3d,
                        normal=t.normal_3d,
                        radius_m=float(t.radius_m),
                        seg=int(args.pc_ring_segments),
                        color=col,
                    )
                    ray = make_normal_ray(
                        center=t.center_3d,
                        normal=t.normal_3d,
                        length_m=float(args.pc_normal_len),
                        color=col,
                    )
                    center_dot = o3d.geometry.TriangleMesh.create_sphere(radius=float(args.pc_center_radius))
                    center_dot.paint_uniform_color(col.tolist())
                    center_dot.translate(t.center_3d.astype(np.float64), relative=False)
                    pc_vis.add_geometry(axis, reset_bounding_box=False)
                    pc_vis.add_geometry(ring, reset_bounding_box=False)
                    pc_vis.add_geometry(ray, reset_bounding_box=False)
                    pc_vis.add_geometry(center_dot, reset_bounding_box=False)
                    pose_geoms.extend([axis, ring, ray, center_dot])

                if not pc_vis.poll_events():
                    break
                pc_vis.update_renderer()

            now = time.time()
            if now - last_log > float(args.log_interval):
                if len(stable) == 0:
                    print(
                        f"[Pose] none | detected2D={len(cands)} c/h/r={n_contour}/{n_hough}/{n_ransac} valid3D={len(poses)} "
                        f"stable=0 warmup={len(warmup)} tracks={len(tracks)}"
                    )
                else:
                    msg = []
                    for t in stable:
                        msg.append(
                            f"#{int(t.tid)} d={t.diameter_mm:.2f} fit={t.diameter_circle_mm:.2f} "
                            f"{t.diameter_mode} t=({t.center_3d[0]:.3f},{t.center_3d[1]:.3f},{t.center_3d[2]:.3f})"
                        )
                    print(
                        f"[Pose] all | detected2D={len(cands)} c/h/r={n_contour}/{n_hough}/{n_ransac} valid3D={len(poses)} "
                        f"stable={len(stable)} warmup={len(warmup)} tracks={len(tracks)} | "
                        + " ; ".join(msg)
                    )
                last_log = now

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        if pc_vis is not None:
            pc_vis.destroy_window()


if __name__ == "__main__":
    main()
