from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class EstimatorConfig:
    conf_thr: float = 0.55
    min_area: int = 700
    border_margin_px: int = 8
    solidity_thr: float = 0.90
    completeness_thr: float = 0.62
    min_axis_ratio: float = 0.18
    max_axis_ratio: float = 1.00
    min_major_px: float = 18.0
    max_ellipse_residual: float = 0.36
    min_ellipse_iou: float = 0.58
    max_occlusion_score: float = 0.48
    symmetry_mod_deg: float = 180.0
    use_edge_refine: bool = True
    edge_canny1: int = 40
    edge_canny2: int = 120
    edge_band_px: int = 3
    mask_open_ksize: int = 3
    mask_close_ksize: int = 5
    mask_blur_ksize: int = 5


def _odd(v: int) -> int:
    v = int(max(1, v))
    return v if v % 2 == 1 else v + 1


def _to_binary_u8(mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8 is None:
        return np.zeros((0, 0), dtype=np.uint8)
    if mask_u8.dtype != np.uint8:
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    else:
        mask_u8 = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
    return mask_u8


def _keep_largest_component(mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8.size == 0:
        return mask_u8
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_lbl <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = int(np.argmax(areas)) + 1
    out = np.zeros_like(mask_u8)
    out[lbl == best_idx] = 255
    return out


def preprocess_mask(mask_u8: np.ndarray, cfg: EstimatorConfig) -> np.ndarray:
    m = _to_binary_u8(mask_u8)
    if m.size == 0:
        return m
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(cfg.mask_open_ksize), _odd(cfg.mask_open_ksize))
    )
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(cfg.mask_close_ksize), _odd(cfg.mask_close_ksize))
    )
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    m = cv2.GaussianBlur(m, (_odd(cfg.mask_blur_ksize), _odd(cfg.mask_blur_ksize)), 0)
    m = np.where(m >= 127, 255, 0).astype(np.uint8)
    m = _keep_largest_component(m)
    return m


def _mask_to_contour(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _edge_refine_contour(mask_u8: np.ndarray, cfg: EstimatorConfig) -> Optional[np.ndarray]:
    c0 = _mask_to_contour(mask_u8)
    if c0 is None:
        return None
    edge = cv2.Canny(mask_u8, cfg.edge_canny1, cfg.edge_canny2)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * int(cfg.edge_band_px) + 1, 2 * int(cfg.edge_band_px) + 1)
    )
    band = cv2.subtract(cv2.dilate(mask_u8, k), cv2.erode(mask_u8, k))
    rim = cv2.bitwise_and(edge, band)
    c1 = _mask_to_contour(rim)
    if c1 is not None and len(c1) >= 15:
        return c1
    return c0


def _touches_border(mask_u8: np.ndarray, margin: int) -> bool:
    h, w = mask_u8.shape[:2]
    m = int(max(0, margin))
    if m <= 0:
        return False
    return bool(
        np.any(mask_u8[:m, :] > 0)
        or np.any(mask_u8[h - m :, :] > 0)
        or np.any(mask_u8[:, :m] > 0)
        or np.any(mask_u8[:, w - m :] > 0)
    )


def _ellipse_residual(contour: np.ndarray, ellipse) -> float:
    (cx, cy), (d1, d2), ang = ellipse
    if d1 >= d2:
        a = max(float(d1) * 0.5, 1e-6)
        b = max(float(d2) * 0.5, 1e-6)
        th = np.deg2rad(float(ang))
    else:
        # Keep theta aligned to major axis when width/height are swapped.
        a = max(float(d2) * 0.5, 1e-6)
        b = max(float(d1) * 0.5, 1e-6)
        th = np.deg2rad(float(ang + 90.0))
    cth = np.cos(th)
    sth = np.sin(th)
    pts = contour.reshape(-1, 2).astype(np.float32)
    x = pts[:, 0] - cx
    y = pts[:, 1] - cy
    xr = x * cth + y * sth
    yr = -x * sth + y * cth
    d = (xr * xr) / (a * a) + (yr * yr) / (b * b)
    return float(np.mean(np.abs(d - 1.0)))


def _pca_angle(contour: np.ndarray) -> Tuple[float, float]:
    pts = contour.reshape(-1, 2).astype(np.float32)
    mu = np.mean(pts, axis=0, keepdims=True)
    q = pts - mu
    cov = (q.T @ q) / max(len(pts) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    v = vecs[:, 0]
    ang = float(np.degrees(np.arctan2(v[1], v[0])))
    evr = float(vals[0] / max(vals[1], 1e-9))
    return ang, evr


def _ellipse_iou(mask_u8: np.ndarray, ellipse) -> float:
    em = np.zeros_like(mask_u8, dtype=np.uint8)
    cv2.ellipse(em, ellipse, 255, -1, cv2.LINE_AA)
    inter = np.count_nonzero(cv2.bitwise_and(mask_u8, em))
    union = np.count_nonzero(cv2.bitwise_or(mask_u8, em))
    if union <= 0:
        return 0.0
    return float(inter / union)


def _occlusion_score(
    solidity: float,
    completeness: float,
    ellipse_iou: float,
    residual: float,
    edge_density: float,
) -> float:
    q_sol = np.clip(1.0 - solidity, 0.0, 1.0)
    q_comp = np.clip(1.0 - completeness, 0.0, 1.0)
    q_iou = np.clip(1.0 - ellipse_iou, 0.0, 1.0)
    q_res = np.clip(residual / 0.6, 0.0, 1.0)
    q_edge = np.clip((0.008 - edge_density) / 0.008, 0.0, 1.0)
    score = 0.34 * q_sol + 0.26 * q_comp + 0.24 * q_iou + 0.10 * q_res + 0.06 * q_edge
    return float(np.clip(score, 0.0, 1.0))


def _normal_to_roll_pitch(normal_vec: np.ndarray) -> Tuple[float, float]:
    nx, ny, nz = float(normal_vec[0]), float(normal_vec[1]), float(normal_vec[2])
    roll = float(np.degrees(np.arctan2(ny, nz)))
    pitch = float(np.degrees(np.arctan2(-nx, nz)))
    return roll, pitch


def _build_output(
    *,
    valid: bool,
    reason: str,
    mask_u8: np.ndarray,
    metrics: Dict,
    contour=None,
    ellipse=None,
    pose2d=None,
    yaw_mod=None,
    tilt=None,
    tilt_dir=None,
    normal_vec=None,
    roll_pitch=None,
) -> Dict:
    out = {
        "valid": bool(valid),
        "reason": reason,
        "mask": mask_u8,
        "metrics": metrics,
    }
    if not valid:
        return out
    out.update(
        {
            "pose2d": pose2d,
            "yaw_mod_deg": yaw_mod,
            "tilt_angle_deg": tilt,
            "tilt_dir_deg": tilt_dir,
            "normal_vec": normal_vec,
            "roll_pitch_deg": roll_pitch,
            "contour": contour,
            "ellipse": ellipse,
        }
    )
    return out


def estimate_tilt_pose_from_mask(
    mask_u8: np.ndarray,
    confidence: float,
    cfg: Optional[EstimatorConfig] = None,
    K: Optional[np.ndarray] = None,
) -> Dict:
    if cfg is None:
        cfg = EstimatorConfig()

    m = preprocess_mask(mask_u8, cfg)
    if m.size == 0:
        return _build_output(valid=False, reason="empty_mask", mask_u8=m, metrics={})

    metrics = {"confidence": float(confidence)}
    if confidence < cfg.conf_thr:
        return _build_output(valid=False, reason="low_conf", mask_u8=m, metrics=metrics)

    area = int(np.count_nonzero(m))
    metrics["area"] = area
    if area < int(cfg.min_area):
        return _build_output(valid=False, reason="small_area", mask_u8=m, metrics=metrics)
    if _touches_border(m, int(cfg.border_margin_px)):
        return _build_output(valid=False, reason="touches_border", mask_u8=m, metrics=metrics)

    contour = _edge_refine_contour(m, cfg) if cfg.use_edge_refine else _mask_to_contour(m)
    if contour is None or len(contour) < 15:
        return _build_output(valid=False, reason="no_contour", mask_u8=m, metrics=metrics)

    c_area = float(max(cv2.contourArea(contour), 1.0))
    hull = cv2.convexHull(contour)
    hull_area = float(max(cv2.contourArea(hull), 1.0))
    solidity = float(c_area / hull_area)
    metrics["solidity"] = solidity
    if solidity < cfg.solidity_thr:
        return _build_output(valid=False, reason="low_solidity", mask_u8=m, metrics=metrics)

    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (d1, d2), e_ang = ellipse
    major = float(max(d1, d2))
    minor = float(min(d1, d2))
    axis_ratio = float(np.clip(minor / max(major, 1e-6), 0.0, 1.0))
    metrics["major_px"] = major
    metrics["minor_px"] = minor
    metrics["axis_ratio"] = axis_ratio
    if major < cfg.min_major_px:
        return _build_output(valid=False, reason="small_major_axis", mask_u8=m, metrics=metrics)
    if axis_ratio < cfg.min_axis_ratio or axis_ratio > cfg.max_axis_ratio:
        return _build_output(valid=False, reason="axis_ratio_gate", mask_u8=m, metrics=metrics)

    e_area = float(np.pi * (major * 0.5) * (minor * 0.5))
    completeness = float(np.clip(c_area / max(e_area, 1.0), 0.0, 2.0))
    metrics["completeness"] = completeness
    if completeness < cfg.completeness_thr:
        return _build_output(valid=False, reason="low_completeness", mask_u8=m, metrics=metrics)

    residual = _ellipse_residual(contour, ellipse)
    e_iou = _ellipse_iou(m, ellipse)
    metrics["ellipse_fit_residual"] = residual
    metrics["ellipse_iou"] = e_iou
    if residual > cfg.max_ellipse_residual:
        return _build_output(valid=False, reason="high_ellipse_residual", mask_u8=m, metrics=metrics)
    if e_iou < cfg.min_ellipse_iou:
        return _build_output(valid=False, reason="low_ellipse_iou", mask_u8=m, metrics=metrics)

    edge = cv2.Canny(m, cfg.edge_canny1, cfg.edge_canny2)
    edge_density = float(np.count_nonzero(edge) / max(area, 1))
    metrics["edge_density"] = edge_density

    occlusion = _occlusion_score(solidity, completeness, e_iou, residual, edge_density)
    metrics["occlusion_score"] = occlusion
    if occlusion > cfg.max_occlusion_score:
        return _build_output(valid=False, reason="high_occlusion_score", mask_u8=m, metrics=metrics)

    tilt = float(np.degrees(np.arccos(np.clip(axis_ratio, 0.0, 1.0))))
    tilt_dir = float(e_ang if d1 >= d2 else (e_ang + 90.0))
    tilt_dir = ((tilt_dir + 180.0) % 360.0) - 180.0

    pca_ang, eig_ratio = _pca_angle(contour)
    metrics["eigenvalue_ratio"] = eig_ratio
    yaw_mod = float((pca_ang + 360.0) % float(cfg.symmetry_mod_deg))

    t = np.deg2rad(tilt)
    a = np.deg2rad(tilt_dir)
    normal_vec = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)], dtype=np.float32)
    normal_vec /= max(float(np.linalg.norm(normal_vec)), 1e-6)
    roll_pitch = _normal_to_roll_pitch(normal_vec)
    if K is not None:
        _ = K

    return _build_output(
        valid=True,
        reason="ok",
        mask_u8=m,
        metrics=metrics,
        contour=contour,
        ellipse=ellipse,
        pose2d=(float(cx), float(cy)),
        yaw_mod=yaw_mod,
        tilt=tilt,
        tilt_dir=tilt_dir,
        normal_vec=(float(normal_vec[0]), float(normal_vec[1]), float(normal_vec[2])),
        roll_pitch=roll_pitch,
    )


def draw_pose_overlay(img_bgr: np.ndarray, result: Dict, prefix: str = "") -> np.ndarray:
    out = img_bgr.copy()
    if not result.get("valid", False):
        txt = f"{prefix} INVALID: {result.get('reason', 'unknown')}".strip()
        cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    contour = result["contour"]
    ellipse = result["ellipse"]
    cx, cy = result["pose2d"]
    tilt = result["tilt_angle_deg"]
    tdir = result["tilt_dir_deg"]
    m = result["metrics"]
    roll, pitch = result["roll_pitch_deg"]

    cv2.drawContours(out, [contour], -1, (0, 255, 255), 2)
    cv2.ellipse(out, ellipse, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(out, (int(round(cx)), int(round(cy))), 4, (0, 255, 0), -1)

    L = 60
    a = np.deg2rad(tdir)
    dx = int(round(L * np.cos(a)))
    dy = int(round(L * np.sin(a)))
    p0 = (int(round(cx)), int(round(cy)))
    p1 = (p0[0] + dx, p0[1] + dy)
    cv2.arrowedLine(out, p0, p1, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.15)

    txt1 = f"{prefix} roll={roll:+.1f} pitch={pitch:+.1f} tilt={tilt:.1f} dir={tdir:.1f}".strip()
    txt2 = (
        f"ar={m.get('axis_ratio', 0):.3f} sol={m.get('solidity', 0):.3f} "
        f"comp={m.get('completeness', 0):.3f} occ={m.get('occlusion_score', 1):.3f}"
    )
    cv2.putText(out, txt1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, txt2, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    return out
