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
    completeness_thr: float = 0.55
    min_axis_ratio: float = 0.18
    max_axis_ratio: float = 1.0
    symmetry_mod_deg: float = 180.0
    use_edge_refine: bool = True
    edge_canny1: int = 40
    edge_canny2: int = 120
    edge_band_px: int = 4


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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.edge_band_px + 1, 2 * cfg.edge_band_px + 1))
    dil = cv2.dilate(mask_u8, k)
    ero = cv2.erode(mask_u8, k)
    band = cv2.bitwise_and(dil, cv2.bitwise_not(ero))
    rim = cv2.bitwise_and(edge, band)
    c1 = _mask_to_contour(rim)
    if c1 is not None and len(c1) >= 15:
        return c1
    return c0


def _touches_border(mask_u8: np.ndarray, margin: int) -> bool:
    h, w = mask_u8.shape[:2]
    m = int(max(0, margin))
    if m == 0:
        return False
    if np.any(mask_u8[:m, :] > 0):
        return True
    if np.any(mask_u8[h - m :, :] > 0):
        return True
    if np.any(mask_u8[:, :m] > 0):
        return True
    if np.any(mask_u8[:, w - m :] > 0):
        return True
    return False


def _ellipse_residual(contour: np.ndarray, ellipse) -> float:
    (cx, cy), (a_len, b_len), ang = ellipse
    a = max(a_len * 0.5, 1e-6)
    b = max(b_len * 0.5, 1e-6)
    th = np.deg2rad(float(ang))
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


def estimate_tilt_pose_from_mask(
    mask_u8: np.ndarray,
    confidence: float,
    cfg: Optional[EstimatorConfig] = None,
    K: Optional[np.ndarray] = None,
) -> Dict:
    if cfg is None:
        cfg = EstimatorConfig()
    if mask_u8 is None:
        return {"valid": False, "reason": "empty_mask"}
    if mask_u8.dtype != np.uint8:
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    if confidence < cfg.conf_thr:
        return {"valid": False, "reason": "low_conf"}

    area = int(np.count_nonzero(mask_u8))
    if area < cfg.min_area:
        return {"valid": False, "reason": "small_area"}
    if _touches_border(mask_u8, cfg.border_margin_px):
        return {"valid": False, "reason": "touches_border"}

    contour = _edge_refine_contour(mask_u8, cfg) if cfg.use_edge_refine else _mask_to_contour(mask_u8)
    if contour is None or len(contour) < 15:
        return {"valid": False, "reason": "no_contour"}

    hull = cv2.convexHull(contour)
    hull_area = float(max(cv2.contourArea(hull), 1.0))
    c_area = float(max(cv2.contourArea(contour), 1.0))
    solidity = float(c_area / hull_area)
    if solidity < cfg.solidity_thr:
        return {"valid": False, "reason": "low_solidity", "solidity": solidity}

    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (d1, d2), e_ang = ellipse
    major = float(max(d1, d2))
    minor = float(min(d1, d2))
    if major <= 1e-6:
        return {"valid": False, "reason": "degenerate_ellipse"}
    axis_ratio = float(np.clip(minor / major, 0.0, 1.0))
    if axis_ratio < cfg.min_axis_ratio or axis_ratio > cfg.max_axis_ratio:
        return {"valid": False, "reason": "axis_ratio_gate", "axis_ratio": axis_ratio}

    e_area = np.pi * (major * 0.5) * (minor * 0.5)
    completeness = float(np.clip(c_area / max(e_area, 1.0), 0.0, 2.0))
    if completeness < cfg.completeness_thr:
        return {"valid": False, "reason": "low_completeness", "completeness": completeness}

    tilt = float(np.degrees(np.arccos(np.clip(axis_ratio, 0.0, 1.0))))
    # cv2 ellipse angle is major-axis angle only if d1 >= d2. Ensure major-axis direction.
    tilt_dir = float(e_ang if d1 >= d2 else (e_ang + 90.0))
    tilt_dir = ((tilt_dir + 180.0) % 360.0) - 180.0

    pca_ang, eig_ratio = _pca_angle(contour)
    yaw_mod = float((pca_ang + 360.0) % float(cfg.symmetry_mod_deg))

    residual = _ellipse_residual(contour, ellipse)

    roll_pitch = None
    if K is not None:
        # Weak-perspective approximation: treat tilt vector as projected surface-normal direction.
        t = np.deg2rad(tilt)
        a = np.deg2rad(tilt_dir)
        nx = np.sin(t) * np.cos(a)
        ny = np.sin(t) * np.sin(a)
        nz = np.cos(t)
        roll = float(np.degrees(np.arctan2(ny, nz)))
        pitch = float(np.degrees(np.arctan2(-nx, nz)))
        roll_pitch = (roll, pitch)

    return {
        "valid": True,
        "pose2d": (float(cx), float(cy)),
        "yaw_mod_deg": yaw_mod,
        "tilt_angle_deg": tilt,
        "tilt_dir_deg": tilt_dir,
        "roll_pitch_deg": roll_pitch,
        "metrics": {
            "axis_ratio": axis_ratio,
            "ellipse_fit_residual": residual,
            "eigenvalue_ratio": eig_ratio,
            "solidity": solidity,
            "completeness": completeness,
            "area": area,
            "confidence": float(confidence),
        },
        "contour": contour,
        "ellipse": ellipse,
    }


def draw_pose_overlay(img_bgr: np.ndarray, result: Dict) -> np.ndarray:
    out = img_bgr.copy()
    if not result.get("valid", False):
        txt = f"INVALID: {result.get('reason', 'unknown')}"
        cv2.putText(out, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    contour = result["contour"]
    ellipse = result["ellipse"]
    cx, cy = result["pose2d"]
    tilt = result["tilt_angle_deg"]
    tdir = result["tilt_dir_deg"]
    ymod = result["yaw_mod_deg"]
    m = result["metrics"]

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

    txt1 = f"yaw_mod={ymod:.1f} tilt={tilt:.1f} dir={tdir:.1f}"
    txt2 = f"ar={m['axis_ratio']:.3f} sol={m['solidity']:.3f} comp={m['completeness']:.3f} res={m['ellipse_fit_residual']:.3f}"
    cv2.putText(out, txt1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, txt2, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    return out
