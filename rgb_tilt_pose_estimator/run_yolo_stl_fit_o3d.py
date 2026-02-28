import argparse
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from run_yolo_tilt_pose import load_roi_yaml, make_mask_from_polygon, suppress_overlaps

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


def configure_realsense(profile, exposure: float, gain: float, wb: float, auto_exp: bool, auto_wb: bool) -> None:
    dev = profile.get_device()
    color_sensor = None
    for s in dev.query_sensors():
        name = s.get_info(rs.camera_info.name).lower()
        if "rgb" in name or "color" in name:
            color_sensor = s
            break
    if color_sensor is None:
        return
    if color_sensor.supports(rs.option.enable_auto_exposure):
        color_sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exp else 0.0)
    if color_sensor.supports(rs.option.exposure):
        color_sensor.set_option(rs.option.exposure, float(exposure))
    if color_sensor.supports(rs.option.gain):
        color_sensor.set_option(rs.option.gain, float(gain))
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0 if auto_wb else 0.0)
    if color_sensor.supports(rs.option.white_balance):
        color_sensor.set_option(rs.option.white_balance, float(wb))


def color_from_index(i: int) -> np.ndarray:
    hue = float((int(i) * 57) % 360) / 360.0
    s = 0.95
    v = 1.0
    h6 = hue * 6.0
    k = int(np.floor(h6)) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if k == 0:
        rgb = np.array([v, t, p], dtype=np.float32)
    elif k == 1:
        rgb = np.array([q, v, p], dtype=np.float32)
    elif k == 2:
        rgb = np.array([p, v, t], dtype=np.float32)
    elif k == 3:
        rgb = np.array([p, q, v], dtype=np.float32)
    elif k == 4:
        rgb = np.array([t, p, v], dtype=np.float32)
    else:
        rgb = np.array([v, p, q], dtype=np.float32)
    return rgb


def mask_to_points(
    mask_full: np.ndarray,
    depth_u16: np.ndarray,
    color_bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    z_min: float,
    z_max: float,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask_full > 0)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    if int(stride) > 1:
        keep = np.arange(0, ys.size, int(stride))
        ys = ys[keep]
        xs = xs[keep]
    z_raw = depth_u16[ys, xs].astype(np.float32)
    z = z_raw * float(depth_scale)
    valid = (z > 0.0) & (z >= float(z_min)) & (z <= float(z_max))
    if int(np.count_nonzero(valid)) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    ys = ys[valid].astype(np.float32)
    xs = xs[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    x = (xs - float(cx)) * z / float(fx)
    y = (ys - float(cy)) * z / float(fy)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    cols = color_bgr[ys.astype(np.int32), xs.astype(np.int32), ::-1].astype(np.float32) / 255.0
    return pts, cols


def to_o3d_pcd(pts: np.ndarray, cols: Optional[np.ndarray] = None) -> "o3d.geometry.PointCloud":
    p = o3d.geometry.PointCloud()
    if pts.shape[0] > 0:
        p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if cols is not None and cols.shape[0] == pts.shape[0]:
        p.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return p


def yolo_mask_roi(pred, idx: int, roi_h: int, roi_w: int) -> Optional[np.ndarray]:
    if pred.masks is None:
        return None
    try:
        mdata = pred.masks.data
        if mdata is not None and int(mdata.shape[0]) > idx:
            m = mdata[idx].detach().cpu().numpy().astype(np.float32)
            if m.shape[0] != roi_h or m.shape[1] != roi_w:
                m = cv2.resize(m, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            mu8 = (m > 0.5).astype(np.uint8) * 255
            # Mild close to remove jagged tiny holes from mask upsampling.
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mu8 = cv2.morphologyEx(mu8, cv2.MORPH_CLOSE, ker)
            return mu8
    except Exception:
        pass
    if pred.masks.xy is not None and len(pred.masks.xy) > idx:
        poly = pred.masks.xy[idx]
        if poly is not None and len(poly) >= 3:
            return make_mask_from_polygon(poly, roi_h, roi_w)
    return None


def preprocess_for_reg(pcd: "o3d.geometry.PointCloud", voxel: float):
    down = pcd.voxel_down_sample(max(1e-4, float(voxel)))
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 2.5, 1e-3), max_nn=30))
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 5.0, 2e-3), max_nn=80)
    )
    return down, feat


def pca_basis(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = np.mean(pts, axis=0)
    x = pts - c
    cov = (x.T @ x) / float(max(1, x.shape[0] - 1))
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    if np.linalg.det(v) < 0.0:
        v[:, 2] *= -1.0
    return w.astype(np.float64), v.astype(np.float64)


def make_basis_from_signs(base: np.ndarray, sx: float, sy: float) -> np.ndarray:
    b = base.copy()
    b[:, 0] *= float(sx)
    b[:, 1] *= float(sy)
    b[:, 2] = np.cross(b[:, 0], b[:, 1])
    n2 = np.linalg.norm(b[:, 2])
    if n2 > 1e-8:
        b[:, 2] /= n2
    return b


def register_mesh_to_segment(
    src_mesh_points: "o3d.geometry.PointCloud",
    tgt_seg_points: "o3d.geometry.PointCloud",
    voxel: float,
    icp_dist: float,
):
    s = np.asarray(src_mesh_points.points, dtype=np.float64)
    t = np.asarray(tgt_seg_points.points, dtype=np.float64)
    if s.shape[0] < 50 or t.shape[0] < 50:
        return np.eye(4), False, 0.0, 0.0

    sw, sb = pca_basis(s)
    tw, tb = pca_basis(t)
    cs = np.mean(s, axis=0)
    ct = np.mean(t, axis=0)

    s_down = src_mesh_points.voxel_down_sample(max(1e-4, voxel * 0.7))
    t_down = tgt_seg_points.voxel_down_sample(max(1e-4, voxel * 0.7))
    if len(s_down.points) < 30 or len(t_down.points) < 30:
        return np.eye(4), False, 0.0, 0.0

    best = None
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            sb2 = make_basis_from_signs(sb, sx, sy)
            R0 = tb @ sb2.T
            t0 = ct - (R0 @ cs)
            T0 = np.eye(4, dtype=np.float64)
            T0[:3, :3] = R0
            T0[:3, 3] = t0
            icp = o3d.pipelines.registration.registration_icp(
                s_down,
                t_down,
                max(float(icp_dist), float(voxel) * 1.5),
                T0,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
            )
            score = float(icp.fitness) - 0.5 * float(icp.inlier_rmse)
            if (best is None) or (score > best[0]):
                best = (score, icp)

    if best is None:
        return np.eye(4), False, 0.0, 0.0
    icp = best[1]
    ok = (icp.fitness >= 0.22) and (icp.inlier_rmse <= max(0.006, icp_dist * 1.5))
    return icp.transformation.copy(), bool(ok), float(icp.fitness), float(icp.inlier_rmse)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO segmented region -> STL fitting in Open3D.")
    p.add_argument("--model", type=str, default="runs/segment/Imported/best.pt")
    p.add_argument("--mesh", type=str, default="tamplate.stl")
    p.add_argument("--mesh_unit", type=str, default="mm", choices=["mm", "m"])
    p.add_argument("--source", type=str, default="rs", choices=["rs"])
    p.add_argument("--roi_yaml", type=str, default="segmentation_bgsub_yolo/config/roi.yaml")
    p.add_argument("--show_roi", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--exposure", type=float, default=140.0)
    p.add_argument("--gain", type=float, default=16.0)
    p.add_argument("--white_balance", type=float, default=4500.0)
    p.add_argument("--auto_exposure", action="store_true")
    p.add_argument("--auto_white_balance", action="store_true")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max_det", type=int, default=20)
    p.add_argument("--max_fit_instances", type=int, default=8)
    p.add_argument("--overlap_thr", type=float, default=0.65)
    p.add_argument("--top_overlap_thr", type=float, default=0.35)
    p.add_argument("--min_visible_ratio", type=float, default=0.55)
    p.add_argument("--prefilter_conf", type=float, default=0.45)
    p.add_argument("--prefilter_min_area", type=int, default=220)
    p.add_argument("--prefilter_max_area_ratio", type=float, default=0.30)
    p.add_argument("--prefilter_min_circularity", type=float, default=0.45)
    p.add_argument("--prefilter_min_solidity", type=float, default=0.80)
    p.add_argument("--prefilter_min_depth_valid_ratio", type=float, default=0.70)
    p.add_argument("--prefilter_min_depth_relief_mm", type=float, default=1.0)
    p.add_argument("--mask_dilate_px", type=int, default=0)
    p.add_argument("--z_min", type=float, default=0.10)
    p.add_argument("--z_max", type=float, default=1.00)
    p.add_argument("--cloud_stride", type=int, default=1)
    p.add_argument("--voxel", type=float, default=0.0035)
    p.add_argument("--icp_dist", type=float, default=0.009)
    p.add_argument("--pc_point_size", type=float, default=2.5)
    p.add_argument("--pc_bg_dark", action="store_true")
    p.add_argument("--hide_full_cloud", action="store_true")
    p.add_argument("--show_mesh_fit", action="store_true")
    p.add_argument("--vector_len", type=float, default=0.03, help="axis vector length in meters")
    p.add_argument("--vector_radius", type=float, default=0.0035, help="arrow shaft radius in meters")
    p.add_argument(
        "--axis_sign_mode",
        type=str,
        default="raw_match",
        choices=["raw_match", "screen_up", "away_camera", "toward_camera"],
        help="resolve ± axis ambiguity",
    )
    return p.parse_args()


def normal_to_roll_pitch(n: np.ndarray) -> Tuple[float, float]:
    nz = float(n[2])
    if abs(nz) < 1e-8:
        nz = 1e-8
    roll = float(np.degrees(np.arctan2(float(n[1]), nz)))
    pitch = float(np.degrees(np.arctan2(-float(n[0]), nz)))
    return roll, pitch


def project_uv(pt: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> Optional[Tuple[float, float]]:
    z = float(pt[2])
    if z <= 1e-8:
        return None
    u = float(fx * float(pt[0]) / z + cx)
    v = float(fy * float(pt[1]) / z + cy)
    return (u, v)


def orient_axis_sign(
    axis_w: np.ndarray,
    center_w: np.ndarray,
    mode: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    test_len: float,
) -> np.ndarray:
    a = axis_w.astype(np.float64).copy()
    n = float(np.linalg.norm(a))
    if n <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    a /= n
    if mode == "raw_match":
        return a
    if mode == "away_camera":
        return a if float(a[2]) >= 0.0 else -a
    if mode == "toward_camera":
        return a if float(a[2]) <= 0.0 else -a
    # screen_up: choose sign so projected arrow goes upward in image (smaller v).
    uv0 = project_uv(center_w, fx, fy, cx, cy)
    if uv0 is None:
        return a
    p1 = center_w + a * float(test_len)
    p2 = center_w - a * float(test_len)
    uv1 = project_uv(p1, fx, fy, cx, cy)
    uv2 = project_uv(p2, fx, fy, cx, cy)
    if uv1 is None and uv2 is None:
        return a
    if uv1 is None:
        return -a
    if uv2 is None:
        return a
    return a if float(uv1[1]) < float(uv2[1]) else -a


def median_depth_in_mask(depth_u16: np.ndarray, mask_u8: np.ndarray, depth_scale: float, z_min: float, z_max: float) -> float:
    v = depth_u16[mask_u8 > 0].astype(np.float32)
    if v.size == 0:
        return float("inf")
    z = v * float(depth_scale)
    z = z[(z > 0.0) & (z >= float(z_min)) & (z <= float(z_max))]
    if z.size == 0:
        return float("inf")
    return float(np.median(z))


def select_top_visible_candidates(
    cands: List[dict],
    depth_u16: np.ndarray,
    depth_scale: float,
    z_min: float,
    z_max: float,
    occ_overlap_thr: float,
    min_visible_ratio: float,
    max_instances: int,
) -> List[dict]:
    for c in cands:
        c["z_med"] = median_depth_in_mask(depth_u16, c["mask_full"], depth_scale, z_min, z_max)
    cands = [c for c in cands if np.isfinite(c["z_med"])]
    # First keep near/top-layer by depth.
    cands.sort(key=lambda z: (z["z_med"], -z["area"], -z["confidence"]))

    selected: List[dict] = []
    occ_union = np.zeros_like(depth_u16, dtype=np.uint8)
    passed: List[dict] = []
    for c in cands:
        m = c["mask_full"]
        a = int(max(1, c["area"]))
        inter = int(np.count_nonzero(cv2.bitwise_and(occ_union, m)))
        vis_ratio = float((a - inter) / float(a))
        overlap_ratio = float(inter / float(a))
        c["visible_ratio"] = vis_ratio
        c["overlap_ratio"] = overlap_ratio
        if overlap_ratio > float(occ_overlap_thr):
            continue
        if vis_ratio < float(min_visible_ratio):
            continue
        passed.append(c)
        occ_union = cv2.bitwise_or(occ_union, m)
    # Among top-layer passed regions, prioritize larger areas for top-N.
    passed.sort(key=lambda z: (-z["area"], z["z_med"], -z["confidence"]))
    selected = passed[: int(max_instances)]
    return selected


def contour_metrics_from_mask(mask_u8: np.ndarray) -> Tuple[float, float]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    c = max(contours, key=cv2.contourArea)
    area = float(max(1.0, cv2.contourArea(c)))
    peri = float(max(1e-6, cv2.arcLength(c, True)))
    circularity = float((4.0 * np.pi * area) / (peri * peri))
    hull = cv2.convexHull(c)
    hull_area = float(max(1e-6, cv2.contourArea(hull)))
    solidity = float(area / hull_area)
    return circularity, solidity


def depth_quality_metrics(
    depth_u16: np.ndarray,
    mask_u8: np.ndarray,
    depth_scale: float,
    z_min: float,
    z_max: float,
) -> Tuple[float, float]:
    vals = depth_u16[mask_u8 > 0].astype(np.float32)
    if vals.size == 0:
        return 0.0, 0.0
    z = vals * float(depth_scale)
    valid = (z > 0.0) & (z >= float(z_min)) & (z <= float(z_max))
    vr = float(np.count_nonzero(valid)) / float(max(1, z.size))
    if np.count_nonzero(valid) < 8:
        return vr, 0.0
    zv = z[valid]
    z_relief = float(np.percentile(zv, 90.0) - np.percentile(zv, 10.0))
    return vr, z_relief


def rot_from_z_to_vec(v: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = v.astype(np.float64)
    n = float(np.linalg.norm(d))
    if n <= 1e-9:
        return np.eye(3, dtype=np.float64)
    d /= n
    c = float(np.clip(np.dot(z, d), -1.0, 1.0))
    if c > 0.999999:
        return np.eye(3, dtype=np.float64)
    if c < -0.999999:
        # 180 deg around x-axis.
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    axis = np.cross(z, d)
    axis /= max(1e-9, float(np.linalg.norm(axis)))
    ang = float(np.arccos(c))
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)
    return R


def make_arrow_mesh(radius: float) -> Tuple["o3d.geometry.TriangleMesh", "o3d.geometry.TriangleMesh"]:
    r = max(0.0006, float(radius))
    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=1.0, resolution=24, split=1)
    tip = o3d.geometry.TriangleMesh.create_sphere(radius=r * 1.9, resolution=16)
    shaft.compute_vertex_normals()
    tip.compute_vertex_normals()
    return shaft, tip


def hide_mesh_far(mesh: "o3d.geometry.TriangleMesh") -> None:
    mesh.translate((0.0, 0.0, -10.0), relative=False)


def make_vector_cylinder(radius: float) -> "o3d.geometry.TriangleMesh":
    r = max(0.0006, float(radius))
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=1.0, resolution=24, split=1)
    cyl.compute_vertex_normals()
    return cyl


def set_vector_cylinder(
    cyl_mesh: "o3d.geometry.TriangleMesh",
    p0: np.ndarray,
    p1: np.ndarray,
    color: Tuple[float, float, float],
    radius: float,
) -> None:
    p0 = p0.astype(np.float64)
    p1 = p1.astype(np.float64)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L <= 1e-8:
        hide_mesh_far(cyl_mesh)
        return
    c2 = make_vector_cylinder(max(0.0006, float(radius)))
    cyl_mesh.vertices = c2.vertices
    cyl_mesh.triangles = c2.triangles
    cyl_mesh.vertex_normals = c2.vertex_normals
    cyl_mesh.paint_uniform_color([float(color[0]), float(color[1]), float(color[2])])
    R = rot_from_z_to_vec(v / L)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    # Cylinder centered at z=0, height=1 => shift to [0, L].
    cyl_mesh.translate((0.0, 0.0, 0.5), relative=True)
    cyl_mesh.scale(L, center=(0.0, 0.0, 0.0))
    cyl_mesh.transform(T)
    cyl_mesh.translate(tuple(p0.tolist()), relative=True)


def set_arrow_mesh(
    shaft_mesh: "o3d.geometry.TriangleMesh",
    tip_mesh: "o3d.geometry.TriangleMesh",
    p0: np.ndarray,
    p1: np.ndarray,
    color: Tuple[float, float, float],
    radius: float,
) -> None:
    p0 = p0.astype(np.float64)
    p1 = p1.astype(np.float64)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L <= 1e-8:
        hide_mesh_far(shaft_mesh)
        hide_mesh_far(tip_mesh)
        return
    # Rebuild canonical meshes each call for robust transforms.
    s2, t2 = make_arrow_mesh(max(0.0006, float(radius)))
    shaft_mesh.vertices = s2.vertices
    shaft_mesh.triangles = s2.triangles
    shaft_mesh.vertex_normals = s2.vertex_normals
    tip_mesh.vertices = t2.vertices
    tip_mesh.triangles = t2.triangles
    tip_mesh.vertex_normals = t2.vertex_normals
    shaft_mesh.paint_uniform_color([float(color[0]), float(color[1]), float(color[2])])
    tip_mesh.paint_uniform_color([float(color[0]), float(color[1]), float(color[2])])

    R = rot_from_z_to_vec(v / L)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R

    # Canonical cylinder is centered at z=0 with height 1, shift to [0, L] first.
    shaft_mesh.translate((0.0, 0.0, 0.5), relative=True)
    shaft_mesh.scale(L, center=(0.0, 0.0, 0.0))
    shaft_mesh.transform(T)
    shaft_mesh.translate(tuple(p0.tolist()), relative=True)

    tip_mesh.transform(T)
    tip_mesh.translate(tuple(p1.tolist()), relative=True)


def main() -> None:
    args = parse_args()
    if rs is None or o3d is None:
        raise RuntimeError("Install pyrealsense2 and open3d in current env.")

    model = YOLO(args.model)
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {args.mesh}")
    if args.mesh_unit == "mm":
        mesh.scale(0.001, center=(0.0, 0.0, 0.0))

    mesh.compute_vertex_normals()
    src_mesh_points = mesh.sample_points_poisson_disk(3500)
    src_mesh_points.paint_uniform_color([0.95, 0.2, 0.1])
    src_pts_np = np.asarray(src_mesh_points.points, dtype=np.float64)
    src_center_local = np.mean(src_pts_np, axis=0).astype(np.float64)
    src_w, src_b = pca_basis(src_pts_np)
    # For a cap-like object, smallest-variance axis is usually cylinder axis.
    axis_local = src_b[:, 2].astype(np.float64)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, int(args.width), int(args.height), rs.format.z16, int(args.fps))
    cfg.enable_stream(rs.stream.color, int(args.width), int(args.height), rs.format.bgr8, int(args.fps))
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    configure_realsense(profile, args.exposure, args.gain, args.white_balance, args.auto_exposure, args.auto_white_balance)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx = float(intr.fx)
    fy = float(intr.fy)
    cx = float(getattr(intr, "ppx", getattr(intr, "cx", 0.0)))
    cy = float(getattr(intr, "ppy", getattr(intr, "cy", 0.0)))

    vis = o3d.visualization.Visualizer()
    vis.create_window("YOLO STL Fit O3D", width=1280, height=780, visible=True)
    full_cloud = o3d.geometry.PointCloud()
    seg_cloud = o3d.geometry.PointCloud()
    fit_meshes: List[o3d.geometry.TriangleMesh] = []
    for _ in range(int(max(1, args.max_fit_instances))):
        fit_meshes.append(o3d.geometry.TriangleMesh(mesh))
    show_full = not bool(args.hide_full_cloud)
    vis.add_geometry(full_cloud)
    vis.add_geometry(seg_cloud)
    if bool(args.show_mesh_fit):
        for gm in fit_meshes:
            vis.add_geometry(gm)
    vector_shafts: List[o3d.geometry.TriangleMesh] = []
    tip_spheres: List[o3d.geometry.TriangleMesh] = []
    for _ in range(int(max(1, args.max_fit_instances))):
        sh = make_vector_cylinder(float(args.vector_radius))
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.0008, float(args.vector_radius) * 0.8), resolution=12)
        sp.compute_vertex_normals()
        hide_mesh_far(sh)
        hide_mesh_far(sp)
        vector_shafts.append(sh)
        tip_spheres.append(sp)
        vis.add_geometry(sh)
        vis.add_geometry(sp)
    ropt = vis.get_render_option()
    if ropt is not None:
        ropt.point_size = float(max(1.0, args.pc_point_size))
        ropt.background_color = np.array([0.03, 0.03, 0.03] if args.pc_bg_dark else [1.0, 1.0, 1.0], dtype=np.float64)
        ropt.mesh_show_back_face = True

    roi_xyxy: Optional[Tuple[int, int, int, int]] = None
    print(f"[Info] model={args.model}")
    print(f"[Info] mesh={args.mesh} unit={args.mesh_unit} (scaled_to_m={args.mesh_unit == 'mm'})")
    print("[Keys] q / ESC to quit")
    view_inited = False
    try:
        while True:
            fr = align.process(pipe.wait_for_frames(5000))
            d = fr.get_depth_frame()
            c = fr.get_color_frame()
            if not d or not c:
                continue
            depth_u16 = np.asanyarray(d.get_data())
            color = np.asanyarray(c.get_data())
            h, w = color.shape[:2]
            if roi_xyxy is None:
                roi_xyxy = load_roi_yaml(args.roi_yaml, w, h) if args.roi_yaml else (0, 0, w, h)
                print(f"[Info] ROI: x={roi_xyxy[0]}:{roi_xyxy[2]} y={roi_xyxy[1]}:{roi_xyxy[3]}")
            x0, y0, x1, y1 = roi_xyxy
            roi = color[y0:y1, x0:x1]

            pred = model.predict(
                source=roi, conf=float(args.conf), iou=float(args.iou), imgsz=int(args.imgsz), max_det=int(args.max_det), verbose=False
            )[0]
            cands = []
            if pred.masks is not None and pred.boxes is not None and len(pred.boxes) > 0:
                boxes = pred.boxes
                n = len(boxes)
                for i in range(n):
                    m = yolo_mask_roi(pred, i, roi.shape[0], roi.shape[1])
                    if m is None:
                        continue
                    area = int(np.count_nonzero(m))
                    if area < 120:
                        continue
                    conf = float(boxes.conf[i].item())
                    mf = np.zeros((h, w), dtype=np.uint8)
                    mm = m.copy()
                    if int(args.mask_dilate_px) > 0:
                        ksz = int(args.mask_dilate_px)
                        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksz + 1, 2 * ksz + 1))
                        mm = cv2.dilate(mm, ker, iterations=1)
                    mf[y0:y1, x0:x1] = mm
                    cands.append({"mask": m, "mask_full": mf, "area": int(np.count_nonzero(mf)), "confidence": conf})
            cands = suppress_overlaps(cands, float(args.overlap_thr))
            # Robust prefilter: remove likely background false positives.
            roi_area = float(max(1, (x1 - x0) * (y1 - y0)))
            filtered: List[dict] = []
            for cand in cands:
                if float(cand["confidence"]) < float(args.prefilter_conf):
                    continue
                if int(cand["area"]) < int(args.prefilter_min_area):
                    continue
                if float(cand["area"]) > float(args.prefilter_max_area_ratio) * roi_area:
                    continue
                circ, solid = contour_metrics_from_mask(cand["mask"].astype(np.uint8))
                if circ < float(args.prefilter_min_circularity):
                    continue
                if solid < float(args.prefilter_min_solidity):
                    continue
                vr, relief_m = depth_quality_metrics(
                    depth_u16=depth_u16,
                    mask_u8=cand["mask_full"],
                    depth_scale=depth_scale,
                    z_min=float(args.z_min),
                    z_max=float(args.z_max),
                )
                if vr < float(args.prefilter_min_depth_valid_ratio):
                    continue
                if (relief_m * 1000.0) < float(args.prefilter_min_depth_relief_mm):
                    continue
                cand["circularity"] = float(circ)
                cand["solidity"] = float(solid)
                cand["depth_valid_ratio"] = float(vr)
                cand["depth_relief_mm"] = float(relief_m * 1000.0)
                filtered.append(cand)
            cands = filtered
            cands.sort(key=lambda z: (z["area"], z["confidence"]), reverse=True)

            mask_full = np.zeros((h, w), dtype=np.uint8)
            selected = select_top_visible_candidates(
                cands=cands,
                depth_u16=depth_u16,
                depth_scale=depth_scale,
                z_min=float(args.z_min),
                z_max=float(args.z_max),
                occ_overlap_thr=float(args.top_overlap_thr),
                min_visible_ratio=float(args.min_visible_ratio),
                max_instances=int(args.max_fit_instances),
            )
            for c in selected:
                mask_full = cv2.bitwise_or(mask_full, c["mask_full"])

            all_mask = np.ones((h, w), dtype=np.uint8) * 255
            all_pts, all_cols = mask_to_points(
                all_mask, depth_u16, color, fx, fy, cx, cy, depth_scale, float(args.z_min), float(args.z_max), stride=2
            )
            if all_pts.shape[0] < 150:
                # Fallback: at least show ROI cloud when full-frame valid points are sparse.
                roi_mask_full = np.zeros((h, w), dtype=np.uint8)
                roi_mask_full[y0:y1, x0:x1] = 255
                all_pts, all_cols = mask_to_points(
                    roi_mask_full, depth_u16, color, fx, fy, cx, cy, depth_scale, float(args.z_min), float(args.z_max), stride=1
                )
            seg_pts, seg_cols = mask_to_points(
                mask_full, depth_u16, color, fx, fy, cx, cy, depth_scale, float(args.z_min), float(args.z_max), stride=max(1, int(args.cloud_stride))
            )

            if show_full:
                full_cloud.points = o3d.utility.Vector3dVector(all_pts.astype(np.float64))
                full_cloud.colors = o3d.utility.Vector3dVector(all_cols.astype(np.float64))
            else:
                full_cloud.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
                full_cloud.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

            # Per-instance segmented colors for clarity.
            seg_pts_all: List[np.ndarray] = []
            seg_cols_all: List[np.ndarray] = []
            for i, c in enumerate(selected[: int(args.max_fit_instances)]):
                ipts, _ = mask_to_points(
                    c["mask_full"],
                    depth_u16,
                    color,
                    fx,
                    fy,
                    cx,
                    cy,
                    depth_scale,
                    float(args.z_min),
                    float(args.z_max),
                    stride=max(1, int(args.cloud_stride)),
                )
                if ipts.shape[0] == 0:
                    continue
                col = color_from_index(i + 1).reshape(1, 3)
                icol = np.repeat(col, ipts.shape[0], axis=0).astype(np.float32)
                seg_pts_all.append(ipts.astype(np.float32))
                seg_cols_all.append(icol)
            if seg_pts_all:
                seg_pts = np.concatenate(seg_pts_all, axis=0)
                seg_cols = np.concatenate(seg_cols_all, axis=0)
            else:
                seg_pts = np.zeros((0, 3), dtype=np.float32)
                seg_cols = np.zeros((0, 3), dtype=np.float32)
            seg_cloud.points = o3d.utility.Vector3dVector(seg_pts.astype(np.float64))
            seg_cloud.colors = o3d.utility.Vector3dVector(seg_cols.astype(np.float64))

            pose_lines: List[str] = []
            valid_fit_count = 0
            for i, c in enumerate(selected[: int(args.max_fit_instances)]):
                obj_pts, _ = mask_to_points(
                    c["mask_full"],
                    depth_u16,
                    color,
                    fx,
                    fy,
                    cx,
                    cy,
                    depth_scale,
                    float(args.z_min),
                    float(args.z_max),
                    stride=max(1, int(args.cloud_stride)),
                )
                gm = fit_meshes[i]
                gm.vertices = mesh.vertices
                gm.triangles = mesh.triangles
                gm.vertex_normals = mesh.vertex_normals
                gm.compute_vertex_normals()
                if obj_pts.shape[0] < 140:
                    gm.paint_uniform_color([0.3, 0.3, 0.3])
                    gm.translate((0.0, 0.0, -8.0), relative=False)
                    hide_mesh_far(vector_shafts[i])
                    hide_mesh_far(tip_spheres[i])
                    continue
                tgt_pcd = to_o3d_pcd(obj_pts, None)
                T, ok, fitness, rmse = register_mesh_to_segment(
                    src_mesh_points=src_mesh_points,
                    tgt_seg_points=tgt_pcd,
                    voxel=float(args.voxel),
                    icp_dist=float(args.icp_dist),
                )
                gm.paint_uniform_color([0.95, 0.2, 0.1] if ok else [0.9, 0.9, 0.15])
                gm.transform(T)
                if ok:
                    valid_fit_count += 1
                Rw = T[:3, :3].astype(np.float64)
                tw = T[:3, 3].astype(np.float64)
                center_w = (Rw @ src_center_local + tw).astype(np.float64)
                axis_w = (Rw @ axis_local).astype(np.float64)
                axis_w = orient_axis_sign(
                    axis_w=axis_w,
                    center_w=center_w,
                    mode=str(args.axis_sign_mode),
                    fx=float(fx),
                    fy=float(fy),
                    cx=float(cx),
                    cy=float(cy),
                    test_len=float(max(0.01, args.vector_len)),
                )
                r_deg, p_deg = normal_to_roll_pitch(axis_w)
                p0 = center_w.astype(np.float64)
                p1 = (center_w + axis_w * float(args.vector_len)).astype(np.float64)
                if ok:
                    set_vector_cylinder(vector_shafts[i], p0, p1, (0.1, 1.0, 0.1), float(args.vector_radius))
                    tip_spheres[i].translate(tuple(p1.tolist()), relative=False)
                    tip_spheres[i].paint_uniform_color([0.1, 1.0, 0.1])
                    pose_lines.append(
                        f"#{i+1} z={c['z_med']:.3f} vis={c['visible_ratio']:.2f} fit={fitness:.2f} rmse={rmse*1000.0:.1f} x={center_w[0]:+.3f} y={center_w[1]:+.3f} z={center_w[2]:+.3f} r={r_deg:+.1f} p={p_deg:+.1f}"
                    )
                else:
                    hide_mesh_far(vector_shafts[i])
                    hide_mesh_far(tip_spheres[i])

            for j in range(len(selected), int(args.max_fit_instances)):
                gm = fit_meshes[j]
                gm.vertices = mesh.vertices
                gm.triangles = mesh.triangles
                gm.vertex_normals = mesh.vertex_normals
                gm.compute_vertex_normals()
                gm.paint_uniform_color([0.3, 0.3, 0.3])
                gm.translate((0.0, 0.0, -8.0), relative=False)
                hide_mesh_far(vector_shafts[j])
                hide_mesh_far(tip_spheres[j])

            vis.update_geometry(full_cloud)
            vis.update_geometry(seg_cloud)
            if bool(args.show_mesh_fit):
                for gm in fit_meshes:
                    vis.update_geometry(gm)
            for sh in vector_shafts:
                vis.update_geometry(sh)
            for sp in tip_spheres:
                vis.update_geometry(sp)
            if not view_inited:
                ctr = vis.get_view_control()
                if ctr is not None:
                    ctr.set_lookat([0.0, 0.0, 0.40])
                    ctr.set_front([0.0, 0.0, -1.0])  # view from camera toward +Z scene
                    ctr.set_up([0.0, -1.0, 0.0])     # camera-style image Y down
                    ctr.set_zoom(0.65)
                view_inited = True
            vis.poll_events()
            vis.update_renderer()

            out = color.copy()
            if args.show_roi:
                cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 1, cv2.LINE_AA)
            if np.count_nonzero(mask_full) > 0:
                ov = out.copy()
                ov[mask_full > 0] = (0.25 * ov[mask_full > 0] + 0.75 * np.array([0, 220, 0])).astype(np.uint8)
                out = ov
            cv2.putText(
                out,
                f"det={len(cands)} top={len(selected)} fit_ok={valid_fit_count} seg_pts={seg_pts.shape[0]}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if valid_fit_count > 0 else (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            yy = 56
            for line in pose_lines[:6]:
                cv2.putText(out, line, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                yy += 22
            cv2.imshow("YOLO STL Fit 2D", out)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
    finally:
        pipe.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
