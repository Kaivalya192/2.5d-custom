import argparse
import copy
import json
import time

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def K_from_intr(intr):
    return np.array([[intr["fx"], 0.0, intr["ppx"]],
                     [0.0, intr["fy"], intr["ppy"]],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def D_from_intr(intr):
    return np.array(intr.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)


def Rt_from_extr(ex):
    R = np.array(ex["R"], dtype=np.float64)
    t = np.array(ex["t_m"], dtype=np.float64).reshape(3, 1)
    return R, t


def make_sgbm(num_disp=256, block_size=5):
    num_disp = int(np.ceil(num_disp / 16) * 16)
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=7,
        speckleWindowSize=120,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def build_rectification(KL, DL, KR, DR, R_LR, t_LR, size_wh):
    W, H = size_wh
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        KL, DL, KR, DR, (W, H), R_LR, t_LR,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )
    mapLx, mapLy = cv2.initUndistortRectifyMap(KL, DL, R1, P1, (W, H), cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(KR, DR, R2, P2, (W, H), cv2.CV_32FC1)
    return (mapLx, mapLy, mapRx, mapRy, R1, Q)


def score_order(irL_raw, irR_raw, maps, matcher):
    mapLx, mapLy, mapRx, mapRy, _, _ = maps
    L = cv2.remap(irL_raw, mapLx, mapLy, cv2.INTER_LINEAR)
    R = cv2.remap(irR_raw, mapRx, mapRy, cv2.INTER_LINEAR)
    disp = matcher.compute(L, R).astype(np.float32) / 16.0
    valid = disp > 1.0
    return float(np.mean(valid)), (float(np.median(disp[valid])) if np.any(valid) else -1.0)


def lr_consistency_mask(dispL, dispR, thresh=1.5):
    H, W = dispL.shape
    xs = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    xr = xs - dispL
    xr_i = np.rint(xr).astype(np.int32)

    valid = dispL > 1.0
    valid &= (xr_i >= 0) & (xr_i < W)

    ys = np.arange(H, dtype=np.int32)[:, None].repeat(W, axis=1)
    dispR_samp = np.zeros_like(dispL, dtype=np.float32)
    dispR_samp[valid] = dispR[ys[valid], xr_i[valid]]

    valid &= (dispR_samp > 1.0)
    valid &= (np.abs(dispL - dispR_samp) <= thresh)
    return valid


def preprocess_for_features(pcd, voxel):
    if len(pcd.points) < 50:
        return None, None
    down = pcd.voxel_down_sample(float(voxel))
    if len(down.points) < 50:
        return None, None

    normal_radius = float(voxel) * 2.0
    feat_radius = float(voxel) * 5.0
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=feat_radius, max_nn=100)
    )
    return down, fpfh


def global_register(model_down, scene_down, model_fpfh, scene_fpfh, voxel, ransac_iters):
    max_corr = float(voxel) * 1.5
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model_down,
        scene_down,
        model_fpfh,
        scene_fpfh,
        mutual_filter=True,
        max_correspondence_distance=max_corr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(int(ransac_iters), 0.999),
    )


def refine_icp(model_down, scene_down, init_T, voxel):
    max_corr = float(voxel) * 1.2
    try:
        return o3d.pipelines.registration.registration_icp(
            model_down,
            scene_down,
            max_corr,
            init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
    except RuntimeError:
        return o3d.pipelines.registration.registration_icp(
            model_down,
            scene_down,
            max_corr,
            init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )


def pose_delta(Ta, Tb):
    dt = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
    R = Ta[:3, :3] @ Tb[:3, :3].T
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    da = float(np.degrees(np.arccos(c)))
    return dt, da


def is_duplicate_pose(T, poses, trans_tol, rot_tol_deg):
    for P in poses:
        dt, da = pose_delta(T, P)
        if dt < trans_tol and da < rot_tol_deg:
            return True
    return False


def detect_instances(scene_pcd, model_down, model_fpfh, args):
    poses = []
    remaining = scene_pcd.voxel_down_sample(float(args.voxel_scene))
    if len(remaining.points) < args.min_scene_points:
        return poses

    normal_radius = float(args.voxel_scene) * 2.0
    remaining.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )

    for _ in range(int(args.max_instances)):
        if len(remaining.points) < args.min_scene_points:
            break

        scene_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            remaining,
            o3d.geometry.KDTreeSearchParamHybrid(radius=float(args.voxel_scene) * 5.0, max_nn=100),
        )
        coarse = global_register(
            model_down, remaining, model_fpfh, scene_fpfh, args.voxel_scene, args.ransac_iters
        )
        if coarse.fitness < args.global_min_fitness:
            break

        fine = refine_icp(model_down, remaining, coarse.transformation, args.voxel_scene)
        if fine.fitness < args.icp_min_fitness:
            break
        if fine.inlier_rmse > args.icp_max_rmse:
            break

        T = fine.transformation.copy()
        if not is_duplicate_pose(T, poses, args.pose_merge_dist, args.pose_merge_angle):
            poses.append(T)

        model_in_scene = copy.deepcopy(model_down)
        model_in_scene.transform(T)
        d = np.asarray(remaining.compute_point_cloud_distance(model_in_scene))
        keep = np.where(d > float(args.instance_suppression_dist))[0]
        if keep.size == d.size:
            break
        if keep.size < args.min_scene_points:
            break
        remaining = remaining.select_by_index(keep)
        remaining.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )

    return poses


def track_instances(scene_pcd, model_down, poses, args):
    if len(poses) == 0:
        return []
    scene_down = scene_pcd.voxel_down_sample(float(args.voxel_scene))
    if len(scene_down.points) < args.min_scene_points:
        return []
    scene_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(args.voxel_scene) * 2.0, max_nn=30)
    )

    tracked = []
    for T in poses:
        fine = refine_icp(model_down, scene_down, T, args.voxel_scene)
        if fine.fitness >= args.track_min_fitness and fine.inlier_rmse <= (args.icp_max_rmse * 1.5):
            if not is_duplicate_pose(fine.transformation, tracked, args.pose_merge_dist, args.pose_merge_angle):
                tracked.append(fine.transformation.copy())
    return tracked


def update_pose_geometries(vis, old_geoms, model_lines, poses, axis_size):
    for g in old_geoms:
        vis.remove_geometry(g, reset_bounding_box=False)

    palette = [
        [1.0, 0.2, 0.2],
        [0.2, 1.0, 0.2],
        [0.2, 0.6, 1.0],
        [1.0, 0.8, 0.2],
        [1.0, 0.2, 1.0],
        [0.2, 1.0, 1.0],
    ]
    new_geoms = []
    for i, T in enumerate(poses):
        lines = copy.deepcopy(model_lines)
        lines.paint_uniform_color(palette[i % len(palette)])
        lines.transform(T)
        vis.add_geometry(lines, reset_bounding_box=False)
        new_geoms.append(lines)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axis_size))
        axis.transform(T)
        vis.add_geometry(axis, reset_bounding_box=False)
        new_geoms.append(axis)
    return new_geoms


def build_raw_legacy_cloud(
    ir_left_raw,
    ir_right_raw,
    rgb_raw,
    maps,
    matcher,
    right_matcher,
    wls_filter,
    use_wls,
    use_lr_check,
    lr_thresh,
    clahe,
    temporal_state,
    temporal_alpha,
    R_L_to_RGB,
    t_L_to_RGB,
    Krgb,
    stride,
    z_min,
    z_max,
):
    mapLx, mapLy, mapRx, mapRy, RrectL, Q = maps

    L = cv2.remap(ir_left_raw, mapLx, mapLy, cv2.INTER_LINEAR)
    R = cv2.remap(ir_right_raw, mapRx, mapRy, cv2.INTER_LINEAR)
    if clahe is not None:
        Lm = clahe.apply(L)
        Rm = clahe.apply(R)
    else:
        Lm, Rm = L, R

    dispL_16 = matcher.compute(Lm, Rm)
    dispL_raw = dispL_16.astype(np.float32) / 16.0

    dispR_16 = None
    if right_matcher is not None:
        dispR_16 = right_matcher.compute(Rm, Lm)
    elif use_lr_check:
        dispR_16 = matcher.compute(Rm, Lm)

    if use_wls and (dispR_16 is not None) and (wls_filter is not None):
        dispF_16 = wls_filter.filter(dispL_16, Lm, None, dispR_16)
        disp = dispF_16.astype(np.float32) / 16.0
    else:
        disp = dispL_raw

    if temporal_alpha > 0:
        prev = temporal_state.get("prev_disp")
        if prev is None:
            temporal_state["prev_disp"] = disp.copy()
        else:
            a = float(temporal_alpha)
            temporal_state["prev_disp"] = (1.0 - a) * disp + a * prev
        disp_use = temporal_state["prev_disp"]
    else:
        disp_use = disp

    valid = disp_use > 1.0
    if use_lr_check and (dispR_16 is not None):
        dispR = dispR_16.astype(np.float32) / 16.0
        finite = dispR[np.isfinite(dispR)]
        if finite.size > 0 and np.median(finite) < 0:
            dispR = -dispR
        valid &= lr_consistency_mask(dispL_raw, dispR, thresh=float(lr_thresh))

    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), 0.0

    xyz = cv2.reprojectImageTo3D(disp_use, Q, handleMissingValues=True)
    H, W = disp_use.shape
    s = max(1, int(stride))
    xyz_s = xyz[0:H:s, 0:W:s, :].reshape(-1, 3)
    valid_s = valid[0:H:s, 0:W:s].reshape(-1).copy()

    Z = xyz_s[:, 2]
    valid_s &= np.isfinite(Z) & (Z >= z_min) & (Z <= z_max)
    if np.count_nonzero(valid_s) < 500:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), float(np.mean(valid))

    pts_rectL = xyz_s[valid_s].astype(np.float32)
    pts_left = (RrectL.T @ pts_rectL.T).T.astype(np.float32)
    pts_rgb = (R_L_to_RGB @ pts_left.T + t_L_to_RGB).T.astype(np.float32)

    Xc, Yc, Zc = pts_rgb[:, 0], pts_rgb[:, 1], pts_rgb[:, 2]
    good = Zc > 0.0

    u = (Krgb[0, 0] * (Xc / Zc) + Krgb[0, 2])
    v = (Krgb[1, 1] * (Yc / Zc) + Krgb[1, 2])
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    Hc, Wc = rgb_raw.shape[:2]
    good &= (ui >= 0) & (ui < Wc) & (vi >= 0) & (vi < Hc)
    if np.count_nonzero(good) < 500:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), float(np.mean(valid))

    pts = pts_rgb[good]
    cols = rgb_raw[vi[good], ui[good], :].astype(np.float32) / 255.0
    return pts, cols, float(np.mean(valid))


def load_model(model_path, model_scale, model_samples, voxel_model):
    mesh = o3d.io.read_triangle_mesh(model_path)
    if mesh.is_empty():
        raise RuntimeError(f"Cannot read model mesh: {model_path}")

    if model_scale != 1.0:
        mesh.scale(float(model_scale), center=mesh.get_center())
    mesh.compute_vertex_normals()

    model_pcd = mesh.sample_points_uniformly(number_of_points=int(model_samples))
    model_down, model_fpfh = preprocess_for_features(model_pcd, voxel_model)
    if model_down is None or model_fpfh is None:
        raise RuntimeError("Model preprocessing failed. Increase --model_samples or lower --voxel_model.")

    model_lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    return mesh, model_down, model_fpfh, model_lines


def print_poses(poses):
    for i, T in enumerate(poses):
        t = T[:3, 3]
        print(f"  [{i}] t=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m")


def main():
    ap = argparse.ArgumentParser(description="Real-time CAD pose on legacy RAW stereo cloud (multi-instance)")
    ap.add_argument("--model", default="tamplate.stl", help="CAD model file (.stl)")
    ap.add_argument("--calib", default="calib_d435i/calibration.json")

    ap.add_argument("--ir_w", type=int, default=848)
    ap.add_argument("--ir_h", type=int, default=480)
    ap.add_argument("--ir_fps", type=int, default=30)

    ap.add_argument("--rgb_w", type=int, default=640)
    ap.add_argument("--rgb_h", type=int, default=480)
    ap.add_argument("--rgb_fps", type=int, default=30)

    ap.add_argument("--num_disp", type=int, default=256)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)
    ap.add_argument("--wls", action="store_true")
    ap.add_argument("--wls_lambda", type=float, default=8000.0)
    ap.add_argument("--wls_sigma", type=float, default=1.5)
    ap.add_argument("--lr_check", action="store_true")
    ap.add_argument("--lr_thresh", type=float, default=1.5)
    ap.add_argument("--temporal_alpha", type=float, default=0.2)

    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--z_min", type=float, default=0.1)
    ap.add_argument("--z_max", type=float, default=1.0)

    ap.add_argument("--model_scale", type=float, default=0.001, help="default assumes CAD units are mm")
    ap.add_argument("--model_samples", type=int, default=35000)
    ap.add_argument("--voxel_model", type=float, default=0.004)
    ap.add_argument("--voxel_scene", type=float, default=0.006)

    ap.add_argument("--max_instances", type=int, default=4)
    ap.add_argument("--detect_interval", type=int, default=6, help="run global detection every N frames")
    ap.add_argument("--min_scene_points", type=int, default=1200)

    ap.add_argument("--ransac_iters", type=int, default=40000)
    ap.add_argument("--global_min_fitness", type=float, default=0.20)
    ap.add_argument("--icp_min_fitness", type=float, default=0.28)
    ap.add_argument("--icp_max_rmse", type=float, default=0.012)
    ap.add_argument("--track_min_fitness", type=float, default=0.18)
    ap.add_argument("--instance_suppression_dist", type=float, default=0.012)
    ap.add_argument("--pose_merge_dist", type=float, default=0.02)
    ap.add_argument("--pose_merge_angle", type=float, default=12.0)

    ap.add_argument("--emitter", type=int, default=1)
    ap.add_argument("--laser_power", type=float, default=None)

    ap.add_argument("--axis_size", type=float, default=0.04)
    ap.add_argument("--point_size", type=float, default=3.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    mesh, model_down, model_fpfh, model_lines = load_model(
        args.model, args.model_scale, args.model_samples, args.voxel_model
    )
    model_extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"[Model] {args.model} extent(m)={model_extent} points(down)={len(model_down.points)}")

    with open(args.calib, "r") as f:
        calib = json.load(f)
    ir1_intr = calib["streams"]["ir1"]["intrinsics"]
    ir2_intr = calib["streams"]["ir2"]["intrinsics"]

    K1, D1 = K_from_intr(ir1_intr), D_from_intr(ir1_intr)
    K2, D2 = K_from_intr(ir2_intr), D_from_intr(ir2_intr)
    R_1to2, t_1to2 = Rt_from_extr(calib["extrinsics"]["ir1_to_ir2"])
    R_2to1, t_2to1 = Rt_from_extr(calib["extrinsics"]["ir2_to_ir1"])
    R_1toRGB, t_1toRGB = Rt_from_extr(calib["extrinsics"]["ir1_to_rgb"])
    R_2toRGB, t_2toRGB = Rt_from_extr(calib["extrinsics"]["ir2_to_rgb"])

    matcher = make_sgbm(args.num_disp, args.block)
    has_ximgproc = hasattr(cv2, "ximgproc")
    right_matcher = None
    wls_filter = None
    use_wls = False
    if args.lr_check or args.wls:
        if has_ximgproc:
            right_matcher = cv2.ximgproc.createRightMatcher(matcher)
        elif args.wls:
            print("[WARN] OpenCV ximgproc is unavailable; --wls disabled.")
    if args.wls and has_ximgproc:
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls_filter.setLambda(float(args.wls_lambda))
        wls_filter.setSigmaColor(float(args.wls_sigma))
        use_wls = True
        print("[Info] WLS enabled")

    maps_A = build_rectification(K1, D1, K2, D2, R_1to2, t_1to2, (args.ir_w, args.ir_h))
    maps_B = build_rectification(K2, D2, K1, D1, R_2to1, t_2to1, (args.ir_w, args.ir_h))
    clahe = None
    if args.clahe:
        clahe = cv2.createCLAHE(
            clipLimit=float(args.clahe_clip),
            tileGridSize=(int(args.clahe_grid), int(args.clahe_grid)),
        )

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)
    cfg.enable_stream(rs.stream.infrared, 2, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)
    cfg.enable_stream(rs.stream.color, args.rgb_w, args.rgb_h, rs.format.rgb8, args.rgb_fps)
    profile = pipe.start(cfg)

    # Use runtime color intrinsics for correct RGB/pointcloud alignment.
    color_sp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    cintr = color_sp.get_intrinsics()
    Krgb = np.array([[cintr.fx, 0, cintr.ppx],
                     [0, cintr.fy, cintr.ppy],
                     [0, 0, 1]], dtype=np.float64)
    print(f"[Info] color intr fx={cintr.fx:.3f} fy={cintr.fy:.3f} cx={cintr.ppx:.3f} cy={cintr.ppy:.3f}")

    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    try:
        depth_sensor.set_option(rs.option.emitter_enabled, float(args.emitter))
    except Exception:
        pass
    if args.laser_power is not None:
        try:
            depth_sensor.set_option(rs.option.laser_power, float(args.laser_power))
        except Exception as e:
            print(f"[WARN] laser_power not set: {e}")

    frames = pipe.wait_for_frames(5000)
    ir1 = frames.get_infrared_frame(1)
    ir2 = frames.get_infrared_frame(2)
    if not ir1 or not ir2:
        raise RuntimeError("IR frames not available.")
    ir1_raw = np.asanyarray(ir1.get_data())
    ir2_raw = np.asanyarray(ir2.get_data())
    vrA, medA = score_order(ir1_raw, ir2_raw, maps_A, matcher)
    vrB, medB = score_order(ir2_raw, ir1_raw, maps_B, matcher)
    print(f"[Order A] LEFT=IR1 RIGHT=IR2  valid={vrA*100:.1f}% medDisp={medA:.2f}")
    print(f"[Order B] LEFT=IR2 RIGHT=IR1  valid={vrB*100:.1f}% medDisp={medB:.2f}")
    if vrB > vrA:
        maps = maps_B
        get_left = lambda fr: np.asanyarray(fr.get_infrared_frame(2).get_data())
        get_right = lambda fr: np.asanyarray(fr.get_infrared_frame(1).get_data())
        R_L_to_RGB, t_L_to_RGB = R_2toRGB, t_2toRGB
        print("[Chosen RAW] LEFT=IR2 RIGHT=IR1")
    else:
        maps = maps_A
        get_left = lambda fr: np.asanyarray(fr.get_infrared_frame(1).get_data())
        get_right = lambda fr: np.asanyarray(fr.get_infrared_frame(2).get_data())
        R_L_to_RGB, t_L_to_RGB = R_1toRGB, t_1toRGB
        print("[Chosen RAW] LEFT=IR1 RIGHT=IR2")

    vis = o3d.visualization.Visualizer()
    vis.create_window("RAW Legacy CAD Pose (multi-instance)", 1400, 850, visible=True)
    render_opt = vis.get_render_option()
    render_opt.point_size = float(args.point_size)

    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(world_axis)

    scene_geom = o3d.geometry.PointCloud()
    scene_geom.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
    scene_geom.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
    vis.add_geometry(scene_geom)

    pose_geoms = []
    temporal_state = {}
    poses = []
    frame_idx = 0
    last_log = 0.0
    first_view = True
    print("[Keys] q / ESC to quit")

    try:
        while True:
            fr = pipe.wait_for_frames(5000)
            color = fr.get_color_frame()
            if not color:
                continue
            rgb_raw = np.asanyarray(color.get_data())
            ir_left = get_left(fr)
            ir_right = get_right(fr)

            pts, cols, valid_ratio = build_raw_legacy_cloud(
                ir_left,
                ir_right,
                rgb_raw,
                maps,
                matcher,
                right_matcher,
                wls_filter,
                use_wls,
                bool(args.lr_check),
                float(args.lr_thresh),
                clahe,
                temporal_state,
                float(args.temporal_alpha),
                R_L_to_RGB,
                t_L_to_RGB,
                Krgb,
                int(args.stride),
                float(args.z_min),
                float(args.z_max),
            )

            if pts.shape[0] < args.min_scene_points:
                vis.poll_events()
                vis.update_renderer()
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue

            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            scene_pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

            do_detect = (frame_idx % max(1, int(args.detect_interval)) == 0) or (len(poses) == 0)
            if do_detect:
                mode = "detect"
                poses = detect_instances(scene_pcd, model_down, model_fpfh, args)
            else:
                mode = "track"
                poses = track_instances(scene_pcd, model_down, poses, args)
                if len(poses) == 0:
                    mode = "recover"
                    poses = detect_instances(scene_pcd, model_down, model_fpfh, args)

            scene_geom.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            scene_geom.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
            vis.update_geometry(scene_geom)

            pose_geoms = update_pose_geometries(
                vis, pose_geoms, model_lines, poses, args.axis_size
            )

            if first_view:
                vis.reset_view_point(True)
                first_view = False
            vis.poll_events()
            vis.update_renderer()

            now = time.time()
            if now - last_log > 1.0:
                print(
                    f"[Pose] mode={mode:7s} instances={len(poses)} "
                    f"raw_pts={pts.shape[0]} disp_valid={valid_ratio*100:.1f}%"
                )
                if mode in ("detect", "recover") and len(poses) > 0:
                    print_poses(poses)
                last_log = now

            frame_idx += 1
            if args.debug:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
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
