import argparse
import copy
import time

import numpy as np
import open3d as o3d
import pyrealsense2 as rs


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


def refine_icp_multistage(model_down, scene_down, init_T, voxel):
    T = init_T
    reg = None
    for mult in (2.5, 1.5, 1.0):
        max_corr = float(voxel) * float(mult)
        try:
            reg = o3d.pipelines.registration.registration_icp(
                model_down,
                scene_down,
                max_corr,
                T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            )
        except RuntimeError:
            reg = o3d.pipelines.registration.registration_icp(
                model_down,
                scene_down,
                max_corr,
                T,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
        T = reg.transformation
    return reg


def best_pose_hypothesis(model_down, scene_down, model_fpfh, scene_fpfh, args):
    best_reg = None
    best_score = None
    trials = max(1, int(args.ransac_hypotheses))
    for _ in range(trials):
        coarse = global_register(
            model_down, scene_down, model_fpfh, scene_fpfh, args.voxel_scene, args.ransac_iters
        )
        if coarse.fitness < args.global_min_fitness:
            continue

        fine = refine_icp_multistage(model_down, scene_down, coarse.transformation, args.voxel_scene)
        if fine.fitness < args.icp_min_fitness:
            continue
        if fine.inlier_rmse > args.icp_max_rmse:
            continue

        score = (float(fine.fitness), float(-fine.inlier_rmse))
        if best_score is None or score > best_score:
            best_score = score
            best_reg = fine
    return best_reg


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
        best = best_pose_hypothesis(model_down, remaining, model_fpfh, scene_fpfh, args)
        if best is None:
            break

        T = best.transformation.copy()
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
        fine = refine_icp_multistage(model_down, scene_down, T, args.voxel_scene)
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


def rs_cloud_from_aligned_depth(depth_frame, depth_scale, K, rgb, stride, z_min, z_max, flip_y=False):
    depth_u16 = np.asanyarray(depth_frame.get_data())
    z = depth_u16[0:depth_u16.shape[0]:stride, 0:depth_u16.shape[1]:stride].reshape(-1).astype(np.float32)
    z_m = z * float(depth_scale)

    valid = (z_m > 0) & (z_m >= z_min) & (z_m <= z_max)
    if np.count_nonzero(valid) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    H, W = depth_u16.shape
    us = np.arange(0, W, stride, dtype=np.float32)
    vs = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    u = uu.reshape(-1)[valid]
    v = vv.reshape(-1)[valid]
    Z = z_m[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    if flip_y:
        Y = -Y
    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)

    ui = u.astype(np.int32)
    vi = v.astype(np.int32)
    cols = rgb[vi, ui, :].astype(np.float32) / 255.0
    return pts, cols


def apply_scene_cleanup(pts, cols, voxel=0.0, sor_nb=0, sor_std=0.0):
    if pts.shape[0] == 0:
        return pts, cols

    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    p.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

    if voxel and voxel > 0:
        p = p.voxel_down_sample(float(voxel))

    if sor_nb and sor_nb > 0 and sor_std and sor_std > 0:
        # remove_statistical_outlier already returns the filtered cloud.
        p, _ = p.remove_statistical_outlier(nb_neighbors=int(sor_nb), std_ratio=float(sor_std))

    pts2 = np.asarray(p.points, dtype=np.float32)
    cols2 = np.asarray(p.colors, dtype=np.float32)
    return pts2, cols2


def make_filters(enable):
    if not enable:
        return None
    # Do not use decimation here; it changes depth image geometry and can break
    # direct RGB/depth pixel correspondence used for colorized clouds.
    spat = rs.spatial_filter()
    temp = rs.temporal_filter()
    hole = rs.hole_filling_filter()
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20)
    temp.set_option(rs.option.filter_smooth_alpha, 0.4)
    temp.set_option(rs.option.filter_smooth_delta, 20)
    return [spat, temp, hole]


def start_rs_pipeline_with_fallback(pipe, req_w, req_h, req_fps, allow_fallback=True):
    tried = []
    candidates = [(int(req_w), int(req_h), int(req_fps))]
    if allow_fallback:
        for c in [(1280, 720, int(req_fps)), (848, 480, int(req_fps)), (640, 480, int(req_fps))]:
            if c not in candidates:
                candidates.append(c)

    last_err = None
    for w, h, fps in candidates:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        try:
            profile = pipe.start(cfg)
            return profile, w, h, fps
        except Exception as e:
            tried.append(f"{w}x{h}@{fps}")
            last_err = e
    raise RuntimeError(f"Failed to start RS pipeline for modes: {', '.join(tried)} | last error: {last_err}")


def print_poses(poses):
    for i, T in enumerate(poses):
        t = T[:3, 3]
        print(f"  [{i}] t=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m")


def main():
    ap = argparse.ArgumentParser(description="Real-time 6D CAD pose from RS SDK point cloud (multi-instance)")
    ap.add_argument("--model", default="tamplate.stl", help="CAD model file (.stl)")

    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--z_min", type=float, default=0.1)
    ap.add_argument("--z_max", type=float, default=1.0)
    ap.add_argument("--flip_y", action="store_true")
    ap.add_argument("--filters", dest="filters", action="store_true",
                    help="enable RS depth post-filters (default: on)")
    ap.add_argument("--no_filters", dest="filters", action="store_false",
                    help="disable RS depth post-filters")
    ap.set_defaults(filters=True)
    ap.add_argument("--render_min_points", type=int, default=500,
                    help="minimum points required to update cloud visualization")
    ap.add_argument("--scene_voxel_vis", type=float, default=0.0,
                    help="optional scene voxel downsample before render/pose (0 disables)")
    ap.add_argument("--sor_nb", type=int, default=0,
                    help="optional scene statistical outlier neighbors (0 disables)")
    ap.add_argument("--sor_std", type=float, default=0.0,
                    help="optional scene statistical outlier std ratio")

    ap.add_argument("--model_scale", type=float, default=0.001, help="default assumes CAD units are mm")
    ap.add_argument("--model_samples", type=int, default=35000)
    ap.add_argument("--voxel_model", type=float, default=0.004)
    ap.add_argument("--voxel_scene", type=float, default=0.006)

    ap.add_argument("--max_instances", type=int, default=4)
    ap.add_argument("--detect_interval", type=int, default=6, help="run global detection every N frames")
    ap.add_argument("--min_scene_points", type=int, default=1200)

    ap.add_argument("--ransac_iters", type=int, default=40000)
    ap.add_argument("--ransac_hypotheses", type=int, default=4,
                    help="number of RANSAC+ICP hypotheses per instance (higher = slower, more robust)")
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
    ap.add_argument("--no_res_fallback", action="store_true",
                    help="disable automatic fallback to lower resolutions if requested mode fails")
    args = ap.parse_args()

    mesh, model_down, model_fpfh, model_lines = load_model(
        args.model, args.model_scale, args.model_samples, args.voxel_model
    )
    model_extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    print(f"[Model] {args.model} extent(m)={model_extent} points(down)={len(model_down.points)}")

    pipe = rs.pipeline()
    profile, run_w, run_h, run_fps = start_rs_pipeline_with_fallback(
        pipe, args.w, args.h, args.fps, allow_fallback=(not args.no_res_fallback)
    )
    print(f"[Info] stream mode depth+color={run_w}x{run_h}@{run_fps} stride={args.stride}")

    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"[Info] depth_scale={depth_scale} m/unit")
    try:
        depth_sensor.set_option(rs.option.emitter_enabled, float(args.emitter))
    except Exception:
        pass
    if args.laser_power is not None:
        try:
            depth_sensor.set_option(rs.option.laser_power, float(args.laser_power))
        except Exception as e:
            print(f"[WARN] laser_power not set: {e}")

    align = rs.align(rs.stream.color)
    filters = make_filters(args.filters)

    # Get runtime color intrinsics from aligned stream.
    first_frames = align.process(pipe.wait_for_frames(5000))
    color0 = first_frames.get_color_frame()
    if not color0:
        raise RuntimeError("Color frame unavailable.")
    cprof = color0.get_profile().as_video_stream_profile()
    intr = cprof.get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    print(f"[Info] intr fx={intr.fx:.3f} fy={intr.fy:.3f} cx={intr.ppx:.3f} cy={intr.ppy:.3f}")
    print("[Info] RS cloud derivation matches rs_builtin_textured_pcd_o3d.py (aligned depth + runtime color intrinsics)")

    vis = o3d.visualization.Visualizer()
    vis.create_window("RS SDK CAD Pose (multi-instance)", 1400, 850, visible=True)
    render_opt = vis.get_render_option()
    render_opt.point_size = float(args.point_size)

    world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(world_axis)
    scene_geom = o3d.geometry.PointCloud()
    scene_geom.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
    scene_geom.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
    vis.add_geometry(scene_geom)

    pose_geoms = []
    poses = []
    frame_idx = 0
    last_log = 0.0
    last_low_log = 0.0
    first_view = True
    print("[Keys] q / ESC to quit")

    try:
        while True:
            fr = pipe.wait_for_frames(5000)
            fr = align.process(fr)

            depth = fr.get_depth_frame()
            color = fr.get_color_frame()
            if not depth or not color:
                continue

            if filters is not None:
                df = depth
                for f in filters:
                    df = f.process(df)
                depth = df.as_depth_frame()

            rgb = np.asanyarray(color.get_data())
            pts_vis, cols_vis = rs_cloud_from_aligned_depth(
                depth,
                depth_scale,
                K,
                rgb,
                int(args.stride),
                float(args.z_min),
                float(args.z_max),
                flip_y=bool(args.flip_y),
            )
            if pts_vis.shape[0] < int(args.render_min_points):
                # Clear stale pose overlays when scene cloud is too sparse/noisy.
                pose_geoms = update_pose_geometries(vis, pose_geoms, model_lines, [], args.axis_size)
                now = time.time()
                if now - last_low_log > 1.0:
                    print(
                        f"[Scene] low points: rs_pts={pts_vis.shape[0]} < render_min_points={args.render_min_points} "
                        f"(z_range={args.z_min:.2f}..{args.z_max:.2f} m)"
                    )
                    last_low_log = now
                vis.poll_events()
                vis.update_renderer()
                continue

            # Render full-detail scene cloud.
            scene_geom.points = o3d.utility.Vector3dVector(pts_vis.astype(np.float64))
            scene_geom.colors = o3d.utility.Vector3dVector(cols_vis.astype(np.float64))
            vis.update_geometry(scene_geom)
            if first_view and pts_vis.shape[0] > 0:
                vis.reset_view_point(True)
                first_view = False

            # Build a matching cloud (optionally denoised/downsampled) without reducing rendered detail.
            pts_match, cols_match = pts_vis, cols_vis
            if args.scene_voxel_vis > 0 or (args.sor_nb > 0 and args.sor_std > 0):
                pts_match, cols_match = apply_scene_cleanup(
                    pts_match, cols_match,
                    voxel=float(args.scene_voxel_vis),
                    sor_nb=int(args.sor_nb),
                    sor_std=float(args.sor_std),
                )

            if pts_match.shape[0] < args.min_scene_points:
                # Clear stale pose overlays when not enough scene support.
                pose_geoms = update_pose_geometries(vis, pose_geoms, model_lines, [], args.axis_size)
                now = time.time()
                if now - last_low_log > 1.0:
                    print(
                        f"[Scene] low points for pose: match_pts={pts_match.shape[0]} < min_scene_points={args.min_scene_points} "
                        f"(z_range={args.z_min:.2f}..{args.z_max:.2f} m)"
                    )
                    last_low_log = now
                vis.poll_events()
                vis.update_renderer()
                continue

            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(pts_match.astype(np.float64))
            scene_pcd.colors = o3d.utility.Vector3dVector(cols_match.astype(np.float64))

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

            pose_geoms = update_pose_geometries(
                vis, pose_geoms, model_lines, poses, args.axis_size
            )
            vis.poll_events()
            vis.update_renderer()

            now = time.time()
            if now - last_log > 1.0:
                print(f"[Pose] mode={mode:7s} instances={len(poses)} rs_pts={pts_vis.shape[0]} match_pts={pts_match.shape[0]}")
                if mode in ("detect", "recover") and len(poses) > 0:
                    print_poses(poses)
                last_log = now

            frame_idx += 1

    finally:
        pipe.stop()
        vis.destroy_window()


if __name__ == "__main__":
    main()
