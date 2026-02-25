import json
import time
import argparse
import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs


# -------------------- Calibration helpers --------------------
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


# -------------------- Stereo pipeline --------------------
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


def raw_cloud_in_rgb(
    ir_left_raw, ir_right_raw,
    maps, matcher, right_matcher, wls_filter,
    use_wls, use_lr_check, lr_thresh,
    clahe, temporal_state, temporal_alpha,
    R_L_to_RGB, t_L_to_RGB,
    Krgb, rgb_w, rgb_h,
    zmin, zmax,
    diag=None
):
    """
    Returns Nx3 float32 points in RGB frame (meters)
    """
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

    dispR_raw = None
    dispR_16 = None
    if right_matcher is not None:
        dispR_16 = right_matcher.compute(Rm, Lm)
        dispR_raw = dispR_16.astype(np.float32) / 16.0
        # SIGN FIX: right matcher may return negative disparity
        med_r = np.median(dispR_raw[np.isfinite(dispR_raw)])
        if med_r < 0:
            dispR_raw = -dispR_raw
            dispR_16 = -dispR_16

    if use_wls and (dispR_16 is not None) and (wls_filter is not None):
        dispF_16 = wls_filter.filter(dispL_16, Lm, None, dispR_16)
        disp = dispF_16.astype(np.float32) / 16.0
    else:
        disp = dispL_raw

    # temporal EMA
    if temporal_alpha > 0:
        prev = temporal_state.get("prev_disp", None)
        if prev is None:
            temporal_state["prev_disp"] = disp.copy()
        else:
            a = float(temporal_alpha)
            temporal_state["prev_disp"] = (1 - a) * disp + a * prev
        disp_use = temporal_state["prev_disp"]
    else:
        disp_use = disp

    valid = disp_use > 1.0
    n_disp_valid = int(np.count_nonzero(valid))
    n_total = int(valid.size)
    if use_lr_check and (dispR_raw is not None):
        valid &= lr_consistency_mask(dispL_raw, dispR_raw, thresh=float(lr_thresh))
    n_lr_valid = int(np.count_nonzero(valid))

    if not np.any(valid):
        if diag is not None:
            diag["frames"] += 1
            diag["raw_total_px"] += n_total
            diag["raw_disp_valid_px"] += n_disp_valid
            diag["raw_lr_valid_px"] += n_lr_valid
        return np.zeros((0, 3), np.float32)

    xyz = cv2.reprojectImageTo3D(disp_use, Q, handleMissingValues=True)  # rectified left
    vv, uu = np.where(valid)
    pts_rectL = xyz[vv, uu, :].astype(np.float32)

    # rectified-left -> original-left
    pts_left = (RrectL.T @ pts_rectL.T).T.astype(np.float32)

    # left -> RGB
    pts_out = (R_L_to_RGB @ pts_left.T + t_L_to_RGB).T.astype(np.float32)

    Z = pts_out[:, 2]
    good = np.isfinite(Z) & (Z >= zmin) & (Z <= zmax)
    if np.any(good):
        pts_g = pts_out[good]
        zg = pts_g[:, 2]
        u = (Krgb[0, 0] * (pts_g[:, 0] / zg) + Krgb[0, 2])
        v = (Krgb[1, 1] * (pts_g[:, 1] / zg) + Krgb[1, 2])
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        inb = (ui >= 0) & (ui < int(rgb_w)) & (vi >= 0) & (vi < int(rgb_h))
    else:
        inb = np.zeros((0,), dtype=bool)
    if diag is not None:
        diag["frames"] += 1
        diag["raw_total_px"] += n_total
        diag["raw_disp_valid_px"] += n_disp_valid
        diag["raw_lr_valid_px"] += n_lr_valid
        diag["raw_zrange_pts"] += int(np.count_nonzero(good))
        diag["raw_overlap_pts"] += int(np.count_nonzero(inb))
    return pts_out[good][inb]


def rs_cloud_in_rgb(depth_frame, depth_scale, K, zmin, zmax, diag=None):
    """
    Convert depth map to point cloud in the camera frame defined by K.
    """
    depth_u16 = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    Z = depth_u16 * float(depth_scale)
    valid = (Z > 0) & (Z >= zmin) & (Z <= zmax)
    if diag is not None:
        diag["frames"] += 1
        diag["rs_total_px"] += int(valid.size)
        diag["rs_zrange_px"] += int(np.count_nonzero(valid))

    if np.count_nonzero(valid) < 500:
        return np.zeros((0, 3), np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    v, u = np.where(valid)

    z = Z[v, u]
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts


def nn_distance_stats(src_pts, dst_pts):
    if src_pts.shape[0] < 1000 or dst_pts.shape[0] < 1000:
        return None
    src = o3d.geometry.PointCloud()
    dst = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts.astype(np.float64))
    dst.points = o3d.utility.Vector3dVector(dst_pts.astype(np.float64))
    d = np.asarray(src.compute_point_cloud_distance(dst), dtype=np.float64)
    if d.size == 0:
        return None
    return {
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "mean": float(np.mean(d)),
    }


def voxel_z_noise_accumulate(voxel_sum, voxel_sumsq, voxel_count, pts, voxel):
    if pts.shape[0] < 1000 or voxel <= 0:
        return
    # voxel key = integer coords
    key = np.floor(pts / voxel).astype(np.int32)
    # we use only Z for temporal noise proxy
    z = pts[:, 2].astype(np.float32)
    # aggregate by unique voxel
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    z_sum = np.bincount(inv, weights=z)
    z_sumsq = np.bincount(inv, weights=z * z)
    z_cnt = np.bincount(inv)

    for i in range(uniq.shape[0]):
        u = (int(uniq[i, 0]), int(uniq[i, 1]), int(uniq[i, 2]))
        voxel_sum[u] = voxel_sum.get(u, 0.0) + float(z_sum[i])
        voxel_sumsq[u] = voxel_sumsq.get(u, 0.0) + float(z_sumsq[i])
        voxel_count[u] = voxel_count.get(u, 0) + int(z_cnt[i])


def voxel_noise_finalize(voxel_sum, voxel_sumsq, voxel_count, min_count=20):
    stds = []
    for k, c in voxel_count.items():
        if c < min_count:
            continue
        mean = voxel_sum[k] / c
        var = voxel_sumsq[k] / c - mean * mean
        if var < 0:
            var = 0
        stds.append(np.sqrt(var))
    if len(stds) == 0:
        return np.nan
    return float(np.median(stds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calib_d435i/calibration.json")

    ap.add_argument("--ir_w", type=int, default=848)
    ap.add_argument("--ir_h", type=int, default=480)
    ap.add_argument("--ir_fps", type=int, default=30)

    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=30)

    ap.add_argument("--z_min", type=float, default=0.1)
    ap.add_argument("--z_max", type=float, default=1.0)

    ap.add_argument("--num_disp", type=int, default=256)
    ap.add_argument("--block", type=int, default=5)

    ap.add_argument("--wls", action="store_true")
    ap.add_argument("--wls_lambda", type=float, default=8000.0)
    ap.add_argument("--wls_sigma", type=float, default=1.5)

    ap.add_argument("--lr_check", action="store_true")
    ap.add_argument("--lr_thresh", type=float, default=1.5)

    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)

    ap.add_argument("--temporal_alpha", type=float, default=0.2)

    ap.add_argument("--voxel", type=float, default=0.01, help="voxel size for noise metric (m)")

    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--point_size", type=float, default=4.0)
    ap.add_argument("--shift_raw_x", type=float, default=0.25)
    ap.add_argument("--diag", action="store_true", help="print RAW/RS stage drop diagnostics")
    ap.add_argument("--emitter", type=int, default=1, help="0=off, 1=on")
    ap.add_argument("--laser_power", type=float, default=None, help="optional projector power")

    args = ap.parse_args()

    zmin, zmax = float(args.z_min), float(args.z_max)

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

    # RealSense streams (depth must match IR mode)
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)
    cfg.enable_stream(rs.stream.infrared, 2, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)
    cfg.enable_stream(rs.stream.depth, args.ir_w, args.ir_h, rs.format.z16, args.ir_fps)
    cfg.enable_stream(rs.stream.color, args.w, args.h, rs.format.rgb8, args.fps)
    profile = pipe.start(cfg)

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

    color_sp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_sp.get_intrinsics()
    Krgb = np.array([[intr.fx, 0, intr.ppx],
                     [0, intr.fy, intr.ppy],
                     [0, 0, 1]], dtype=np.float64)

    # Raw stereo setup
    matcher = make_sgbm(args.num_disp, args.block)

    right_matcher = None
    wls_filter = None
    use_wls = False
    if args.wls:
        right_matcher = cv2.ximgproc.createRightMatcher(matcher)

    if args.wls:
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls_filter.setLambda(float(args.wls_lambda))
        wls_filter.setSigmaColor(float(args.wls_sigma))
        use_wls = True
    if args.lr_check and not use_wls:
        print("[WARN] legacy mode ignores --lr_check unless --wls is enabled.")

    maps_A = build_rectification(K1, D1, K2, D2, R_1to2, t_1to2, (args.ir_w, args.ir_h))  # LEFT=IR1
    maps_B = build_rectification(K2, D2, K1, D1, R_2to1, t_2to1, (args.ir_w, args.ir_h))  # LEFT=IR2

    clahe = None
    if args.clahe:
        clahe = cv2.createCLAHE(clipLimit=float(args.clahe_clip),
                                tileGridSize=(int(args.clahe_grid), int(args.clahe_grid)))

    # Decide ordering once
    frames = pipe.wait_for_frames(5000)
    _ = align.process(frames)
    ir1 = frames.get_infrared_frame(1)
    ir2 = frames.get_infrared_frame(2)
    ir1_raw = np.asanyarray(ir1.get_data())
    ir2_raw = np.asanyarray(ir2.get_data())

    vrA, medA = score_order(ir1_raw, ir2_raw, maps_A, matcher)
    vrB, medB = score_order(ir2_raw, ir1_raw, maps_B, matcher)
    print(f"[Order A] LEFT=IR1 RIGHT=IR2  valid={vrA*100:.1f}%  medDisp={medA:.2f}")
    print(f"[Order B] LEFT=IR2 RIGHT=IR1  valid={vrB*100:.1f}%  medDisp={medB:.2f}")

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

    # Visualization
    vis = None
    pcd_rs = pcd_raw = None
    viz_first = True
    if args.viz:
        vis = o3d.visualization.Visualizer()
        vis.create_window("RAW cloud vs RS cloud", 1280, 720, visible=True)
        opt = vis.get_render_option()
        opt.point_size = float(args.point_size)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(axis)

        pcd_rs = o3d.geometry.PointCloud()
        pcd_raw = o3d.geometry.PointCloud()
        pcd_rs.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
        pcd_raw.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
        vis.add_geometry(pcd_rs)
        vis.add_geometry(pcd_raw)

    # Metrics accumulators
    dens_rs = []
    dens_raw = []
    nn_stats_list = []

    # voxel noise accumulators
    rs_sum, rs_sumsq, rs_cnt = {}, {}, {}
    raw_sum, raw_sumsq, raw_cnt = {}, {}, {}

    temporal_state = {}
    raw_diag = {
        "frames": 0,
        "raw_total_px": 0,
        "raw_disp_valid_px": 0,
        "raw_lr_valid_px": 0,
        "raw_zrange_pts": 0,
        "raw_overlap_pts": 0,
    } if args.diag else None
    rs_diag = {
        "frames": 0,
        "rs_total_px": 0,
        "rs_zrange_px": 0,
    } if args.diag else None

    print(f"[Warmup] {args.warmup} frames ...")
    for _ in range(int(args.warmup)):
        fr = pipe.wait_for_frames(5000)
        _ = align.process(fr)

    print(f"[Eval] {args.frames} frames in {zmin:.2f}..{zmax:.2f} m")
    t0 = time.time()

    try:
        for _ in range(int(args.frames)):
            fr = pipe.wait_for_frames(5000)
            fr_a = align.process(fr)

            depth = fr_a.get_depth_frame()
            color = fr_a.get_color_frame()
            if not depth or not color:
                continue

            # RS cloud in RGB frame
            pts_rs = rs_cloud_in_rgb(depth, depth_scale, Krgb, zmin, zmax, diag=rs_diag)

            # RAW cloud in RGB frame
            ir_left = get_left(fr)
            ir_right = get_right(fr)
            pts_raw = raw_cloud_in_rgb(
                ir_left, ir_right,
                maps, matcher, right_matcher, wls_filter,
                use_wls, bool(args.lr_check), float(args.lr_thresh),
                clahe, temporal_state, float(args.temporal_alpha),
                R_L_to_RGB, t_L_to_RGB,
                Krgb, args.w, args.h,
                zmin, zmax,
                diag=raw_diag
            )

            dens_rs.append(float(pts_rs.shape[0]))
            dens_raw.append(float(pts_raw.shape[0]))

            voxel_z_noise_accumulate(rs_sum, rs_sumsq, rs_cnt, pts_rs, float(args.voxel))
            voxel_z_noise_accumulate(raw_sum, raw_sumsq, raw_cnt, pts_raw, float(args.voxel))

            st = nn_distance_stats(pts_raw, pts_rs)
            if st is not None:
                nn_stats_list.append(st)

            if vis is not None:
                # shift RAW for side-by-side
                pts_raw_v = pts_raw.copy()
                pts_raw_v[:, 0] += float(args.shift_raw_x)

                pcd_rs.points = o3d.utility.Vector3dVector(pts_rs.astype(np.float64))
                pcd_raw.points = o3d.utility.Vector3dVector(pts_raw_v.astype(np.float64))

                vis.update_geometry(pcd_rs)
                vis.update_geometry(pcd_raw)
                if viz_first and (pts_rs.shape[0] > 1000 or pts_raw.shape[0] > 1000):
                    vis.reset_view_point(True)
                    viz_first = False
                vis.poll_events()
                vis.update_renderer()

    finally:
        pipe.stop()
        if vis is not None:
            vis.destroy_window()

    dt = time.time() - t0

    # Density
    mean_rs = float(np.mean(dens_rs)) if len(dens_rs) else 0.0
    mean_raw = float(np.mean(dens_raw)) if len(dens_raw) else 0.0

    # Noise
    noise_rs = voxel_noise_finalize(rs_sum, rs_sumsq, rs_cnt, min_count=max(20, int(args.frames * 0.2)))
    noise_raw = voxel_noise_finalize(raw_sum, raw_sumsq, raw_cnt, min_count=max(20, int(args.frames * 0.2)))

    # Accuracy (NN distances)
    if len(nn_stats_list):
        med = float(np.median([s["median"] for s in nn_stats_list]))
        p95 = float(np.median([s["p95"] for s in nn_stats_list]))
        rmse = float(np.median([s["rmse"] for s in nn_stats_list]))
        mean = float(np.median([s["mean"] for s in nn_stats_list]))
    else:
        med = p95 = rmse = mean = np.nan

    print("\n================== CLOUD-TO-CLOUD REPORT ==================")
    print(f"Range: {zmin:.2f}..{zmax:.2f} m | frames: {len(dens_rs)} | elapsed: {dt:.2f}s")
    print("\n[DENSITY] points per frame (higher is better)")
    print(f"  RS  : {mean_rs:.0f} pts/frame")
    print(f"  RAW : {mean_raw:.0f} pts/frame")

    print("\n[NOISE] median voxel-Z std (m) (lower is better)")
    print(f"  RS  : {noise_rs:.6f} m")
    print(f"  RAW : {noise_raw:.6f} m")

    print("\n[ACCURACY vs RS] NN distance RAW->RS (m) (lower is better)")
    print(f"  median: {med:.6f} m")
    print(f"  p95   : {p95:.6f} m")
    print(f"  rmse  : {rmse:.6f} m")
    print(f"  mean  : {mean:.6f} m")
    if args.diag and raw_diag is not None and rs_diag is not None:
        print("\n[DIAG] stage retention")
        if raw_diag["raw_total_px"] > 0:
            disp_pct = 100.0 * raw_diag["raw_disp_valid_px"] / raw_diag["raw_total_px"]
            lr_pct = 100.0 * raw_diag["raw_lr_valid_px"] / raw_diag["raw_total_px"]
            print(f"  RAW disp>1 px   : {disp_pct:.1f}%")
            print(f"  RAW LR-cons px  : {lr_pct:.1f}%")
        if raw_diag["frames"] > 0:
            print(f"  RAW z-range pts : {raw_diag['raw_zrange_pts'] / raw_diag['frames']:.0f} pts/frame")
            print(f"  RAW overlap pts : {raw_diag['raw_overlap_pts'] / raw_diag['frames']:.0f} pts/frame")
        if rs_diag["rs_total_px"] > 0:
            rs_pct = 100.0 * rs_diag["rs_zrange_px"] / rs_diag["rs_total_px"]
            print(f"  RS z-range px   : {rs_pct:.1f}%")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
