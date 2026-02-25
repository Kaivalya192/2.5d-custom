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


def depth_map_from_raw_stereo(
    ir_left_raw, ir_right_raw,
    maps, matcher, right_matcher, wls_filter,
    use_wls, use_lr_check, lr_thresh,
    clahe, temporal_state, temporal_alpha,
    R_L_to_RGB, t_L_to_RGB, Krgb_runtime,
    out_h, out_w,
    debug=False
):
    """
    Returns:
      depth_raw_m (HxW float32 with NaN for invalid) in RGB camera Z (meters)
      valid_mask (HxW bool)
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

    # raw disparities for LR-check
    dispL_raw = dispL_16.astype(np.float32) / 16.0
    dispR_raw = None
    dispR_16 = None
    if right_matcher is not None:
        dispR_16 = right_matcher.compute(Rm, Lm)
        dispR_raw = dispR_16.astype(np.float32) / 16.0

        # ---- FIX
        med_r = np.median(dispR_raw[np.isfinite(dispR_raw)])
        if med_r < 0:
            dispR_raw = -dispR_raw
            dispR_16 = -dispR_16

    # WLS disparity (used for reproject)
    if use_wls and (dispR_16 is not None) and (wls_filter is not None):
        dispF_16 = wls_filter.filter(dispL_16, Lm, None, dispR_16)
        disp = dispF_16.astype(np.float32) / 16.0
    else:
        disp = dispL_raw

    # temporal EMA on the disparity used for reproject
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

    # validity mask
    valid = disp_use > 1.0
    n_disp = int(np.count_nonzero(valid))

    if use_lr_check and (dispR_raw is not None):
        valid &= lr_consistency_mask(dispL_raw, dispR_raw, thresh=float(lr_thresh))
    n_lr = int(np.count_nonzero(valid))

    if n_lr == 0:
        if debug:
            print(f"[RAW DBG] disp_valid={n_disp} lr_valid=0")
        depth = np.full((out_h, out_w), np.nan, dtype=np.float32)
        return depth, np.zeros_like(depth, dtype=bool)

    # 3D in rectified-left coordinates
    xyz = cv2.reprojectImageTo3D(disp_use, Q, handleMissingValues=True)

    vv, uu = np.where(valid)
    pts_rectL = xyz[vv, uu, :].astype(np.float32)

    # rectified-left -> original-left
    pts_left = (RrectL.T @ pts_rectL.T).T  # Nx3

    # left -> RGB frame
    pts_rgb = (R_L_to_RGB @ pts_left.T + t_L_to_RGB).T.astype(np.float32)

    Zc = pts_rgb[:, 2]
    good_z = Zc > 0.0
    n_z = int(np.count_nonzero(good_z))
    if n_z == 0:
        if debug:
            print(f"[RAW DBG] disp_valid={n_disp} lr_valid={n_lr} z>0=0")
        depth = np.full((out_h, out_w), np.nan, dtype=np.float32)
        return depth, np.zeros_like(depth, dtype=bool)

    pts_rgb = pts_rgb[good_z]
    Zc = Zc[good_z]

    # project to RGB pixels (runtime intrinsics!)
    u = (Krgb_runtime[0, 0] * (pts_rgb[:, 0] / Zc) + Krgb_runtime[0, 2])
    v = (Krgb_runtime[1, 1] * (pts_rgb[:, 1] / Zc) + Krgb_runtime[1, 2])
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    inb = (ui >= 0) & (ui < out_w) & (vi >= 0) & (vi < out_h)
    n_inb = int(np.count_nonzero(inb))
    if n_inb == 0:
        if debug:
            print(f"[RAW DBG] disp_valid={n_disp} lr_valid={n_lr} z>0={n_z} inbounds=0")
        depth = np.full((out_h, out_w), np.nan, dtype=np.float32)
        return depth, np.zeros_like(depth, dtype=bool)

    ui = ui[inb]
    vi = vi[inb]
    z = Zc[inb]

    # z-buffer into RGB-sized depth map
    depth_flat = np.full(out_h * out_w, np.inf, dtype=np.float32)
    idx = vi * out_w + ui
    np.minimum.at(depth_flat, idx, z)
    depth = depth_flat.reshape(out_h, out_w)

    valid_map = np.isfinite(depth) & (depth != np.inf)
    n_pix = int(np.count_nonzero(valid_map))

    depth[~valid_map] = np.nan

    if debug:
        print(f"[RAW DBG] disp_valid={n_disp} lr_valid={n_lr} z>0={n_z} inb={n_inb} pix={n_pix}")

    return depth, valid_map


# -------------------- RS built-in depth (reference) --------------------
def depth_map_from_rs_aligned(depth_frame, depth_scale):
    depth_u16 = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_m = depth_u16 * float(depth_scale)
    valid = depth_m > 0
    depth_m[~valid] = np.nan
    return depth_m, valid


# -------------------- Metrics --------------------
def accumulate_stats(sum_img, sumsq_img, count_img, depth_img, valid_mask):
    d = depth_img.copy()
    d[~valid_mask] = 0.0
    sum_img += d
    sumsq_img += d * d
    count_img += valid_mask.astype(np.uint32)


def finalize_noise(sum_img, sumsq_img, count_img, min_count=5):
    valid = count_img >= min_count
    mean = np.zeros_like(sum_img, dtype=np.float32)
    mean[valid] = (sum_img[valid] / count_img[valid]).astype(np.float32)
    var = np.zeros_like(sum_img, dtype=np.float32)
    var[valid] = (sumsq_img[valid] / count_img[valid]).astype(np.float32) - mean[valid] ** 2
    var[var < 0] = 0
    std = np.sqrt(var)
    std[~valid] = np.nan
    return std


def summarize_error(err):
    e = err[np.isfinite(err)]
    if e.size == 0:
        return None
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    med = float(np.median(np.abs(e)))
    p95 = float(np.percentile(np.abs(e), 95))
    bias = float(np.mean(e))
    return {"mae": mae, "rmse": rmse, "median_abs": med, "p95_abs": p95, "bias": bias}


def range_mask(depth, zmin, zmax):
    return np.isfinite(depth) & (depth >= zmin) & (depth <= zmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calib_d435i/calibration.json")

    ap.add_argument("--ir_w", type=int, default=848)
    ap.add_argument("--ir_h", type=int, default=480)
    ap.add_argument("--ir_fps", type=int, default=30)

    ap.add_argument("--w", type=int, default=640)   # color width
    ap.add_argument("--h", type=int, default=480)   # color height
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

    ap.add_argument("--emitter", type=int, default=1)
    ap.add_argument("--laser_power", type=float, default=None)

    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--point_size", type=float, default=5.0)
    ap.add_argument("--shift_raw_x", type=float, default=0.25)

    ap.add_argument("--debug_raw", action="store_true", help="print RAW pipeline stage counts")
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

    # ---- RealSense pipeline: IR1, IR2, DEPTH, COLOR together
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)
    cfg.enable_stream(rs.stream.infrared, 2, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)

    # IMPORTANT: depth tied to IR mode -> match it
    cfg.enable_stream(rs.stream.depth, args.ir_w, args.ir_h, rs.format.z16, args.ir_fps)

    # color
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

    # runtime color intrinsics (use THIS everywhere)
    color_sp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_sp.get_intrinsics()
    Krgb = np.array([[intr.fx, 0, intr.ppx],
                     [0, intr.fy, intr.ppy],
                     [0, 0, 1]], dtype=np.float64)
    print(f"[Info] color intr fx={intr.fx:.3f} fy={intr.fy:.3f} cx={intr.ppx:.3f} cy={intr.ppy:.3f}")

    # ---- raw stereo setup
    matcher = make_sgbm(args.num_disp, args.block)

    # right matcher is needed for LR-check even if WLS is off
    right_matcher = None
    wls_filter = None
    use_wls = False

    if args.lr_check or args.wls:
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(matcher)
        except Exception as e:
            right_matcher = None
            print(f"[WARN] right matcher not available (opencv-contrib missing?). LR-check will be skipped. {e}")

    if args.wls:
        try:
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher)
            wls_filter.setLambda(float(args.wls_lambda))
            wls_filter.setSigmaColor(float(args.wls_sigma))
            use_wls = True
            print("[Info] WLS enabled")
        except Exception as e:
            wls_filter = None
            use_wls = False
            print(f"[WARN] WLS not available (need opencv-contrib-python). Continuing without WLS. {e}")

    maps_A = build_rectification(K1, D1, K2, D2, R_1to2, t_1to2, (args.ir_w, args.ir_h))  # LEFT=IR1
    maps_B = build_rectification(K2, D2, K1, D1, R_2to1, t_2to1, (args.ir_w, args.ir_h))  # LEFT=IR2

    clahe = None
    if args.clahe:
        clahe = cv2.createCLAHE(clipLimit=float(args.clahe_clip),
                                tileGridSize=(int(args.clahe_grid), int(args.clahe_grid)))

    # Decide raw ordering once
    frames = pipe.wait_for_frames(5000)
    _ = align.process(frames)
    ir1 = frames.get_infrared_frame(1)
    ir2 = frames.get_infrared_frame(2)
    if not ir1 or not ir2:
        raise RuntimeError("IR frames not available")

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

    # ---- accumulators
    Hc, Wc = args.h, args.w
    sum_rs = np.zeros((Hc, Wc), dtype=np.float64)
    sumsq_rs = np.zeros((Hc, Wc), dtype=np.float64)
    cnt_rs = np.zeros((Hc, Wc), dtype=np.uint32)

    sum_raw = np.zeros((Hc, Wc), dtype=np.float64)
    sumsq_raw = np.zeros((Hc, Wc), dtype=np.float64)
    cnt_raw = np.zeros((Hc, Wc), dtype=np.uint32)

    densities_rs, densities_raw = [], []
    err_list = []
    temporal_state = {}

    # ---- visualization
    vis = None
    pcd_raw = None
    pcd_rs = None
    viz_first = True

    if args.viz:
        vis = o3d.visualization.Visualizer()
        vis.create_window("RAW vs RS point cloud (side-by-side)", 1280, 720, visible=True)
        opt = vis.get_render_option()
        opt.point_size = float(args.point_size)
        opt.background_color = np.asarray([0.0, 0.0, 0.0])

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(axis)

        pcd_rs = o3d.geometry.PointCloud()
        pcd_raw = o3d.geometry.PointCloud()

        pcd_rs.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
        pcd_rs.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1]], dtype=np.float64))
        pcd_raw.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
        pcd_raw.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 1]], dtype=np.float64))

        vis.add_geometry(pcd_rs)
        vis.add_geometry(pcd_raw)
        vis.poll_events()
        vis.update_renderer()

    def depth_to_pcd(depth_m, rgb_img, Krgb_use, shift_x=0.0):
        fx, fy = Krgb_use[0, 0], Krgb_use[1, 1]
        cx, cy = Krgb_use[0, 2], Krgb_use[1, 2]
        valid = range_mask(depth_m, zmin, zmax)
        if np.count_nonzero(valid) < 500:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

        v, u = np.where(valid)
        Z = depth_m[v, u].astype(np.float32)
        X = (u.astype(np.float32) - cx) * Z / fx
        Y = (v.astype(np.float32) - cy) * Z / fy
        X = X + float(shift_x)

        pts = np.stack([X, Y, Z], axis=1)
        cols = rgb_img[v, u, :].astype(np.float32) / 255.0
        return pts, cols

    # ---- warmup
    print(f"[Warmup] {args.warmup} frames ...")
    for _ in range(int(args.warmup)):
        fr = pipe.wait_for_frames(5000)
        _ = align.process(fr)

    print(f"[Eval] collecting {args.frames} frames in range {zmin:.2f}..{zmax:.2f} m")
    t0 = time.time()

    try:
        for _ in range(int(args.frames)):
            frames = pipe.wait_for_frames(5000)
            frames_aligned = align.process(frames)

            depth = frames_aligned.get_depth_frame()
            color = frames_aligned.get_color_frame()
            if not depth or not color:
                continue

            rgb = np.asanyarray(color.get_data())  # RGB8

            # RS depth (aligned to color)
            depth_rs, valid_rs0 = depth_map_from_rs_aligned(depth, depth_scale)
            valid_rs = valid_rs0 & range_mask(depth_rs, zmin, zmax)

            # RAW depth (projected into color pixels)
            ir_left = get_left(frames)
            ir_right = get_right(frames)

            depth_raw, valid_raw0 = depth_map_from_raw_stereo(
                ir_left, ir_right,
                maps, matcher, right_matcher, wls_filter,
                use_wls, bool(args.lr_check), float(args.lr_thresh),
                clahe, temporal_state, float(args.temporal_alpha),
                R_L_to_RGB, t_L_to_RGB, Krgb,  # runtime K
                Hc, Wc,
                debug=args.debug_raw
            )
            valid_raw = valid_raw0 & range_mask(depth_raw, zmin, zmax)

            densities_rs.append(float(np.mean(valid_rs)))
            densities_raw.append(float(np.mean(valid_raw)))

            accumulate_stats(sum_rs, sumsq_rs, cnt_rs, depth_rs, valid_rs)
            accumulate_stats(sum_raw, sumsq_raw, cnt_raw, depth_raw, valid_raw)

            overlap = valid_rs & valid_raw
            if np.any(overlap):
                err_list.append((depth_raw[overlap] - depth_rs[overlap]).astype(np.float32))

            if vis is not None:
                pts_rs, col_rs = depth_to_pcd(depth_rs, rgb, Krgb, shift_x=0.0)
                pts_raw, col_raw = depth_to_pcd(depth_raw, rgb, Krgb, shift_x=float(args.shift_raw_x))

                if pts_rs.shape[0] > 500:
                    pcd_rs.points = o3d.utility.Vector3dVector(pts_rs.astype(np.float64))
                    pcd_rs.colors = o3d.utility.Vector3dVector(col_rs.astype(np.float64))
                    vis.update_geometry(pcd_rs)

                if pts_raw.shape[0] > 500:
                    pcd_raw.points = o3d.utility.Vector3dVector(pts_raw.astype(np.float64))
                    pcd_raw.colors = o3d.utility.Vector3dVector(col_raw.astype(np.float64))
                    vis.update_geometry(pcd_raw)

                if viz_first and (pts_rs.shape[0] > 500 or pts_raw.shape[0] > 500):
                    vis.reset_view_point(True)
                    viz_first = False

                vis.poll_events()
                vis.update_renderer()

    finally:
        pipe.stop()
        if vis is not None:
            vis.destroy_window()

    dt = time.time() - t0

    density_rs = float(np.mean(densities_rs)) * 100.0
    density_raw = float(np.mean(densities_raw)) * 100.0

    std_rs = finalize_noise(sum_rs, sumsq_rs, cnt_rs, min_count=max(5, int(args.frames * 0.5)))
    std_raw = finalize_noise(sum_raw, sumsq_raw, cnt_raw, min_count=max(5, int(args.frames * 0.2)))

    noise_rs = float(np.nanmedian(std_rs)) if np.any(np.isfinite(std_rs)) else np.nan
    noise_raw = float(np.nanmedian(std_raw)) if np.any(np.isfinite(std_raw)) else np.nan

    acc = None
    if len(err_list) > 0:
        err_all = np.concatenate(err_list, axis=0)
        acc = summarize_error(err_all)

    print("\n================== COMPARISON REPORT ==================")
    print(f"Range: {zmin:.2f} m .. {zmax:.2f} m | frames used: {len(densities_rs)} | elapsed: {dt:.2f}s")

    print("\n[DENSITY] (valid pixels within range)")
    print(f"  RS  : {density_rs:.2f}%")
    print(f"  RAW : {density_raw:.2f}%")

    print("\n[NOISE] (median temporal std of depth per pixel, meters; lower is better)")
    print(f"  RS  : {noise_rs:.6f} m")
    print(f"  RAW : {noise_raw:.6f} m")

    print("\n[ACCURACY vs RS reference] (RAW - RS on overlapping valid pixels)")
    if acc is None:
        print("  Not enough overlapping valid pixels to compute error stats.")
    else:
        print(f"  MAE      : {acc['mae']:.6f} m")
        print(f"  RMSE     : {acc['rmse']:.6f} m")
        print(f"  MedianAbs: {acc['median_abs']:.6f} m")
        print(f"  P95Abs   : {acc['p95_abs']:.6f} m")
        print(f"  Bias     : {acc['bias']:.6f} m  (positive => RAW farther than RS)")

    print("=======================================================\n")


if __name__ == "__main__":
    main()