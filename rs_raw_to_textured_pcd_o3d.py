import json
import time
import argparse
import numpy as np
import cv2
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

    fx = P1[0, 0]
    baseline = -P2[0, 3] / fx  # sign can vary; we rely on Q for reproject
    return (mapLx, mapLy, mapRx, mapRy, R1, Q, fx, baseline)


def score_order(irL_raw, irR_raw, maps, matcher):
    mapLx, mapLy, mapRx, mapRy, _, _, _, _ = maps
    L = cv2.remap(irL_raw, mapLx, mapLy, cv2.INTER_LINEAR)
    R = cv2.remap(irR_raw, mapRx, mapRy, cv2.INTER_LINEAR)
    disp16 = matcher.compute(L, R)
    disp = disp16.astype(np.float32) / 16.0
    valid = disp > 1.0
    return float(np.mean(valid)), (float(np.median(disp[valid])) if np.any(valid) else -1.0)


def lr_consistency_mask(dispL, dispR, thresh=1.5):
    """
    dispL: HxW float disparity (left->right)
    dispR: HxW float disparity (right->left)
    Checks: dispL(x,y) ~= dispR(x - dispL, y)
    """
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


def apply_o3d_cleanup(pts, cols, voxel=0.0, sor_nb=0, sor_std=0.0):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    p.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

    if voxel and voxel > 0:
        p = p.voxel_down_sample(voxel_size=float(voxel))

    if sor_nb and sor_nb > 0 and sor_std and sor_std > 0:
        # remove_statistical_outlier already returns the filtered cloud.
        p, _ = p.remove_statistical_outlier(nb_neighbors=int(sor_nb), std_ratio=float(sor_std))

    pts2 = np.asarray(p.points, dtype=np.float64)
    cols2 = np.asarray(p.colors, dtype=np.float64)
    return pts2, cols2


def cloud_from_disparity(
    disp_use, valid, Q, RrectL,
    R_L_to_RGB, t_L_to_RGB,
    Krgb, rgb_raw,
    stride, z_min, z_max
):
    H, W = disp_use.shape
    xyz = cv2.reprojectImageTo3D(disp_use, Q, handleMissingValues=True)  # HxWx3

    xyz_s = xyz[0:H:stride, 0:W:stride, :].reshape(-1, 3)
    valid_s = valid[0:H:stride, 0:W:stride].reshape(-1).copy()

    Z = xyz_s[:, 2]
    valid_s &= np.isfinite(Z) & (Z >= z_min) & (Z <= z_max)
    if np.count_nonzero(valid_s) < 1000:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    pts_rectL = xyz_s[valid_s].astype(np.float32)
    pts_left = (RrectL.T @ pts_rectL.T).T
    pts_rgb = (R_L_to_RGB @ pts_left.T + t_L_to_RGB).T.astype(np.float32)

    Xc, Yc, Zc = pts_rgb[:, 0], pts_rgb[:, 1], pts_rgb[:, 2]
    good = Zc > 0.0

    u = (Krgb[0, 0] * (Xc / Zc) + Krgb[0, 2])
    v = (Krgb[1, 1] * (Yc / Zc) + Krgb[1, 2])
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    Hc, Wc = rgb_raw.shape[:2]
    good &= (ui >= 0) & (ui < Wc) & (vi >= 0) & (vi < Hc)
    if np.count_nonzero(good) < 1000:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    pts_final = pts_rgb[good]
    cols = rgb_raw[vi[good], ui[good], :].astype(np.float32) / 255.0
    return pts_final, cols


def rs_cloud_from_aligned_depth(depth_frame, depth_scale, Krgb, rgb_raw, stride, z_min, z_max):
    depth_u16 = np.asanyarray(depth_frame.get_data())
    z = depth_u16[0:depth_u16.shape[0]:stride, 0:depth_u16.shape[1]:stride].reshape(-1).astype(np.float32)
    z_m = z * float(depth_scale)

    valid = (z_m > 0) & (z_m >= z_min) & (z_m <= z_max)
    if np.count_nonzero(valid) < 1000:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    H, W = depth_u16.shape
    us = np.arange(0, W, stride, dtype=np.float32)
    vs = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    u = uu.reshape(-1)[valid]
    v = vv.reshape(-1)[valid]
    Z = z_m[valid]

    fx, fy = Krgb[0, 0], Krgb[1, 1]
    cx, cy = Krgb[0, 2], Krgb[1, 2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)

    ui = u.astype(np.int32)
    vi = v.astype(np.int32)
    cols = rgb_raw[vi, ui, :].astype(np.float32) / 255.0
    return pts, cols


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--calib", default="calib_d435i/calibration.json")

    ap.add_argument("--ir_w", type=int, default=848)
    ap.add_argument("--ir_h", type=int, default=480)
    ap.add_argument("--ir_fps", type=int, default=30)

    ap.add_argument("--rgb_w", type=int, default=640)
    ap.add_argument("--rgb_h", type=int, default=480)
    ap.add_argument("--rgb_fps", type=int, default=30)

    ap.add_argument("--num_disp", type=int, default=256)
    ap.add_argument("--block", type=int, default=5)

    ap.add_argument("--wls", action="store_true")
    ap.add_argument("--wls_lambda", type=float, default=8000.0)
    ap.add_argument("--wls_sigma", type=float, default=1.5)

    ap.add_argument("--clahe", action="store_true", help="contrast-limited histogram equalization on IR")
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)

    ap.add_argument("--lr_check", action="store_true", help="left-right consistency mask (legacy path uses this with --wls)")
    ap.add_argument("--lr_thresh", type=float, default=1.5)

    ap.add_argument("--median", type=int, default=3, help="median filter size on disparity (0 disables)")
    ap.add_argument("--speckle", action="store_true", help="opencv speckle filter on raw disp16")

    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--z_min", type=float, default=0.1)
    ap.add_argument("--z_max", type=float, default=1.0)

    ap.add_argument("--temporal_alpha", type=float, default=0.2, help="EMA on disparity, 0 disables")

    ap.add_argument("--emitter", type=int, default=1)
    ap.add_argument("--laser_power", type=float, default=None)

    ap.add_argument("--compare_rs", action="store_true",
                    help="show manual legacy RAW cloud vs RealSense SDK depth cloud side-by-side")
    ap.add_argument("--shift_rs_x", type=float, default=0.25,
                    help="x shift (m) for RS cloud when --compare_rs is enabled")

    # Open3D cleanup for nicer visuals
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel downsample size in meters (0 disables)")
    ap.add_argument("--sor_nb", type=int, default=0, help="statistical outlier removal nb_neighbors (0 disables)")
    ap.add_argument("--sor_std", type=float, default=0.0, help="statistical outlier removal std_ratio")

    ap.add_argument("--debug", action="store_true", help="show rectified left + disparity windows")
    args = ap.parse_args()

    stride = max(1, int(args.stride))

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

    W, H = args.ir_w, args.ir_h

    matcher = make_sgbm(args.num_disp, args.block)

    # Legacy behavior: right matcher is only used when WLS is enabled.
    use_wls = False
    wls_filter = None
    right_matcher = None
    if args.wls:
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(matcher)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher)
            wls_filter.setLambda(args.wls_lambda)
            wls_filter.setSigmaColor(args.wls_sigma)
            use_wls = True
            print("[OK] WLS enabled")
        except Exception as e:
            print(f"[WARN] WLS requested but not available: {e}")
    if args.lr_check and not use_wls:
        print("[WARN] legacy mode ignores --lr_check unless --wls is enabled.")

    maps_A = build_rectification(K1, D1, K2, D2, R_1to2, t_1to2, (W, H))  # LEFT=IR1
    maps_B = build_rectification(K2, D2, K1, D1, R_2to1, t_2to1, (W, H))  # LEFT=IR2

    clahe = None
    if args.clahe:
        clahe = cv2.createCLAHE(clipLimit=float(args.clahe_clip),
                                tileGridSize=(int(args.clahe_grid), int(args.clahe_grid)))

    # RealSense streams
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)
    cfg.enable_stream(rs.stream.infrared, 2, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)
    cfg.enable_stream(rs.stream.color, args.rgb_w, args.rgb_h, rs.format.rgb8, args.rgb_fps)
    if args.compare_rs:
        # Depth mode is tied to IR stereo mode on D435i; request depth in IR resolution/FPS.
        cfg.enable_stream(rs.stream.depth, args.ir_w, args.ir_h, rs.format.z16, args.ir_fps)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color) if args.compare_rs else None

    # Use runtime color intrinsics for correct RGB/pointcloud alignment.
    color_sp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    cintr = color_sp.get_intrinsics()
    Krgb = np.array([[cintr.fx, 0, cintr.ppx],
                     [0, cintr.fy, cintr.ppy],
                     [0, 0, 1]], dtype=np.float64)
    print(f"[Info] color intr fx={cintr.fx:.3f} fy={cintr.fy:.3f} cx={cintr.ppx:.3f} cy={cintr.ppy:.3f}")

    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_scale = None
    if args.compare_rs:
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

    # Decide ordering once (highest valid ratio)
    frames = pipe.wait_for_frames(5000)
    ir1 = frames.get_infrared_frame(1)
    ir2 = frames.get_infrared_frame(2)
    if not ir1 or not ir2:
        raise RuntimeError("IR frames not available")

    ir1_raw = np.asanyarray(ir1.get_data())
    ir2_raw = np.asanyarray(ir2.get_data())

    vrA, medA = score_order(ir1_raw, ir2_raw, maps_A, matcher)  # IR1 left
    vrB, medB = score_order(ir2_raw, ir1_raw, maps_B, matcher)  # IR2 left
    print(f"[Order A] LEFT=IR1 RIGHT=IR2  valid={vrA*100:.1f}%  medDisp={medA:.2f}")
    print(f"[Order B] LEFT=IR2 RIGHT=IR1  valid={vrB*100:.1f}%  medDisp={medB:.2f}")

    if vrB > vrA:
        maps = maps_B
        get_left = lambda fr: np.asanyarray(fr.get_infrared_frame(2).get_data())
        get_right = lambda fr: np.asanyarray(fr.get_infrared_frame(1).get_data())
        R_L_to_RGB, t_L_to_RGB = R_2toRGB, t_2toRGB
        print("[Chosen] LEFT=IR2 RIGHT=IR1")
    else:
        maps = maps_A
        get_left = lambda fr: np.asanyarray(fr.get_infrared_frame(1).get_data())
        get_right = lambda fr: np.asanyarray(fr.get_infrared_frame(2).get_data())
        R_L_to_RGB, t_L_to_RGB = R_1toRGB, t_1toRGB
        print("[Chosen] LEFT=IR1 RIGHT=IR2")

    mapLx, mapLy, mapRx, mapRy, RrectL, Q, fx_rect, baseline = maps
    print(f"[Info] fx_rect={fx_rect:.3f} baseline(from P2)={baseline:.6f} (sign may vary; Q used for 3D)")
    print("[Keys] q / ESC to quit")

    # Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window("Manual RAW legacy stereo -> textured point cloud", width=1280, height=720)

    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
    pcd_raw.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
    vis.add_geometry(pcd_raw)

    pcd_rs = None
    if args.compare_rs:
        pcd_rs = o3d.geometry.PointCloud()
        pcd_rs.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
        pcd_rs.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
        vis.add_geometry(pcd_rs)
        print(f"[Compare] manual RAW (left) vs RS SDK (right shifted by +{float(args.shift_rs_x):.2f} m)")

    vis.get_render_option().point_size = 3.0

    prev_disp = None
    first_nonempty = True
    last_stat = 0.0

    try:
        while True:
            frames = pipe.wait_for_frames(5000)
            fr_for_rs = align.process(frames) if align is not None else frames

            rgb = fr_for_rs.get_color_frame()
            if not rgb:
                continue
            depth = fr_for_rs.get_depth_frame() if args.compare_rs else None
            if args.compare_rs and not depth:
                continue

            L_raw = get_left(frames)
            R_raw = get_right(frames)
            rgb_raw = np.asanyarray(rgb.get_data())  # RGB8

            # Rectify
            L = cv2.remap(L_raw, mapLx, mapLy, cv2.INTER_LINEAR)
            R = cv2.remap(R_raw, mapRx, mapRy, cv2.INTER_LINEAR)

            # Contrast boost
            if clahe is not None:
                Lm = clahe.apply(L)
                Rm = clahe.apply(R)
            else:
                Lm, Rm = L, R

            # Legacy disparity pipeline
            dispL_16 = matcher.compute(Lm, Rm)
            if args.speckle:
                cv2.filterSpeckles(dispL_16, newVal=0, maxSpeckleSize=200, maxDiff=16)

            if use_wls:
                dispR_16 = right_matcher.compute(Rm, Lm)
                dispF_16 = wls_filter.filter(dispL_16, Lm, None, dispR_16)
                disp = dispF_16.astype(np.float32) / 16.0
                dispR = dispR_16.astype(np.float32) / 16.0
            else:
                disp = dispL_16.astype(np.float32) / 16.0
                dispR = None

            if args.median and args.median >= 3 and (args.median % 2 == 1):
                disp = cv2.medianBlur(disp, int(args.median))

            if args.temporal_alpha > 0:
                if prev_disp is None:
                    prev_disp = disp.copy()
                else:
                    a = float(args.temporal_alpha)
                    prev_disp = (1 - a) * disp + a * prev_disp
                disp_use = prev_disp
            else:
                disp_use = disp

            valid = disp_use > 1.0
            if args.lr_check and (dispR is not None):
                valid &= lr_consistency_mask(disp_use, dispR, thresh=float(args.lr_thresh))

            if not np.any(valid):
                if args.debug:
                    disp_vis = np.uint8(np.clip(disp_use / float(args.num_disp) * 255.0, 0, 255))
                    cv2.imshow("Rectified LEFT", L)
                    cv2.imshow("Disparity", cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET))
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27):
                        break
                vis.poll_events()
                vis.update_renderer()
                continue

            now = time.time()
            if now - last_stat > 1.0:
                vratio = float(np.mean(valid)) * 100.0
                med = float(np.median(disp_use[valid])) if np.any(valid) else -1.0
                print(f"[Disp] valid={vratio:.1f}% med={med:.2f}")
                last_stat = now

            pts_raw, cols_raw = cloud_from_disparity(
                disp_use, valid, Q, RrectL,
                R_L_to_RGB, t_L_to_RGB, Krgb, rgb_raw,
                stride, args.z_min, args.z_max
            )

            if args.voxel > 0 or (args.sor_nb > 0 and args.sor_std > 0):
                pts_raw, cols_raw = apply_o3d_cleanup(
                    pts_raw, cols_raw,
                    voxel=float(args.voxel),
                    sor_nb=int(args.sor_nb),
                    sor_std=float(args.sor_std)
                )

            pcd_raw.points = o3d.utility.Vector3dVector(pts_raw.astype(np.float64))
            pcd_raw.colors = o3d.utility.Vector3dVector(cols_raw.astype(np.float64))
            vis.update_geometry(pcd_raw)

            has_points = pts_raw.shape[0] > 0
            if args.compare_rs and pcd_rs is not None:
                pts_rs, cols_rs = rs_cloud_from_aligned_depth(
                    depth, depth_scale, Krgb, rgb_raw, stride, args.z_min, args.z_max
                )
                pts_rs_view = pts_rs.copy()
                if pts_rs_view.shape[0] > 0:
                    pts_rs_view[:, 0] += float(args.shift_rs_x)
                pcd_rs.points = o3d.utility.Vector3dVector(pts_rs_view.astype(np.float64))
                pcd_rs.colors = o3d.utility.Vector3dVector(cols_rs.astype(np.float64))
                vis.update_geometry(pcd_rs)
                has_points = has_points or (pts_rs_view.shape[0] > 0)

            if first_nonempty and has_points:
                vis.reset_view_point(True)
                first_nonempty = False
            vis.poll_events()
            vis.update_renderer()

            if args.debug:
                disp_vis = np.uint8(np.clip(disp_use / float(args.num_disp) * 255.0, 0, 255))
                cv2.imshow("Rectified LEFT", L)
                cv2.imshow("Disparity", cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET))
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
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
