import argparse
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def make_filters(enable: bool):
    """Optional RealSense post-filters (still 'builtin' pipeline, just improves depth)."""
    if not enable:
        return None

    # Keep pixel grid consistent with aligned color; avoid decimation here.
    spat = rs.spatial_filter()
    temp = rs.temporal_filter()
    hole = rs.hole_filling_filter()

    # Reasonable defaults (tune later if you want)
    # dec.set_option(rs.option.filter_magnitude, 2)  # 2 = decimate (optional)
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20)
    temp.set_option(rs.option.filter_smooth_alpha, 0.4)
    temp.set_option(rs.option.filter_smooth_delta, 20)

    return [spat, temp, hole]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--stride", type=int, default=2, help="subsample pixels for speed (2 = every 2nd pixel)")
    ap.add_argument("--z_min", type=float, default=0.2, help="meters")
    ap.add_argument("--z_max", type=float, default=5.0, help="meters")

    ap.add_argument("--flip_y", action="store_true",
                    help="RealSense uses +Y down; flip to make it look upright in Open3D")

    ap.add_argument("--filters", action="store_true",
                    help="enable RealSense depth post-filters (recommended)")

    ap.add_argument("--emitter", type=int, default=1, help="0=off, 1=on")
    ap.add_argument("--laser_power", type=float, default=None, help="optional: set projector power if supported")

    args = ap.parse_args()

    stride = max(1, int(args.stride))

    # --- RealSense pipeline (builtin stereo matching gives depth)
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.w, args.h, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.w, args.h, rs.format.rgb8, args.fps)

    profile = pipe.start(cfg)

    # Depth scale (z16 units -> meters)
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"[Info] depth_scale={depth_scale} meters/unit")

    # Turn emitter on (helps stereo depth in low texture)
    try:
        depth_sensor.set_option(rs.option.emitter_enabled, float(args.emitter))
    except Exception:
        pass
    if args.laser_power is not None:
        try:
            depth_sensor.set_option(rs.option.laser_power, float(args.laser_power))
        except Exception as e:
            print(f"[WARN] laser_power not set: {e}")

    # Align depth -> color so every depth pixel matches RGB pixel
    align = rs.align(rs.stream.color)

    # Optional post-processing filters
    filters = make_filters(args.filters)

    # --- Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("RealSense Depth (builtin) -> Textured PointCloud (Open3D)", 1280, 720)

    pcd = o3d.geometry.PointCloud()
    # seed a point so Open3D doesn't complain on first frame
    pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]], dtype=np.float64))
    vis.add_geometry(pcd)

    first = True

    # We'll grab intrinsics after first aligned frame (guarantees correct intrinsics for aligned depth)
    K = None
    uu = vv = None

    try:
        while True:
            frames = pipe.wait_for_frames()
            frames = align.process(frames)

            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            # Apply optional filters on depth (still using RS built-in pipeline output)
            if filters is not None:
                df = depth
                for f in filters:
                    df = f.process(df)
                depth = df.as_depth_frame()

            # Get intrinsics for aligned color/depth grid
            if K is None:
                cprof = color.get_profile().as_video_stream_profile()
                intr = cprof.get_intrinsics()
                fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
                K = (fx, fy, cx, cy)
                print(f"[Info] intrinsics fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")

                # Precompute pixel grid at stride
                H, W = intr.height, intr.width
                us = np.arange(0, W, stride, dtype=np.float32)
                vs = np.arange(0, H, stride, dtype=np.float32)
                uu, vv = np.meshgrid(us, vs)
                uu = uu.reshape(-1)
                vv = vv.reshape(-1)

            fx, fy, cx, cy = K

            # Numpy images
            depth_u16 = np.asanyarray(depth.get_data())  # HxW uint16 (aligned to color)
            rgb = np.asanyarray(color.get_data())        # HxW x3 uint8 (RGB)

            # Subsample depth
            z = depth_u16[0:depth_u16.shape[0]:stride, 0:depth_u16.shape[1]:stride].reshape(-1).astype(np.float32)
            z_m = z * depth_scale

            # Mask valid depth range
            valid = (z_m > 0) & (z_m >= args.z_min) & (z_m <= args.z_max)
            if np.count_nonzero(valid) < 500:
                vis.poll_events()
                vis.update_renderer()
                continue

            u = uu[valid]
            v = vv[valid]
            Z = z_m[valid]

            # Back-project to 3D (camera coordinates)
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            if args.flip_y:
                Y = -Y  # make upright in Open3D for many users

            pts = np.stack([X, Y, Z], axis=1)

            # Colors from RGB (aligned pixel)
            ui = u.astype(np.int32)
            vi = v.astype(np.int32)
            cols = rgb[vi, ui, :].astype(np.float32) / 255.0

            # Update Open3D
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

            vis.update_geometry(pcd)
            if first:
                vis.reset_view_point(True)
                first = False
            vis.poll_events()
            vis.update_renderer()

    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        vis.destroy_window()


if __name__ == "__main__":
    main()
