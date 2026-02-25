import os
import json
import argparse
from datetime import datetime

import pyrealsense2 as rs


def intrinsics_to_dict(intr: rs.intrinsics):
    # intr.model is an enum; intr.model.name may not exist in older bindings
    model = str(intr.model).replace("distortion.", "")
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": model,
        "coeffs": list(intr.coeffs),  # k1,k2,p1,p2,k3... (depends on model)
    }


def extrinsics_to_dict(ex: rs.extrinsics):
    # rotation is row-major 3x3 (9 floats), translation is 3 floats in meters
    R = [list(ex.rotation[0:3]), list(ex.rotation[3:6]), list(ex.rotation[6:9])]
    t = list(ex.translation)
    return {"R": R, "t_m": t}


def find_stream_profile(profile: rs.pipeline_profile, stream_type, stream_index, width, height, fps, fmt):
    """
    Returns the first matching stream profile if exact match exists.
    Otherwise returns None.
    """
    for sp in profile.get_streams():
        if sp.stream_type() != stream_type:
            continue
        if sp.stream_index() != stream_index:
            continue
        vsp = sp.as_video_stream_profile()
        if vsp.width() == width and vsp.height() == height and vsp.fps() == fps and sp.format() == fmt:
            return sp
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="realsense_calib", help="Output folder")
    parser.add_argument("--ir_w", type=int, default=848)
    parser.add_argument("--ir_h", type=int, default=480)
    parser.add_argument("--ir_fps", type=int, default=30)
    parser.add_argument("--rgb_w", type=int, default=640)
    parser.add_argument("--rgb_h", type=int, default=480)
    parser.add_argument("--rgb_fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()

    # Request streams. Using Y8 for IR is typical on D435i.
    cfg.enable_stream(rs.stream.infrared, 1, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)
    cfg.enable_stream(rs.stream.infrared, 2, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)
    cfg.enable_stream(rs.stream.color, args.rgb_w, args.rgb_h, rs.format.bgr8, args.rgb_fps)

    profile = pipe.start(cfg)

    # Grab stream profiles
    ir1_sp = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2_sp = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    rgb_sp = profile.get_stream(rs.stream.color).as_video_stream_profile()

    # Intrinsics
    ir1_intr = ir1_sp.get_intrinsics()
    ir2_intr = ir2_sp.get_intrinsics()
    rgb_intr = rgb_sp.get_intrinsics()

    # Extrinsics (sensor-to-sensor)
    # These come from the device calibration (factory).
    ir1_to_ir2 = ir1_sp.get_extrinsics_to(ir2_sp)
    ir2_to_ir1 = ir2_sp.get_extrinsics_to(ir1_sp)
    rgb_to_ir1 = rgb_sp.get_extrinsics_to(ir1_sp)
    ir1_to_rgb = ir1_sp.get_extrinsics_to(rgb_sp)
    rgb_to_ir2 = rgb_sp.get_extrinsics_to(ir2_sp)
    ir2_to_rgb = ir2_sp.get_extrinsics_to(rgb_sp)

    calib = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "device_name": profile.get_device().get_info(rs.camera_info.name),
            "serial": profile.get_device().get_info(rs.camera_info.serial_number),
        },
        "streams": {
            "ir1": {
                "stream": "infrared",
                "index": 1,
                "width": ir1_sp.width(),
                "height": ir1_sp.height(),
                "fps": ir1_sp.fps(),
                "format": str(ir1_sp.format()),
                "intrinsics": intrinsics_to_dict(ir1_intr),
            },
            "ir2": {
                "stream": "infrared",
                "index": 2,
                "width": ir2_sp.width(),
                "height": ir2_sp.height(),
                "fps": ir2_sp.fps(),
                "format": str(ir2_sp.format()),
                "intrinsics": intrinsics_to_dict(ir2_intr),
            },
            "rgb": {
                "stream": "color",
                "index": 0,
                "width": rgb_sp.width(),
                "height": rgb_sp.height(),
                "fps": rgb_sp.fps(),
                "format": str(rgb_sp.format()),
                "intrinsics": intrinsics_to_dict(rgb_intr),
            },
        },
        "extrinsics": {
            "ir1_to_ir2": extrinsics_to_dict(ir1_to_ir2),
            "ir2_to_ir1": extrinsics_to_dict(ir2_to_ir1),
            "rgb_to_ir1": extrinsics_to_dict(rgb_to_ir1),
            "ir1_to_rgb": extrinsics_to_dict(ir1_to_rgb),
            "rgb_to_ir2": extrinsics_to_dict(rgb_to_ir2),
            "ir2_to_rgb": extrinsics_to_dict(ir2_to_rgb),
        },
        "notes": [
            "Extrinsics are factory calibration from RealSense.",
            "Rotation is 3x3 row-major, translation is meters.",
            "Use ir1_to_ir2 as your stereo baseline transform.",
        ],
    }

    out_json = os.path.join(args.out, "calibration.json")
    with open(out_json, "w") as f:
        json.dump(calib, f, indent=2)

    # Optional: also save a small human-readable summary
    summary_txt = os.path.join(args.out, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"Device: {calib['meta']['device_name']}\n")
        f.write(f"Serial: {calib['meta']['serial']}\n")
        f.write(f"Generated: {calib['meta']['generated_at']}\n\n")
        f.write("Streams:\n")
        for k in ["ir1", "ir2", "rgb"]:
            s = calib["streams"][k]
            intr = s["intrinsics"]
            f.write(f"  {k}: {s['width']}x{s['height']} @ {s['fps']} {s['format']}\n")
            f.write(f"     fx={intr['fx']:.3f} fy={intr['fy']:.3f} cx={intr['ppx']:.3f} cy={intr['ppy']:.3f}\n")
            f.write(f"     model={intr['model']} coeffs={intr['coeffs']}\n")
        f.write("\nExtrinsics:\n")
        for k, ex in calib["extrinsics"].items():
            f.write(f"  {k}: t(m)={ex['t_m']}\n")

    pipe.stop()
    print(f"[OK] Saved calibration to: {out_json}")
    print(f"[OK] Saved summary to: {summary_txt}")


if __name__ == "__main__":
    main()