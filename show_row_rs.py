import argparse
import numpy as np
import cv2
import pyrealsense2 as rs


def normalize_ir(ir8: np.ndarray) -> np.ndarray:
    # IR is already 8-bit usually; but contrast can be low. Optional normalize for display:
    # Comment out normalization if you want "raw as-is".
    return cv2.equalizeHist(ir8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir_w", type=int, default=848)
    parser.add_argument("--ir_h", type=int, default=480)
    parser.add_argument("--ir_fps", type=int, default=30)
    parser.add_argument("--rgb_w", type=int, default=640)
    parser.add_argument("--rgb_h", type=int, default=480)
    parser.add_argument("--rgb_fps", type=int, default=30)
    parser.add_argument("--tile", action="store_true", help="Show as single tiled window instead of 3 windows")
    args = parser.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)
    cfg.enable_stream(rs.stream.infrared, 2, args.ir_w, args.ir_h, rs.format.y8, args.ir_fps)
    cfg.enable_stream(rs.stream.color, args.rgb_w, args.rgb_h, rs.format.bgr8, args.rgb_fps)

    pipe.start(cfg)

    try:
        while True:
            frames = pipe.wait_for_frames()
            ir1 = frames.get_infrared_frame(1)
            ir2 = frames.get_infrared_frame(2)
            rgb = frames.get_color_frame()

            if not ir1 or not ir2 or not rgb:
                continue

            ir1_img = np.asanyarray(ir1.get_data())
            ir2_img = np.asanyarray(ir2.get_data())
            rgb_img = np.asanyarray(rgb.get_data())  # already BGR8 as configured

            # Display tweaks (optional)
            ir1_disp = normalize_ir(ir1_img)
            ir2_disp = normalize_ir(ir2_img)

            if args.tile:
                # convert IR to BGR for tiling
                ir1_bgr = cv2.cvtColor(ir1_disp, cv2.COLOR_GRAY2BGR)
                ir2_bgr = cv2.cvtColor(ir2_disp, cv2.COLOR_GRAY2BGR)

                # Resize to same height for a clean tile
                H = max(ir1_bgr.shape[0], rgb_img.shape[0], ir2_bgr.shape[0])

                def resize_to_h(img, Ht):
                    h, w = img.shape[:2]
                    if h == Ht:
                        return img
                    scale = Ht / float(h)
                    return cv2.resize(img, (int(w * scale), Ht), interpolation=cv2.INTER_AREA)

                ir1_bgr = resize_to_h(ir1_bgr, H)
                rgb_r = resize_to_h(rgb_img, H)
                ir2_bgr = resize_to_h(ir2_bgr, H)

                tiled = cv2.hconcat([ir1_bgr, rgb_r, ir2_bgr])
                cv2.imshow("IR1 | RGB | IR2  (press q to quit)", tiled)
            else:
                cv2.imshow("IR1 (press q to quit)", ir1_disp)
                cv2.imshow("IR2 (press q to quit)", ir2_disp)
                cv2.imshow("RGB (press q to quit)", rgb_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()