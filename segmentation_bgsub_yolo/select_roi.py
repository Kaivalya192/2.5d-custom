import argparse
import os

import cv2


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_roi_yaml(output_yaml: str, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> None:
    fs = cv2.FileStorage(output_yaml, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open yaml for write: {output_yaml}")
    fs.write("x", int(x))
    fs.write("y", int(y))
    fs.write("w", int(w))
    fs.write("h", int(h))
    fs.write("image_width", int(img_w))
    fs.write("image_height", int(img_h))
    fs.write("x_norm", float(x) / float(img_w))
    fs.write("y_norm", float(y) / float(img_h))
    fs.write("w_norm", float(w) / float(img_w))
    fs.write("h_norm", float(h) / float(img_h))
    fs.release()


def main():
    parser = argparse.ArgumentParser(description="Select rectangular ROI in OpenCV and save to YAML.")
    parser.add_argument("--image", type=str, required=True, help="Reference image path (usually background_mean.png)")
    parser.add_argument("--output_yaml", type=str, default="segmentation_bgsub_yolo/config/roi.yaml")
    parser.add_argument("--window", type=str, default="Select ROI")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    print("[Info] Drag rectangle and press ENTER/SPACE. Press 'c' to cancel.")
    roi = cv2.selectROI(args.window, img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI not selected.")

    out_dir = os.path.dirname(args.output_yaml)
    if out_dir:
        ensure_dir(out_dir)
    save_roi_yaml(args.output_yaml, x, y, w, h, img.shape[1], img.shape[0])
    print(f"[OK] ROI saved: {args.output_yaml}")
    print(f"[ROI] x={x} y={y} w={w} h={h}")


if __name__ == "__main__":
    main()
