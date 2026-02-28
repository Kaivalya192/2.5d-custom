import argparse
import os

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _select_polygon_4pt(img: np.ndarray, win_name: str) -> np.ndarray:
    pts: list[tuple[int, int]] = []
    disp = img.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal disp
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) < 4:
                pts.append((int(x), int(y)))
            disp = img.copy()
            for i, p in enumerate(pts):
                cv2.circle(disp, p, 5, (0, 255, 255), -1)
                cv2.putText(disp, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if len(pts) >= 2:
                cv2.polylines(disp, [np.array(pts, dtype=np.int32)], False, (0, 255, 255), 2)
            if len(pts) == 4:
                cv2.polylines(disp, [np.array(pts, dtype=np.int32)], True, (0, 255, 255), 2)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)
    print("[Info] Click 4 polygon points in order (clockwise or counterclockwise).")
    print("[Info] Keys: ENTER/SPACE=save, r=reset, q/ESC=cancel")
    while True:
        cv2.imshow(win_name, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32):  # enter / space
            if len(pts) == 4:
                break
            print(f"[Warn] Need exactly 4 points, currently {len(pts)}.")
        elif k in (ord("r"), ord("R")):
            pts.clear()
            disp = img.copy()
        elif k in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(win_name)
            raise RuntimeError("ROI selection cancelled.")

    cv2.destroyWindow(win_name)
    return np.array(pts, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Select and save ROI for two-side segmentation pipeline.")
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--output_yaml", type=str, default="segmentation_two_side_yolo/config/roi.yaml")
    ap.add_argument("--mode", type=str, default="polygon4", choices=["polygon4", "rect"])
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    if args.mode == "rect":
        print("[Info] Drag ROI rectangle and press ENTER/SPACE. Press c to cancel.")
        x, y, w, h = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        if w <= 0 or h <= 0:
            raise RuntimeError("Invalid ROI selected.")
        poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        roi_mode = "rect"
    else:
        poly = _select_polygon_4pt(img, "Select ROI Polygon")
        x = int(np.min(poly[:, 0]))
        y = int(np.min(poly[:, 1]))
        x1 = int(np.max(poly[:, 0]))
        y1 = int(np.max(poly[:, 1]))
        w = int(max(1, x1 - x))
        h = int(max(1, y1 - y))
        roi_mode = "polygon4"

    out_dir = os.path.dirname(args.output_yaml)
    if out_dir:
        ensure_dir(out_dir)

    fs = cv2.FileStorage(args.output_yaml, cv2.FILE_STORAGE_WRITE)
    fs.write("mode", roi_mode)
    fs.write("x", int(x))
    fs.write("y", int(y))
    fs.write("w", int(w))
    fs.write("h", int(h))
    fs.write("points", poly.astype(np.int32))
    fs.write("image_width", int(img.shape[1]))
    fs.write("image_height", int(img.shape[0]))
    fs.write("x_norm", float(x) / float(img.shape[1]))
    fs.write("y_norm", float(y) / float(img.shape[0]))
    fs.write("w_norm", float(w) / float(img.shape[1]))
    fs.write("h_norm", float(h) / float(img.shape[0]))
    fs.release()

    print(f"[OK] ROI saved: {args.output_yaml} -> mode={roi_mode} x={x} y={y} w={w} h={h}")
    if roi_mode == "polygon4":
        print(f"[OK] points={poly.tolist()}")


if __name__ == "__main__":
    main()
