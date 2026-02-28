from typing import Tuple

import cv2


def load_roi_yaml(path: str, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open ROI yaml: {path}")
    x = int(fs.getNode("x").real())
    y = int(fs.getNode("y").real())
    w = int(fs.getNode("w").real())
    h = int(fs.getNode("h").real())
    fs.release()
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Invalid ROI or ROI outside frame.")
    return x0, y0, x1, y1

