from typing import Generator

import cv2
import numpy as np


def source_video(path: str) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()

