from typing import Generator

import cv2
import numpy as np


def source_webcam(index: int) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {index}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()

