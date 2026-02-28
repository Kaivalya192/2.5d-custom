import glob
import os
from typing import Generator, List

import cv2
import numpy as np


def source_images(folder: str) -> Generator[np.ndarray, None, None]:
    files: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    files.sort()
    if not files:
        raise RuntimeError(f"No images found in: {folder}")
    for fp in files:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is not None:
            yield img

