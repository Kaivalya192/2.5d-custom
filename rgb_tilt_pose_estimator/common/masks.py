from typing import Dict, List

import cv2
import numpy as np


def make_mask_from_polygon(poly_xy: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if poly_xy is None or len(poly_xy) < 3:
        return m
    pts = np.round(poly_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 255)
    return m


def overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    ia = int(np.count_nonzero(mask_a))
    ib = int(np.count_nonzero(mask_b))
    if ia <= 0 or ib <= 0:
        return 0.0
    inter = int(np.count_nonzero(cv2.bitwise_and(mask_a, mask_b)))
    return float(inter / max(min(ia, ib), 1))


def suppress_overlaps(cands: List[Dict], ratio_thr: float) -> List[Dict]:
    if len(cands) <= 1:
        return cands
    order = sorted(range(len(cands)), key=lambda i: (cands[i]["confidence"], cands[i]["area"]), reverse=True)
    keep: List[int] = []
    for idx in order:
        blocked = False
        for k in keep:
            if overlap_ratio(cands[idx]["mask"], cands[k]["mask"]) > ratio_thr:
                blocked = True
                break
        if not blocked:
            keep.append(idx)
    keep.sort()
    return [cands[i] for i in keep]

