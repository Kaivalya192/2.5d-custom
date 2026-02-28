from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TrackState:
    track_id: int
    center: np.ndarray
    normal: np.ndarray
    confidence: float
    hits: int
    missed: int
    metrics: Dict


class PoseTracker:
    def __init__(self, alpha: float, max_dist_px: float, max_missed: int, min_hits: int) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.max_dist_px = float(max_dist_px)
        self.max_missed = int(max(0, max_missed))
        self.min_hits = int(max(1, min_hits))
        self.tracks: Dict[int, TrackState] = {}
        self.next_id = 1

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= 1e-8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return (v / n).astype(np.float32)

    def _new_track(self, det: Dict) -> int:
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = TrackState(
            track_id=tid,
            center=det["center"].astype(np.float32),
            normal=self._normalize(det["normal"]),
            confidence=float(det["confidence"]),
            hits=1,
            missed=0,
            metrics=dict(det["metrics"]),
        )
        return tid

    def _update_track(self, track: TrackState, det: Dict) -> None:
        c_new = det["center"].astype(np.float32)
        n_new = self._normalize(det["normal"])
        # Resolve 180-degree ambiguity of ellipse major-axis direction.
        if float(np.dot(n_new[:2], track.normal[:2])) < 0.0:
            n_new[0] *= -1.0
            n_new[1] *= -1.0
        track.center = ((1.0 - self.alpha) * track.center + self.alpha * c_new).astype(np.float32)
        n_mix = (1.0 - self.alpha) * track.normal + self.alpha * n_new
        track.normal = self._normalize(n_mix)
        track.confidence = float((1.0 - self.alpha) * track.confidence + self.alpha * float(det["confidence"]))
        track.metrics = dict(det["metrics"])
        track.hits += 1
        track.missed = 0

    def update(self, detections: List[Dict]) -> List[int]:
        assigned_track_ids = [-1] * len(detections)
        if not self.tracks and not detections:
            return assigned_track_ids
        if not self.tracks:
            for i, d in enumerate(detections):
                assigned_track_ids[i] = self._new_track(d)
            return assigned_track_ids

        track_ids = list(self.tracks.keys())
        if detections:
            det_centers = np.stack([d["center"] for d in detections], axis=0).astype(np.float32)
            trk_centers = np.stack([self.tracks[tid].center for tid in track_ids], axis=0).astype(np.float32)
            dmat = np.linalg.norm(trk_centers[:, None, :] - det_centers[None, :, :], axis=2)
        else:
            dmat = np.empty((len(track_ids), 0), dtype=np.float32)

        used_tracks = set()
        used_dets = set()
        while dmat.size > 0:
            t_idx, d_idx = np.unravel_index(int(np.argmin(dmat)), dmat.shape)
            best = float(dmat[t_idx, d_idx])
            if best > self.max_dist_px:
                break
            tid = track_ids[t_idx]
            if tid in used_tracks or d_idx in used_dets:
                dmat[t_idx, d_idx] = np.inf
                continue
            self._update_track(self.tracks[tid], detections[d_idx])
            assigned_track_ids[d_idx] = tid
            used_tracks.add(tid)
            used_dets.add(d_idx)
            dmat[t_idx, :] = np.inf
            dmat[:, d_idx] = np.inf

        for i, d in enumerate(detections):
            if assigned_track_ids[i] < 0:
                assigned_track_ids[i] = self._new_track(d)

        for tid in list(self.tracks.keys()):
            if tid not in used_tracks and tid not in assigned_track_ids:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]

        return assigned_track_ids


def random_color(seed: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(int(seed))
    c = rng.integers(70, 255, size=3, dtype=np.uint8)
    return int(c[0]), int(c[1]), int(c[2])


def normal_to_pose(normal_vec: np.ndarray) -> Tuple[float, float, float, float]:
    nx, ny, nz = float(normal_vec[0]), float(normal_vec[1]), float(normal_vec[2])
    nz = max(nz, 1e-8)
    tilt = float(np.degrees(np.arctan2(np.hypot(nx, ny), nz)))
    tilt_dir = float(np.degrees(np.arctan2(ny, nx)))
    roll = float(np.degrees(np.arctan2(ny, nz)))
    pitch = float(np.degrees(np.arctan2(-nx, nz)))
    return roll, pitch, tilt, tilt_dir

