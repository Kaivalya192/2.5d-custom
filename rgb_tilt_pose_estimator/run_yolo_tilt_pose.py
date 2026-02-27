import argparse
import glob
import os
import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from estimator import EstimatorConfig, estimate_tilt_pose_from_mask

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


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


def _configure_realsense_color_sensor(
    profile,
    exposure: float,
    gain: float,
    white_balance: float,
    auto_exposure: bool,
    auto_white_balance: bool,
) -> None:
    dev = profile.get_device()
    sensors = dev.query_sensors()
    color_sensor = None
    for s in sensors:
        name = s.get_info(rs.camera_info.name).lower()
        if "rgb" in name or "color" in name:
            color_sensor = s
            break
    if color_sensor is None:
        return
    if color_sensor.supports(rs.option.enable_auto_exposure):
        color_sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto_exposure else 0.0)
    if color_sensor.supports(rs.option.exposure):
        color_sensor.set_option(rs.option.exposure, float(exposure))
    if color_sensor.supports(rs.option.gain):
        color_sensor.set_option(rs.option.gain, float(gain))
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0 if auto_white_balance else 0.0)
    if color_sensor.supports(rs.option.white_balance):
        color_sensor.set_option(rs.option.white_balance, float(white_balance))


def source_realsense(
    width: int,
    height: int,
    fps: int,
    exposure: float,
    gain: float,
    white_balance: float,
    auto_exposure: bool,
    auto_white_balance: bool,
) -> Generator[np.ndarray, None, None]:
    if rs is None:
        raise RuntimeError("pyrealsense2 is not available.")
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipe.start(cfg)
    _configure_realsense_color_sensor(
        profile,
        exposure=exposure,
        gain=gain,
        white_balance=white_balance,
        auto_exposure=auto_exposure,
        auto_white_balance=auto_white_balance,
    )
    try:
        for _ in range(15):
            _ = pipe.wait_for_frames(5000)
        while True:
            frames = pipe.wait_for_frames(5000)
            color = frames.get_color_frame()
            if color:
                yield np.asanyarray(color.get_data())
    finally:
        pipe.stop()


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robust RGB-only roll/pitch estimator from YOLOv8 segmentation masks with ROI and smoothing."
    )
    p.add_argument("--model", type=str, default="runs/segment/Imported/best.pt")
    p.add_argument("--source", type=str, default="rs", choices=["rs", "webcam", "video", "images"])
    p.add_argument("--video", type=str, default="")
    p.add_argument("--image_dir", type=str, default="")
    p.add_argument("--webcam_index", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--roi_yaml", type=str, default="segmentation_bgsub_yolo/config/roi.yaml")
    p.add_argument("--show_roi", action="store_true")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max_det", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--overlap_suppress_ratio", type=float, default=0.35)
    p.add_argument("--overlay_alpha", type=float, default=0.30)
    p.add_argument("--track_alpha", type=float, default=0.40)
    p.add_argument("--track_max_dist_px", type=float, default=75.0)
    p.add_argument("--track_max_missed", type=int, default=10)
    p.add_argument("--track_min_hits", type=int, default=2)
    p.add_argument("--show_invalid", action="store_true")
    p.add_argument("--save_video", type=str, default="")
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--out_fps", type=float, default=20.0)
    p.add_argument("--window", type=str, default="YOLO Tilt Roll/Pitch")
    p.add_argument("--exposure", type=float, default=140.0)
    p.add_argument("--gain", type=float, default=16.0)
    p.add_argument("--white_balance", type=float, default=4500.0)
    p.add_argument("--auto_exposure", action="store_true")
    p.add_argument("--auto_white_balance", action="store_true")
    p.add_argument("--conf_thr", type=float, default=0.55)
    p.add_argument("--min_area", type=int, default=700)
    p.add_argument("--border_margin_px", type=int, default=8)
    p.add_argument("--solidity_thr", type=float, default=0.90)
    p.add_argument("--completeness_thr", type=float, default=0.62)
    p.add_argument("--min_axis_ratio", type=float, default=0.18)
    p.add_argument("--max_axis_ratio", type=float, default=1.00)
    p.add_argument("--min_major_px", type=float, default=18.0)
    p.add_argument("--max_ellipse_residual", type=float, default=0.36)
    p.add_argument("--min_ellipse_iou", type=float, default=0.58)
    p.add_argument("--max_occlusion_score", type=float, default=0.48)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.source == "video" and not args.video:
        raise RuntimeError("--video is required when --source video")
    if args.source == "images" and not args.image_dir:
        raise RuntimeError("--image_dir is required when --source images")
    if args.roi_yaml and not os.path.exists(args.roi_yaml):
        raise RuntimeError(f"ROI file not found: {args.roi_yaml}")

    model = YOLO(args.model)
    print(f"[Info] model: {args.model}")
    if args.source == "rs":
        print(
            f"[Info] RealSense color config: {args.width}x{args.height}@{args.fps} "
            f"exp={args.exposure} gain={args.gain} wb={args.white_balance} "
            f"auto_exp={args.auto_exposure} auto_wb={args.auto_white_balance}"
        )

    if args.source == "rs":
        frames = source_realsense(
            width=args.width,
            height=args.height,
            fps=args.fps,
            exposure=args.exposure,
            gain=args.gain,
            white_balance=args.white_balance,
            auto_exposure=args.auto_exposure,
            auto_white_balance=args.auto_white_balance,
        )
    elif args.source == "webcam":
        frames = source_webcam(args.webcam_index)
    elif args.source == "video":
        frames = source_video(args.video)
    else:
        frames = source_images(args.image_dir)

    cfg = EstimatorConfig(
        conf_thr=float(args.conf_thr),
        min_area=int(args.min_area),
        border_margin_px=int(args.border_margin_px),
        solidity_thr=float(args.solidity_thr),
        completeness_thr=float(args.completeness_thr),
        min_axis_ratio=float(args.min_axis_ratio),
        max_axis_ratio=float(args.max_axis_ratio),
        min_major_px=float(args.min_major_px),
        max_ellipse_residual=float(args.max_ellipse_residual),
        min_ellipse_iou=float(args.min_ellipse_iou),
        max_occlusion_score=float(args.max_occlusion_score),
    )

    tracker = PoseTracker(
        alpha=float(args.track_alpha),
        max_dist_px=float(args.track_max_dist_px),
        max_missed=int(args.track_max_missed),
        min_hits=int(args.track_min_hits),
    )

    roi_xyxy: Optional[Tuple[int, int, int, int]] = None
    writer: Optional[cv2.VideoWriter] = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    frame_idx = 0
    prev_t = time.time()
    while True:
        try:
            frame = next(frames)
        except StopIteration:
            break
        frame_idx += 1
        h, w = frame.shape[:2]

        if args.roi_yaml:
            if roi_xyxy is None:
                roi_xyxy = load_roi_yaml(args.roi_yaml, w, h)
                print(f"[Info] ROI loaded: x={roi_xyxy[0]}:{roi_xyxy[2]} y={roi_xyxy[1]}:{roi_xyxy[3]}")
            x0, y0, x1, y1 = roi_xyxy
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        roi = frame[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        if rh <= 0 or rw <= 0:
            raise RuntimeError("Empty ROI after cropping.")

        pred = model.predict(
            source=roi,
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=int(args.imgsz),
            max_det=int(args.max_det),
            device=args.device,
            verbose=False,
        )[0]

        raw_candidates: List[Dict] = []
        if pred.masks is not None and pred.boxes is not None and len(pred.masks.xy) > 0:
            polys = pred.masks.xy
            confs = pred.boxes.conf.cpu().numpy() if pred.boxes.conf is not None else np.zeros((len(polys),), dtype=np.float32)
            for i, poly in enumerate(polys):
                if poly is None or len(poly) < 3:
                    continue
                mask = make_mask_from_polygon(poly, rh, rw)
                area = int(np.count_nonzero(mask))
                if area <= 0:
                    continue
                raw_candidates.append(
                    {
                        "poly": np.round(poly).astype(np.int32),
                        "mask": mask,
                        "area": area,
                        "confidence": float(confs[i]),
                    }
                )

        candidates = suppress_overlaps(raw_candidates, ratio_thr=float(args.overlap_suppress_ratio))
        valid_dets: List[Dict] = []
        invalid_dets: List[Dict] = []
        for c in candidates:
            est = estimate_tilt_pose_from_mask(c["mask"], c["confidence"], cfg=cfg, K=None)
            if est.get("valid", False):
                n = np.asarray(est["normal_vec"], dtype=np.float32)
                ctr = np.asarray(est["pose2d"], dtype=np.float32)
                valid_dets.append(
                    {
                        "center": ctr,
                        "normal": n,
                        "confidence": float(c["confidence"]),
                        "metrics": dict(est.get("metrics", {})),
                        "result": est,
                    }
                )
            else:
                invalid_dets.append({"result": est})

        det_track_ids = tracker.update(valid_dets)

        overlay = frame.copy()
        out = frame.copy()

        for det, tid in zip(valid_dets, det_track_ids):
            tr = tracker.tracks.get(tid)
            if tr is None or tr.missed > 0:
                continue
            color = random_color(tid * 11939)

            contour = det["result"]["contour"].reshape(-1, 2).astype(np.int32)
            contour[:, 0] += int(x0)
            contour[:, 1] += int(y0)
            cv2.fillPoly(overlay, [contour.reshape(-1, 1, 2)], color)
            cv2.polylines(out, [contour.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)

            cxg = float(tr.center[0] + x0)
            cyg = float(tr.center[1] + y0)
            roll, pitch, tilt, tdir = normal_to_pose(tr.normal)
            L = 52
            ax = np.deg2rad(tdir)
            p0 = (int(round(cxg)), int(round(cyg)))
            p1 = (int(round(cxg + L * np.cos(ax))), int(round(cyg + L * np.sin(ax))))
            cv2.circle(out, p0, 4, (0, 255, 0), -1)
            cv2.arrowedLine(out, p0, p1, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.16)

            stable = tr.hits >= tracker.min_hits
            stxt = "OK" if stable else "WARM"
            label = (
                f"#{tid} {stxt} r={roll:+.1f} p={pitch:+.1f} "
                f"tilt={tilt:.1f} conf={tr.confidence:.2f}"
            )
            tx = max(8, min(w - 360, p0[0] - 24))
            ty = max(20, p0[1] - 10)
            cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

            if args.debug:
                m = tr.metrics
                dbg = (
                    f"sol={m.get('solidity', 0):.2f} comp={m.get('completeness', 0):.2f} "
                    f"iouE={m.get('ellipse_iou', 0):.2f} occ={m.get('occlusion_score', 1):.2f}"
                )
                cv2.putText(out, dbg, (tx, min(h - 10, ty + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

        if args.show_invalid:
            ytxt = 58
            for inv in invalid_dets[:8]:
                r = inv["result"].get("reason", "invalid")
                cv2.putText(out, f"reject: {r}", (10, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
                ytxt += 16

        out = cv2.addWeighted(overlay, float(args.overlay_alpha), out, 1.0 - float(args.overlay_alpha), 0.0)

        if args.show_roi and roi_xyxy is not None:
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 0), 1, cv2.LINE_AA)

        now_t = time.time()
        dt = max(now_t - prev_t, 1e-6)
        prev_t = now_t
        fps = 1.0 / dt
        stable_count = sum(1 for t in tracker.tracks.values() if t.missed == 0 and t.hits >= tracker.min_hits)
        live_count = sum(1 for t in tracker.tracks.values() if t.missed == 0)
        cv2.putText(
            out,
            f"frame={frame_idx} det={len(raw_candidates)} keep={len(candidates)} valid={len(valid_dets)} tracks={live_count}/{stable_count} fps={fps:.1f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if args.save_video:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save_video, fourcc, float(args.out_fps), (w, h))
                print(f"[Info] writing video: {args.save_video}")
            writer.write(out)
        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir, f"pose_{frame_idx:06d}.png"), out)

        cv2.imshow(args.window, out)
        wait = 1 if args.source in ("rs", "webcam", "video") else 0
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27):
            break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
