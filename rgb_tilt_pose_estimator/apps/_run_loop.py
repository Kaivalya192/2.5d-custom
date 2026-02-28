import os
import time
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from rgb_tilt_pose_estimator.common.estimator import EstimatorConfig, estimate_tilt_pose_from_mask
from rgb_tilt_pose_estimator.common.masks import make_mask_from_polygon, suppress_overlaps
from rgb_tilt_pose_estimator.common.roi import load_roi_yaml
from rgb_tilt_pose_estimator.common.tracker import PoseTracker, normal_to_pose, random_color


def run_pose_loop(args, model, frames: Generator[np.ndarray, None, None]) -> None:
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
        wait = 1 if args.source in ("rs", "oak", "webcam", "video") else 0
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27):
            break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

