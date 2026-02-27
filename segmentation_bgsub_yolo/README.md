# Automatic YOLOv8 Segmentation Dataset Generation (Background Subtraction)

Pipeline for fixed-camera industrial bin-picking scenes (Intel RealSense RGB, fixed lighting, fixed bin).

## Features
- Fixed-parameter RealSense capture (manual exposure, gain, white balance).
- Background model from averaged empty-bin frames.
- Foreground extraction via RGB absolute difference.
- Shadow suppression + morphology cleanup + median filtering.
- Largest connected component filtering (single-object mask).
- YOLOv8 segmentation polygon export (`class_id x1 y1 x2 y2 ...`, normalized).
- Batch processing with optional overlay quality-check images.
- Shared ROI rectangle selection and YAML save/load.

## Scripts
- `rs_capture.py`: Capture background frames and object frames from RealSense.
- `bg_model.py`: Build averaged background image from folder.
- `select_roi.py`: Draw rectangle ROI in OpenCV and save YAML.
- `generate_dataset.py`: Generate `images/train` + `labels/train` + `data.yaml`.
- `synth_overlap_augment.py`: Create synthetic multi-object overlap scenes from labeled single-object data.

## Quick Start (Windows PowerShell / CMD)
Run commands from repo root:
- `C:\Users\asus\dexsent\test\2.5d-custom`

Notes:
- Use one-line commands below for maximum Windows compatibility.
- In PowerShell multiline mode, use backtick `` ` `` (not `^`).
- Do not press Enter after `python ...` and then type flags on a new line unless you use backtick continuation.

## 1) Capture empty bin background (30 frames)
```bash
python segmentation_bgsub_yolo/rs_capture.py background --out_dir segmentation_bgsub_yolo/captured/background --num_frames 30 --width 1280 --height 720 --fps 30 --exposure 140 --gain 16 --white_balance 4500 --save_individual
```

PowerShell multiline:
```powershell
python segmentation_bgsub_yolo/rs_capture.py background `
  --out_dir segmentation_bgsub_yolo/captured/background `
  --num_frames 30 `
  --width 1280 --height 720 --fps 30 `
  --exposure 140 --gain 16 --white_balance 4500 `
  --save_individual
```

Expected output:
- `segmentation_bgsub_yolo/captured/background/background_*.png`
- `segmentation_bgsub_yolo/captured/background/background_mean.png`
- `segmentation_bgsub_yolo/captured/background/background_mean.npy`

## 2) Capture object images
```bash
python segmentation_bgsub_yolo/rs_capture.py objects --out_dir segmentation_bgsub_yolo/captured/objects --num_frames 300 --width 1280 --height 720 --fps 30 --exposure 140 --gain 16 --white_balance 4500
```

Controls in window:
- `s`: save current frame
- `q` or `ESC`: quit

## 3) (Optional) Recompute averaged background from folder
```bash
python segmentation_bgsub_yolo/bg_model.py --input_dir segmentation_bgsub_yolo/captured/background --output_path segmentation_bgsub_yolo/captured/background/background_mean.png
```

PowerShell multiline:
```powershell
python segmentation_bgsub_yolo/bg_model.py `
  --input_dir segmentation_bgsub_yolo/captured/background `
  --output_path segmentation_bgsub_yolo/captured/background/background_mean.png
```

## 4) Define shared ROI once and save YAML
```bash
python segmentation_bgsub_yolo/select_roi.py --image segmentation_bgsub_yolo/captured/background/background_mean.png --output_yaml segmentation_bgsub_yolo/config/roi.yaml
```

## 5) Generate YOLOv8 segmentation dataset
```bash
python segmentation_bgsub_yolo/generate_dataset.py --background segmentation_bgsub_yolo/captured/background/background_mean.png --input_dir segmentation_bgsub_yolo/captured/objects --dataset_root segmentation_bgsub_yolo/dataset --roi_yaml segmentation_bgsub_yolo/config/roi.yaml --class_name part --class_id 0 --diff_thresh 28 --channel_thresh 18 --shadow_tol 14 --min_area 1200 --open_ksize 3 --close_ksize 7 --median_ksize 5 --polygon_epsilon_ratio 0.002 --save_overlay
```

PowerShell multiline:
```powershell
python segmentation_bgsub_yolo/generate_dataset.py `
  --background segmentation_bgsub_yolo/captured/background/background_mean.png `
  --input_dir segmentation_bgsub_yolo/captured/objects `
  --dataset_root segmentation_bgsub_yolo/dataset `
  --roi_yaml segmentation_bgsub_yolo/config/roi.yaml `
  --class_name part `
  --class_id 0 `
  --diff_thresh 28 `
  --channel_thresh 18 `
  --shadow_tol 14 `
  --min_area 1200 `
  --open_ksize 3 `
  --close_ksize 7 `
  --median_ksize 5 `
  --polygon_epsilon_ratio 0.002 `
  --save_overlay
```

## Dataset Output
- Images: `segmentation_bgsub_yolo/dataset/images/train`
- Labels: `segmentation_bgsub_yolo/dataset/labels/train`
- Overlays: `segmentation_bgsub_yolo/dataset/overlays` (if `--save_overlay`)
- Config: `segmentation_bgsub_yolo/dataset/data.yaml`
- ROI config: `segmentation_bgsub_yolo/config/roi.yaml`

## ROI YAML Fields
- `x`, `y`, `w`, `h`: rectangle in pixel coordinates
- `image_width`, `image_height`: reference image size
- `x_norm`, `y_norm`, `w_norm`, `h_norm`: normalized values

## YOLOv8 Training
```bash
yolo task=segment mode=train model=yolov8n-seg.pt data=segmentation_bgsub_yolo/dataset/data.yaml imgsz=640 epochs=100 batch=16
```

## Real-Time Instance Segmentation Inference
Annotate all detected objects in each frame individually.

RealSense live:
```bash
python segmentation_bgsub_yolo/yolo_seg_infer_realtime.py --model runs/segment/train/weights/best.pt --source rs --conf 0.25 --iou 0.45 --imgsz 640
```

Webcam live:
```bash
python segmentation_bgsub_yolo/yolo_seg_infer_realtime.py --model runs/segment/train/weights/best.pt --source webcam --webcam_index 0
```

Image folder:
```bash
python segmentation_bgsub_yolo/yolo_seg_infer_realtime.py --model runs/segment/train/weights/best.pt --source images --image_dir segmentation_bgsub_yolo/dataset/images/train
```

## Synthetic Overlap Augmentation (for bin-picking)
Use this to create occluded multi-instance training images from your labeled single-object dataset.

```bash
python segmentation_bgsub_yolo/synth_overlap_augment.py --src_dataset_root segmentation_bgsub_yolo/dataset --background segmentation_bgsub_yolo/captured/background/background_mean.png --out_dataset_root segmentation_bgsub_yolo/dataset_aug --roi_yaml segmentation_bgsub_yolo/config/roi.yaml --class_id 0 --class_name part --num_images 400 --min_instances 2 --max_instances 30 --count_mode mixed --dense_probability 0.6 --min_success_ratio 0.45 --min_scale 0.85 --max_scale 1.2 --max_rotation 180 --jitter 0.15 --min_visible_area 80 --min_visible_ratio 0.15 --max_tries_per_image 12 --val_ratio 0.2 --test_ratio 0.1 --copy_original --save_overlays
```

Train on augmented set:
```bash
yolo task=segment mode=train model=yolov8n-seg.pt data=segmentation_bgsub_yolo/dataset_aug/data.yaml imgsz=640 epochs=120 batch=8 workers=0
```

## Tuning Guide
- Too much noise: increase `--diff_thresh`, `--channel_thresh`, `--open_ksize`.
- Holes in mask: increase `--close_ksize`.
- Object edges jagged: increase `--median_ksize` (odd values).
- Shadows detected as object: increase `--shadow_tol`.
- Small false blobs: increase `--min_area`.
- Polygon too detailed: increase `--polygon_epsilon_ratio`.

## Robustness Notes
- Keep camera rigidly fixed.
- Keep bin fixed and clean.
- Keep lighting stable (no flicker).
- Lock camera controls (already done by `rs_capture.py`).
