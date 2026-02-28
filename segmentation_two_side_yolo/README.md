# Two-Side YOLO Segmentation (Front/Back)

Separate pipeline for training a 2-class segmentation model:
- `front` (class `0`)
- `back` (class `1`)

This is isolated from `segmentation_bgsub_yolo`.

## Folder Layout
- `segmentation_two_side_yolo/rs_capture_two_side.py`
- `segmentation_two_side_yolo/generate_two_side_dataset.py`
- output:
  - `segmentation_two_side_yolo/captured/background`
  - `segmentation_two_side_yolo/captured/front`
  - `segmentation_two_side_yolo/captured/back`
  - `segmentation_two_side_yolo/dataset/images/{train,val,test}`
  - `segmentation_two_side_yolo/dataset/labels/{train,val,test}`
  - `segmentation_two_side_yolo/dataset/data.yaml`

## 1) Capture Background
```powershell
python segmentation_two_side_yolo/rs_capture_two_side.py background --out_dir segmentation_two_side_yolo/captured/background --num_frames 30 --width 1280 --height 720 --fps 30 --exposure 140 --gain 16 --white_balance 4500 --save_individual
```

## 2) Capture Front Samples (around 20)
```powershell
python segmentation_two_side_yolo/rs_capture_two_side.py front --out_dir segmentation_two_side_yolo/captured/front --num_frames 20 --width 1280 --height 720 --fps 30 --exposure 140 --gain 16 --white_balance 4500
```

## 3) Capture Back Samples (around 20)
```powershell
python segmentation_two_side_yolo/rs_capture_two_side.py back --out_dir segmentation_two_side_yolo/captured/back --num_frames 20 --width 1280 --height 720 --fps 30 --exposure 140 --gain 16 --white_balance 4500
```

Controls in capture window:
- `s`: save frame
- `q` or `ESC`: quit

## 4) Build 2-Class Dataset
First define separate ROI for this pipeline:
```powershell
python segmentation_two_side_yolo/select_roi.py --image segmentation_two_side_yolo/captured/background/background_mean.png --output_yaml segmentation_two_side_yolo/config/roi.yaml --mode polygon4
```
`polygon4` is default. Click 4 points and press `Enter/Space`.

If you already have ROI yaml:
```powershell
python segmentation_two_side_yolo/generate_two_side_dataset.py --background segmentation_two_side_yolo/captured/background/background_mean.png --front_dir segmentation_two_side_yolo/captured/front --back_dir segmentation_two_side_yolo/captured/back --dataset_root segmentation_two_side_yolo/dataset --roi_yaml segmentation_two_side_yolo/config/roi.yaml --crop_to_roi --diff_thresh 28 --channel_thresh 18 --shadow_tol 14 --min_area 900 --open_ksize 3 --close_ksize 7 --median_ksize 5 --polygon_epsilon_ratio 0.002 --save_overlay --skip_empty
```
If some images are unexpectedly skipped, add `--save_reject_overlay` to inspect reject reasons in `dataset/reject_overlays`.

Without ROI:
```powershell
python segmentation_two_side_yolo/generate_two_side_dataset.py --background segmentation_two_side_yolo/captured/background/background_mean.png --front_dir segmentation_two_side_yolo/captured/front --back_dir segmentation_two_side_yolo/captured/back --dataset_root segmentation_two_side_yolo/dataset --diff_thresh 28 --channel_thresh 18 --shadow_tol 14 --min_area 900 --open_ksize 3 --close_ksize 7 --median_ksize 5 --polygon_epsilon_ratio 0.002 --save_overlay --skip_empty
```

## 5) Train YOLOv8 Segmentation
```powershell
yolo task=segment mode=train model=yolov8n-seg.pt data=segmentation_two_side_yolo/dataset/data.yaml imgsz=640 epochs=120 batch=8 workers=0
```

## 6) Synthetic Overlap Augmentation (400 Front + 400 Back)
This creates a two-class overlapped synthetic dataset similar to your previous pipeline, but balanced for both sides:
- 400 front-dominant images
- 400 back-dominant images
- total synthetic images = 800

```powershell
python segmentation_two_side_yolo/synth_overlap_augment_two_side.py --src_dataset_root segmentation_two_side_yolo/dataset --background segmentation_two_side_yolo/captured/background/background_mean.png --out_dataset_root segmentation_two_side_yolo/dataset_aug --num_images_per_class 400 --min_instances 2 --max_instances 24 --dominant_class_ratio 0.65 --min_scale 0.85 --max_scale 1.2 --max_rotation 180 --jitter 0.15 --min_visible_area 80 --min_visible_ratio 0.15 --min_success_ratio 0.45 --max_tries_per_image 12 --val_ratio 0.2 --test_ratio 0.1 --roi_yaml segmentation_two_side_yolo/config/roi.yaml --save_overlays --copy_original
```

Then train on augmented dataset:
```powershell
yolo task=segment mode=train model=yolov8n-seg.pt data=segmentation_two_side_yolo/dataset_aug/data.yaml imgsz=640 epochs=120 batch=8 workers=0
```

## Notes
- Keep one object per frame during capture for clean mask generation.
- Use fixed light and fixed camera settings for best front/back consistency.
- If masks miss the part, lower `--diff_thresh`/`--channel_thresh` slightly.
