# Real-Time ADAS Perception and Risk Warning System

Computer-vision pipeline: YOLO object detection → multi-object tracking → risk-zone analysis → distance estimation → ADAS warnings on driving video, producing an annotated demo video, frame-level CSV logs, and runtime metrics.

## Overview

| Phase | Status | Key output |
|-------|--------|------------|
| Phase 1 — Detection | Done | `outputs/detections/predict/` |
| Phase 2 — Tracking | Done | `outputs/logs/tracking_results.csv` |
| Phase 3 — ADAS warning engine | Done | `outputs/logs/warnings.csv`, `outputs/demo/adas_demo_v1.mp4` |
| Phase 4 — Metrics & polish | Done | `outputs/reports/demo_metrics.csv`, `outputs/figures/fps_plot.png` |
| V2 — Evaluation | Done | `outputs/figures/confidence_hist.png`, tracker comparison, failure-case screenshots |
| V3 — KITTI fine-tuning | Done | `outputs/reports/kitti_training_results.csv`, `outputs/figures/kitti_training_curve.png` |
| V4 — Distance estimation | Done | `~Xm` overlay on video, `distance_m` column in `warnings.csv` |
| V5 — Time-to-collision | Upcoming | TTC from tracked centroid velocity |
| V6 — Lane/zone awareness | Upcoming | Road-area-aware risk boundaries |
| V7 — Publication package | Upcoming | Report, slides, CV bullets, thesis proposal |

## Architecture

```
source video
    │
    ▼
Frame loader (cv2.VideoCapture)
    │
    ▼
YOLO detector  (yolo11s_kitti.pt fine-tuned, or yolo11s.pt pretrained)
    │
    ▼
ByteTrack tracker  (persistent IDs across frames)
    │
    ▼
Risk-zone analyzer  (polygon intersection, ratio-scaled zones)
    │
    ▼
Distance estimator  (D = f_y × H_real / h_bbox, heuristic or KITTI-calibrated)
    │
    ▼
ADAS warning engine  (PEDESTRIAN RISK / CYCLIST RISK / VEHICLE TOO CLOSE / FRONT OBJECT)
    │
    ▼
Annotator  (zone overlay + color-coded bboxes + distance labels + HUD)
    │
    ├──▶  outputs/demo/adas_demo_v1.mp4
    ├──▶  outputs/logs/warnings.csv          (frame, track_id, class, warning, distance_m, bbox)
    ├──▶  outputs/logs/tracking_results.csv
    ├──▶  outputs/logs/frame_timings.csv
    └──▶  outputs/reports/  +  outputs/figures/
```

## Installation

Requires Python 3.10 or 3.11. GPU (CUDA) recommended; CPU fallback works.

```powershell
git clone https://github.com/AmanouNasri1/adas-perception-risk-system.git
cd adas-perception-risk-system
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch — use the official selector for your CUDA version:
# https://pytorch.org/get-started/locally/
# Example for CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install -e .
```

Place a short driving clip at `data/samples/test_video.mp4` (see [data/README.md](data/README.md)).
The fine-tuned KITTI model (`yolo11s_kitti.pt`) is stored on Google Drive — see V3 below. The pretrained `yolo11s.pt` downloads automatically on first run.

## Usage

### Detection (Phase 1)
```powershell
python scripts/run_detection.py --source data/samples/test_video.mp4
```
Output: `outputs/detections/predict/`. Config: `configs/detector.yaml`; all values overridable via CLI.

### Tracking (Phase 2)
```powershell
python scripts/run_tracking.py --source data/samples/test_video.mp4
# BoT-SORT comparison:
python scripts/run_tracking.py --source data/samples/test_video.mp4 --tracker botsort.yaml
```
Output: tracked video + `outputs/logs/tracking_results.csv`.

### ADAS demo with distance estimation (Phase 3 / V4)
```powershell
# Heuristic distance (default focal length from configs/camera.yaml):
python scripts/run_adas_demo.py --source data/samples/test_video.mp4

# Calibration-based distance (KITTI P2 matrix):
python scripts/run_adas_demo.py --source data/samples/test_video.mp4 `
    --calib D:\kitti\training\calib\000042.txt
```
Output: `outputs/demo/adas_demo_v1.mp4`, `outputs/logs/warnings.csv` (with `distance_m`),
`outputs/logs/tracking_results.csv`, `outputs/logs/frame_timings.csv`.

### Metrics (Phase 4)
```powershell
python scripts/compute_metrics.py
```
Output: `outputs/reports/demo_metrics.csv`, `outputs/figures/fps_plot.png`.

### Evaluation (V2)
```powershell
python scripts/run_evaluation.py
```
Output: confidence histogram, ByteTrack/BoT-SORT comparison chart and CSV, three failure-case screenshots.

### KITTI label conversion (V3 prep)
```powershell
python scripts/kitti_to_yolo.py `
    --kitti-images D:\kitti\training\image_2 `
    --kitti-labels D:\kitti\training\label_2 `
    --out D:\kitti\kitti_yolo
```
Output: YOLO-format dataset at `--out`, ready to upload to Google Drive for Colab fine-tuning.

## Dataset

See [data/README.md](data/README.md). KITTI data and model weights are stored on an external HDD and Google Drive — not committed to this repo.

## Results

### V1 baseline — pretrained `yolo11s.pt` on `test_video.mp4` (640×360, 900 frames, RTX 3050)

| Metric | Value |
|--------|-------|
| Average FPS | 52.0 |
| Median FPS | 55.0 |
| p5 FPS (worst 5%) | 32.5 |
| Avg YOLO inference | 13.3 ms/frame |
| Avg annotation time | 3.6 ms/frame |
| Unique track IDs | 126 |
| Average track length | 27.9 frames |
| Total warning events | 1330 (577/900 frames) |

### V3 — KITTI fine-tuning (`yolo11s_kitti.pt`)

Fine-tuned on 7481 KITTI images (80/20 split, 50 epochs, batch 16, imgsz 640) in Google Colab on a T4 GPU. Weights on Google Drive — not in this repo.

| Metric | Value |
|--------|-------|
| Best mAP@0.5 | **0.922** (epoch 49) |
| Best mAP@0.5:0.95 | **0.700** (epoch 50) |
| Final precision | 0.924 |
| Final recall | 0.855 |
| Training time | 116 min (Colab T4) |

Training curve: `outputs/figures/kitti_training_curve.png`.

### V4 — Distance estimation on `test_video.mp4` (1280×720, 4967 frames, `yolo11s_kitti.pt`)

| Metric | Value |
|--------|-------|
| Average FPS | 33.8 |
| Avg YOLO inference | 13.7 ms/frame |
| Distance method | Heuristic (`f_y = 900 px`) |
| Distance range observed | 2.6 – 55.4 m (mean 15.6 m) |
| Total warning events | 6087 |
| FRONT OBJECT | 3637 |
| VEHICLE TOO CLOSE | 2329 |
| PEDESTRIAN RISK | 116 |
| CYCLIST RISK | 5 |

### Tracker comparison (ByteTrack vs BoT-SORT)

| Metric | ByteTrack | BoT-SORT |
|--------|-----------|---------|
| Unique track IDs (fewer = stable) | 126 | **115** |
| Avg track length (frames) | 27.9 | **31.8** |
| Median track length (frames) | 6 | **9** |
| Tracks ≥ 10 frames | 51 | **53** |

BoT-SORT shows better stability; ByteTrack is kept as default for speed and as the V3–V5 baseline.

## Limitations

Three documented failure cases measured on the V1 baseline video.

**1. FRONT OBJECT fires almost continuously.**
The trapezoidal front zone covers the full lane width at the bottom of the frame, so any vehicle ahead triggers it regardless of distance. V4 now provides distance estimates; V5 TTC will allow gating warnings by approach velocity.

**2. PEDESTRIAN RISK fires for distant pavement pedestrians.**
The risk zone uses image coordinates only, with no depth cue. A pedestrian at 20 m triggers the same warning as one at 3 m. Fix: apply a minimum bbox-size gate or scale zone boundaries by distance estimate.

**3. VEHICLE TOO CLOSE misclassifies large stationary vehicles.**
The bbox-height heuristic fires when a large bus/truck fills the frame even when stationary. V4 provides explicit distance; V5 TTC (approach velocity) will distinguish stationary from approaching targets.

## Roadmap

| Phase | Status | Scope |
|-------|--------|-------|
| V1 (Phases 1–4) | Done | Detection + tracking + ADAS warnings + metrics |
| V2 | Done | Evaluation: confidence histograms, tracker comparison, failure cases |
| V3 | Done | KITTI fine-tuning in Colab |
| V4 | Done | Monocular distance estimation (heuristic + calib) |
| V5 | Upcoming | Time-to-collision from tracked centroid velocity history |
| V6 | Upcoming | Lane/road-zone-aware risk boundaries |
| V7 | Upcoming | Technical report, slides, CV bullets, thesis proposal |

Full 12-week plan: `docs/ADAS_Perception_Risk_System_Project_Plan.pdf` §6.

## References

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/)
- [PyTorch — Get Started Locally](https://pytorch.org/get-started/locally/)
- Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022.
