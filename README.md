# Real-Time ADAS Perception and Risk Warning System

Computer-vision pipeline: YOLO object detection → multi-object tracking → risk-zone analysis → ADAS warnings on driving video, producing an annotated demo video, frame-level CSV logs, and runtime metrics.

## Overview

| Phase | Status | Key output |
|-------|--------|------------|
| Phase 1 — Detection | Done | `outputs/detections/predict/` |
| Phase 2 — Tracking  | Done | `outputs/logs/tracking_results.csv` |
| Phase 3 — ADAS warning engine | Done | `outputs/logs/warnings.csv`, `outputs/demo/adas_demo_v1.mp4` |
| Phase 4 — Metrics & polish | Done | `outputs/reports/demo_metrics.csv`, `outputs/figures/fps_plot.png` |
| V2 — Evaluation | Done | `outputs/figures/confidence_hist.png`, `outputs/figures/tracker_comparison.png`, `outputs/figures/failure_case_*.png`, `outputs/reports/tracker_comparison.csv` |

## Architecture

```
source video
    │
    ▼
Frame loader (cv2.VideoCapture)
    │
    ▼
YOLO detector  (yolo11s.pt, 6-class COCO subset)
    │
    ▼
ByteTrack tracker  (persistent IDs across frames)
    │
    ▼
Risk-zone analyzer  (polygon intersection, ratio-scaled zones)
    │
    ▼
ADAS warning engine  (PEDESTRIAN RISK / CYCLIST RISK / VEHICLE TOO CLOSE / FRONT OBJECT)
    │
    ▼
Annotator  (zone overlay + color-coded bboxes + HUD)
    │
    ├──▶  outputs/demo/adas_demo_v1.mp4
    ├──▶  outputs/logs/warnings.csv
    ├──▶  outputs/logs/tracking_results.csv
    └──▶  outputs/reports/demo_metrics.csv + outputs/figures/fps_plot.png
```

## Installation

Requires Python 3.10 or 3.11. GPU (CUDA) recommended; CPU fallback works.

```powershell
git clone <repo-url>
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

## Usage

### Phase 1 — Detection

```powershell
python scripts/run_detection.py --source data/samples/test_video.mp4
```

Output: `outputs/detections/predict/`. Config lives in `configs/detector.yaml`; all values
are overridable via CLI flags (`--model`, `--conf`, `--iou`, `--imgsz`).

### Phase 2 — Tracking

```powershell
python scripts/run_tracking.py --source data/samples/test_video.mp4
# BoT-SORT comparison:
python scripts/run_tracking.py --source data/samples/test_video.mp4 --tracker botsort.yaml
```

Output: tracked video + `outputs/logs/tracking_results.csv`.

### Phase 3 — ADAS demo

```powershell
python scripts/run_adas_demo.py --source data/samples/test_video.mp4
```

Output: `outputs/demo/adas_demo_v1.mp4`, `outputs/logs/warnings.csv`,
`outputs/logs/tracking_results.csv`, `outputs/logs/frame_timings.csv`.

### Phase 4 — Metrics

```powershell
python scripts/compute_metrics.py
```

Output: `outputs/reports/demo_metrics.csv`, `outputs/figures/fps_plot.png`.

### V2 — Evaluation

```powershell
python scripts/run_evaluation.py
```

Output: confidence histogram per class, ByteTrack/BoT-SORT comparison chart and CSV,
and three annotated failure-case screenshots extracted from the demo video.

## Dataset

See [data/README.md](data/README.md).

## Results

Measured on `test_video.mp4` (640×360, 29.97 fps, 900 frames) with `yolo11s.pt` on an RTX 3050.

### Runtime

| Metric | Value |
|--------|-------|
| Average FPS | 52.0 |
| Median FPS | 55.0 |
| p5 FPS (worst 5%) | 32.5 |
| Avg YOLO inference | 13.3 ms/frame |
| Avg annotation time | 3.6 ms/frame |

### Tracking (ByteTrack)

| Metric | Value |
|--------|-------|
| Unique track IDs | 126 |
| Average track length | 27.9 frames |
| Longest track | 270 frames |

**ByteTrack vs BoT-SORT quantitative comparison** (`outputs/reports/tracker_comparison.csv`):

| Metric | ByteTrack | BoT-SORT |
|--------|-----------|---------|
| Unique track IDs (fewer = stable) | 126 | **115** |
| Total detections | 3517 | **3660** |
| Avg track length (frames) | 27.9 | **31.8** |
| Median track length (frames) | 6 | **9** |
| Tracks ≥ 10 frames | 51 | **53** |
| Longest track (frames) | 270 | 271 |

BoT-SORT wins on all stability metrics for this clip. ByteTrack is kept as default because it is
marginally faster and is the established baseline for later V3–V5 tracker comparisons.

### Detections

| Class | Detections |
|-------|-----------|
| person | 1918 |
| car | 1229 |
| truck | 263 |
| bus | 68 |
| bicycle | 38 |
| motorcycle | 1 |

### ADAS Warnings

| Warning type | Events | Avg duration (frames) |
|--------------|--------|----------------------|
| FRONT OBJECT | 619 | 20.6 |
| PEDESTRIAN RISK | 526 | 15.9 |
| VEHICLE TOO CLOSE | 174 | 14.5 |
| CYCLIST RISK | 11 | 3.7 |

Warnings fired in 577 of 900 frames (64%). The high rate is expected at V1 and reflects
the image-coordinate zone design without distance estimation — see Limitations.

## Limitations

Three documented failure cases (per §10.4 / §17.1).

**1. FRONT OBJECT fires almost continuously.**
The trapezoidal front zone covers the full driving-lane width at the bottom of the frame,
so any vehicle visible ahead triggers it regardless of real distance. 619 FRONT OBJECT events
in 900 frames means the warning is effectively always active, making it meaningless as an
alert. Fix: distance estimation (V4) or tighter zone narrowed to only the immediate path.

**2. PEDESTRIAN RISK fires for distant pedestrians on the pavement.**
The risk zone uses image coordinates only, with no depth cue. A pedestrian walking along the
far pavement at 20 m distance enters the polygon and fires a warning identical to one at 3 m.
526 events from 1918 person detections (27% trigger rate). Fix: scale zone boundaries by
approximate depth, or apply a minimum bbox-size gate.

**3. VEHICLE TOO CLOSE misclassifies large stationary vehicles.**
The heuristic (`bbox_height ≥ 25% of frame height AND bottom_y ≥ 65% of frame height`)
fires when a large bus or truck fills the frame even if it is not on a collision course.
174 events despite only 68 bus detections in the clip. Fix: time-to-collision using tracked
centroid velocity (V5), or add a minimum confidence and trajectory filter.

## Roadmap

| Phase | Scope |
|-------|-------|
| V2 | Evaluation: mAP, confidence histograms, tracker comparison metrics |
| V3 | KITTI fine-tuning in Colab |
| V4 | Improved distance estimation from KITTI calibration or bbox heuristics |
| V5 | Time-to-collision from tracked object velocity |
| V6 | Lane/road-zone-aware risk boundaries |
| V7 | Technical report PDF, slides, CV bullets, thesis proposal |

Full 12-week plan: `docs/ADAS_Perception_Risk_System_Project_Plan.pdf` §6.

## References

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/)
- [PyTorch — Get Started Locally](https://pytorch.org/get-started/locally/)
- Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022.
