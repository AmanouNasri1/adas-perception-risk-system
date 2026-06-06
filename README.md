# Real-Time ADAS Perception and Risk Warning System

Computer-vision pipeline: YOLO object detection → multi-object tracking → risk-zone analysis → ADAS warnings on driving video.

## Overview

| Phase | Status | Output |
|-------|--------|--------|
| Phase 1 — Detection | Done | `outputs/detections/predict/` |
| Phase 2 — Tracking | Pending | `outputs/logs/tracking_results.csv` |
| Phase 3 — ADAS warning engine | Pending | `outputs/logs/warnings.csv` |
| Phase 4 — Demo polish & metrics | Pending | `outputs/demo/adas_demo_v1.mp4` |

## Installation

Requires Python 3.10 or 3.11. GPU (CUDA) recommended; CPU fallback works.

```powershell
# 1. Clone and activate venv
git clone <repo-url>
cd adas-perception-risk-system
.\.venv\Scripts\Activate.ps1   # or create: py -3.11 -m venv .venv

# 2. Install PyTorch with CUDA — use the official selector:
#    https://pytorch.org/get-started/locally/
#    Example for CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .
```

## Usage

### Phase 1 — Detection

```powershell
python scripts/run_detection.py --source data/samples/test_video.mp4
```

Config lives in `configs/detector.yaml`. CLI flags override config values:

```powershell
python scripts/run_detection.py --source data/samples/test_video.mp4 --conf 0.4 --model yolo11n.pt
```

Annotated output is saved to `outputs/detections/predict/`.

## Dataset

See [data/README.md](data/README.md).

## Results

*(Phase 4 — to be filled)*

## Limitations

*(Phase 4 — to be filled; failure cases will be documented here)*

## Roadmap

See `docs/ADAS_Perception_Risk_System_Project_Plan.pdf` §6 for the full 12-week plan.

## References

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/)
- [PyTorch](https://pytorch.org/)
