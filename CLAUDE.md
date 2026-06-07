# CLAUDE.md

This file guides Claude Code when working in this repository.

## Project

Real-Time ADAS Perception and Risk Warning System: a computer-vision pipeline that runs YOLO object detection → multi-object tracking → risk-zone analysis → ADAS warnings on driving video, producing annotated video plus CSV logs and runtime metrics. It is a portfolio and bachelor-thesis-foundation project, so the value comes from clean, measured engineering, not a flashy demo.

The full plan is at `docs/ADAS_Perception_Risk_System_Project_Plan.pdf`. That PDF is the specification — consult it for detail (section numbers are referenced below). This file is the working contract for *how* to build it.

Current state: V1–V7 are fully built and committed — the roadmap is complete. The repo is clean and all audit issues from session 1 are resolved. Further work is thesis-grade evaluation (see `docs/thesis_proposal.md`), not new pipeline phases.

## Completed phases (do not re-audit or re-implement)

- **Phase 1 (Detection):** `scripts/run_detection.py` reads `configs/detector.yaml`, outputs to `outputs/detections/predict/`.
- **Phase 2 (Tracking):** `scripts/run_tracking.py`, ByteTrack default, BoT-SORT flag. Outputs `outputs/logs/tracking_results.csv`.
- **Phase 3 (ADAS demo):** `scripts/run_adas_demo.py` — full pipeline with zone drawing, warnings, annotated video, and per-frame timing.
- **Phase 4 (Metrics):** `scripts/compute_metrics.py` — FPS plot and metrics CSV.
- **V2 (Evaluation):** `scripts/run_evaluation.py` — confidence histogram, tracker comparison, failure-case screenshots.
- **V3 (KITTI fine-tuning):** Completed in Google Colab. `scripts/kitti_to_yolo.py` converts labels. `configs/kitti.yaml` is the Colab dataset config. Fine-tuned weights at `D:\kitti\weights\yolo11s_kitti.pt` (external HDD, gitignored). Best mAP@0.5 = 0.922 after 50 epochs.
- **V4 (Distance estimation):** `src/adas_perception/risk/distance.py` — `D = f_y × H_real / h_bbox`. `configs/camera.yaml` holds default focal length and per-class heights. `--calib` flag accepts a KITTI P2 file. Distance shown on bboxes and in HUD; `distance_m` column added to `warnings.csv`.
- **V5 (Time-to-collision):** `src/adas_perception/risk/ttc.py` — `TTCEstimator` keeps a per-track ring buffer of `(video_time, distance)` and fits closing speed by least squares; `TTC = distance / approach_speed`. `configs/ttc.yaml` holds the threshold and buffer params. TTC is a Priority-5 warning that *escalates* an existing in-path zone warning (gated by `require_in_path`, which suppresses the monocular false positive of roadside objects the ego passes). `ttc_s` column added to `warnings.csv`; shown on the bbox and as per-track HUD alerts.
- **V6 (Lane/road-area awareness):** `src/adas_perception/risk/lane.py` — `LaneSegmenter` wraps the **YOLOPv2** TorchScript model (drivable-area + lane-line segmentation). The EMA-smoothed drivable area becomes the dynamic FRONT zone (largest contour → polygon), replacing the static trapezoid; falls back to the static zone when drivable coverage is low (intersections). `configs/lane.yaml` holds weights path, device, fp16, thresholds, smoothing, fallback. Weights `D:\kitti\weights\yolopv2.pt` (external HDD, gitignored). Flags: `--no-lane`, `--lane-weights`. Adds `lane_ms` to `frame_timings.csv`. Measured: 98% lane-detection rate, FRONT OBJECT false positives cut 67% (3637→1215), 34→15.6 FPS.
- **V7 (Publication package):** Reproducible doc generators — `scripts/build_report.py` (reportlab) → `docs/technical_report.pdf` (6 pages, structure per PDF §15.3), `scripts/build_slides.py` (python-pptx) → `docs/slides.pptx` (8 slides per §15.4). Plus `docs/cv_bullets.md` and `docs/thesis_proposal.md`. Regenerate with `python scripts/build_report.py` / `build_slides.py`. Note: no PDF/PPTX renderer in this environment, so QA was content/geometry-based, not pixel-level.

## Environment and commands (Windows 11 / PowerShell)

- Activate the venv each session: `.\.venv\Scripts\Activate.ps1`
- Python is 3.10/3.11 for PyTorch + CUDA compatibility; don't switch interpreters casually.
- Install PyTorch from the official selector (https://pytorch.org/get-started/locally/) — that is the source of truth, never hardcode a wheel/CUDA version. GPU is an RTX 3050; CPU fallback is fine for development.
- Detection: `python scripts/run_detection.py --source data/samples/test_video.mp4`
- Tracking: `python scripts/run_tracking.py --source data/samples/test_video.mp4`
- ADAS demo (main script): `python scripts/run_adas_demo.py --source data/samples/test_video.mp4`
- ADAS demo with KITTI calib: add `--calib D:\kitti\training\calib\000042.txt`
- Metrics: `python scripts/compute_metrics.py`
- Evaluation: `python scripts/run_evaluation.py`
- Active model: `D:\kitti\weights\yolo11s_kitti.pt` (fine-tuned, set in `configs/detector.yaml`). Use `yolo11s.pt` if the HDD path is unavailable.
- Active classes: car, truck, bus, motorcycle (person and bicycle removed from `configs/detector.yaml` — intentional for V4 vehicle-distance focus).
- Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Training runs in Google Colab, not locally.

## Architecture

Build it as one pipeline, not disconnected scripts:

frame loader → YOLO detector → tracker (ByteTrack now, BoT-SORT later) → lane/road-area segmentation (YOLOPv2, dynamic front zone) → risk-zone analyzer → approximate distance estimator → time-to-collision → ADAS warning engine → annotated video + CSV logs + metrics/plots.

Reusable logic lives in the package `src/adas_perception/` (submodules `detection/`, `tracking/`, `risk/`, `visualization/`, `utils/`); `scripts/` holds thin entrypoints that import from it. Generated artifacts go under `outputs/{detections,tracking,demo,logs,figures,reports}/` and are gitignored.

## Roadmap — one phase at a time

Each phase has explicit acceptance criteria in the PDF; meet them before moving on.

- **Phase 1–4 and V2–V7:** Done. See "Completed phases" above. The roadmap is complete.
- **Beyond the roadmap:** thesis-grade evaluation (ground-truth distance/TTC/in-path metrics on KITTI) per `docs/thesis_proposal.md`; optional ONNX/fp16 deployment study; optional ROS 2 / CARLA integration.

## Conventions

- Config-driven: thresholds, zones, model names, and class lists live in `configs/*.yaml`, not hardcoded in scripts.
- No absolute paths in committed code; use relative paths and config.
- Honest metrics: measure FPS and inference time, and never call the system "real-time" without numbers. Save screenshots of failure cases.
- Git: never commit datasets, weights, or output videos. Commit messages describe engineering milestones ("Add object tracking and CSV logging"), not feelings ("finally works").
- KITTI: controlled layers only, stored on external HDD/Drive, never in the repo; keep `data/README.md` explaining how to obtain it.
- Test each new feature on one image, one short clip, and one driving sequence before calling it done.

## Definition of done — V1 (PDF §17)

Detection, tracking, and ADAS demo scripts all run from the command line; the annotated demo video exists; `tracking_results.csv` and `warnings.csv` exist; average FPS is reported; the README covers installation and usage; at least three failure cases are documented; and the repo is clean (no datasets or heavy weights).
