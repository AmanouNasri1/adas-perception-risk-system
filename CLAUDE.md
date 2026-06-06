# CLAUDE.md

This file guides Claude Code when working in this repository.

## Project

Real-Time ADAS Perception and Risk Warning System: a computer-vision pipeline that runs YOLO object detection → multi-object tracking → risk-zone analysis → ADAS warnings on driving video, producing annotated video plus CSV logs and runtime metrics. It is a portfolio and bachelor-thesis-foundation project, so the value comes from clean, measured engineering, not a flashy demo.

The full plan is at `docs/ADAS_Perception_Risk_System_Project_Plan.pdf`. That PDF is the specification — consult it for detail (section numbers are referenced below). This file is the working contract for *how* to build it.

Current state: Phase 1 detection has been run — `scripts/run_detection.py` was executed on a sample video. Phases 2–4 are not built yet.

## Before any new work: audit and reconcile the repo

The PDF was written as a plan in advance, so the code on disk may have drifted from it or contain bugs that only surface at runtime. On your first task in this repo, run the audit below and fix what is broken **before** starting the next phase. Summarize what you changed and why.

1. **Read the actual repo first.** Print the tree and read every file under `scripts/`, `src/`, and `configs/`. Compare with the intended structure in PDF §7.8 and note anything missing (folders, `__init__.py` files, configs).
2. **COCO class names.** The pretrained YOLO model emits COCO names: `person, bicycle, car, motorcycle, bus, truck`. It never emits `pedestrian`, `cyclist`, or `van`. The warning engine (§10.3) references those three, which are dead branches that silently never fire. Confirm the intended mapping (`person` → pedestrian risk, `bicycle` → cyclist risk) and remove or comment the dead names so the logic isn't misleading. Note that a cycling person is detected as both `person` and `bicycle`, so the same human can trigger two warnings — decide how to handle that.
3. **Config vs. code.** `scripts/run_detection.py` (§8.2) takes CLI args but never loads `configs/detector.yaml`. Pick one source of truth — config-driven is preferred — and make scripts read the YAML with CLI flags as overrides.
4. **Risk-zone resolution.** `configs/risk_zones.yaml` (§10.1) hardcodes polygon points for 1280×720, but the sample video may have a different resolution, which misaligns the zones. Scale zones to the actual frame size at runtime, or define them as 0–1 ratios.
5. **Repo hygiene.** Confirm `.gitignore` (§7.7) exists and that no model weights (`*.pt`/`*.onnx`/`*.engine`), datasets, or output videos are tracked. Flag anything heavy that is already staged or committed.
6. **Reproducibility.** Confirm `requirements.txt` exists and matches the installed packages, and that the README records the exact detection command that worked (§8.4).

Proceed to phase work only once the audit is clean or the fixes are committed.

## Environment and commands (Windows 11 / PowerShell)

- Activate the venv each session: `.\.venv\Scripts\Activate.ps1`
- Python is 3.10/3.11 for PyTorch + CUDA compatibility; don't switch interpreters casually.
- Install PyTorch from the official selector (https://pytorch.org/get-started/locally/) — that is the source of truth, never hardcode a wheel/CUDA version. GPU is an RTX 3050; CPU fallback is fine for development.
- Detection: `python scripts/run_detection.py --source data/samples/test_video.mp4`
- Tracking (Phase 2): `python scripts/run_tracking.py --source data/samples/test_video.mp4 --tracker bytetrack.yaml`
- Models: `yolo11s.pt` by default, `yolo11n.pt` if inference is slow.
- Verify GPU when needed: `python -c "import torch; print(torch.cuda.is_available())"`
- Training runs in Google Colab, not locally, and only after V1 is stable.

## Architecture

Build it as one pipeline, not disconnected scripts:

frame loader → YOLO detector → tracker (ByteTrack now, BoT-SORT later) → risk-zone analyzer → approximate distance estimator → ADAS warning engine → annotated video + CSV logs + metrics/plots.

Reusable logic lives in the package `src/adas_perception/` (submodules `detection/`, `tracking/`, `risk/`, `visualization/`, `utils/`); `scripts/` holds thin entrypoints that import from it. Generated artifacts go under `outputs/{detections,tracking,demo,logs,figures,reports}/` and are gitignored.

## Roadmap — one phase at a time

Each phase has explicit acceptance criteria in the PDF; meet them before moving on.

- **Phase 1 — Detection** (done; verify during the audit): YOLO on an image and a video, annotated output saved (§8.4).
- **Phase 2 — Tracking:** `scripts/run_tracking.py` with stable IDs across frames and `outputs/logs/tracking_results.csv` (frame, track_id, class_id, class_name, confidence, x1, y1, x2, y2). Compare ByteTrack vs. BoT-SORT qualitatively (§9.4).
- **Phase 3 — ADAS warning engine:** implement `src/adas_perception/risk/warning_engine.py` and `scripts/run_adas_demo.py`; draw the front zone and pedestrian/cyclist zone, fire warnings, and log `outputs/logs/warnings.csv`. False positives are expected — document them, don't hide them (§10.4).
- **Phase 4 — Demo polish and metrics:** `outputs/demo/adas_demo_v1.mp4` with FPS and warning overlays, an FPS plot at `outputs/figures/fps_plot.png`, `outputs/reports/demo_metrics.csv`, and README v1 (§11).
- **Later (V2–V7, §6):** evaluation, KITTI fine-tuning in Colab, distance estimation, time-to-collision, lane-aware risk, then the publication package. Do not pull these into V1.

## Conventions

- Config-driven: thresholds, zones, model names, and class lists live in `configs/*.yaml`, not hardcoded in scripts.
- No absolute paths in committed code; use relative paths and config.
- Honest metrics: measure FPS and inference time, and never call the system "real-time" without numbers. Save screenshots of failure cases.
- Git: never commit datasets, weights, or output videos. Commit messages describe engineering milestones ("Add object tracking and CSV logging"), not feelings ("finally works").
- KITTI: controlled layers only, stored on external HDD/Drive, never in the repo; keep `data/README.md` explaining how to obtain it.
- Test each new feature on one image, one short clip, and one driving sequence before calling it done.

## Definition of done — V1 (PDF §17)

Detection, tracking, and ADAS demo scripts all run from the command line; the annotated demo video exists; `tracking_results.csv` and `warnings.csv` exist; average FPS is reported; the README covers installation and usage; at least three failure cases are documented; and the repo is clean (no datasets or heavy weights).
