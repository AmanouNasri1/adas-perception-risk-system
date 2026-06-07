# Bachelor Thesis Proposal

## Working title

**Real-Time Monocular ADAS Perception and Risk Assessment: Lane-Aware Collision Warning from a Single Forward Camera**

## 1. Background and motivation

Advanced Driver-Assistance Systems (ADAS) reduce collisions by perceiving the road scene
and warning the driver about imminent risk. Production systems fuse radar, lidar and
cameras, but a large fraction of the perceptual signal is recoverable from a **single
forward camera** — making monocular ADAS an attractive, low-cost research setting and a
strong proxy for the perception stack of autonomous vehicles.

This thesis builds on a working engineering foundation (developed as a portfolio project,
phases V1–V6) that already implements a full monocular pipeline: object detection,
multi-object tracking, lane/road-area segmentation, approximate distance estimation,
time-to-collision (TTC), and a tiered warning engine. The proposed thesis turns this
foundation into a measured scientific study of **how far monocular, camera-only ADAS risk
assessment can be pushed**, and where it breaks.

## 2. Problem statement

Monocular ADAS risk warning suffers from three coupled weaknesses:

1. **Depth is ill-posed.** Bounding-box-height distance is biased by object-size priors and
   pitch; calibration helps but does not generalise across cameras.
2. **Risk zones are heuristic.** Fixed image-space polygons mislabel adjacent-lane and
   roadside objects as in-path; learned drivable-area segmentation helps but degrades at
   intersections and in poor conditions.
3. **Warnings are hard to validate.** Without ground truth, false-positive / false-negative
   rates for TTC and front-object warnings are usually reported anecdotally.

**Research question:** *Can a camera-only pipeline produce collision warnings whose
false-positive and false-negative rates are low enough to be useful, and which design
choices (calibrated vs heuristic depth, learned vs fixed risk zones, tracking quality)
contribute most to that performance?*

## 3. Objectives

- **O1.** Quantify monocular distance error against KITTI ground truth (3D boxes / lidar)
  for the heuristic and calibrated estimators, by class and by range.
- **O2.** Evaluate the lane/road-area-aware risk zone against fixed zones using labelled
  in-path / not-in-path object annotations, reporting precision/recall of FRONT-OBJECT.
- **O3.** Validate TTC against ground-truth closing speed on KITTI tracking sequences;
  characterise false positives/negatives across traffic scenarios.
- **O4.** Establish a real-time operating point (accuracy vs FPS) on commodity hardware,
  including model-size and precision (fp16/ONNX) trade-offs.

## 4. Proposed method

Extend the existing pipeline rather than rebuild it:

- **Depth.** Compare the bbox-height heuristic and KITTI-calibrated focal length against a
  learned monocular-depth baseline; analyse residual error vs range/class.
- **Risk zones.** Keep the YOLOPv2 drivable-area dynamic front zone; add an explicit
  intersection/low-confidence detector to characterise (not just trigger) the fallback.
- **TTC.** Retain the least-squares closing-speed estimator; add Kalman smoothing on the
  per-track distance signal and compare.
- **Evaluation harness.** Build a ground-truth-aligned evaluator on KITTI tracking that
  scores distance, in-path classification, and TTC against labels.

## 5. Datasets and tools

- **KITTI** object-detection and tracking benchmarks (already used for V3 fine-tuning;
  calibration files available). Optionally **BDD100K** for weather/lighting diversity.
- **Stack:** PyTorch, Ultralytics YOLO, YOLOPv2, OpenCV, Google Colab (training), local
  RTX 3050 (inference). The existing repo is modular and config-driven.

## 6. Evaluation plan

| Component | Metric | Ground truth |
|-----------|--------|--------------|
| Detection | mAP@0.5, mAP@0.5:0.95, P, R | KITTI labels |
| Tracking | ID switches, track length, MOTA-style | KITTI tracking |
| Distance | MAE / RMSE by class and range | KITTI 3D boxes / lidar |
| Risk zone | FRONT-OBJECT precision / recall | in-path annotations |
| TTC | FP / FN rate, timing error | ground-truth closing speed |
| Runtime | FPS, latency, fp16/ONNX speedup | wall-clock |

## 7. Expected contributions

1. An open, reproducible monocular ADAS pipeline with measured component-level accuracy.
2. A quantified comparison of heuristic vs calibrated depth and fixed vs learned risk zones.
3. A ground-truth-aligned evaluation of monocular TTC warnings across traffic scenarios.
4. A documented real-time operating point and failure-mode analysis on commodity hardware.

## 8. Risks and mitigations

- *Ground-truth alignment effort* → start from KITTI's existing calibration/label tooling.
- *Real-time budget on RTX 3050* → fp16/ONNX, model-size sweep, frame-skip for the lane model.
- *Scope creep* → the V1–V6 pipeline already works; the thesis is evaluation and analysis,
  not a rebuild.

## 9. Indicative timeline (12–14 weeks)

| Weeks | Work |
|-------|------|
| 1–2 | Literature review; KITTI ground-truth evaluation harness |
| 3–5 | Distance error study (O1) |
| 6–8 | Risk-zone evaluation, learned vs fixed (O2) |
| 9–10 | TTC validation across scenarios (O3) |
| 11–12 | Runtime/deployment study (O4); ablations |
| 13–14 | Writing, figures, defence preparation |

## 10. Foundation already in place (V1–V6)

Detection (YOLO11s, KITTI-fine-tuned to mAP@0.5 = 0.922), ByteTrack/BoT-SORT tracking,
YOLOPv2 lane/road-area segmentation, monocular distance, least-squares TTC, and a tiered
warning engine — all config-driven with CSV logging and measured FPS. See the technical
report (`docs/technical_report.pdf`) and the repository README for details.
