# CV bullets — ADAS Perception and Risk Warning System

Tailored phrasings of the same project for different contexts. Pick the set that fits the
role; numbers are measured on a 1280×720 / 30 fps dashcam clip (4967 frames) on an RTX 3050.

## Concise (3 bullets, general SWE/CV CV)

- Built a real-time ADAS perception pipeline (YOLO11 detection → ByteTrack multi-object
  tracking → YOLOPv2 lane/road segmentation → risk-zone, monocular-distance and
  time-to-collision reasoning → tiered ADAS warnings) on driving video, with annotated
  output, frame-level CSV logs, and FPS/latency metrics; **34 FPS** baseline, **15.6 FPS**
  with deep lane segmentation.
- Fine-tuned YOLO11s on KITTI in Colab to **mAP@0.5 = 0.922** (P = 0.92, R = 0.86), and
  added monocular distance (`D = f_y·H/h`) and least-squares time-to-collision estimation
  with a per-track history buffer.
- Drove down false positives with measured engineering: an *in-path* gate cut spurious
  TTC alerts by **75%**, and a YOLOPv2 drivable-area dynamic front zone cut FRONT-OBJECT
  false positives by **67%** vs a fixed-polygon baseline — each change quantified by A/B.

## Detailed (portfolio / thesis-foundation)

- Designed and implemented a modular, config-driven ADAS perception pipeline in Python
  (Ultralytics YOLO, OpenCV, PyTorch): detection → tracking → lane/road-area segmentation
  → risk-zone analysis → distance estimation → time-to-collision → tiered warning engine,
  producing annotated video, `tracking_results.csv`, `warnings.csv`, and per-frame timing logs.
- Fine-tuned a YOLO11s detector on the KITTI benchmark (50 epochs, Colab T4) reaching
  **mAP@0.5 = 0.922 / mAP@0.5:0.95 = 0.700**, and benchmarked ByteTrack vs BoT-SORT for
  track-ID stability.
- Implemented monocular distance estimation (bounding-box-height heuristic and KITTI-P2
  calibrated focal length) and a time-to-collision module using a per-track distance
  ring buffer with a least-squares closing-speed fit; surfaced a Priority-5 collision
  warning tier.
- Added lane/road-area awareness with the YOLOPv2 panoptic model: the segmented ego-drivable
  area becomes a dynamic front risk zone (EMA-smoothed, with automatic fallback to a static
  zone at intersections), reducing front-object false positives by **67%**.
- Emphasised honest, reproducible evaluation throughout: every claim ("real-time", "fewer
  false positives") is backed by measured FPS/latency and A/B warning statistics, with
  documented failure cases and screenshots.

## One-liner (LinkedIn headline / elevator)

Real-time ADAS perception pipeline (detection, tracking, lane segmentation, distance,
time-to-collision, risk warnings) on driving video — built as clean, measured engineering
and a bachelor-thesis foundation.

## Skills evidenced

Computer vision · deep learning (YOLO, YOLOPv2) · multi-object tracking · monocular geometry ·
PyTorch · OpenCV · Python packaging · config-driven design · reproducible evaluation ·
KITTI · Google Colab · Git.
