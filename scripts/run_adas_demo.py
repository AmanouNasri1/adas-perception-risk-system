import csv
import time
from collections import deque
import cv2
from pathlib import Path
import argparse
from ultralytics import YOLO

from adas_perception.utils.config import load_config
from adas_perception.risk.warning_engine import (
    load_zones, get_warnings, top_warning, compute_iou,
    PRIORITY, BICYCLE_PERSON_OVERLAP_IOU, TTC_WARNING,
)
from adas_perception.risk.distance import parse_kitti_calib, estimate_distance
from adas_perception.risk.ttc import TTCEstimator
from adas_perception.visualization.annotator import draw_zones, draw_box, draw_hud


def parse_args():
    p = argparse.ArgumentParser(description="ADAS demo — tracking + risk zones + warnings + distance")
    p.add_argument("--source",         required=True)
    p.add_argument("--config",         default="configs/detector.yaml")
    p.add_argument("--tracker-config", default="configs/tracker.yaml")
    p.add_argument("--zone-config",    default="configs/risk_zones.yaml")
    p.add_argument("--camera-config",  default="configs/camera.yaml")
    p.add_argument("--ttc-config",     default="configs/ttc.yaml")
    p.add_argument("--calib",          default=None,
                   help="KITTI calib .txt for calibration-based f_y (optional)")
    p.add_argument("--model",          default=None)
    p.add_argument("--conf",           type=float, default=None)
    p.add_argument("--iou",            type=float, default=None)
    p.add_argument("--imgsz",          type=int,   default=None)
    p.add_argument("--out-video",      default="outputs/demo/adas_demo_v1.mp4")
    p.add_argument("--out-warnings",   default="outputs/logs/warnings.csv")
    p.add_argument("--out-tracking",   default="outputs/logs/tracking_results.csv")
    p.add_argument("--out-timings",    default="outputs/logs/frame_timings.csv")
    return p.parse_args()


def main():
    args = parse_args()
    det_cfg  = load_config(args.config)
    trk_cfg  = load_config(args.tracker_config)
    zone_cfg = load_config(args.zone_config)
    cam_cfg  = load_config(args.camera_config)
    ttc_cfg  = load_config(args.ttc_config)

    model_name  = args.model or det_cfg["model"]
    conf        = args.conf  if args.conf  is not None else det_cfg["conf"]
    iou_thresh  = args.iou   if args.iou   is not None else det_cfg["iou"]
    imgsz       = args.imgsz if args.imgsz is not None else det_cfg["imgsz"]
    tracker     = trk_cfg["tracker_type"]
    class_names = det_cfg.get("classes_of_interest", [])

    # Distance estimation setup.
    if args.calib:
        intrinsics = parse_kitti_calib(args.calib)
        fy = intrinsics["fy"]
        dist_mode = f"calibrated (f_y={fy:.1f})"
    else:
        fy = cam_cfg["default_focal_length_y"]
        dist_mode = f"heuristic (f_y={fy:.1f})"
    known_heights = cam_cfg["known_heights_m"]
    min_dist = cam_cfg["min_distance_m"]
    max_dist = cam_cfg["max_distance_m"]

    # Time-to-collision setup (V5).
    ttc_estimator = TTCEstimator(
        history_size=ttc_cfg["history_size"],
        min_samples=ttc_cfg["min_samples"],
        min_approach_speed_mps=ttc_cfg["min_approach_speed_mps"],
        max_ttc_s=ttc_cfg["max_ttc_s"],
    )
    ttc_threshold = ttc_cfg["warn_threshold_s"]
    ttc_in_path_only = ttc_cfg.get("require_in_path", True)

    cap = cv2.VideoCapture(args.source)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # TTC uses video time (frame_idx * dt), independent of processing speed.
    dt_per_frame = 1.0 / src_fps if src_fps and src_fps > 0 else 1.0 / 30.0

    front_poly, ped_cyc_poly, close_cfg = load_zones(zone_cfg, frame_w, frame_h)

    out_video    = Path(args.out_video).resolve()
    out_warnings = Path(args.out_warnings).resolve()
    out_tracking = Path(args.out_tracking).resolve()
    out_timings  = Path(args.out_timings).resolve()
    for p in (out_video, out_warnings, out_timings):
        p.parent.mkdir(parents=True, exist_ok=True)

    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    vwriter = cv2.VideoWriter(str(out_video), fourcc, src_fps, (frame_w, frame_h))

    model = YOLO(model_name)
    name_to_idx = {v: k for k, v in model.names.items()}
    classes = [name_to_idx[n] for n in class_names if n in name_to_idx] or None

    results = model.track(
        source=args.source,
        tracker=tracker,
        conf=conf,
        iou=iou_thresh,
        imgsz=imgsz,
        classes=classes,
        save=False,
        stream=True,
        persist=True,
    )

    fps_buf       = deque(maxlen=30)
    frame_timings = []
    frame_idx     = 0
    warning_rows  = 0
    ttc_warn_rows = 0

    print(f"Distance mode : {dist_mode}")
    print(f"TTC threshold : {ttc_threshold:.1f}s")
    print(f"Source        : {args.source}  ({frame_w}x{frame_h} @ {src_fps:.0f} fps)")

    with (
        out_warnings.open("w", newline="", encoding="utf-8") as wf,
        out_tracking.open("w", newline="", encoding="utf-8") as tf,
    ):
        warn_w = csv.writer(wf)
        warn_w.writerow(["frame", "track_id", "class_name", "warning_type",
                         "distance_m", "ttc_s", "x1", "y1", "x2", "y2"])
        track_w = csv.writer(tf)
        track_w.writerow(["frame", "track_id", "class_id", "class_name",
                           "confidence", "x1", "y1", "x2", "y2"])

        for result in results:
            t_frame = time.perf_counter()
            frame   = result.orig_img.copy()
            names   = result.names
            speed   = result.speed

            boxes_this_frame: list[tuple] = []
            if result.boxes is not None:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    tid    = int(box.id.item())
                    cls_id = int(box.cls.item())
                    cls_nm = names[cls_id]
                    conf_v = float(box.conf.item())
                    coords = tuple(float(v) for v in box.xyxy[0].tolist())
                    boxes_this_frame.append((tid, cls_id, cls_nm, conf_v, coords))
                    track_w.writerow([
                        frame_idx, tid, cls_id, cls_nm,
                        f"{conf_v:.4f}",
                        *[f"{v:.1f}" for v in coords],
                    ])

            bicycle_boxes = [b[4] for b in boxes_this_frame if b[2] == "bicycle"]

            draw_zones(frame, front_poly, ped_cyc_poly)

            active_warnings: list[str] = []
            ttc_alerts: list[tuple[int, float]] = []
            nearest_m: float | None = None
            video_t = frame_idx * dt_per_frame

            for tid, cls_id, cls_nm, conf_v, coords in boxes_this_frame:
                dist_m = estimate_distance(coords, cls_nm, fy, known_heights,
                                           min_dist, max_dist)
                if dist_m is not None:
                    nearest_m = dist_m if nearest_m is None else min(nearest_m, dist_m)

                ttc_s = ttc_estimator.update(tid, dist_m, video_t)

                warns = get_warnings(cls_nm, coords, frame_h,
                                     front_poly, ped_cyc_poly, close_cfg)

                if "PEDESTRIAN RISK" in warns and cls_nm == "person":
                    if any(compute_iou(coords, bb) > BICYCLE_PERSON_OVERLAP_IOU
                           for bb in bicycle_boxes):
                        warns.remove("PEDESTRIAN RISK")
                        warns.append("CYCLIST RISK")

                # TTC escalates an existing in-path warning to imminent. Gating on a
                # prior zone warning suppresses roadside objects the ego merely passes.
                if (ttc_s is not None and ttc_s < ttc_threshold
                        and (warns or not ttc_in_path_only)):
                    warns.append(TTC_WARNING)
                    ttc_alerts.append((tid, ttc_s))

                best = top_warning(warns)
                draw_box(frame, coords, tid, cls_nm, best,
                         distance_m=dist_m, ttc_s=ttc_s)

                dist_str = f"{dist_m:.1f}" if dist_m is not None else ""
                ttc_str  = f"{ttc_s:.1f}"  if ttc_s  is not None else ""
                for w in warns:
                    warn_w.writerow([frame_idx, tid, cls_nm, w, dist_str, ttc_str,
                                     *[f"{v:.1f}" for v in coords]])
                    warning_rows += 1
                    if w == TTC_WARNING:
                        ttc_warn_rows += 1
                    if w not in active_warnings:
                        active_warnings.append(w)

            ttc_estimator.prune({b[0] for b in boxes_this_frame})

            fps_hud = sum(fps_buf) / len(fps_buf) if fps_buf else 0.0
            active_warnings.sort(key=lambda w: -PRIORITY.get(w, 0))
            draw_hud(frame, frame_idx, fps_hud, active_warnings,
                     nearest_m=nearest_m, ttc_alerts=ttc_alerts)
            vwriter.write(frame)

            annotation_ms = (time.perf_counter() - t_frame) * 1000
            total_ms = (speed.get("preprocess", 0.0) + speed.get("inference", 0.0)
                        + speed.get("postprocess", 0.0) + annotation_ms)
            fps_buf.append(1000.0 / total_ms if total_ms > 0 else 0.0)
            frame_timings.append({
                "frame":          frame_idx,
                "preprocess_ms":  round(speed.get("preprocess",  0.0), 2),
                "inference_ms":   round(speed.get("inference",   0.0), 2),
                "postprocess_ms": round(speed.get("postprocess", 0.0), 2),
                "annotation_ms":  round(annotation_ms, 2),
                "total_ms":       round(total_ms, 2),
            })
            frame_idx += 1

    vwriter.release()

    with out_timings.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "frame", "preprocess_ms", "inference_ms",
            "postprocess_ms", "annotation_ms", "total_ms",
        ])
        w.writeheader()
        w.writerows(frame_timings)

    total_ms_sum = sum(t["total_ms"] for t in frame_timings)
    avg_fps = 1000 * frame_idx / total_ms_sum if total_ms_sum > 0 else 0.0
    avg_inf = sum(t["inference_ms"] for t in frame_timings) / frame_idx if frame_idx else 0.0

    print(f"\nADAS demo complete")
    print(f"  Frames processed     : {frame_idx}")
    print(f"  Average FPS          : {avg_fps:.1f}")
    print(f"  Avg inference (ms)   : {avg_inf:.1f}")
    print(f"  Warning rows         : {warning_rows}")
    print(f"  TTC warning rows     : {ttc_warn_rows}")
    print(f"  Distance mode        : {dist_mode}")
    print(f"  Demo video           : {out_video}")
    print(f"  Warnings CSV         : {out_warnings}")
    print(f"  Tracking CSV         : {out_tracking}")
    print(f"  Frame timings CSV    : {out_timings}")


if __name__ == "__main__":
    main()
