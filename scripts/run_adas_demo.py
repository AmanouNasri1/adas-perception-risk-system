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
    PRIORITY, BICYCLE_PERSON_OVERLAP_IOU,
)
from adas_perception.visualization.annotator import draw_zones, draw_box, draw_hud


def parse_args():
    p = argparse.ArgumentParser(description="ADAS demo — tracking + risk zones + warnings")
    p.add_argument("--source",         required=True)
    p.add_argument("--config",         default="configs/detector.yaml")
    p.add_argument("--tracker-config", default="configs/tracker.yaml")
    p.add_argument("--zone-config",    default="configs/risk_zones.yaml")
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

    model_name  = args.model or det_cfg["model"]
    conf        = args.conf  if args.conf  is not None else det_cfg["conf"]
    iou_thresh  = args.iou   if args.iou   is not None else det_cfg["iou"]
    imgsz       = args.imgsz if args.imgsz is not None else det_cfg["imgsz"]
    tracker     = trk_cfg["tracker_type"]
    class_names = det_cfg.get("classes_of_interest", [])

    cap = cv2.VideoCapture(args.source)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

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

    fps_buf       = deque(maxlen=30)   # rolling window for HUD display
    frame_timings = []
    frame_idx     = 0
    warning_rows  = 0

    with (
        out_warnings.open("w", newline="", encoding="utf-8") as wf,
        out_tracking.open("w", newline="", encoding="utf-8") as tf,
    ):
        warn_w = csv.writer(wf)
        warn_w.writerow(["frame", "track_id", "class_name", "warning_type",
                         "x1", "y1", "x2", "y2"])
        track_w = csv.writer(tf)
        track_w.writerow(["frame", "track_id", "class_id", "class_name",
                           "confidence", "x1", "y1", "x2", "y2"])

        for result in results:
            t_frame = time.perf_counter()
            frame   = result.orig_img.copy()
            names   = result.names
            speed   = result.speed   # {'preprocess': ms, 'inference': ms, 'postprocess': ms}

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
            for tid, cls_id, cls_nm, conf_v, coords in boxes_this_frame:
                warns = get_warnings(cls_nm, coords, frame_h,
                                     front_poly, ped_cyc_poly, close_cfg)

                if "PEDESTRIAN RISK" in warns and cls_nm == "person":
                    if any(compute_iou(coords, bb) > BICYCLE_PERSON_OVERLAP_IOU
                           for bb in bicycle_boxes):
                        warns.remove("PEDESTRIAN RISK")
                        warns.append("CYCLIST RISK")

                best = top_warning(warns)
                draw_box(frame, coords, tid, cls_nm, best)

                for w in warns:
                    warn_w.writerow([frame_idx, tid, cls_nm, w,
                                     *[f"{v:.1f}" for v in coords]])
                    warning_rows += 1
                    if w not in active_warnings:
                        active_warnings.append(w)

            # HUD shows rolling FPS from previous frames (one-frame lag — imperceptible).
            fps_hud = sum(fps_buf) / len(fps_buf) if fps_buf else 0.0
            active_warnings.sort(key=lambda w: -PRIORITY.get(w, 0))
            draw_hud(frame, frame_idx, fps_hud, active_warnings)
            vwriter.write(frame)

            # Compute full per-frame time (YOLO stages + annotation + video write).
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

    # Save per-frame timings.
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
    print(f"  Demo video           : {out_video}")
    print(f"  Warnings CSV         : {out_warnings}")
    print(f"  Tracking CSV         : {out_tracking}")
    print(f"  Frame timings CSV    : {out_timings}")


if __name__ == "__main__":
    main()
