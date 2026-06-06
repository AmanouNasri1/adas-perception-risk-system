import csv
import time
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
    p.add_argument("--source",        required=True)
    p.add_argument("--config",        default="configs/detector.yaml")
    p.add_argument("--tracker-config",default="configs/tracker.yaml")
    p.add_argument("--zone-config",   default="configs/risk_zones.yaml")
    p.add_argument("--model",         default=None)
    p.add_argument("--conf",          type=float, default=None)
    p.add_argument("--iou",           type=float, default=None)
    p.add_argument("--imgsz",         type=int,   default=None)
    p.add_argument("--out-video",     default="outputs/demo/adas_demo_v1.mp4")
    p.add_argument("--out-warnings",  default="outputs/logs/warnings.csv")
    p.add_argument("--out-tracking",  default="outputs/logs/tracking_results.csv")
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

    # Read source video properties for VideoWriter.
    cap = cv2.VideoCapture(args.source)
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps  = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    front_poly, ped_cyc_poly, close_cfg = load_zones(zone_cfg, frame_w, frame_h)

    out_video    = Path(args.out_video).resolve()
    out_warnings = Path(args.out_warnings).resolve()
    out_tracking = Path(args.out_tracking).resolve()
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_warnings.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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
        save=False,   # we write our own annotated video
        stream=True,
        persist=True,
    )

    frame_idx     = 0
    warning_rows  = 0
    t_start       = time.perf_counter()

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
            frame = result.orig_img.copy()
            names = result.names

            # Collect all valid tracked boxes this frame.
            boxes_this_frame: list[tuple] = []
            if result.boxes is not None:
                for box in result.boxes:
                    if box.id is None:
                        continue
                    tid     = int(box.id.item())
                    cls_id  = int(box.cls.item())
                    cls_nm  = names[cls_id]
                    conf_v  = float(box.conf.item())
                    coords  = tuple(float(v) for v in box.xyxy[0].tolist())
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

                # Person overlapping a bicycle → emit CYCLIST RISK, not PEDESTRIAN RISK.
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

            fps = (frame_idx + 1) / (time.perf_counter() - t_start)
            active_warnings.sort(key=lambda w: -PRIORITY.get(w, 0))
            draw_hud(frame, frame_idx, fps, active_warnings)

            vwriter.write(frame)
            frame_idx += 1

    vwriter.release()

    avg_fps = frame_idx / (time.perf_counter() - t_start)
    print(f"\nADAS demo complete")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Average FPS      : {avg_fps:.1f}")
    print(f"  Warning rows     : {warning_rows}")
    print(f"  Demo video       : {out_video}")
    print(f"  Warnings CSV     : {out_warnings}")
    print(f"  Tracking CSV     : {out_tracking}")


if __name__ == "__main__":
    main()
