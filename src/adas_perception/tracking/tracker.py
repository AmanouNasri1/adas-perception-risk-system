import csv
from pathlib import Path
from ultralytics import YOLO


def run_tracking(
    source: str,
    model_name: str,
    tracker: str,
    conf: float,
    iou: float,
    imgsz: int,
    class_names: list,
    out_dir: Path,
    log_path: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    name_to_idx = {v: k for k, v in model.names.items()}
    classes = [name_to_idx[n] for n in class_names if n in name_to_idx] or None

    results = model.track(
        source=source,
        tracker=tracker,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=classes,
        save=True,
        project=str(out_dir),
        name="track",
        exist_ok=True,
        stream=True,
        persist=True,
    )

    track_ids_seen: set[int] = set()
    rows_written = 0
    frames_processed = 0

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "track_id", "class_id", "class_name",
            "confidence", "x1", "y1", "x2", "y2",
        ])
        for frame_idx, result in enumerate(results):
            frames_processed += 1
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                cls_id = int(box.cls.item())
                conf_val = float(box.conf.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                writer.writerow([
                    frame_idx, track_id, cls_id, names[cls_id],
                    f"{conf_val:.4f}",
                    f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                ])
                track_ids_seen.add(track_id)
                rows_written += 1

    return {
        "frames_processed": frames_processed,
        "unique_tracks": len(track_ids_seen),
        "rows_written": rows_written,
        "log_path": log_path,
        "out_dir": out_dir,
    }
