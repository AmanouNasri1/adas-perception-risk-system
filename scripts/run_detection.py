from pathlib import Path
import argparse
from ultralytics import YOLO
from adas_perception.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO detection — config-driven with CLI overrides")
    parser.add_argument("--source", required=True, help="Image, video, webcam index, or folder")
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--model", default=None, help="Override config model")
    parser.add_argument("--conf", type=float, default=None, help="Override config conf threshold")
    parser.add_argument("--iou", type=float, default=None, help="Override config IoU threshold")
    parser.add_argument("--imgsz", type=int, default=None, help="Override config image size")
    parser.add_argument("--out", default="outputs/detections")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_name = args.model or cfg["model"]
    conf = args.conf if args.conf is not None else cfg["conf"]
    iou = args.iou if args.iou is not None else cfg["iou"]
    imgsz = args.imgsz if args.imgsz is not None else cfg["imgsz"]
    class_names = cfg.get("classes_of_interest", [])

    model = YOLO(model_name)
    # Resolve class names → COCO indices via the model's own name table.
    name_to_idx = {v: k for k, v in model.names.items()}
    classes = [name_to_idx[n] for n in class_names if n in name_to_idx] or None

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=args.source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=classes,
        save=True,
        project=str(out_dir),
        name="predict",
        exist_ok=True,
    )
    print(f"Finished. Results saved under: {out_dir / 'predict'}")
    print(f"Frames processed: {len(results)}")


if __name__ == "__main__":
    main()
