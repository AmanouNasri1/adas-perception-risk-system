from pathlib import Path
import argparse
from adas_perception.utils.config import load_config
from adas_perception.tracking.tracker import run_tracking


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO tracking — config-driven with CLI overrides")
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--tracker", default=None,
                        help="Override tracker type: bytetrack.yaml or botsort.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--out", default="outputs/tracking")
    parser.add_argument("--log", default="outputs/logs/tracking_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    det_cfg = load_config(args.config)
    trk_cfg = load_config(args.tracker_config)

    model_name = args.model or det_cfg["model"]
    conf = args.conf if args.conf is not None else det_cfg["conf"]
    iou = args.iou if args.iou is not None else det_cfg["iou"]
    imgsz = args.imgsz if args.imgsz is not None else det_cfg["imgsz"]
    tracker = args.tracker or trk_cfg["tracker_type"]
    class_names = det_cfg.get("classes_of_interest", [])

    out_dir = Path(args.out).resolve()
    log_path = Path(args.log).resolve()

    summary = run_tracking(
        source=args.source,
        model_name=model_name,
        tracker=tracker,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        class_names=class_names,
        out_dir=out_dir,
        log_path=log_path,
    )

    print(f"\nTracking complete ({tracker})")
    print(f"  Frames processed : {summary['frames_processed']}")
    print(f"  Unique track IDs : {summary['unique_tracks']}")
    print(f"  CSV rows written : {summary['rows_written']}")
    print(f"  Video saved to   : {summary['out_dir'] / 'track'}")
    print(f"  CSV saved to     : {summary['log_path']}")


if __name__ == "__main__":
    main()
