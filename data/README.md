# Data

This repository does not include full datasets or sample videos.

## Local sample videos

Place short driving clips in `data/samples/` for quick debugging. These files are gitignored and must be obtained separately.

```
data/
  samples/
    test_video.mp4    # short driving clip for development
  kitti/              # gitignored — see below
    images/
    labels/
```

The detection command expects `data/samples/test_video.mp4`:

```powershell
python scripts/run_detection.py --source data/samples/test_video.mp4
```

## KITTI dataset (later phases)

Used for fine-tuning (Phase V3) and evaluation. Store on an external HDD or Google Drive — do not commit to this repo.

- [KITTI Object Detection](https://www.cvlibs.net/datasets/kitti/eval_object.php)
- [KITTI Object Tracking](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)

If your dataset lives on an external drive, point to it via CLI:

```powershell
python scripts/run_detection.py --source D:\kitti\images\val\
```

## Model weights (gitignored)

Weights are never committed. Store them on the external HDD and point to them via config or CLI.

| Weight | Used by | Default path | Source |
|--------|---------|--------------|--------|
| `yolo11s_kitti.pt` | Detector (V3 fine-tuned) | `D:\kitti\weights\yolo11s_kitti.pt` (`configs/detector.yaml`) | Trained in Colab (V3); falls back to `yolo11s.pt` |
| `yolopv2.pt` | Lane/road-area awareness (V6) | `D:\kitti\weights\yolopv2.pt` (`configs/lane.yaml`) | [YOLOPv2 release](https://github.com/CAIC-AD/YOLOPv2) — TorchScript `yolopv2.pt` (~149 MB) |

YOLOPv2 is a panoptic driving-perception model (detection + drivable-area + lane-line
segmentation); V6 uses only its drivable-area and lane-line heads. Download `yolopv2.pt`
from the official release and place it at the path above, or override with
`--lane-weights <path>`. If the file is absent, the demo logs a notice and runs with the
static front zone (`--no-lane` forces this).
