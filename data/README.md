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
