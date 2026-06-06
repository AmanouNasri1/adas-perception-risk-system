"""
Convert the KITTI object-detection training split to YOLO format and produce
an 80/20 train/val split ready for Ultralytics fine-tuning.

Class mapping (KITTI → our 6-class YOLO setup):
  Pedestrian / Person_sitting → 0  person
  Cyclist                     → 1  bicycle
  Car / Van                   → 2  car
  Motorcycle                  → 3  motorcycle  (no KITTI source — slot stays empty)
  Tram                        → 4  bus
  Truck                       → 5  truck
  Misc / DontCare             →    skipped

Usage (run from the repo root, venv active):
  python scripts/kitti_to_yolo.py \\
      --kitti-images D:\\kitti\\training\\image_2 \\
      --kitti-labels D:\\kitti\\training\\label_2 \\
      --out D:\\kitti\\kitti_yolo \\
      --val-split 0.2 --seed 42
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


KITTI_TO_CLS = {
    "Pedestrian":    0,
    "Person_sitting":0,
    "Cyclist":       1,
    "Car":           2,
    "Van":           2,
    "Tram":          4,
    "Truck":         5,
}
SKIP_CLASSES = {"Misc", "DontCare"}
CLASS_NAMES  = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def convert_label(label_path: Path, img_w: int, img_h: int) -> list[str]:
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        kitti_cls = parts[0]
        if kitti_cls in SKIP_CLASSES or kitti_cls not in KITTI_TO_CLS:
            continue
        x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        # Clip to image bounds (some KITTI labels extend slightly outside).
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        rows.append(f"{KITTI_TO_CLS[kitti_cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kitti-images", required=True,
                   help="Path to KITTI training/image_2 folder")
    p.add_argument("--kitti-labels", required=True,
                   help="Path to KITTI training/label_2 folder")
    p.add_argument("--out",          required=True,
                   help="Output root (will contain images/ and labels/ sub-trees)")
    p.add_argument("--val-split",    type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    img_src  = Path(args.kitti_images)
    lbl_src  = Path(args.kitti_labels)
    out_root = Path(args.out)

    img_paths = sorted(img_src.glob("*.png"))
    if not img_paths:
        img_paths = sorted(img_src.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No PNG/JPG images found in {img_src}")

    random.seed(args.seed)
    shuffled = img_paths[:]
    random.shuffle(shuffled)
    n_val    = int(len(shuffled) * args.val_split)
    val_set  = set(p.stem for p in shuffled[:n_val])

    stats = {"train": 0, "val": 0, "boxes": 0, "skipped_imgs": 0}

    for img_path in img_paths:
        stem      = img_path.stem
        lbl_path  = lbl_src / f"{stem}.txt"
        split     = "val" if stem in val_set else "train"

        img_out = out_root / "images" / split / img_path.name
        lbl_out = out_root / "labels" / split / f"{stem}.txt"
        img_out.parent.mkdir(parents=True, exist_ok=True)
        lbl_out.parent.mkdir(parents=True, exist_ok=True)

        # Read image dimensions without loading pixel data.
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        yolo_rows = convert_label(lbl_path, img_w, img_h) if lbl_path.exists() else []

        shutil.copy2(img_path, img_out)
        lbl_out.write_text("\n".join(yolo_rows))

        stats[split] += 1
        stats["boxes"] += len(yolo_rows)
        if not yolo_rows:
            stats["skipped_imgs"] += 1

    print(f"\nConversion complete → {out_root}")
    print(f"  Train images  : {stats['train']}")
    print(f"  Val   images  : {stats['val']}")
    print(f"  Total boxes   : {stats['boxes']}")
    print(f"  Images with 0 valid boxes: {stats['skipped_imgs']}")
    print(f"\nNext: upload {out_root} to Google Drive, then run fine-tuning in Colab.")


if __name__ == "__main__":
    main()
