"""
V2 evaluation: confidence histogram, ByteTrack vs BoT-SORT comparison,
and failure-case screenshots extracted from the annotated demo video.

Outputs:
  outputs/figures/confidence_hist.png
  outputs/figures/tracker_comparison.png
  outputs/figures/failure_case_1_front_object.png
  outputs/figures/failure_case_2_distant_pedestrian.png
  outputs/figures/failure_case_3_vehicle_too_close.png
  outputs/reports/tracker_comparison.csv
"""
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path


TRACKING_BT  = Path("outputs/logs/tracking_results.csv")
TRACKING_BS  = Path("outputs/logs/tracking_results_botsort.csv")
WARNINGS     = Path("outputs/logs/warnings.csv")
DEMO_VIDEO   = Path("outputs/demo/adas_demo_v1.mp4")
FIGURES_DIR  = Path("outputs/figures")
REPORTS_DIR  = Path("outputs/reports")
CONF_THRESH  = 0.35   # must match configs/detector.yaml


# ── helpers ──────────────────────────────────────────────────────────────────

def save_fig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def tracker_metrics(df: pl.DataFrame) -> dict:
    lengths = df.group_by("track_id").len()
    return {
        "unique_ids":      int(df["track_id"].n_unique()),
        "total_rows":      len(df),
        "avg_len":         float(lengths["len"].mean()),
        "median_len":      float(lengths["len"].median()),
        "long_ge10":       int((lengths["len"] >= 10).sum()),
        "max_len":         int(lengths["len"].max()),
    }


# ── 1. Confidence histogram ───────────────────────────────────────────────────

def make_confidence_hist(df: pl.DataFrame) -> None:
    classes = sorted(df["class_name"].unique().to_list())
    n = len(classes)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3.2), sharey=False)
    axes_flat = axes.flatten()

    for i, cls in enumerate(classes):
        ax = axes_flat[i]
        conf_vals = df.filter(pl.col("class_name") == cls)["confidence"].to_numpy()
        ax.hist(conf_vals, bins=20, range=(CONF_THRESH, 1.0),
                color="#1f77b4", edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(CONF_THRESH, color="#d62728", linestyle="--", linewidth=1.0,
                   label=f"thresh {CONF_THRESH}")
        ax.set_title(f"{cls}  (n={len(conf_vals)})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Confidence score", fontsize=9)
        ax.set_ylabel("Detection count", fontsize=9)
        ax.set_xlim(CONF_THRESH - 0.02, 1.02)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, rows * cols):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Detection confidence distribution by class  (YOLO yolo11s.pt, conf >= 0.35)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, FIGURES_DIR / "confidence_hist.png")


# ── 2. Tracker comparison ─────────────────────────────────────────────────────

def make_tracker_comparison(bt_df: pl.DataFrame, bs_df: pl.DataFrame) -> None:
    bt = tracker_metrics(bt_df)
    bs = tracker_metrics(bs_df)

    # Save CSV
    field_labels = [
        ("unique_track_ids",          "unique_ids"),
        ("total_detection_rows",      "total_rows"),
        ("avg_track_length_frames",   "avg_len"),
        ("median_track_length_frames","median_len"),
        ("tracks_ge_10_frames",       "long_ge10"),
        ("max_track_length_frames",   "max_len"),
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "tracker_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "bytetrack", "botsort", "winner"])
        for label, key in field_labels:
            bv, sv = bt[key], bs[key]
            # For unique IDs: fewer is better (more stable). For everything else: more is better.
            if key == "unique_ids":
                winner = "botsort" if sv < bv else "bytetrack" if bv < sv else "tie"
            else:
                winner = "botsort" if sv > bv else "bytetrack" if bv > sv else "tie"
            w.writerow([label, f"{bv:.1f}", f"{sv:.1f}", winner])
    print(f"  Saved: {REPORTS_DIR / 'tracker_comparison.csv'}")

    # Bar chart
    bar_metrics = [
        ("Unique IDs\n(fewer=stable)", "unique_ids"),
        ("Avg track\nlength (frames)", "avg_len"),
        ("Median track\nlength (frames)", "median_len"),
        ("Tracks >=10\nframes",         "long_ge10"),
    ]
    labels = [m[0] for m in bar_metrics]
    bt_vals = [bt[m[1]] for m in bar_metrics]
    bs_vals = [bs[m[1]] for m in bar_metrics]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, bt_vals, width, label="ByteTrack", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, bs_vals, width, label="BoT-SORT",  color="#ff7f0e")
    ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(
        f"ByteTrack vs BoT-SORT — tracking stability  "
        f"(ByteTrack: {bt['unique_ids']} IDs / BoT-SORT: {bs['unique_ids']} IDs)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    save_fig(fig, FIGURES_DIR / "tracker_comparison.png")


# ── 3. Failure-case screenshots ───────────────────────────────────────────────

def _grab_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def save_failure_frame(cap: cv2.VideoCapture, frame_idx: int,
                       out_path: Path, description: str) -> None:
    frame = _grab_frame(cap, frame_idx)
    if frame is None:
        print(f"  [skip] Could not read frame {frame_idx}")
        return
    # Burn a description label into the screenshot.
    label = f"Failure case: {description}  (frame {frame_idx})"
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 30), (frame.shape[1], h), (30, 30, 30), -1)
    cv2.putText(frame, label, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"  Saved: {out_path}  (frame {frame_idx})")


def make_failure_cases(warn_df: pl.DataFrame) -> None:
    if not DEMO_VIDEO.exists():
        print("  [skip] Demo video not found — skipping failure-case screenshots.")
        return

    cap = cv2.VideoCapture(str(DEMO_VIDEO))
    if not cap.isOpened():
        print("  [skip] Cannot open demo video.")
        return

    # Case 1: FRONT OBJECT fires with no real hazard — pick an early frame with
    # only FRONT OBJECT (no PEDESTRIAN RISK or VEHICLE TOO CLOSE on the same frame).
    front_frames = set(
        warn_df.filter(pl.col("warning_type") == "FRONT OBJECT")["frame"].to_list()
    )
    high_frames = set(
        warn_df.filter(
            pl.col("warning_type").is_in(["PEDESTRIAN RISK", "CYCLIST RISK", "VEHICLE TOO CLOSE"])
        )["frame"].to_list()
    )
    front_only = sorted(front_frames - high_frames)
    f1 = front_only[len(front_only) // 4] if front_only else sorted(front_frames)[0]
    save_failure_frame(cap, f1, FIGURES_DIR / "failure_case_1_front_object.png",
                       "FRONT OBJECT fires without proximity danger")

    # Case 2: PEDESTRIAN RISK on a distant (small) pedestrian — smallest bbox height.
    ped_df = (
        warn_df.filter(pl.col("warning_type") == "PEDESTRIAN RISK")
        .with_columns((pl.col("y2") - pl.col("y1")).alias("bbox_h"))
        .sort("bbox_h")
    )
    f2 = int(ped_df["frame"][len(ped_df) // 5])   # 20th-percentile smallest bbox
    save_failure_frame(cap, f2, FIGURES_DIR / "failure_case_2_distant_pedestrian.png",
                       "PEDESTRIAN RISK fires for distant pedestrian (small bbox)")

    # Case 3: VEHICLE TOO CLOSE for a large bus or truck that is not a collision threat.
    vc_df = warn_df.filter(pl.col("warning_type") == "VEHICLE TOO CLOSE")
    bus_df = vc_df.filter(pl.col("class_name") == "bus")
    target_df = bus_df if len(bus_df) > 0 else vc_df
    f3 = int(target_df["frame"][len(target_df) // 2])
    cls_label = "bus" if len(bus_df) > 0 else "vehicle"
    save_failure_frame(cap, f3, FIGURES_DIR / "failure_case_3_vehicle_too_close.png",
                       f"VEHICLE TOO CLOSE fires for large {cls_label} (heuristic false positive)")

    cap.release()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    bt_df   = pl.read_csv(TRACKING_BT)
    bs_df   = pl.read_csv(TRACKING_BS)
    warn_df = pl.read_csv(WARNINGS)

    print("\n[1] Confidence histogram")
    make_confidence_hist(bt_df)

    print("\n[2] Tracker comparison")
    make_tracker_comparison(bt_df, bs_df)

    print("\n[3] Failure-case screenshots")
    make_failure_cases(warn_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
