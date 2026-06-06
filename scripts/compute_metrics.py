"""
Reads outputs/logs/frame_timings.csv, tracking_results.csv, warnings.csv and produces:
  outputs/reports/demo_metrics.csv
  outputs/figures/fps_plot.png
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl


TIMINGS_PATH  = Path("outputs/logs/frame_timings.csv")
TRACKING_PATH = Path("outputs/logs/tracking_results.csv")
WARNINGS_PATH = Path("outputs/logs/warnings.csv")
METRICS_OUT   = Path("outputs/reports/demo_metrics.csv")
FPS_PLOT_OUT  = Path("outputs/figures/fps_plot.png")


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = arr[max(0, i - window + 1): i + 1].mean()
    return out


def main():
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    FPS_PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

    timings  = pl.read_csv(TIMINGS_PATH)
    tracking = pl.read_csv(TRACKING_PATH)
    warnings = pl.read_csv(WARNINGS_PATH)

    # ── FPS / timing metrics ──────────────────────────────────────────────────
    total_ms_arr   = timings["total_ms"].to_numpy()
    inf_ms_arr     = timings["inference_ms"].to_numpy()
    annot_ms_arr   = timings["annotation_ms"].to_numpy()
    fps_per_frame  = 1000.0 / np.maximum(total_ms_arr, 0.01)

    n_frames     = len(timings)
    avg_fps      = float(np.mean(fps_per_frame))
    median_fps   = float(np.median(fps_per_frame))
    p5_fps       = float(np.percentile(fps_per_frame, 5))   # worst-case
    avg_inf_ms   = float(np.mean(inf_ms_arr))
    avg_annot_ms = float(np.mean(annot_ms_arr))
    avg_total_ms = float(np.mean(total_ms_arr))

    # ── Tracking metrics ──────────────────────────────────────────────────────
    unique_tracks    = int(tracking["track_id"].n_unique())
    track_lengths    = tracking.group_by("track_id").len()
    avg_track_len    = float(track_lengths["len"].mean())
    median_track_len = float(track_lengths["len"].median())
    max_track_len    = int(track_lengths["len"].max())

    class_counts = (
        tracking.group_by("class_name").len()
        .sort("len", descending=True)
    )

    # ── Warning metrics ───────────────────────────────────────────────────────
    total_warnings = len(warnings)
    frames_with_warnings = int(warnings["frame"].n_unique())
    warn_counts = (
        warnings.group_by("warning_type").len()
        .sort("len", descending=True)
    )
    # Average duration per (track_id, warning_type) pair, then mean by type.
    duration_by_pair = warnings.group_by(["track_id", "warning_type"]).len()
    avg_warn_dur = (
        duration_by_pair.group_by("warning_type")
        .agg(pl.col("len").mean().alias("avg_duration_frames"))
        .sort("warning_type")
    )

    # ── Build metrics CSV ─────────────────────────────────────────────────────
    rows: list[tuple[str, str]] = [
        ("frames_processed",         str(n_frames)),
        ("avg_fps",                  f"{avg_fps:.2f}"),
        ("median_fps",               f"{median_fps:.2f}"),
        ("p5_fps_worst_5pct",        f"{p5_fps:.2f}"),
        ("avg_inference_ms",         f"{avg_inf_ms:.2f}"),
        ("avg_annotation_ms",        f"{avg_annot_ms:.2f}"),
        ("avg_total_frame_ms",       f"{avg_total_ms:.2f}"),
        ("unique_track_ids",         str(unique_tracks)),
        ("avg_track_length_frames",  f"{avg_track_len:.1f}"),
        ("median_track_length",      f"{median_track_len:.0f}"),
        ("max_track_length_frames",  str(max_track_len)),
        ("total_warning_events",     str(total_warnings)),
        ("frames_with_warnings",     str(frames_with_warnings)),
    ]
    for class_name, count in class_counts.iter_rows():
        rows.append((f"detections_{class_name}", str(count)))
    for warn_type, count in warn_counts.iter_rows():
        rows.append((f"warnings_{warn_type.replace(' ', '_')}", str(count)))
    for warn_type, avg_dur in avg_warn_dur.iter_rows():
        rows.append((f"avg_duration_frames_{warn_type.replace(' ', '_')}", f"{avg_dur:.1f}"))

    with METRICS_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    print(f"Saved: {METRICS_OUT}  ({len(rows)} metrics)")

    # ── FPS plot ──────────────────────────────────────────────────────────────
    frames    = timings["frame"].to_numpy()
    fps_roll  = rolling_mean(fps_per_frame, window=30)
    fps_std   = rolling_mean(
        np.array([fps_per_frame[max(0, i-29):i+1].std() for i in range(n_frames)]),
        window=1,
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(frames, fps_roll - fps_std, fps_roll + fps_std,
                    alpha=0.18, color="#1f77b4", label="_nolegend_")
    ax.plot(frames, fps_roll, color="#1f77b4", linewidth=1.4,
            label=f"Rolling avg FPS (window=30)")
    ax.axhline(avg_fps, color="#1f77b4", linestyle="--", linewidth=1.0,
               label=f"Mean {avg_fps:.1f} FPS")
    ax.axhline(30,      color="#2ca02c", linestyle=":",  linewidth=1.2,
               label="30 FPS reference")
    ax.axhline(25,      color="#ff7f0e", linestyle=":",  linewidth=1.2,
               label="25 FPS reference")

    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("FPS", fontsize=11)
    ax.set_title(
        f"Processing speed — ADAS demo  |  {avg_fps:.1f} avg FPS  |  "
        f"{avg_inf_ms:.1f} ms avg inference  |  {n_frames} frames",
        fontsize=11,
    )
    ax.set_xlim(0, n_frames - 1)
    ax.set_ylim(0, min(fps_per_frame.max() * 1.15, 120))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="major", linewidth=0.5, alpha=0.5)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FPS_PLOT_OUT, dpi=150)
    plt.close(fig)
    print(f"Saved: {FPS_PLOT_OUT}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n--- Runtime ---")
    print(f"  Frames          : {n_frames}")
    print(f"  Avg FPS         : {avg_fps:.1f}  (median {median_fps:.1f}, p5 {p5_fps:.1f})")
    print(f"  Avg inference   : {avg_inf_ms:.1f} ms")
    print(f"  Avg annotation  : {avg_annot_ms:.1f} ms")
    print(f"\n--- Tracking ---")
    print(f"  Unique tracks   : {unique_tracks}")
    print(f"  Avg track len   : {avg_track_len:.1f} frames  (max {max_track_len})")
    print(f"\n--- Class counts ---")
    for name, cnt in class_counts.iter_rows():
        print(f"  {name:12s}  : {cnt}")
    print(f"\n--- Warnings ---")
    print(f"  Total events    : {total_warnings}  across {frames_with_warnings} frames")
    for wt, cnt in warn_counts.iter_rows():
        print(f"  {wt:20s}: {cnt}")
    print(f"\n--- Warning duration (avg frames per track per warning type) ---")
    for wt, dur in avg_warn_dur.iter_rows():
        print(f"  {wt:20s}: {dur:.1f} frames")


if __name__ == "__main__":
    main()
