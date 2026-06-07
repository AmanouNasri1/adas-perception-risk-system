"""
Monocular distance estimation via bounding-box height (V4).

Formula: D = f_y * H_real / h_bbox_pixels
  f_y        — camera focal length in pixels (from KITTI P2 or camera.yaml default)
  H_real     — known real-world object height in metres (per class, from camera.yaml)
  h_bbox_px  — bounding-box pixel height = y2 - y1

Outputs are approximate and labelled accordingly; not suitable for safety decisions.
"""
from __future__ import annotations
from pathlib import Path


def parse_kitti_calib(calib_path: str | Path) -> dict[str, float]:
    """
    Read a KITTI calibration .txt and return intrinsics from the P2 matrix.
    Returns {'fx': float, 'fy': float, 'cx': float, 'cy': float}.
    """
    with open(calib_path) as f:
        for line in f:
            if not line.startswith("P2:"):
                continue
            nums = list(map(float, line.strip().split()[1:]))
            # P2 layout: [fx 0 cx tx | 0 fy cy ty | 0 0 1 tz]
            return {
                "fx": nums[0],
                "fy": nums[5],
                "cx": nums[2],
                "cy": nums[6],
            }
    raise ValueError(f"P2 line not found in {calib_path}")


def estimate_distance(
    bbox: tuple,
    class_name: str,
    fy: float,
    known_heights: dict[str, float],
    min_dist: float = 2.0,
    max_dist: float = 80.0,
) -> float | None:
    """
    Return estimated distance in metres, or None if not computable.
    Rounds to one decimal place.
    """
    h_real = known_heights.get(class_name)
    if h_real is None:
        return None
    _, y1, _, y2 = bbox
    h_px = y2 - y1
    if h_px < 2:
        return None
    d = fy * h_real / h_px
    if d < min_dist or d > max_dist:
        return None
    return round(d, 1)
