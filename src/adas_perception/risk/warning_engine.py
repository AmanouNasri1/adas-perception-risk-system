from __future__ import annotations

# Warning priority — higher number = higher severity.
PRIORITY: dict[str, int] = {
    "PEDESTRIAN RISK": 4,
    "CYCLIST RISK":    4,
    "VEHICLE TOO CLOSE": 3,
    "FRONT OBJECT":    1,
}

VEHICLE_CLASSES: frozenset[str] = frozenset({"car", "truck", "bus", "motorcycle"})

# Suppress PEDESTRIAN RISK for a person box that overlaps a bicycle box above this IoU.
# Rationale: a cycling person is detected as both "person" and "bicycle"; we emit CYCLIST only.
BICYCLE_PERSON_OVERLAP_IOU = 0.30


def load_zones(cfg: dict, frame_w: int, frame_h: int) -> tuple[list, list, dict]:
    """Scale ratio-based zone config to pixel coordinates for a given frame size."""
    def scale(pts: list) -> list[tuple[int, int]]:
        return [(int(x * frame_w), int(y * frame_h)) for x, y in pts]

    front = scale(cfg["front_zone"]["points_ratio"])
    ped_cyc = scale(cfg["pedestrian_cyclist_zone"]["points_ratio"])
    close_cfg = cfg["close_vehicle"]
    return front, ped_cyc, close_cfg


def bbox_bottom_center(box: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def point_in_polygon(point: tuple, polygon: list) -> bool:
    """Ray-casting test (even-odd rule)."""
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def get_warnings(
    class_name: str,
    box: tuple,
    frame_h: int,
    front_poly: list,
    ped_cyc_poly: list,
    close_cfg: dict,
) -> list[str]:
    warnings: list[str] = []
    bottom = bbox_bottom_center(box)

    if point_in_polygon(bottom, front_poly):
        warnings.append("FRONT OBJECT")

    if class_name == "person":
        if point_in_polygon(bottom, ped_cyc_poly):
            warnings.append("PEDESTRIAN RISK")
    elif class_name == "bicycle":
        if point_in_polygon(bottom, ped_cyc_poly):
            warnings.append("CYCLIST RISK")

    if class_name in VEHICLE_CLASSES:
        x1, y1, x2, y2 = box
        height_ratio = (y2 - y1) / frame_h
        bottom_ratio = y2 / frame_h
        if (
            height_ratio >= close_cfg["min_bbox_height_ratio"]
            and bottom_ratio >= close_cfg["min_bottom_y_ratio"]
        ):
            warnings.append("VEHICLE TOO CLOSE")

    return warnings


def top_warning(warnings: list[str]) -> str | None:
    if not warnings:
        return None
    return max(warnings, key=lambda w: PRIORITY.get(w, 0))


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)
