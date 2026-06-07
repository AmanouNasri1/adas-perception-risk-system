import cv2
import numpy as np

# BGR colors
_FRONT_COLOR   = (0, 200, 255)   # orange
_PED_CYC_COLOR = (0,  40, 220)   # red
_ZONE_ALPHA    = 0.28

_BOX_COLORS = {
    "TTC WARNING":       (0,   0, 255),   # pure red — highest severity (V5)
    "PEDESTRIAN RISK":   (0,  40, 220),   # red
    "CYCLIST RISK":      (0,  40, 220),
    "VEHICLE TOO CLOSE": (0, 120, 255),   # amber
    "FRONT OBJECT":      (0, 200, 255),   # orange
    None:                (0, 200,   0),   # green
}


_LANE_LINE_COLOR = (0, 0, 255)   # red — YOLOPv2 lane lines
_DRIVABLE_COLOR  = (0, 180, 0)   # green — YOLOPv2 drivable area


def _fill_poly(frame: np.ndarray, poly: list, color: tuple, alpha: float = _ZONE_ALPHA) -> None:
    pts = np.array(poly, np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], True, color, 2)


def draw_pedcyc_zone(frame: np.ndarray, ped_cyc_poly: list) -> None:
    _fill_poly(frame, ped_cyc_poly, _PED_CYC_COLOR)


def draw_zones(
    frame: np.ndarray,
    front_poly: list,
    ped_cyc_poly: list,
) -> None:
    _fill_poly(frame, ped_cyc_poly, _PED_CYC_COLOR)
    _fill_poly(frame, front_poly,   _FRONT_COLOR)


def draw_lane_overlay(
    frame: np.ndarray,
    drivable_mask: np.ndarray,
    lane_mask: np.ndarray,
    polygon: list | None = None,
    draw_drivable: bool = True,
    draw_lane_lines: bool = True,
) -> None:
    """Draw the YOLOPv2 drivable-area fill, lane-line pixels, and dynamic zone outline (V6)."""
    if draw_drivable and drivable_mask is not None:
        overlay = frame.copy()
        overlay[drivable_mask > 0] = _DRIVABLE_COLOR
        cv2.addWeighted(overlay, _ZONE_ALPHA, frame, 1 - _ZONE_ALPHA, 0, frame)
    if draw_lane_lines and lane_mask is not None:
        frame[lane_mask > 0] = _LANE_LINE_COLOR
    if polygon is not None and len(polygon) >= 3:
        cv2.polylines(frame, [np.array(polygon, np.int32)], True, _FRONT_COLOR, 2)


def draw_box(
    frame: np.ndarray,
    box: tuple,
    track_id: int,
    class_name: str,
    warning: str | None,
    distance_m: float | None = None,
    ttc_s: float | None = None,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    color = _BOX_COLORS.get(warning, _BOX_COLORS[None])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"#{track_id} {class_name}"
    if distance_m is not None:
        label += f" ~{distance_m:.0f}m"
    if warning == "TTC WARNING" and ttc_s is not None:
        label += f" | TTC ~{ttc_s:.1f}s"
    elif warning:
        label += f" | {warning}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.42, 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

    # Place label above bbox if room, else just inside top edge.
    if y1 - th - 8 >= 0:
        bg_y1, text_y = y1 - th - 8, y1 - 4
    else:
        bg_y1, text_y = y1, y1 + th + 4

    bg_x2 = min(x1 + tw + 4, frame.shape[1] - 1)
    cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y1 + th + 6), color, -1)
    cv2.putText(frame, label, (x1 + 2, text_y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    fps: float | None,
    active_warnings: list[str],
    nearest_m: float | None = None,
    ttc_alerts: list[tuple[int, float]] | None = None,
) -> None:
    lines: list[tuple[str, tuple]] = []
    lines.append((f"Frame {frame_idx}", (220, 220, 220)))
    if fps is not None:
        lines.append((f"FPS  {fps:.1f}", (220, 220, 220)))
    if nearest_m is not None:
        lines.append((f"Nearest ~{nearest_m:.0f} m (approx)", (50, 210, 255)))
    # Per-track imminent-collision alerts (V5), sorted soonest-first, in bright red.
    for tid, ttc_s in sorted(ttc_alerts or [], key=lambda a: a[1]):
        lines.append((f"TTC #{tid} ~{ttc_s:.1f}s", (0, 0, 255)))
    for w in active_warnings:
        lines.append((f"!! {w}", (60, 60, 255)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 22
    for text, color in lines:
        # Dark outline for readability on any background.
        cv2.putText(frame, text, (10, y), font, 0.52, (0, 0, 0),   3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), font, 0.52, color,        1, cv2.LINE_AA)
        y += 22
