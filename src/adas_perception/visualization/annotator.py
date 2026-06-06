import cv2
import numpy as np

# BGR colors
_FRONT_COLOR   = (0, 200, 255)   # orange
_PED_CYC_COLOR = (0,  40, 220)   # red
_ZONE_ALPHA    = 0.28

_BOX_COLORS = {
    "PEDESTRIAN RISK":   (0,  40, 220),   # red
    "CYCLIST RISK":      (0,  40, 220),
    "VEHICLE TOO CLOSE": (0, 120, 255),   # amber
    "FRONT OBJECT":      (0, 200, 255),   # orange
    None:                (0, 200,   0),   # green
}


def draw_zones(
    frame: np.ndarray,
    front_poly: list,
    ped_cyc_poly: list,
) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(ped_cyc_poly, np.int32)], _PED_CYC_COLOR)
    cv2.fillPoly(overlay, [np.array(front_poly,   np.int32)], _FRONT_COLOR)
    cv2.addWeighted(overlay, _ZONE_ALPHA, frame, 1 - _ZONE_ALPHA, 0, frame)
    cv2.polylines(frame, [np.array(ped_cyc_poly, np.int32)], True, _PED_CYC_COLOR, 2)
    cv2.polylines(frame, [np.array(front_poly,   np.int32)], True, _FRONT_COLOR,   2)


def draw_box(
    frame: np.ndarray,
    box: tuple,
    track_id: int,
    class_name: str,
    warning: str | None,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    color = _BOX_COLORS.get(warning, _BOX_COLORS[None])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"#{track_id} {class_name}"
    if warning:
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
) -> None:
    lines: list[tuple[str, tuple]] = []
    lines.append((f"Frame {frame_idx}", (220, 220, 220)))
    if fps is not None:
        lines.append((f"FPS  {fps:.1f}", (220, 220, 220)))
    for w in active_warnings:
        lines.append((f"!! {w}", (60, 60, 255)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 22
    for text, color in lines:
        # Dark outline for readability on any background.
        cv2.putText(frame, text, (10, y), font, 0.52, (0, 0, 0),   3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), font, 0.52, color,        1, cv2.LINE_AA)
        y += 22
