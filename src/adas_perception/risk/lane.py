"""
Lane / road-area awareness (V6) via the YOLOPv2 panoptic driving-perception model.

YOLOPv2 is a TorchScript model that, for a letterboxed RGB frame, returns:
  (detection, drivable_area_seg, lane_line_seg)
We ignore the detection branch (YOLO already handles detection) and use:
  - drivable_area_seg : (1, 2, H, W) already-softmaxed class probabilities; channel 1
                        is the ego drivable area. This becomes the dynamic FRONT zone.
  - lane_line_seg     : (1, 1, H, W) already-activated lane-line probability, drawn
                        for context.

A per-frame EMA on the drivable probability stabilises the zone against flicker. When
the drivable coverage in the lower band of the frame is too low to trust (e.g. at an
intersection with no clear ego lane), `process()` returns ok=False and the caller falls
back to the static front zone in configs/risk_zones.yaml.

Outputs are approximate and labelled accordingly; not suitable for safety decisions.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass
class LaneResult:
    ok: bool                     # True if segmentation is confident enough to use
    polygon: list | None         # largest drivable contour, full-frame (x, y) ints, or None
    drivable_mask: np.ndarray    # full-frame uint8 {0, 1}
    lane_mask: np.ndarray        # full-frame uint8 {0, 1}
    coverage: float              # drivable fraction within the lower-ROI band


class LaneSegmenter:
    """YOLOPv2 drivable-area + lane-line segmentation with temporal smoothing (V6)."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "cuda",
        half: bool = True,
        input_height: int = 384,
        input_width: int = 640,
        drivable_threshold: float = 0.5,
        lane_threshold: float = 0.5,
        ema_alpha: float = 0.4,
        roi_bottom_fraction: float = 0.5,
        min_drivable_coverage: float = 0.05,
    ) -> None:
        if not Path(weights).exists():
            raise FileNotFoundError(f"YOLOPv2 weights not found: {weights}")
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.half = bool(half) and self.device == "cuda"
        self.model = torch.jit.load(str(weights), map_location=self.device).eval()
        if self.half:
            self.model = self.model.half()
        self.in_h, self.in_w = input_height, input_width
        self.da_thr = drivable_threshold
        self.ll_thr = lane_threshold
        self.alpha = ema_alpha
        self.roi_bottom_fraction = roi_bottom_fraction
        self.min_cov = min_drivable_coverage
        self._ema: np.ndarray | None = None  # EMA of drivable prob at model resolution

        # Warm up CUDA (JIT specialization + cuDNN autotune) so the first real frame
        # isn't a multi-second outlier that skews the timing metrics.
        self._warmup()

    @torch.no_grad()
    def _warmup(self, iters: int = 2) -> None:
        dummy = torch.zeros(1, 3, self.in_h, self.in_w, device=self.device)
        dummy = dummy.half() if self.half else dummy.float()
        for _ in range(iters):
            self.model(dummy)
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _letterbox(self, img: np.ndarray, color=(114, 114, 114)):
        h, w = img.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (self.in_h - nh) // 2
        bottom = self.in_h - nh - top
        left = (self.in_w - nw) // 2
        right = self.in_w - nw - left
        out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return out, left, top

    @torch.no_grad()
    def process(self, frame: np.ndarray) -> LaneResult:
        h, w = frame.shape[:2]
        lb, padx, pady = self._letterbox(frame)
        x = torch.from_numpy(lb[:, :, ::-1].transpose(2, 0, 1).copy()).to(self.device)
        x = (x.half() if self.half else x.float()).div(255.0).unsqueeze(0)
        _, da, ll = self.model(x)
        da_prob = da.float()[0, 1].cpu().numpy()   # drivable-class probability
        ll_prob = ll.float()[0, 0].cpu().numpy()   # lane-line probability

        # Temporal EMA on the drivable probability map (model resolution).
        if self._ema is None or self._ema.shape != da_prob.shape:
            self._ema = da_prob
        else:
            self._ema = self.alpha * da_prob + (1.0 - self.alpha) * self._ema
        da_bin = (self._ema > self.da_thr).astype(np.uint8)
        ll_bin = (ll_prob > self.ll_thr).astype(np.uint8)

        # Undo the letterbox (crop padding, scale back to the full frame).
        def unpad(m: np.ndarray) -> np.ndarray:
            crop = m[pady:self.in_h - pady if pady else self.in_h,
                     padx:self.in_w - padx if padx else self.in_w]
            return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)

        da_full = unpad(da_bin)
        ll_full = unpad(ll_bin)

        # Confidence: drivable fraction within the lower band of the frame.
        roi_y = int(h * (1.0 - self.roi_bottom_fraction))
        roi = da_full[roi_y:, :]
        coverage = float(roi.mean()) if roi.size else 0.0

        polygon = self._largest_polygon(da_full) if coverage >= self.min_cov else None
        ok = polygon is not None
        return LaneResult(ok, polygon, da_full, ll_full, coverage)

    @staticmethod
    def _largest_polygon(mask: np.ndarray) -> list | None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 1.0:
            return None
        eps = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 3:
            return None
        return [(int(p[0][0]), int(p[0][1])) for p in approx]
