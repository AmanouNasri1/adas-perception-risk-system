"""
Time-to-collision (TTC) estimation (V5).

TTC = distance / approach_speed, where approach_speed = -d(distance)/dt is how fast an
object's monocular distance (V4) is shrinking. We keep a short ring buffer of
(time, distance) samples per track_id and fit the closing speed by least squares, which
is far less noisy than differencing two consecutive frames.

Time is video time (frame_idx / source_fps), not wall-clock, so TTC stays correct
regardless of how fast the pipeline processes frames.

Estimates inherit the approximation of the underlying distance model and are labelled
accordingly; not suitable for safety decisions.
"""
from __future__ import annotations
from collections import deque


class TTCEstimator:
    """Per-track closing-speed and TTC estimator backed by ring buffers."""

    def __init__(
        self,
        history_size: int = 10,
        min_samples: int = 5,
        min_approach_speed_mps: float = 0.5,
        max_ttc_s: float = 60.0,
    ) -> None:
        self.history_size = history_size
        self.min_samples = min_samples
        self.min_approach_speed_mps = min_approach_speed_mps
        self.max_ttc_s = max_ttc_s
        self._history: dict[int, deque[tuple[float, float]]] = {}

    def update(self, track_id: int, distance_m: float | None, t_s: float) -> float | None:
        """
        Record a (time, distance) sample for a track and return the current TTC in
        seconds, or None when not yet computable (too few samples, or the object is
        stationary / receding).
        """
        if distance_m is None:
            return None

        hist = self._history.get(track_id)
        if hist is None:
            hist = deque(maxlen=self.history_size)
            self._history[track_id] = hist
        hist.append((t_s, distance_m))

        if len(hist) < self.min_samples:
            return None

        approach_speed = self._approach_speed(hist)
        if approach_speed < self.min_approach_speed_mps:
            return None

        ttc = distance_m / approach_speed
        return round(min(ttc, self.max_ttc_s), 1)

    @staticmethod
    def _approach_speed(hist: deque[tuple[float, float]]) -> float:
        """
        Least-squares slope of distance vs time; approach speed = -slope (m/s).
        Positive means the object is closing in, negative means receding.
        """
        n = len(hist)
        mean_t = sum(t for t, _ in hist) / n
        mean_d = sum(d for _, d in hist) / n
        num = sum((t - mean_t) * (d - mean_d) for t, d in hist)
        den = sum((t - mean_t) ** 2 for t, _ in hist)
        if den == 0:
            return 0.0
        slope = num / den
        return -slope

    def prune(self, active_ids: set[int]) -> None:
        """Forget tracks no longer present to bound memory on long videos."""
        for tid in [t for t in self._history if t not in active_ids]:
            del self._history[tid]
