from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


from . import audio
from .stream import _count


def vad_config(values: dict | None = None) -> dict:
    """Validate energy hysteresis and integer speech/hangover durations."""
    result = dict(on_threshold=0.02, off_threshold=0.01, min_frames=2, hangover=2)
    if values is not None:
        if not isinstance(values, dict) or set(values) - set(result):
            raise ValueError("unknown VAD fields")
        result.update(values)
    for key in ["on_threshold", "off_threshold"]:
        if (
            isinstance(result[key], bool)
            or not isinstance(result[key], (int, float))
            or not math.isfinite(result[key])
        ):
            raise ValueError("thresholds must be finite numbers")
    if not 0 <= result["off_threshold"] <= result["on_threshold"] or result["on_threshold"] <= 0:
        raise ValueError("invalid hysteresis thresholds")
    _count(result["min_frames"], maximum=1000)
    _count(result["hangover"], maximum=1000)
    return result


class EnergyVAD:
    """Offline energy baseline with hysteresis, minimum speech and terminal flush."""

    def __init__(self, settings: dict | None = None):
        self.config = vad_config(settings)
        self.index = self.run = self.silence = 0
        self.start: int | None = None
        self.closed = False

    def feed(self, level: float) -> list[dict]:
        if self.closed:
            raise RuntimeError("VAD stream is closed")
        if not np.isfinite(level) or level < 0:
            raise ValueError("energy must be finite and nonnegative")
        events = []
        if self.start is None:
            self.run = self.run + 1 if level >= self.config["on_threshold"] else 0
            if self.run >= self.config["min_frames"]:
                self.start = self.index - self.run + 1
                events.append({"kind": "start", "frame": self.start})
        else:
            self.silence = self.silence + 1 if level <= self.config["off_threshold"] else 0
            if self.silence >= self.config["hangover"]:
                events.append({"kind": "end", "frame": self.index - self.silence + 1})
                self.start, self.run, self.silence = None, 0, 0
        self.index += 1
        return events

    def flush(self) -> list[dict]:
        if self.closed:
            return []
        result = (
            [{"kind": "end", "frame": self.index - self.silence}] if self.start is not None else []
        )
        self.closed = True
        return result
