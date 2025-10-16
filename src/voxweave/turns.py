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
