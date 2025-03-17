from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_audio(samples: ArrayLike) -> NDArray[np.float64]:
    """Copy finite real audio into a (frames, channels) float64 array."""
    raw = np.asarray(samples)
    if raw.dtype.kind not in "iuf" or raw.ndim not in (1, 2):
        raise ValueError("audio must be a real numeric vector or matrix")
    array = np.array(raw, dtype=np.float64, copy=True)
    if array.ndim == 1:
        array = array[:, None]
    if not 1 <= array.shape[1] <= 8 or not np.isfinite(array).all():
        raise ValueError("audio needs 1..8 channels and finite samples")
    return array


def sample_rate(value: int) -> int:
    """Validate an integral sample rate between 1 Hz and 768 kHz."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("sample rate must be an integer")
    if not 1 <= value <= 768000:
        raise ValueError("sample rate must be in 1..768000")
    return int(value)


def duration(samples: ArrayLike, rate: int) -> float:
    """Return duration in seconds, independent of channel count."""
    return len(as_audio(samples)) / sample_rate(rate)
