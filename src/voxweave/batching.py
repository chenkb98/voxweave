from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


from . import audio
from .stream import _count


def pad_audio_batch(clips: list[ArrayLike]) -> dict:
    """Pad variable-length clips with explicit lengths and validity masks."""
    if not isinstance(clips, list) or not 1 <= len(clips) <= 1024:
        raise ValueError("batch needs 1..1024 clips")
    arrays = [audio.as_audio(clip) for clip in clips]
    channels = arrays[0].shape[1]
    if any(value.shape[1] != channels for value in arrays):
        raise ValueError("batch channel counts differ")
    lengths = np.array([len(value) for value in arrays], dtype=np.int64)
    maximum = int(lengths.max())
    if len(arrays) * maximum * channels > 10000000:
        raise ValueError("padded batch is too large")
    samples = np.zeros((len(arrays), maximum, channels))
    for i, value in enumerate(arrays):
        samples[i, : len(value)] = value
    return {
        "samples": samples,
        "lengths": lengths,
        "mask": np.arange(maximum)[None, :] < lengths[:, None],
    }


def attention_mask(lengths: ArrayLike, maximum: int | None = None) -> NDArray[np.bool_]:
    """Build a boolean sequence mask without accepting fractional lengths."""
    values = np.asarray(lengths)
    if values.dtype.kind not in "iu" or values.ndim != 1 or np.any(values < 0):
        raise ValueError("lengths must be a nonnegative integer vector")
    required = int(values.max(initial=0))
    maximum = required if maximum is None else _count(maximum, True)
    if maximum < required or len(values) * maximum > 10000000:
        raise ValueError("invalid mask width or allocation")
    return np.arange(maximum)[None, :] < values[:, None]
