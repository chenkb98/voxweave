from __future__ import annotations

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
    if len(values) == 0:
        if maximum is None:
            return np.empty((0, 0), dtype=bool)
        maximum = _count(maximum, True)
        return np.empty((0, maximum), dtype=bool)
    maximum = required if maximum is None else _count(maximum, True)
    if maximum < required or len(values) * maximum > 10000000:
        raise ValueError("invalid mask width or allocation")
    return np.arange(maximum)[None, :] < values[:, None]


def pack_tokens(tokens: ArrayLike, vocabulary: int = 1024) -> NDArray[np.int64]:
    """Interleave frame/codebook tokens using disjoint vocabulary offsets."""
    values, vocabulary = np.asarray(tokens), _count(vocabulary, maximum=1000000)
    if (
        values.dtype.kind not in "iu"
        or values.ndim != 2
        or not 1 <= values.shape[1] <= 16
        or np.any(values < 0)
        or np.any(values >= vocabulary)
    ):
        raise ValueError("tokens must be bounded integer (frames, codebooks)")
    return (values.astype(np.int64) + np.arange(values.shape[1]) * vocabulary).ravel()


def unpack_tokens(packed: ArrayLike, codebooks: int, vocabulary: int = 1024) -> NDArray[np.int64]:
    """Invert token packing and reject tokens in the wrong codebook position."""
    values = np.asarray(packed)
    codebooks, vocabulary = _count(codebooks, maximum=16), _count(vocabulary, maximum=1000000)
    if values.dtype.kind not in "iu" or values.ndim != 1 or len(values) % codebooks:
        raise ValueError("invalid packed token shape")
    matrix = values.astype(np.int64).reshape(-1, codebooks)
    offsets = np.arange(codebooks) * vocabulary
    if np.any(matrix < offsets) or np.any(matrix >= offsets + vocabulary):
        raise ValueError("token belongs to the wrong codebook")
    return matrix - offsets


def token_intervals(
    count: int, samples_per_token: int, rate: int, offset: int = 0
) -> NDArray[np.float64]:
    """Map fixed-stride audio tokens to contiguous half-open time intervals."""
    count, stride = _count(count, True), _count(samples_per_token)
    rate, offset = audio.sample_rate(rate), _count(offset, True, 2**53 - 1)
    _count(offset + count * stride, True, 2**53 - 1)
    starts = (offset + np.arange(count, dtype=np.int64) * stride) / rate
    ends = (offset + (np.arange(count, dtype=np.int64) + 1) * stride) / rate
    return np.column_stack([starts, ends])
