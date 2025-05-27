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


def mono(samples: ArrayLike) -> NDArray[np.float64]:
    """Average channels, retaining a singleton channel dimension."""
    audio = as_audio(samples)
    return audio.mean(axis=1, keepdims=True)


def channels(samples: ArrayLike, count: int) -> NDArray[np.float64]:
    """Duplicate mono channels; reject ambiguous multichannel remapping."""
    audio = as_audio(samples)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= count <= 8:
        raise ValueError("channel count must be an integer in 1..8")
    if count == audio.shape[1]:
        return audio
    if count == 1:
        return mono(audio)
    if audio.shape[1] != 1:
        raise ValueError("only mono audio can be expanded")
    return np.repeat(audio, count, axis=1)


def crop(samples: ArrayLike, start: int, stop: int) -> NDArray[np.float64]:
    """Copy a validated half-open frame interval."""
    audio = as_audio(samples)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in [start, stop]):
        raise ValueError("frame boundaries must be integers")
    if not 0 <= start <= stop <= len(audio):
        raise ValueError("crop lies outside audio")
    return audio[start:stop].copy()


def pad(samples: ArrayLike, before: int = 0, after: int = 0) -> NDArray[np.float64]:
    """Add zero-valued frames before and after audio."""
    audio = as_audio(samples)
    if any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v <= 10000000
        for v in [before, after]
    ):
        raise ValueError("padding counts must be bounded nonnegative integers")
    return np.pad(audio, ((before, after), (0, 0)))


def trim(samples: ArrayLike, threshold: float = 1e-4) -> NDArray[np.float64]:
    """Remove leading/trailing frames silent in every channel."""
    audio = as_audio(samples)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    active = np.flatnonzero(np.max(np.abs(audio), axis=1) > threshold)
    return audio[active[0] : active[-1] + 1].copy() if len(active) else audio[:0].copy()


def gain(samples: ArrayLike, decibels: float) -> NDArray[np.float64]:
    """Apply amplitude gain in dB without automatic clipping."""
    if not np.isfinite(decibels) or abs(decibels) > 600:
        raise ValueError("gain must be finite and within +/-600 dB")
    return as_audio(as_audio(samples) * 10.0 ** (decibels / 20.0))


def peak(samples: ArrayLike) -> float:
    """Return the largest absolute sample, or zero for empty audio."""
    return float(np.max(np.abs(as_audio(samples)), initial=0.0))


def rms(samples: ArrayLike) -> float:
    """Compute stable root-mean-square amplitude over all samples."""
    audio = as_audio(samples)
    scale = peak(audio)
    return float(scale * np.sqrt(np.mean((audio / scale) ** 2))) if scale else 0.0


def normalize_peak(samples: ArrayLike, target: float = 0.95) -> NDArray[np.float64]:
    """Scale non-silent audio to a chosen peak in [0, 1]."""
    audio = as_audio(samples)
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("target peak must be in [0, 1]")
    maximum = peak(audio)
    return audio / maximum * target if maximum else audio


def normalize_rms(samples: ArrayLike, target: float = 0.1) -> NDArray[np.float64]:
    """Scale non-silent audio to RMS; clipping prevention is explicit."""
    audio = as_audio(samples)
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("target RMS must be in [0, 1]")
    value = rms(audio)
    return as_audio(audio / value * target) if value else audio


def remove_dc(samples: ArrayLike) -> NDArray[np.float64]:
    """Subtract each channel's mean without mixing channels."""
    audio = as_audio(samples)
    return audio - audio.mean(axis=0, keepdims=True) if len(audio) else audio


def reverse(samples: ArrayLike) -> NDArray[np.float64]:
    """Reverse frame order while preserving channel order."""
    return as_audio(samples)[::-1].copy()


def repeat(samples: ArrayLike, count: int) -> NDArray[np.float64]:
    """Repeat a clip an explicitly bounded number of times."""
    audio = as_audio(samples)
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or not 0 <= count <= 10000
    ):
        raise ValueError("repeat count must be an integer in 0..10000")
    if len(audio) * count > 10000000:
        raise ValueError("repeated clip exceeds ten million frames")
    return np.tile(audio, (count, 1))


def mix(left: ArrayLike, right: ArrayLike, weight: float = 0.5) -> NDArray[np.float64]:
    """Blend equal-shaped clips without implicit padding or broadcasting."""
    a, b = as_audio(left), as_audio(right)
    if a.shape != b.shape or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("mix requires equal shapes and a weight in [0, 1]")
    return as_audio((1 - weight) * a + weight * b)
