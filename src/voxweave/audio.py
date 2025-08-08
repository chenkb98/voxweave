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


def concatenate(clips: list[ArrayLike]) -> NDArray[np.float64]:
    """Join clips with matching channel counts in time order."""
    arrays = [as_audio(clip) for clip in clips]
    if not arrays:
        return np.empty((0, 1), dtype=np.float64)
    if any(a.shape[1] != arrays[0].shape[1] for a in arrays):
        raise ValueError("channel counts must match")
    return np.concatenate(arrays, axis=0)


def fade_in(samples: ArrayLike, frames: int) -> NDArray[np.float64]:
    """Apply a linear incoming ramp in place on a copy."""
    audio = as_audio(samples)
    if (
        isinstance(frames, bool)
        or not isinstance(frames, (int, np.integer))
        or not 0 <= frames <= len(audio)
    ):
        raise ValueError("fade length must be an integer within the clip")
    if frames:
        ramp = np.linspace(0, 1, frames) if frames > 1 else np.zeros(1)
        audio[:frames] *= ramp[:, None]
    return audio


def fade_out(samples: ArrayLike, frames: int) -> NDArray[np.float64]:
    """Apply a linear outgoing ramp in place on a copy."""
    audio = as_audio(samples)
    if (
        isinstance(frames, bool)
        or not isinstance(frames, (int, np.integer))
        or not 0 <= frames <= len(audio)
    ):
        raise ValueError("fade length must be an integer within the clip")
    if frames:
        ramp = np.linspace(1, 0, frames) if frames > 1 else np.zeros(1)
        audio[-frames:] *= ramp[:, None]
    return audio


def crossfade(left: ArrayLike, right: ArrayLike, frames: int) -> NDArray[np.float64]:
    """Join clips with a complementary linear crossfade."""
    a, b = as_audio(left), as_audio(right)
    if (
        a.shape[1] != b.shape[1]
        or isinstance(frames, bool)
        or not isinstance(frames, (int, np.integer))
        or not 0 <= frames <= min(len(a), len(b))
    ):
        raise ValueError("crossfade requires matching channels and valid overlap")
    if not frames:
        return concatenate([a, b])
    ramp = np.linspace(0, 1, frames)[:, None] if frames > 1 else np.full((1, 1), 0.5)
    return concatenate([a[:-frames], a[-frames:] * (1 - ramp) + b[:frames] * ramp, b[frames:]])


def resample(samples: ArrayLike, source_rate: int, target_rate: int) -> NDArray[np.float64]:
    """Linearly resample; no anti-alias filter is implied."""
    audio = as_audio(samples)
    source_rate, target_rate = sample_rate(source_rate), sample_rate(target_rate)
    length = int(math.floor(len(audio) * target_rate / source_rate + 0.5))
    if length > 10000000:
        raise ValueError("resampled clip exceeds ten million frames")
    if not length or not len(audio):
        return np.empty((length, audio.shape[1]), dtype=np.float64)
    positions = np.arange(length) * source_rate / target_rate
    return np.column_stack(
        [
            np.interp(positions, np.arange(len(audio)), audio[:, channel])
            for channel in range(audio.shape[1])
        ]
    )


def frames(samples: ArrayLike, size: int, hop: int, pad_end: bool = True) -> NDArray[np.float64]:
    """Frame audio as (windows, frames, channels), optionally zero-padding the tail."""
    audio = as_audio(samples)
    if any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 1 <= v <= 1000000
        for v in [size, hop]
    ):
        raise ValueError("size and hop must be bounded positive integers")
    starts = list(range(0, len(audio) if pad_end else max(0, len(audio) - size + 1), hop))
    if len(starts) * size * audio.shape[1] > 10000000:
        raise ValueError("framing would allocate too many samples")
    output = np.zeros((len(starts), size, audio.shape[1]))
    for i, start in enumerate(starts):
        part = audio[start : start + size]
        output[i, : len(part)] = part
    return output


def overlap_add(windows: ArrayLike, hop: int, length: int | None = None) -> NDArray[np.float64]:
    """Reconstruct rectangular windows by averaging overlapping samples."""
    array = np.asarray(windows, dtype=np.float64)
    if (
        array.ndim != 3
        or not 1 <= array.shape[2] <= 8
        or array.shape[1] < 1
        or not np.isfinite(array).all()
    ):
        raise ValueError("windows must be finite (windows, frames, channels)")
    if (
        isinstance(hop, bool)
        or not isinstance(hop, (int, np.integer))
        or not 1 <= hop <= array.shape[1]
    ):
        raise ValueError("hop must be between one and the window size")
    total = (len(array) - 1) * hop + array.shape[1] if len(array) else 0
    if total > 10000000:
        raise ValueError("reconstruction is too large")
    if length is None:
        length = total
    if (
        isinstance(length, bool)
        or not isinstance(length, (int, np.integer))
        or not 0 <= length <= total
    ):
        raise ValueError("invalid reconstruction length")
    output, counts = np.zeros((total, array.shape[2])), np.zeros((total, 1))
    for i, part in enumerate(array):
        output[i * hop : i * hop + len(part)] += part
        counts[i * hop : i * hop + len(part)] += 1
    return (output / np.maximum(counts, 1))[:length]


def pcm16_encode(samples: ArrayLike) -> bytes:
    """Interleave channels as little-endian signed PCM16 with saturation."""
    audio = as_audio(samples)
    integers = np.clip(np.rint(np.clip(audio, -1, 1) * 32768), -32768, 32767)
    return integers.astype("<i2").tobytes()


def pcm16_decode(data: bytes, channel_count: int = 1) -> NDArray[np.float64]:
    """Decode complete interleaved little-endian PCM16 frames."""
    if not isinstance(data, bytes):
        raise ValueError("PCM payload must be bytes")
    if (
        isinstance(channel_count, bool)
        or not isinstance(channel_count, int)
        or not 1 <= channel_count <= 8
    ):
        raise ValueError("channel count must be in 1..8")
    if len(data) % (2 * channel_count):
        raise ValueError("PCM payload contains an incomplete frame")
    return np.frombuffer(data, dtype="<i2").astype(np.float64).reshape(-1, channel_count) / 32768


def wav_encode(samples: ArrayLike, rate: int) -> bytes:
    """Serialize a self-contained PCM16 WAV in memory."""
    import io
    import wave

    audio, rate = as_audio(samples), sample_rate(rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(audio.shape[1])
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm16_encode(audio))
    return buffer.getvalue()


def wav_decode(payload: bytes) -> tuple[NDArray[np.float64], int]:
    """Read uncompressed PCM16 WAV and reject truncated frame payloads."""
    import io
    import wave

    if not isinstance(payload, bytes):
        raise ValueError("WAV payload must be bytes")
    try:
        with wave.open(io.BytesIO(payload), "rb") as source:
            if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                raise ValueError("only uncompressed PCM16 WAV is supported")
            rate, count, channels = (
                sample_rate(source.getframerate()),
                source.getnframes(),
                source.getnchannels(),
            )
            data = source.readframes(count)
            if len(data) != count * channels * 2:
                raise ValueError("truncated WAV frames")
            return pcm16_decode(data, channels), rate
    except (wave.Error, EOFError) as error:
        raise ValueError("invalid PCM16 WAV") from error


def clipping_fraction(samples: ArrayLike, threshold: float = 1.0) -> float:
    """Fraction of samples at or beyond an absolute clipping threshold."""
    audio = as_audio(samples)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    return float(np.mean(np.abs(audio) >= threshold)) if audio.size else 0.0


def zero_crossing_rate(samples: ArrayLike) -> float:
    """Mean sign-bit changes between adjacent samples within each channel."""
    audio = as_audio(samples)
    if len(audio) < 2:
        return 0.0
    signs = audio < 0
    return float(np.mean(signs[1:] != signs[:-1]))
