from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


from . import audio


def _count(value: int, zero: bool = False, maximum: int = 10000000) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not int(not zero) <= value <= maximum
    ):
        raise ValueError("invalid bounded integer count")
    return value


def byte_chunks(payload: bytes, size: int) -> list[bytes]:
    """Split bytes without dropping the final incomplete network chunk."""
    if not isinstance(payload, bytes):
        raise ValueError("payload must be bytes")
    size = _count(size)
    return [payload[start : start + size] for start in range(0, len(payload), size)]


class PCMDecoder:
    """Incrementally decode PCM16, retaining incomplete sample/channel frames."""

    def __init__(self, channels: int = 1):
        audio.channels([], channels)
        self.channels, self.pending = int(channels), b""

    def feed(self, payload: bytes) -> NDArray[np.float64]:
        if not isinstance(payload, bytes) or len(payload) > 20000000:
            raise ValueError("invalid PCM packet")
        combined = self.pending + payload
        complete = len(combined) // (2 * self.channels) * (2 * self.channels)
        result = audio.pcm16_decode(combined[:complete], self.channels)
        self.pending = combined[complete:]
        return result

    def flush(self) -> None:
        if self.pending:
            raise ValueError("stream ended with an incomplete PCM frame")


class AudioFramer:
    """Turn arbitrary audio packets into exact contiguous frames."""

    def __init__(self, size: int, channels: int = 1):
        self.size = _count(size)
        self.pending = audio.channels([], channels)
        self.emitted = 0

    def feed(self, samples: ArrayLike) -> list[NDArray[np.float64]]:
        array = audio.as_audio(samples)
        if array.shape[1] != self.pending.shape[1]:
            raise ValueError("channel count changed midstream")
        combined = audio.concatenate([self.pending, array])
        count = len(combined) // self.size
        result = [combined[i * self.size : (i + 1) * self.size].copy() for i in range(count)]
        self.pending = combined[count * self.size :].copy()
        self.emitted += count * self.size
        return result

    def flush(self, pad_end: bool = False) -> list[NDArray[np.float64]]:
        if not isinstance(pad_end, bool):
            raise ValueError("pad_end must be boolean")
        if not len(self.pending):
            return []
        result = (
            audio.pad(self.pending, after=self.size - len(self.pending))
            if pad_end
            else self.pending.copy()
        )
        self.emitted += len(result)
        self.pending = self.pending[:0].copy()
        return [result]


class RingBuffer:
    """Bounded audio queue with explicit rejection or oldest-frame eviction."""

    def __init__(self, capacity: int, channels: int = 1, overflow: str = "reject"):
        self.capacity = _count(capacity)
        if overflow not in ["reject", "drop_oldest"]:
            raise ValueError("unknown overflow policy")
        self.overflow, self.samples = overflow, audio.channels([], channels)

    def __len__(self) -> int:
        return len(self.samples)

    def append(self, samples: ArrayLike) -> int:
        values = audio.as_audio(samples)
        if values.shape[1] != self.samples.shape[1]:
            raise ValueError("channel mismatch")
        excess = max(0, len(self.samples) + len(values) - self.capacity)
        if excess and self.overflow == "reject":
            raise BufferError("audio queue is full")
        combined = audio.concatenate([self.samples, values])
        self.samples = combined[-self.capacity :].copy()
        return excess

    def read(self, count: int) -> NDArray[np.float64]:
        _count(count, True)
        if count > len(self.samples):
            raise ValueError("not enough buffered frames")
        result = self.samples[:count].copy()
        self.samples = self.samples[count:].copy()
        return result
