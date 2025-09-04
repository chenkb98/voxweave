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
