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
