from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


from . import audio
from .stream import _count


def text_message(role: str, text: str) -> dict:
    """Construct a text segment with an explicit conversation role."""
    if (
        role not in ["system", "user", "assistant"]
        or not isinstance(text, str)
        or len(text) > 100000
    ):
        raise ValueError("invalid role or bounded text")
    return {"role": role, "content": [{"type": "text", "text": text}]}
