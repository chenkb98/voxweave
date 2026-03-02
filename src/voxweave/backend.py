from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


from . import audio, conversation, turns
from .stream import _count


def model_request(messages: list[dict], generation: dict | None = None) -> dict:
    """Construct a backend-neutral request ending with a user turn."""
    history = conversation.validate_conversation(messages)
    if history[-1]["role"] != "user":
        raise ValueError("generation requires a final user message")
    settings = dict(max_tokens=128, temperature=0.0, seed=0)
    if generation is not None:
        if not isinstance(generation, dict) or set(generation) - set(settings):
            raise ValueError("unknown generation fields")
        settings.update(generation)
    _count(settings["max_tokens"], maximum=8192)
    _count(settings["seed"], True, 2**53 - 1)
    value = settings["temperature"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 2
    ):
        raise ValueError("invalid temperature")
    return {"messages": history, "generation": settings}
