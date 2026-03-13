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


def offline_response(request: dict) -> str:
    """Return deterministic audio diagnostics, explicitly labeled as a test backend."""
    import base64

    if not isinstance(request, dict) or set(request) != {"messages", "generation"}:
        raise ValueError("invalid request fields")
    value = model_request(request["messages"], request["generation"])
    count, seconds, characters = 0, 0.0, 0
    for part in value["messages"][-1]["content"]:
        if part["type"] == "text":
            characters += len(part["text"])
        else:
            samples, rate = audio.wav_decode(base64.b64decode(part["wav"], validate=True))
            count += 1
            seconds += audio.duration(samples, rate)
    return f"offline test backend: audio_segments={count}; duration_s={seconds:.6f}; text_characters={characters}"


def stream_response(text: str, chunk_size: int = 16) -> list[dict]:
    """Split response text on Unicode codepoint boundaries with one final event."""
    conversation.text_message("assistant", text)
    size = _count(chunk_size)
    chunks = [text[i : i + size] for i in range(0, len(text), size)] or [""]
    return [
        {"kind": "text", "sequence": i, "text": chunk, "final": i == len(chunks) - 1}
        for i, chunk in enumerate(chunks)
    ]
