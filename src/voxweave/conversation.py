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


def audio_message(samples: ArrayLike, rate: int, text: str = "") -> dict:
    """Build a user message containing an embedded PCM16 WAV and optional text."""
    import base64

    text_part = text_message("user", text)["content"]
    payload = audio.wav_encode(samples, rate)
    if len(payload) > 4000000:
        raise ValueError("audio message is too large")
    content = [{"type": "audio", "wav": base64.b64encode(payload).decode("ascii")}]
    if text:
        content.extend(text_part)
    return {"role": "user", "content": content}


def validate_message(message: dict) -> dict:
    """Copy and validate the complete local multimodal message schema."""
    import base64
    import binascii
    import copy

    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        raise ValueError("invalid message fields")
    text_message(message["role"], "")
    content = message["content"]
    if not isinstance(content, list) or not 1 <= len(content) <= 20:
        raise ValueError("message needs 1..20 content segments")
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("content segment must be an object")
        if set(part) == {"type", "text"} and part["type"] == "text":
            text_message(message["role"], part["text"])
        elif set(part) == {"type", "wav"} and part["type"] == "audio" and message["role"] == "user":
            if not isinstance(part["wav"], str) or len(part["wav"]) > 5400000:
                raise ValueError("invalid audio payload")
            try:
                audio.wav_decode(base64.b64decode(part["wav"], validate=True))
            except (binascii.Error, ValueError) as error:
                raise ValueError("invalid embedded WAV") from error
        else:
            raise ValueError("unsupported content schema or audio role")
    return copy.deepcopy(message)
