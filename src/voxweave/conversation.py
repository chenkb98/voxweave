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
