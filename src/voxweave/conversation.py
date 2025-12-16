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


def validate_conversation(messages: list[dict]) -> list[dict]:
    """Validate optional leading system context followed by alternating turns."""
    if not isinstance(messages, list) or not 1 <= len(messages) <= 200:
        raise ValueError("conversation needs 1..200 messages")
    result = [validate_message(value) for value in messages]
    expected = "user"
    for index, message in enumerate(result):
        if message["role"] == "system":
            if index != 0:
                raise ValueError("system context must be first")
        else:
            if message["role"] != expected:
                raise ValueError("user and assistant turns must alternate")
            expected = "assistant" if expected == "user" else "user"
    return result


def append_message(messages: list[dict], message: dict) -> list[dict]:
    """Append a validated turn to a copied conversation, including an empty start."""
    if not isinstance(messages, list):
        raise ValueError("history must be a list")
    return validate_conversation([*messages, message])


def replace_transcript(message: dict, text: str) -> dict:
    """Replace text segments while preserving the user's original audio."""
    value = validate_message(message)
    if value["role"] != "user":
        raise ValueError("only user transcripts may be revised")
    text_part = text_message("user", text)["content"]
    value["content"] = [part for part in value["content"] if part["type"] != "text"] + text_part
    return validate_message(value)


def apply_delta(state: dict, update: dict) -> dict:
    """Apply the next full-text revision and make finalization terminal."""
    for value in [state, update]:
        if not isinstance(value, dict) or set(value) != {"text", "revision", "final"}:
            raise ValueError("invalid transcript update fields")
        text_message("user", value["text"])
        _count(value["revision"], True, 2**53 - 1)
        if type(value["final"]) is not bool:
            raise ValueError("final must be boolean")
    if state["final"] or update["revision"] != state["revision"] + 1:
        raise ValueError("stale, skipped or post-final revision")
    return update.copy()
