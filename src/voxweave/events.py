from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


from . import audio, conversation
from .stream import _count


def _event(value: dict) -> dict:
    import base64
    import binascii

    if not isinstance(value, dict):
        raise ValueError("event must be an object")
    if value.get("kind") == "text":
        if set(value) != {"kind", "sequence", "text", "final"} or type(value["final"]) is not bool:
            raise ValueError("invalid text event fields")
        _count(value["sequence"], True, 2**53 - 1)
        conversation.text_message("assistant", value["text"])
    elif value.get("kind") == "audio":
        if set(value) != {
            "kind",
            "sequence",
            "offset",
            "sample_rate",
            "frames",
            "channels",
            "pcm16",
        }:
            raise ValueError("invalid audio event fields")
        for key in ["sequence", "offset"]:
            _count(value[key], True, 2**53 - 1)
        _count(value["frames"], True)
        audio.sample_rate(value["sample_rate"])
        if not isinstance(value["pcm16"], str) or len(value["pcm16"]) > 30000000:
            raise ValueError("invalid PCM field")
        try:
            decoded = audio.pcm16_decode(
                base64.b64decode(value["pcm16"], validate=True), value["channels"]
            )
        except (ValueError, binascii.Error) as error:
            raise ValueError("invalid PCM event payload") from error
        if len(decoded) != value["frames"]:
            raise ValueError("declared frame count differs from payload")
    else:
        raise ValueError("unsupported event kind")
    return value.copy()


def event_encode(event: dict) -> str:
    """Encode one exact, validated audio/text event as canonical JSONL."""
    import json

    return (
        json.dumps(
            _event(event),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def event_decode(text: str) -> dict:
    """Read strict JSON events with duplicate-key and nonfinite rejection."""
    import json

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate event field")
            value[key] = item
        return value

    def constant(value):
        raise ValueError("invalid JSON constant")

    if not isinstance(text, str) or len(text) > 31000000:
        raise ValueError("event must be bounded text")
    return _event(json.loads(text, object_pairs_hook=pairs, parse_constant=constant))
