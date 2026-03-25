import numpy as np
import pytest
from voxweave import events as m


def test_event_encode_example():
    event = {"kind": "text", "sequence": 0, "text": "你好", "final": True}
    assert m.event_encode(event).endswith("\n")


def test_event_encode_boundaries():
    from voxweave.stream import frame_event

    value = frame_event([0], 8000, 0, 0)
    for key, invalid in [
        ("frames", 2),
        ("pcm16", "!"),
        ("channels", 0),
        ("sequence", True),
        ("offset", -1),
    ]:
        with pytest.raises(ValueError):
            m.event_encode(dict(value, **{key: invalid}))
