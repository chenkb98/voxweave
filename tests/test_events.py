import numpy as np
import pytest
from voxweave import events as m


def test_event_encode_example():
    event = {"kind": "text", "sequence": 0, "text": "你好", "final": True}
    assert m.event_encode(event).endswith("\n")
