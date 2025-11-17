import numpy as np
import pytest
from voxweave import conversation as m


def test_text_message_example():
    assert m.text_message("user", "你好") == {
        "role": "user",
        "content": [{"type": "text", "text": "你好"}],
    }


def test_text_message_boundaries():
    for role, text in [("tool", "x"), ("user", None), ("assistant", "x" * 100001)]:
        with pytest.raises(ValueError):
            m.text_message(role, text)
    assert m.text_message("assistant", "")["content"][0]["text"] == ""
