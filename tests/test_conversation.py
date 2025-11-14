import numpy as np
import pytest
from voxweave import conversation as m


def test_text_message_example():
    assert m.text_message("user", "你好") == {
        "role": "user",
        "content": [{"type": "text", "text": "你好"}],
    }
