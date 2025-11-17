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


def test_text_message_invariants():
    for role in ["system", "user", "assistant"]:
        result = m.text_message(role, "语音\n🎵")
        assert set(result) == {"role", "content"}
        assert set(result["content"][0]) == {"type", "text"}
        assert result["role"] == role and result["content"][0]["text"] == "语音\n🎵"


def test_audio_message_example():
    result = m.audio_message([0, 0.5], 16000, "听到了什么？")
    assert [x["type"] for x in result["content"]] == ["audio", "text"]
