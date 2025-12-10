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


def test_audio_message_boundaries():
    with pytest.raises(ValueError):
        m.audio_message([0], 0)
    with pytest.raises(ValueError):
        m.audio_message([np.nan], 16000)
    with pytest.raises(ValueError):
        m.audio_message([0], 16000, None)
    assert len(m.audio_message([], 16000)["content"]) == 1


def test_audio_message_invariants():
    import base64
    from voxweave import audio

    result = m.audio_message([[0, -1], [0.5, 0]], 8000)
    samples, rate = audio.wav_decode(base64.b64decode(result["content"][0]["wav"]))
    assert rate == 8000 and samples.shape == (2, 2)
    np.testing.assert_array_equal(samples, [[0, -1], [0.5, 0]])


def test_validate_message_example():
    message = m.audio_message([0], 8000, "test")
    assert m.validate_message(message) == message


def test_validate_message_boundaries():
    for value in [
        {},
        {"role": "user", "content": []},
        {"role": "tool", "content": [{"type": "text", "text": "x"}]},
        {"role": "user", "content": [{"type": "audio", "wav": "!"}]},
    ]:
        with pytest.raises(ValueError):
            m.validate_message(value)
    value = m.audio_message([0], 8000)
    value["role"] = "assistant"
    with pytest.raises(ValueError):
        m.validate_message(value)


def test_validate_message_invariants():
    value = m.audio_message([0], 8000, "hello")
    copy = m.validate_message(value)
    copy["content"][-1]["text"] = "changed"
    assert value["content"][-1]["text"] == "hello"
    for key in value:
        candidate = dict(value)
        candidate.pop(key)
        with pytest.raises(ValueError):
            m.validate_message(candidate)
    with pytest.raises(ValueError):
        m.validate_message(dict(value, extra=1))


def test_validate_conversation_example():
    messages = [
        m.text_message("system", "local"),
        m.text_message("user", "hi"),
        m.text_message("assistant", "hello"),
    ]
    assert m.validate_conversation(messages) == messages


def test_validate_conversation_boundaries():
    for messages in [
        [],
        [m.text_message("assistant", "x")],
        [m.text_message("user", "a"), m.text_message("user", "b")],
        [m.text_message("user", "x"), m.text_message("system", "x")],
    ]:
        with pytest.raises(ValueError):
            m.validate_conversation(messages)


def test_validate_conversation_invariants():
    for count in range(1, 10):
        messages = [
            m.text_message("user" if i % 2 == 0 else "assistant", str(i)) for i in range(count)
        ]
        assert m.validate_conversation(messages) == messages
        assert (
            m.validate_conversation([m.text_message("system", "context")] + messages)[1:]
            == messages
        )


def test_append_message_example():
    history = m.append_message([], m.text_message("user", "hi"))
    assert len(m.append_message(history, m.text_message("assistant", "hello"))) == 2


def test_append_message_boundaries():
    history = [m.text_message("user", "hi")]
    with pytest.raises(ValueError):
        m.append_message(history, m.text_message("user", "again"))
    assert len(history) == 1
    with pytest.raises(ValueError):
        m.append_message(None, m.text_message("user", "hi"))


def test_append_message_invariants():
    message = m.text_message("user", "hello")
    history = m.append_message([], message)
    result = m.append_message(history, m.text_message("assistant", "world"))
    result[0]["content"][0]["text"] = "changed"
    assert history[0]["content"][0]["text"] == message["content"][0]["text"] == "hello"
