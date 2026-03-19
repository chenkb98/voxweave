import numpy as np
import pytest
from voxweave import backend as m


def test_model_request_example():
    from voxweave import conversation

    request = m.model_request([conversation.text_message("user", "hello")])
    assert request["generation"] == {"max_tokens": 128, "temperature": 0.0, "seed": 0}


def test_model_request_boundaries():
    from voxweave import conversation

    history = [conversation.text_message("user", "hello")]
    for values in [
        {"max_tokens": 0},
        {"seed": True},
        {"temperature": np.nan},
        {"temperature": 3},
        {"extra": 1},
    ]:
        with pytest.raises(ValueError):
            m.model_request(history, values)
    with pytest.raises(ValueError):
        m.model_request(history + [conversation.text_message("assistant", "world")])


def test_model_request_invariants():
    from voxweave import conversation

    history = [conversation.audio_message([0], 8000)]
    request = m.model_request(history)
    assert set(request) == {"messages", "generation"}
    assert set(request["generation"]) == {"max_tokens", "temperature", "seed"}
    request["messages"][0]["content"][0]["wav"] = "changed"
    assert history[0]["content"][0]["wav"] != "changed"


def test_offline_response_example():
    from voxweave import conversation

    response = m.offline_response(m.model_request([conversation.audio_message([0] * 8, 8000)]))
    assert (
        response == "offline test backend: audio_segments=1; duration_s=0.001000; text_characters=0"
    )


def test_offline_response_boundaries():
    for request in [
        {},
        {"messages": [], "generation": {}},
        {"messages": [], "generation": {}, "extra": 1},
    ]:
        with pytest.raises(ValueError):
            m.offline_response(request)


def test_offline_response_invariants():
    from voxweave import conversation

    for text in ["", "hello", "你好🎵"]:
        request = m.model_request([conversation.text_message("user", text)])
        response = m.offline_response(request)
        assert response.endswith(f"text_characters={len(text)}")
        assert response == m.offline_response(request)
        assert response.startswith("offline test backend:")


def test_stream_response_example():
    assert m.stream_response("你好世界", 2) == [
        {"kind": "text", "sequence": 0, "text": "你好", "final": False},
        {"kind": "text", "sequence": 1, "text": "世界", "final": True},
    ]


def test_stream_response_boundaries():
    for size in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            m.stream_response("text", size)
    assert m.stream_response("", 2) == [{"kind": "text", "sequence": 0, "text": "", "final": True}]


def test_stream_response_invariants():
    text = "语音 hello 🎵\n结束"
    for size in range(1, 12):
        events = m.stream_response(text, size)
        assert "".join(event["text"] for event in events) == text
        assert sum(event["final"] for event in events) == 1 and events[-1]["final"]
        assert [event["sequence"] for event in events] == list(range(len(events)))


def test_run_conversation_example():
    result = m.run_conversation([0, 0, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0], 8000, 2)
    assert result["turns"] == [{"start": 2, "end": 6}] and len(result["messages"]) == 2
    assert result["messages"][-1]["content"][0]["text"].startswith("offline test backend:")


def test_run_conversation_boundaries():
    assert m.run_conversation([], 8000, 2) == {"turns": [], "messages": []}
    with pytest.raises(ValueError):
        m.run_conversation([1], 0)
    with pytest.raises(ValueError):
        m.run_conversation([1], 8000, backend="not callable")
