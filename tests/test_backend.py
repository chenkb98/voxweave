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


def test_run_conversation_invariants():
    seen = []

    def backend(request):
        seen.append(request)
        return "observed"

    result = m.run_conversation([0.1] * 4 + [0] * 4 + [0.1] * 4, 8000, 2, backend=backend)
    assert len(seen) == 2 and len(result["messages"]) == 4
    assert [len(value["messages"]) for value in seen] == [1, 3]
    assert all(value["messages"][-1]["role"] == "user" for value in seen)


def test_cancel_stream_example():
    assert list(m.cancel_stream([1, 2, 3], lambda: False)) == [1, 2, 3]


def test_cancel_stream_boundaries():
    assert list(m.cancel_stream([1, 2, 3], lambda: True)) == []
    assert list(m.cancel_stream([], lambda: False)) == []
    with pytest.raises(ValueError):
        list(m.cancel_stream([1], False))


def test_cancel_stream_invariants():
    state = {"cancelled": False, "closed": False, "pulled": 0}

    def source():
        try:
            for value in range(5):
                state["pulled"] += 1
                yield value
        finally:
            state["closed"] = True

    stream = m.cancel_stream(source(), lambda: state["cancelled"])
    assert next(stream) == 0
    state["cancelled"] = True
    assert list(stream) == [] and state["pulled"] == 1 and state["closed"]



def test_offline_response_with_mixed_audio_and_text_content():
    from voxweave import conversation

    message = conversation.audio_message([0] * 16000, 16000, "describe this audio")
    request = m.model_request([message])
    response = m.offline_response(request)
    # offline_response processes only the last message content parts
    assert "audio_segments=1" in response
    assert "duration_s=1.000000" in response
    assert "text_characters=19" in response


def test_offline_response_counts_only_final_message_content():
    from voxweave import conversation

    # offline_response iterates over messages[-1]["content"] only
    history = [
        conversation.audio_message([0] * 8000, 8000),
        conversation.text_message("assistant", "acknowledged"),
        conversation.audio_message([0] * 16000, 8000, "transcribe"),
    ]
    request = m.model_request(history)
    response = m.offline_response(request)
    # Only the last message is counted: 1 audio segment (2s) + 10 text chars
    assert "audio_segments=1" in response
    assert "duration_s=2.000000" in response
    assert "text_characters=10" in response
