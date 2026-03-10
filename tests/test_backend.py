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
