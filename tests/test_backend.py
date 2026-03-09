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
