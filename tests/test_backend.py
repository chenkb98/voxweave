import numpy as np
import pytest
from voxweave import backend as m


def test_model_request_example():
    from voxweave import conversation

    request = m.model_request([conversation.text_message("user", "hello")])
    assert request["generation"] == {"max_tokens": 128, "temperature": 0.0, "seed": 0}
