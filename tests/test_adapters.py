from types import SimpleNamespace

import numpy as np
import pytest

from voxweave.adapters import qwen2_audio_response
from voxweave.backend import model_request
from voxweave.conversation import audio_message, text_message


class Inputs(dict):
    input_ids = SimpleNamespace(size=lambda axis: 2)

    def to(self, device):
        assert device == "cpu"
        return self


class Processor:
    feature_extractor = SimpleNamespace(sampling_rate=16000)

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **arguments):
        self.messages = messages
        assert arguments == {"add_generation_prompt": True, "tokenize": False}
        return "rendered prompt"

    def __call__(self, **arguments):
        self.calls.append(arguments)
        return Inputs(input_ids=np.array([[1, 2]]))

    def batch_decode(self, ids, **arguments):
        np.testing.assert_array_equal(ids, [[3, 4]])
        assert arguments == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        return ["decoded response"]


class Model:
    device, training = "cpu", False

    def generate(self, **arguments):
        assert arguments["max_new_tokens"] == 16 and arguments["do_sample"] is False
        return np.array([[1, 2, 3, 4]])


def test_qwen_protocol_uses_current_audio_keyword_and_trims_prompt():
    request = model_request(
        [text_message("system", "instructions"), audio_message([0, 0.5], 16000, "describe")],
        {"max_tokens": 16},
    )
    processor = Processor()
    assert qwen2_audio_response(request, processor, Model()) == "decoded response"
    assert set(processor.calls[0]) == {"text", "audio", "return_tensors", "padding"}
    np.testing.assert_array_equal(processor.calls[0]["audio"][0], [0, 0.5])
    assert processor.messages[0]["content"] == "instructions"
    assert processor.messages[1]["content"][0] == {"type": "audio", "audio_url": "embedded://0"}


def test_rejects_resampling_and_sampling_ambiguity_before_model_calls():
    processor = Processor()
    with pytest.raises(ValueError):
        qwen2_audio_response(model_request([audio_message([0], 8000)]), processor, Model())
    with pytest.raises(ValueError):
        qwen2_audio_response(
            model_request([text_message("user", "x")], {"temperature": 0.5}), processor, Model()
        )
    assert processor.calls == []


def test_text_only_request_and_eval_requirement():
    request = model_request([text_message("user", "hello")], {"max_tokens": 16})
    processor = Processor()
    assert qwen2_audio_response(request, processor, Model()) == "decoded response"
    assert "audio" not in processor.calls[0]
    model = Model()
    model.training = True
    with pytest.raises(ValueError):
        qwen2_audio_response(request, Processor(), model)


def test_rejects_unexpected_request_fields():
    with pytest.raises(ValueError):
        qwen2_audio_response({}, Processor(), Model())
