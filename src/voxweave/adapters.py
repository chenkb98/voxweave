from __future__ import annotations

import base64
from typing import Any

import numpy as np

from . import audio, backend, conversation


def qwen2_audio_response(request: dict, processor: Any, model: Any) -> str:
    """Run greedy Qwen2-Audio generation using explicitly supplied model objects."""
    if not isinstance(request, dict) or set(request) != {"messages", "generation"}:
        raise ValueError("invalid request schema")
    value = backend.model_request(request["messages"], request["generation"])
    if value["generation"]["temperature"] != 0:
        raise ValueError("this adapter supports deterministic greedy decoding only")
    if getattr(model, "training", False):
        raise ValueError("set model.eval() before inference")
    rate = audio.sample_rate(processor.feature_extractor.sampling_rate)
    messages, waveforms = [], []
    for message in value["messages"]:
        content = []
        for part in message["content"]:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            else:
                samples, source_rate = audio.wav_decode(
                    base64.b64decode(part["wav"], validate=True)
                )
                if source_rate != rate:
                    raise ValueError("resample audio explicitly to the processor sample rate")
                waveforms.append(audio.mono(samples)[:, 0].astype(np.float32))
                content.append({"type": "audio", "audio_url": f"embedded://{len(waveforms) - 1}"})
        messages.append(
            {
                "role": message["role"],
                "content": content
                if message["role"] == "user"
                else "".join(part["text"] for part in content),
            }
        )
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    arguments = {"text": text, "return_tensors": "pt", "padding": True}
    if waveforms:
        arguments["audio"] = waveforms
    inputs = processor(**arguments).to(model.device)
    generated = model.generate(
        **inputs, max_new_tokens=value["generation"]["max_tokens"], do_sample=False
    )
    generated = generated[:, inputs.input_ids.size(1) :]
    responses = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if not isinstance(responses, list) or len(responses) != 1:
        raise ValueError("processor must decode exactly one response")
    conversation.text_message("assistant", responses[0])
    return responses[0]
