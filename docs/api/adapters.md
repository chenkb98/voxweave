# adapters


## `qwen2_audio_response`

Run greedy Qwen2-Audio generation using explicitly supplied processor and model
objects. The function validates the request schema through `backend.model_request`,
rejects non-zero temperature (only deterministic decoding is supported), and
requires the model to be in eval mode before inference.

Audio content is decoded via `audio.wav_decode` and must already match the
processors sampling rate — no implicit resampling is performed. The processors
chat template renders the conversation, and `batch_decode` must return exactly
one response string. The function does not download model weights or contact
external services; all model objects are caller-provided.

See [model-adapter.md](../model-adapter.md) for the full adapter contract and
upstream transformer library references.
