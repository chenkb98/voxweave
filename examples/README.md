# Executable examples

Install the project dev dependencies first. All examples use synthetic data
and run on CPU without external models or network services.

```sh
python examples/pcm_stream.py
python examples/voice_turns.py
python examples/event_replay.py
python examples/jitter_buffer.py
```

Each example prints its result to stdout. The outputs are synthetic
diagnostics, not real speech model benchmarks.
