import numpy as np
import pytest

from voxweave import events as m


def test_event_encode_example():
    event = {"kind": "text", "sequence": 0, "text": "你好", "final": True}
    assert m.event_encode(event).endswith("\n")


def test_event_encode_boundaries():
    from voxweave.stream import frame_event

    value = frame_event([0], 8000, 0, 0)
    for key, invalid in [
        ("frames", 2),
        ("pcm16", "!"),
        ("channels", 0),
        ("sequence", True),
        ("offset", -1),
    ]:
        with pytest.raises(ValueError):
            m.event_encode(dict(value, **{key: invalid}))


def test_event_encode_invariants():
    import json

    from voxweave.stream import frame_event

    for value in [
        frame_event([0], 8000, 0, 0),
        {"kind": "text", "sequence": 1, "text": "ok", "final": True},
    ]:
        assert json.loads(m.event_encode(value)) == value
        for key in value:
            candidate = dict(value)
            candidate.pop(key)
            with pytest.raises(ValueError):
                m.event_encode(candidate)
        with pytest.raises(ValueError):
            m.event_encode(dict(value, unexpected=1))


def test_event_decode_example():
    event = {"kind": "text", "sequence": 0, "text": "ok", "final": True}
    assert m.event_decode(m.event_encode(event)) == event


def test_event_decode_boundaries():
    for text in ["", "null", "[]", "{}", '{"kind":"text","kind":"audio"}', '{"value":NaN}']:
        with pytest.raises(ValueError):
            m.event_decode(text)


def test_event_decode_invariants():
    from voxweave.stream import frame_event

    for channels in [1, 2, 8]:
        event = frame_event(np.zeros((5, channels)), 16000, 4, 123)
        encoded = m.event_encode(event)
        assert m.event_encode(m.event_decode(encoded)) == encoded


def test_replay_events_example():
    from voxweave.stream import frame_event

    result = m.replay_events(
        [
            frame_event([0, 0.5], 8000, 0, 0),
            {"kind": "text", "sequence": 1, "text": "ok", "final": True},
        ]
    )
    assert result["text"] == "ok" and result["final"] and result["sample_rate"] == 8000


def test_replay_events_boundaries():
    from voxweave.stream import frame_event

    for events in [
        [frame_event([0], 8000, 1, 0)],
        [frame_event([0], 8000, 0, 1)],
        [
            {"kind": "text", "sequence": 0, "text": "", "final": True},
            {"kind": "text", "sequence": 1, "text": "", "final": True},
        ],
    ]:
        with pytest.raises(ValueError):
            m.replay_events(events)
    assert m.replay_events([])["audio"].shape == (0, 1)


def test_replay_events_invariants():
    from voxweave import audio
    from voxweave.stream import frame_event

    x = np.linspace(-0.5, 0.5, 17)
    for size in range(1, 8):
        events = [
            frame_event(x[start : start + size], 16000, i, start)
            for i, start in enumerate(range(0, len(x), size))
        ]
        result = m.replay_events([m.event_decode(m.event_encode(value)) for value in events])
        np.testing.assert_array_equal(result["audio"], audio.pcm16_decode(audio.pcm16_encode(x)))
