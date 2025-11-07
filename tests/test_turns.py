import numpy as np
import pytest
from voxweave import turns as m


def test_vad_config_example():
    assert m.vad_config()["min_frames"] == 2


def test_vad_config_boundaries():
    for values in [
        {"on_threshold": 0},
        {"off_threshold": -0.1},
        {"off_threshold": 0.5},
        {"min_frames": 0},
        {"hangover": True},
        {"unknown": 1},
        {"on_threshold": np.nan},
    ]:
        with pytest.raises(ValueError):
            m.vad_config(values)


def test_vad_config_invariants():
    value = m.vad_config()
    assert set(value) == {"on_threshold", "off_threshold", "min_frames", "hangover"}
    assert m.vad_config(value) == value
    for field in value:
        with pytest.raises(ValueError):
            m.vad_config({field + "_typo": value[field]})


def test_EnergyVAD_example():
    vad = m.EnergyVAD()
    events = [event for level in [0, 0.1, 0.1, 0, 0] for event in vad.feed(level)]
    assert events == [{"kind": "start", "frame": 1}, {"kind": "end", "frame": 3}]


def test_EnergyVAD_boundaries():
    vad = m.EnergyVAD()
    with pytest.raises(ValueError):
        vad.feed(np.nan)
    assert vad.index == 0
    assert vad.flush() == [] and vad.flush() == []
    with pytest.raises(RuntimeError):
        vad.feed(0.1)


def test_EnergyVAD_invariants():
    vad = m.EnergyVAD()
    events = [event for level in [0.1, 0.1, 0.015, 0.005] for event in vad.feed(level)]
    events += vad.flush()
    assert events == [{"kind": "start", "frame": 0}, {"kind": "end", "frame": 3}]
    vad = m.EnergyVAD()
    assert vad.feed(0.1) == [] and vad.feed(0) == [] and vad.flush() == []


def test_detect_turns_example():
    assert m.detect_turns([0, 0, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0], 2) == [{"start": 2, "end": 6}]


def test_detect_turns_boundaries():
    assert m.detect_turns([], 2) == []
    assert m.detect_turns(np.zeros(10), 2) == []
    with pytest.raises(ValueError):
        m.detect_turns([1], 0)
    with pytest.raises(ValueError):
        m.detect_turns([np.nan], 2)


def test_detect_turns_invariants():
    assert m.detect_turns([0.1] * 5, 2) == [{"start": 0, "end": 5}]
    for channels in [1, 2, 8]:
        x = np.zeros((12, channels))
        x[2:8] = 0.1
        assert m.detect_turns(x, 2) == [{"start": 2, "end": 8}]


def test_merge_turns_example():
    assert m.merge_turns([{"start": 0, "end": 2}, {"start": 3, "end": 5}], 1) == [
        {"start": 0, "end": 5}
    ]


def test_merge_turns_boundaries():
    for turns in [
        [{"start": 1, "end": 1}],
        [{"start": -1, "end": 2}],
        [{"start": 0, "end": 3}, {"start": 2, "end": 4}],
        [{"start": 0, "end": 2, "extra": 1}],
    ]:
        with pytest.raises(ValueError):
            m.merge_turns(turns, 1)
    with pytest.raises(ValueError):
        m.merge_turns([], -1)


def test_merge_turns_invariants():
    turns = [{"start": 0, "end": 2}, {"start": 4, "end": 6}, {"start": 10, "end": 12}]
    for gap in range(6):
        result = m.merge_turns(turns, gap)
        assert m.merge_turns(result, gap) == result
        assert len(result) <= len(turns)
    assert turns[0]["end"] == 2


def test_extract_turns_example():
    result = m.extract_turns([1, 2, 3, 4], [{"start": 1, "end": 3}])
    np.testing.assert_array_equal(result[0].ravel(), [2, 3])


def test_extract_turns_boundaries():
    assert m.extract_turns([], []) == []
    with pytest.raises(ValueError):
        m.extract_turns([1], [{"start": 0, "end": 2}])
    with pytest.raises(ValueError):
        m.extract_turns([np.nan], [])


def test_extract_turns_invariants():
    x = np.arange(12).reshape(6, 2)
    turns = [{"start": 0, "end": 2}, {"start": 2, "end": 6}]
    result = m.extract_turns(x, turns)
    np.testing.assert_array_equal(np.concatenate(result), x)
    result[0][0, 0] = 99
    assert x[0, 0] == 0 and result[1][0, 0] == 4


def test_frame_labels_example():
    np.testing.assert_array_equal(
        m.frame_labels(6, [{"start": 2, "end": 4}], 2), [False, True, False]
    )


def test_frame_labels_boundaries():
    assert m.frame_labels(0, [], 2).shape == (0,)
    with pytest.raises(ValueError):
        m.frame_labels(1, [{"start": 0, "end": 2}], 1)
    with pytest.raises(ValueError):
        m.frame_labels(2, [], 0)
