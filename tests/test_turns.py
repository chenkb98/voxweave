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
