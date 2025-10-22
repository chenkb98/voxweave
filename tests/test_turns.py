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
