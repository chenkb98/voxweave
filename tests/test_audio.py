import numpy as np
import pytest
from voxweave import audio as m


def test_as_audio_example():
    np.testing.assert_array_equal(m.as_audio([1, -1]), [[1], [-1]])


def test_as_audio_boundaries():
    for invalid in [
        1,
        [[[1]]],
        ["1"],
        [True],
        [complex(1)],
        [np.nan],
        [np.inf],
        np.zeros((2, 0)),
        np.zeros((1, 9)),
    ]:
        with pytest.raises(ValueError):
            m.as_audio(invalid)
    assert m.as_audio([]).shape == (0, 1)


def test_as_audio_invariants():
    x = np.arange(12).reshape(6, 2)
    y = m.as_audio(x)
    y[0, 0] = 99
    assert x[0, 0] == 0
    np.testing.assert_array_equal(m.as_audio(m.as_audio(x)), x)
    assert y.dtype == np.float64


def test_sample_rate_example():
    assert m.sample_rate(16000) == 16000


def test_sample_rate_boundaries():
    for value in [0, -1, 768001, 1.5, True, "16000", np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.sample_rate(value)
    assert m.sample_rate(1) == 1
    assert m.sample_rate(768000) == 768000


def test_sample_rate_invariants():
    for value in [8000, 16000, 22050, 24000, 44100, 48000, 96000]:
        assert m.sample_rate(np.int64(value)) == value
        assert type(m.sample_rate(np.int64(value))) is int


def test_duration_example():
    assert m.duration(np.zeros((320, 2)), 16000) == 0.02
