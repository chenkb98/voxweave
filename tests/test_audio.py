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


def test_duration_boundaries():
    assert m.duration([], 8000) == 0
    with pytest.raises(ValueError):
        m.duration([1], 0)
    with pytest.raises(ValueError):
        m.duration([np.nan], 8000)


def test_duration_invariants():
    for channels in [1, 2, 8]:
        assert m.duration(np.zeros((441, channels)), 44100) == 0.01
    assert m.duration([1, 2], 2) == 2 * m.duration([1], 2)


def test_mono_example():
    np.testing.assert_array_equal(m.mono([[1, -1], [0.5, 0.5]]), [[0], [0.5]])


def test_mono_boundaries():
    assert m.mono(np.empty((0, 2))).shape == (0, 1)
    with pytest.raises(ValueError):
        m.mono([[1, np.nan]])
    np.testing.assert_array_equal(m.mono([0.5]), [[0.5]])


def test_mono_invariants():
    x = np.array([[0.2, 0.8], [-0.6, 0.4]])
    np.testing.assert_allclose(m.mono(x), m.mono(x[:, ::-1]))
    np.testing.assert_array_equal(m.mono(m.mono(x)), m.mono(x))
    np.testing.assert_array_equal(x, [[0.2, 0.8], [-0.6, 0.4]])


def test_channels_example():
    np.testing.assert_array_equal(m.channels([1, 2], 2), [[1, 1], [2, 2]])
