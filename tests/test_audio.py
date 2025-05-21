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


def test_channels_boundaries():
    for count in [0, 9, 1.5, True]:
        with pytest.raises(ValueError):
            m.channels([1], count)
    with pytest.raises(ValueError):
        m.channels([[1, 2]], 3)
    assert m.channels([], 8).shape == (0, 8)


def test_channels_invariants():
    x = np.array([[0.25], [-0.75]])
    for count in range(1, 9):
        np.testing.assert_array_equal(m.channels(m.channels(x, count), 1), x)
    np.testing.assert_array_equal(m.channels([[1, -1]], 1), [[0]])


def test_crop_example():
    np.testing.assert_array_equal(m.crop([1, 2, 3, 4], 1, 3), [[2], [3]])


def test_crop_boundaries():
    for start, stop in [(-1, 1), (2, 1), (0, 5), (0.5, 1), (True, 1)]:
        with pytest.raises(ValueError):
            m.crop([1, 2], start, stop)
    assert m.crop([], 0, 0).shape == (0, 1)


def test_crop_invariants():
    x = np.arange(5)
    for start in range(6):
        for stop in range(start, 6):
            assert len(m.crop(x, start, stop)) == stop - start
            np.testing.assert_array_equal(m.crop(x, start, stop).ravel(), x[start:stop])


def test_pad_example():
    np.testing.assert_array_equal(m.pad([1, 2], 1, 2).ravel(), [0, 1, 2, 0, 0])


def test_pad_boundaries():
    for value in [-1, 0.5, True, 10000001]:
        with pytest.raises(ValueError):
            m.pad([1], value, 0)
        with pytest.raises(ValueError):
            m.pad([1], 0, value)
    assert m.pad([], 2, 3).shape == (5, 1)


def test_pad_invariants():
    x = [[1, -1], [0.2, 0.4]]
    for before in range(4):
        for after in range(4):
            y = m.pad(x, before, after)
            np.testing.assert_array_equal(m.crop(y, before, before + 2), x)


def test_trim_example():
    np.testing.assert_array_equal(m.trim([0, 0.2, 0, -0.3, 0]).ravel(), [0.2, 0, -0.3])


def test_trim_boundaries():
    assert m.trim([0, 0]).shape == (0, 1)
    assert m.trim([]).shape == (0, 1)
    for threshold in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.trim([0], threshold)
    np.testing.assert_array_equal(m.trim([0.1, 0.2], 0.1).ravel(), [0.2])


def test_trim_invariants():
    x = [[0, 0], [0, 0.5], [0, 0]]
    np.testing.assert_array_equal(m.trim(x), [[0, 0.5]])
    np.testing.assert_array_equal(m.trim(m.trim(x)), m.trim(x))
    np.testing.assert_array_equal(m.trim(m.pad([0.5], 3, 4)), [[0.5]])


def test_gain_example():
    np.testing.assert_allclose(m.gain([0.1, -0.1], 20).ravel(), [1, -1])


def test_gain_boundaries():
    for value in [np.nan, np.inf, -np.inf, 601, -601]:
        with pytest.raises(ValueError):
            m.gain([1], value)
    assert m.gain([], 0).shape == (0, 1)
    np.testing.assert_array_equal(m.gain([0], 600), [[0]])


def test_gain_invariants():
    x = np.linspace(-0.5, 0.5, 9)
    for value in [-60, -3, 0, 3, 60]:
        np.testing.assert_allclose(m.gain(m.gain(x, value), -value).ravel(), x, atol=1e-15)
    np.testing.assert_array_equal(m.gain([2], 0), [[2]])


def test_peak_example():
    assert m.peak([0.2, -0.8, 0.5]) == 0.8


def test_peak_boundaries():
    assert m.peak([]) == 0
    assert m.peak([0]) == 0
    with pytest.raises(ValueError):
        m.peak([np.nan])
    assert m.peak([2]) == 2


def test_peak_invariants():
    x = np.array([[0.1, -0.9], [0.4, 0.2]])
    assert m.peak(x) == m.peak(-x) == m.peak(x[::-1])
    assert m.peak(x * 2) == 2 * m.peak(x)


def test_rms_example():
    assert m.rms([1, -1, 1, -1]) == 1


def test_rms_boundaries():
    assert m.rms([]) == 0
    assert m.rms([0, 0]) == 0
    assert m.rms([1e200, -1e200]) == 1e200
    with pytest.raises(ValueError):
        m.rms([np.inf])


def test_rms_invariants():
    x = [3, 4]
    assert m.rms(x) == pytest.approx(np.sqrt(12.5))
    assert m.rms(x) == m.rms([-3, -4])
    assert m.rms(x) == m.rms([3, 4, 3, 4])
    assert m.rms(x) <= m.peak(x)


def test_normalize_peak_example():
    np.testing.assert_allclose(m.normalize_peak([1, -2], 1).ravel(), [0.5, -1])


def test_normalize_peak_boundaries():
    for target in [-0.1, 1.1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.normalize_peak([1], target)
    np.testing.assert_array_equal(m.normalize_peak([0, 0]), [[0], [0]])
    assert m.normalize_peak([]).shape == (0, 1)


def test_normalize_peak_invariants():
    x = [0.2, -0.4, 0.1]
    for target in [0, 0.3, 1]:
        y = m.normalize_peak(x, target)
        assert m.peak(y) == pytest.approx(target)
        np.testing.assert_allclose(m.normalize_peak(y, target), y)


def test_normalize_rms_example():
    np.testing.assert_allclose(m.normalize_rms([1, -1], 0.25).ravel(), [0.25, -0.25])


def test_normalize_rms_boundaries():
    for target in [-1, 2, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.normalize_rms([1], target)
    np.testing.assert_array_equal(m.normalize_rms([0, 0]), [[0], [0]])
    assert m.normalize_rms([]).shape == (0, 1)


def test_normalize_rms_invariants():
    for x in [[1, 2, 3], [-1, 0, 1], [0.001]]:
        y = m.normalize_rms(x, 0.2)
        assert m.rms(y) == pytest.approx(0.2)
        np.testing.assert_allclose(m.normalize_rms(y, 0.2), y)


def test_remove_dc_example():
    np.testing.assert_array_equal(m.remove_dc([[1, 10], [3, 20]]), [[-1, -5], [1, 5]])
