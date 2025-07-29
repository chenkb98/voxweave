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


def test_remove_dc_boundaries():
    assert m.remove_dc([]).shape == (0, 1)
    np.testing.assert_array_equal(m.remove_dc([[2, 3]]), [[0, 0]])
    with pytest.raises(ValueError):
        m.remove_dc([np.nan])


def test_remove_dc_invariants():
    x = np.arange(12).reshape(6, 2)
    y = m.remove_dc(x)
    np.testing.assert_allclose(y.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(m.remove_dc(x + 100), y)
    np.testing.assert_allclose(m.remove_dc(y), y)


def test_reverse_example():
    np.testing.assert_array_equal(m.reverse([[1, 2], [3, 4]]), [[3, 4], [1, 2]])


def test_reverse_boundaries():
    assert m.reverse([]).shape == (0, 1)
    np.testing.assert_array_equal(m.reverse([1]), [[1]])
    with pytest.raises(ValueError):
        m.reverse([np.inf])


def test_reverse_invariants():
    x = np.arange(15).reshape(5, 3)
    np.testing.assert_array_equal(m.reverse(m.reverse(x)), x)
    assert m.rms(m.reverse(x)) == m.rms(x)
    y = m.reverse(x)
    y[0, 0] = 99
    assert x[-1, 0] == 12


def test_repeat_example():
    np.testing.assert_array_equal(m.repeat([1, -1], 2).ravel(), [1, -1, 1, -1])


def test_repeat_boundaries():
    for count in [-1, 1.2, True, 10001]:
        with pytest.raises(ValueError):
            m.repeat([1], count)
    assert m.repeat([[1, 2]], 0).shape == (0, 2)
    assert m.repeat([], 10000).shape == (0, 1)


def test_repeat_invariants():
    x = [[0.1, 0.2], [0.3, 0.4]]
    for count in range(1, 5):
        result = m.repeat(x, count)
        assert len(result) == 2 * count
        assert m.rms(result) == pytest.approx(m.rms(x))
        for i in range(count):
            np.testing.assert_array_equal(result[2 * i : 2 * i + 2], x)


def test_mix_example():
    np.testing.assert_allclose(m.mix([0, 1], [1, 0], 0.25).ravel(), [0.25, 0.75])


def test_mix_boundaries():
    for weight in [-1, 2, np.nan, np.inf]:
        with pytest.raises(ValueError):
            m.mix([1], [1], weight)
    with pytest.raises(ValueError):
        m.mix([1], [1, 2])
    with pytest.raises(ValueError):
        m.mix([[1, 2]], [1])
    assert m.mix([], []).shape == (0, 1)


def test_mix_invariants():
    a, b = [0.1, 0.5], [-0.3, 0.2]
    for weight in [0, 0.2, 0.5, 1]:
        np.testing.assert_allclose(m.mix(a, b, weight), m.mix(b, a, 1 - weight))
    np.testing.assert_array_equal(m.mix(a, b, 0).ravel(), a)
    np.testing.assert_array_equal(m.mix(a, b, 1).ravel(), b)


def test_concatenate_example():
    np.testing.assert_array_equal(m.concatenate([[1, 2], [3]]).ravel(), [1, 2, 3])


def test_concatenate_boundaries():
    assert m.concatenate([]).shape == (0, 1)
    assert m.concatenate([np.empty((0, 2))]).shape == (0, 2)
    with pytest.raises(ValueError):
        m.concatenate([[1], [[1, 2]]])
    with pytest.raises(ValueError):
        m.concatenate([[1], [np.nan]])


def test_concatenate_invariants():
    x = np.arange(7)
    for split in range(8):
        np.testing.assert_array_equal(m.concatenate([x[:split], x[split:]]).ravel(), x)
    a, b, c = [1], [2, 3], [4]
    np.testing.assert_array_equal(
        m.concatenate([m.concatenate([a, b]), c]), m.concatenate([a, m.concatenate([b, c])])
    )


def test_fade_in_example():
    np.testing.assert_allclose(m.fade_in([1, 1, 1, 1], 3).ravel(), [0, 0.5, 1, 1])


def test_fade_in_boundaries():
    for frames in [-1, 5, 1.5, True]:
        with pytest.raises(ValueError):
            m.fade_in([1, 2], frames)
    np.testing.assert_array_equal(m.fade_in([1], 1), [[0]])
    np.testing.assert_array_equal(m.fade_in([1, 2], 0).ravel(), [1, 2])


def test_fade_in_invariants():
    x = np.arange(10).reshape(5, 2)
    for frames in range(6):
        y = m.fade_in(x, frames)
        assert y.shape == x.shape
        assert m.peak(y) <= m.peak(x)
    np.testing.assert_array_equal(x, np.arange(10).reshape(5, 2))


def test_fade_out_example():
    np.testing.assert_allclose(m.fade_out([1, 1, 1, 1], 3).ravel(), [1, 1, 0.5, 0])


def test_fade_out_boundaries():
    for frames in [-1, 5, 1.5, True]:
        with pytest.raises(ValueError):
            m.fade_out([1, 2], frames)
    np.testing.assert_array_equal(m.fade_out([1], 1), [[0]])
    np.testing.assert_array_equal(m.fade_out([1, 2], 0).ravel(), [1, 2])


def test_fade_out_invariants():
    x = np.arange(10).reshape(5, 2)
    for frames in range(6):
        y = m.fade_out(x, frames)
        assert y.shape == x.shape
        assert m.peak(y) <= m.peak(x)
    np.testing.assert_array_equal(x, np.arange(10).reshape(5, 2))


def test_crossfade_example():
    np.testing.assert_allclose(m.crossfade([1, 1, 1], [0, 0, 0], 2).ravel(), [1, 1, 0, 0])


def test_crossfade_boundaries():
    for frames in [-1, 3, 0.5, True]:
        with pytest.raises(ValueError):
            m.crossfade([1, 1], [0, 0], frames)
    np.testing.assert_array_equal(m.crossfade([1], [0], 1), [[0.5]])
    assert m.crossfade([], [], 0).shape == (0, 1)


def test_crossfade_invariants():
    for frames in range(5):
        output = m.crossfade(np.ones(5), np.ones(4), frames)
        assert len(output) == 9 - frames
        np.testing.assert_allclose(output, 1)
    a, b = [1, 2, 3], [4, 5, 6]
    np.testing.assert_allclose(
        m.reverse(m.crossfade(a, b, 2)), m.crossfade(m.reverse(b), m.reverse(a), 2)
    )


def test_resample_example():
    np.testing.assert_allclose(m.resample([0, 1, 0], 2, 4).ravel(), [0, 0.5, 1, 0.5, 0, 0])


def test_resample_boundaries():
    assert m.resample([], 16000, 24000).shape == (0, 1)
    assert m.resample([1], 4, 1).shape == (0, 1)
    with pytest.raises(ValueError):
        m.resample([1], 0, 4)
    with pytest.raises(ValueError):
        m.resample([1], 4, True)


def test_resample_invariants():
    x = np.arange(12).reshape(6, 2)
    np.testing.assert_array_equal(m.resample(x, 16000, 16000), x)
    for a, b in [(8000, 16000), (16000, 8000), (44100, 48000)]:
        np.testing.assert_allclose(m.resample(np.ones((20, 2)), a, b), 1)


def test_frames_example():
    np.testing.assert_array_equal(m.frames([1, 2, 3], 2, 2)[:, :, 0], [[1, 2], [3, 0]])


def test_frames_boundaries():
    for size, hop in [(0, 1), (1, 0), (-1, 1), (1, 1.5), (True, 1)]:
        with pytest.raises(ValueError):
            m.frames([1], size, hop)
    assert m.frames([], 4, 2).shape == (0, 4, 1)
    assert m.frames([1], 4, 2, False).shape == (0, 4, 1)


def test_frames_invariants():
    x = np.arange(11)
    for size in range(1, 6):
        chunks = m.frames(x, size, size)
        np.testing.assert_array_equal(chunks[:, :, 0].ravel()[: len(x)], x)
        assert np.all(chunks[:, :, 0].ravel()[len(x) :] == 0)
    np.testing.assert_array_equal(m.frames([1, 2, 3], 2, 1, False)[:, :, 0], [[1, 2], [2, 3]])


def test_overlap_add_example():
    np.testing.assert_array_equal(m.overlap_add([[[1], [2]], [[2], [3]]], 1).ravel(), [1, 2, 3])


def test_overlap_add_boundaries():
    for windows, hop in [([[1]], 1), ([[[np.nan]]], 1), ([[[1]]], 0), ([[[1]]], 2)]:
        with pytest.raises(ValueError):
            m.overlap_add(windows, hop)
    assert m.overlap_add(np.empty((0, 3, 2)), 1).shape == (0, 2)
    with pytest.raises(ValueError):
        m.overlap_add([[[1]]], 1, 2)


def test_overlap_add_invariants():
    x = np.arange(26).reshape(13, 2)
    for size in range(1, 7):
        for hop in range(1, size + 1):
            np.testing.assert_allclose(m.overlap_add(m.frames(x, size, hop), hop, len(x)), x)


def test_pcm16_encode_example():
    assert m.pcm16_encode([-1, 0, 1]) == b"\x00\x80\x00\x00\xff\x7f"


def test_pcm16_encode_boundaries():
    assert m.pcm16_encode([]) == b""
    assert m.pcm16_encode([-2, 2]) == m.pcm16_encode([-1, 1])
    with pytest.raises(ValueError):
        m.pcm16_encode([np.nan])


def test_pcm16_encode_invariants():
    assert m.pcm16_encode([[0, -1], [1, 0]]) == b"\x00\x00\x00\x80\xff\x7f\x00\x00"
    values = np.linspace(-1, 1, 101)
    decoded = np.frombuffer(m.pcm16_encode(values), dtype="<i2").astype(float) / 32768
    assert np.max(np.abs(decoded - values)) <= 1 / 32768


def test_pcm16_decode_example():
    np.testing.assert_array_equal(m.pcm16_decode(b"\x00\x80\x00\x00").ravel(), [-1, 0])


def test_pcm16_decode_boundaries():
    for payload, channels in [(b"x", 1), (b"xx", 2), (b"", 0), (b"", True), ("xx", 1)]:
        with pytest.raises(ValueError):
            m.pcm16_decode(payload, channels)
    assert m.pcm16_decode(b"", 2).shape == (0, 2)


def test_pcm16_decode_invariants():
    for count in range(1, 9):
        x = np.linspace(-1, 0.9, 5 * count).reshape(5, count)
        y = m.pcm16_decode(m.pcm16_encode(x), count)
        assert y.shape == x.shape
        assert np.max(np.abs(x - y)) <= 1 / 32768
        assert m.pcm16_encode(y) == m.pcm16_encode(x)


def test_wav_encode_example():
    assert m.wav_encode([0, 1], 16000)[:4] == b"RIFF"


def test_wav_encode_boundaries():
    with pytest.raises(ValueError):
        m.wav_encode([1], 0)
    with pytest.raises(ValueError):
        m.wav_encode([np.nan], 16000)
    assert len(m.wav_encode([], 16000)) >= 44


def test_wav_encode_invariants():
    import io, wave

    for channels in [1, 2, 8]:
        payload = m.wav_encode(np.zeros((7, channels)), 24000)
        with wave.open(io.BytesIO(payload), "rb") as handle:
            assert handle.getparams()[:4] == (channels, 2, 24000, 7)
            assert handle.readframes(7) == bytes(14 * channels)
        assert payload == m.wav_encode(np.zeros((7, channels)), 24000)


def test_wav_decode_example():
    audio, rate = m.wav_decode(m.wav_encode([0, -1], 8000))
    assert rate == 8000
    np.testing.assert_array_equal(audio.ravel(), [0, -1])


def test_wav_decode_boundaries():
    for payload in [b"", b"RIFF", b"not wav", m.wav_encode([1, 2], 16000)[:-1]]:
        with pytest.raises(ValueError):
            m.wav_decode(payload)
    audio, rate = m.wav_decode(m.wav_encode([], 24000))
    assert audio.shape == (0, 1) and rate == 24000


def test_wav_decode_invariants():
    for channels in [1, 2, 8]:
        x = np.linspace(-0.8, 0.8, 11 * channels).reshape(11, channels)
        payload = m.wav_encode(x, 44100)
        y, rate = m.wav_decode(payload)
        assert rate == 44100 and y.shape == x.shape
        assert np.max(np.abs(x - y)) <= 1 / 32768
        assert m.wav_encode(y, rate) == payload
