import numpy as np
import pytest

from voxweave import audio


def test_channel_and_frame_averages_remain_finite_at_large_scale():
    samples = np.full((8, 2), 1e308)
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(audio.mono(samples), samples[:, :1])
        np.testing.assert_array_equal(audio.channels(samples, 1), samples[:, :1])
        np.testing.assert_array_equal(audio.remove_dc(samples), np.zeros_like(samples))
        windows = audio.frames(samples, 4, 2)
        np.testing.assert_array_equal(audio.overlap_add(windows, 2, len(samples)), samples)
        np.testing.assert_allclose(
            audio.resample([1e308, -1e308], 2, 4).ravel(), [1e308, 0, -1e308, -1e308]
        )


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e308])
def test_relative_signal_metrics_are_scale_invariant(scale):
    signal = np.array([1.0, -1.0, 1.0, -1.0])
    with np.errstate(over="raise", invalid="raise"):
        assert audio.snr(signal * scale, -signal * scale) == pytest.approx(-20 * np.log10(2))
        assert audio.spectral_centroid(signal * scale, 8) == pytest.approx(4)
        assert audio.band_energy(signal * scale, 8, 4, 4) == pytest.approx(1)
        assert audio.spectral_flatness(signal * scale, 8) == pytest.approx(
            audio.spectral_flatness(signal, 8)
        )


def test_spectrum_preserves_representable_large_dc_and_tones():
    with np.errstate(over="raise", invalid="raise"):
        _, dc = audio.spectrum(np.full((8, 2), 1e308), 8)
        np.testing.assert_array_equal(dc[0], [1e308, 1e308])
        assert not np.any(dc[1:])
        _, tone = audio.spectrum([1e308, -1e308, 1e308, -1e308], 8)
        assert tone[-1, 0] == pytest.approx(1e308)


def test_unrepresentable_centered_audio_is_rejected():
    with pytest.raises(ValueError):
        audio.remove_dc([1.7e308, 1.7e308, -1.7e308])
