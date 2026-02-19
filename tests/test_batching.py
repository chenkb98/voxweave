import numpy as np
import pytest
from voxweave import batching as m


def test_pad_audio_batch_example():
    result = m.pad_audio_batch([[1, 2], [3]])
    np.testing.assert_array_equal(result["samples"][:, :, 0], [[1, 2], [3, 0]])
    np.testing.assert_array_equal(result["mask"], [[True, True], [True, False]])


def test_pad_audio_batch_boundaries():
    with pytest.raises(ValueError):
        m.pad_audio_batch([])
    with pytest.raises(ValueError):
        m.pad_audio_batch([[1], [[1, 2]]])
    with pytest.raises(ValueError):
        m.pad_audio_batch([[np.nan]])
    assert m.pad_audio_batch([[], []])["samples"].shape == (2, 0, 1)


def test_pad_audio_batch_invariants():
    clips = [np.arange(size) for size in range(7)]
    result = m.pad_audio_batch(clips)
    assert set(result) == {"samples", "lengths", "mask"}
    for i, clip in enumerate(clips):
        np.testing.assert_array_equal(result["samples"][i, result["mask"][i], 0], clip)
        assert result["mask"][i].sum() == result["lengths"][i] == len(clip)
        assert np.all(result["samples"][i, ~result["mask"][i]] == 0)


def test_attention_mask_example():
    np.testing.assert_array_equal(
        m.attention_mask([0, 2, 1]), [[False, False], [True, True], [True, False]]
    )


def test_attention_mask_boundaries():
    for lengths in [[-1], [1.5], [True], [[1]]]:
        with pytest.raises(ValueError):
            m.attention_mask(lengths)
    with pytest.raises(ValueError):
        m.attention_mask([2], 1)
    assert m.attention_mask(np.array([], dtype=int), 3).shape == (0, 3)


def test_attention_mask_invariants():
    for width in range(1, 12):
        lengths = np.arange(width + 1)
        mask = m.attention_mask(lengths, width)
        np.testing.assert_array_equal(mask.sum(axis=1), lengths)
        assert np.all(np.diff(mask.astype(int), axis=1) <= 0)


def test_pack_tokens_example():
    np.testing.assert_array_equal(m.pack_tokens([[1, 2], [3, 0]], 4), [1, 6, 3, 4])


def test_pack_tokens_boundaries():
    for values in [[[4]], [[-1]], [[1.5]], [[True]], [1, 2], np.empty((2, 0), dtype=int)]:
        with pytest.raises(ValueError):
            m.pack_tokens(values, 4)
    assert m.pack_tokens(np.empty((0, 2), dtype=int), 4).shape == (0,)
