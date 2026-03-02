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


def test_pack_tokens_invariants():
    for books in range(1, 8):
        tokens = np.arange(5 * books).reshape(5, books) % 8
        packed = m.pack_tokens(tokens, 8).reshape(5, books)
        for book in range(books):
            assert np.all((packed[:, book] >= book * 8) & (packed[:, book] < (book + 1) * 8))
            np.testing.assert_array_equal(packed[:, book] % 8, tokens[:, book])


def test_unpack_tokens_example():
    np.testing.assert_array_equal(m.unpack_tokens([1, 6, 3, 4], 2, 4), [[1, 2], [3, 0]])


def test_unpack_tokens_boundaries():
    for values, books in [([1], 2), ([1, 2], 2), ([1.5], 1), ([-1], 1), ([True], 1)]:
        with pytest.raises(ValueError):
            m.unpack_tokens(values, books, 4)
    assert m.unpack_tokens(np.array([], dtype=int), 2).shape == (0, 2)


def test_unpack_tokens_invariants():
    for vocabulary in [1, 4, 17]:
        for books in range(1, 9):
            tokens = np.arange(7 * books).reshape(7, books) % vocabulary
            packed = m.pack_tokens(tokens, vocabulary)
            np.testing.assert_array_equal(m.unpack_tokens(packed, books, vocabulary), tokens)
            np.testing.assert_array_equal(
                m.pack_tokens(m.unpack_tokens(packed, books, vocabulary), vocabulary), packed
            )


def test_token_intervals_example():
    np.testing.assert_allclose(m.token_intervals(2, 320, 16000), [[0, 0.02], [0.02, 0.04]])


def test_token_intervals_boundaries():
    assert m.token_intervals(0, 320, 16000).shape == (0, 2)
    for args in [(-1, 1, 8000), (1, 0, 8000), (1, 1, 0)]:
        with pytest.raises(ValueError):
            m.token_intervals(*args)
    with pytest.raises(ValueError):
        m.token_intervals(1, 1, 8000, 2**53 - 1)


def test_token_intervals_invariants():
    values = m.token_intervals(100, 441, 44100, 882)
    np.testing.assert_allclose(values[:, 1] - values[:, 0], 0.01)
    np.testing.assert_array_equal(values[:-1, 1], values[1:, 0])
    assert values[0, 0] == 0.02 and values[-1, 1] == 1.02
