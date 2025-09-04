import numpy as np
import pytest
from voxweave import stream as m


def test_byte_chunks_example():
    assert m.byte_chunks(b"abcdefg", 3) == [b"abc", b"def", b"g"]


def test_byte_chunks_boundaries():
    for size in [0, -1, 1.5, True]:
        with pytest.raises(ValueError):
            m.byte_chunks(b"abc", size)
    with pytest.raises(ValueError):
        m.byte_chunks("abc", 2)
    assert m.byte_chunks(b"", 2) == []


def test_byte_chunks_invariants():
    payload = bytes(range(255))
    for size in range(1, 33):
        chunks = m.byte_chunks(payload, size)
        assert b"".join(chunks) == payload
        assert all(0 < len(chunk) <= size for chunk in chunks)


def test_PCMDecoder_example():
    decoder = m.PCMDecoder()
    assert decoder.feed(b"\x00").shape == (0, 1)
    np.testing.assert_array_equal(decoder.feed(b"\x80"), [[-1]])
    decoder.flush()
