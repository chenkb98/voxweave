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
