import numpy as np
import pytest
from voxweave import stream as m


def test_byte_chunks_example():
    assert m.byte_chunks(b"abcdefg", 3) == [b"abc", b"def", b"g"]
