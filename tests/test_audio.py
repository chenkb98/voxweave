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
