import numpy as np
import pytest
from voxweave import batching as m


def test_pad_audio_batch_example():
    result = m.pad_audio_batch([[1, 2], [3]])
    np.testing.assert_array_equal(result["samples"][:, :, 0], [[1, 2], [3, 0]])
    np.testing.assert_array_equal(result["mask"], [[True, True], [True, False]])
