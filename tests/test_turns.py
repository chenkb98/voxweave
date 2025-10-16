import numpy as np
import pytest
from voxweave import turns as m


def test_vad_config_example():
    assert m.vad_config()["min_frames"] == 2
