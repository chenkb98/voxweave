import voxweave


def test_version_contract():
    assert voxweave.__version__ == "0.1.0"
    assert voxweave.__all__ == [
        "__version__",
        "AudioFramer",
        "PCMDecoder",
        "EnergyVAD",
        "run_conversation",
        "replay_events",
    ]
