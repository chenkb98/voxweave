import voxweave


def test_version_contract():
    assert voxweave.__version__ == "0.1.2"
    assert voxweave.__all__ == [
        "__version__",
        "AudioFramer",
        "PCMDecoder",
        "EnergyVAD",
        "run_conversation",
        "replay_events",
    ]


def test_pep561_typed_marker_exists():
    """PEP 561 requires py.typed for type checkers to use inline annotations."""
    from importlib import resources
    import voxweave

    assert resources.files(voxweave).joinpath("py.typed").is_file()
