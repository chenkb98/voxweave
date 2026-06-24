import json

import pytest

from voxweave import audio, cli, events, stream


@pytest.fixture
def command(capsys):
    def invoke(*arguments):
        assert cli.main(list(arguments)) == 0
        return json.loads(capsys.readouterr().out)

    return invoke


def test_synthetic_demo_and_wav_analysis(command, tmp_path):
    demo = command("demo")
    assert demo["backend"] == "offline test backend" and len(demo["turns"]) == 2
    assert len(demo["responses"]) == 2
    target = tmp_path / "voice.wav"
    target.write_bytes(audio.wav_encode([0.0] * 160 + [0.1] * 320 + [0.0] * 320, 8000))
    result = command("analyze", str(target), "--frame-size", "80")
    assert len(result["turns"]) == len(result["responses"]) == 1


def test_event_file_replay_includes_audio_text_and_terminal_state(command, tmp_path):
    target = tmp_path / "events.jsonl"
    values = [
        stream.frame_event([0.0, 0.5], 16000, 0, 0),
        {"kind": "text", "sequence": 1, "text": "你好", "final": True},
    ]
    target.write_text("".join(events.event_encode(value) for value in values))
    assert command("replay", str(target)) == {
        "frames": 2,
        "sample_rate": 16000,
        "text": "你好",
        "final": True,
    }
    target.write_text("{invalid}")
    with pytest.raises(SystemExit) as error:
        cli.main(["replay", str(target)])
    assert error.value.code == 2


def test_invalid_frame_size_is_a_cli_error():
    with pytest.raises(SystemExit) as error:
        cli.main(["demo", "--frame-size", "0"])
    assert error.value.code == 2
