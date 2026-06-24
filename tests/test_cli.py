import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from voxweave.cli import build_parser


def invoke(*arguments, cwd=None):
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    return subprocess.run(
        [sys.executable, "-m", "voxweave.cli", *arguments],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("command", ["demo", "analyze", "replay"])
def test_command_help(command):
    result = invoke(command, "--help")
    assert result.returncode == 0 and "usage:" in result.stdout


def test_command_surface():
    parser = build_parser()
    subcommands = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subcommands) == 1
    assert set(subcommands[0].choices) == set(["demo", "analyze", "replay"])


def test_invalid_command_does_not_write(tmp_path):
    result = invoke("unknown-command", cwd=tmp_path)
    assert result.returncode == 2 and list(tmp_path.iterdir()) == []


def test_installed_style_version():
    result = invoke("--version")
    assert result.returncode == 0 and result.stdout.strip() == "0.1.2"
