from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import __version__, audio, backend, events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxweave")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Run synthetic audio through the offline conversation backend"
    )
    demo.add_argument("--frame-size", type=int, default=80)
    analyze = commands.add_parser("analyze", help="Detect turns in a PCM16 WAV")
    analyze.add_argument("path")
    analyze.add_argument("--frame-size", type=int, default=320)
    replay = commands.add_parser("replay", help="Replay validated event JSONL")
    replay.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            values = [
                events.event_decode(line)
                for line in Path(args.path).read_text().splitlines()
                if line.strip()
            ]
            replayed = events.replay_events(values)
            result = {
                "frames": len(replayed["audio"]),
                "sample_rate": replayed["sample_rate"],
                "text": replayed["text"],
                "final": replayed["final"],
            }
        else:
            if args.command == "demo":
                rate = 8000
                tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(320) / rate)
                samples = np.concatenate([np.zeros(160), tone, np.zeros(320), tone, np.zeros(320)])
            else:
                samples, rate = audio.wav_decode(Path(args.path).read_bytes())
            value = backend.run_conversation(samples, rate, args.frame_size)
            result = {
                "turns": value["turns"],
                "responses": [
                    message["content"][0]["text"]
                    for message in value["messages"]
                    if message["role"] == "assistant"
                ],
                "backend": "offline test backend",
            }
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
