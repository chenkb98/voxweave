"""Streaming speech language model plumbing: audio frames, turns, conversation schemas, batching, and adapters."""

from .backend import run_conversation
from .events import replay_events
from .stream import AudioFramer, PCMDecoder
from .turns import EnergyVAD

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "AudioFramer",
    "PCMDecoder",
    "EnergyVAD",
    "run_conversation",
    "replay_events",
]
