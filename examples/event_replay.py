from voxweave.events import event_encode, event_decode, replay_events
from voxweave.stream import frame_event

events = [
    frame_event([0, 0.25], 16000, 0, 0),
    frame_event([-0.25, 0], 16000, 1, 2),
    {"kind": "text", "sequence": 2, "text": "offline example", "final": True},
]
result = replay_events([event_decode(event_encode(value)) for value in events])
print({"frames": len(result["audio"]), "text": result["text"], "final": result["final"]})
