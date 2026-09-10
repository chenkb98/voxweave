"""Demonstrate JitterBuffer reordering out-of-order network packets."""

from voxweave.stream import JitterBuffer

buffer = JitterBuffer(capacity=8)

# Simulate out-of-order packet arrival
packets = [
    (2, b"packet-2"),
    (0, b"packet-0"),
    (1, b"packet-1"),
    (3, b"packet-3"),
]

released = []
for sequence, payload in packets:
    ready = buffer.push(sequence, payload)
    if ready:
        released.extend(ready)

buffer.flush()
print({"received": len(packets), "released": len(released), "order": [p.decode() for p in released]})
