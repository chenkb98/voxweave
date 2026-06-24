import numpy as np

from voxweave.audio import concatenate, pcm16_decode, pcm16_encode
from voxweave.stream import PCMDecoder, byte_chunks

signal = np.linspace(-0.5, 0.5, 64).reshape(32, 2)
payload = pcm16_encode(signal)
decoder = PCMDecoder(2)
rebuilt = concatenate([decoder.feed(chunk) for chunk in byte_chunks(payload, 7)])
decoder.flush()
np.testing.assert_array_equal(rebuilt, pcm16_decode(payload, 2))
print({"frames": len(rebuilt), "channels": 2, "packet_bytes": 7})
