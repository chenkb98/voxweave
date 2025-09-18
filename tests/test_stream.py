import numpy as np
import pytest
from voxweave import stream as m


def test_byte_chunks_example():
    assert m.byte_chunks(b"abcdefg", 3) == [b"abc", b"def", b"g"]


def test_byte_chunks_boundaries():
    for size in [0, -1, 1.5, True]:
        with pytest.raises(ValueError):
            m.byte_chunks(b"abc", size)
    with pytest.raises(ValueError):
        m.byte_chunks("abc", 2)
    assert m.byte_chunks(b"", 2) == []


def test_byte_chunks_invariants():
    payload = bytes(range(255))
    for size in range(1, 33):
        chunks = m.byte_chunks(payload, size)
        assert b"".join(chunks) == payload
        assert all(0 < len(chunk) <= size for chunk in chunks)


def test_PCMDecoder_example():
    decoder = m.PCMDecoder()
    assert decoder.feed(b"\x00").shape == (0, 1)
    np.testing.assert_array_equal(decoder.feed(b"\x80"), [[-1]])
    decoder.flush()


def test_PCMDecoder_boundaries():
    with pytest.raises(ValueError):
        m.PCMDecoder(0)
    decoder = m.PCMDecoder(2)
    decoder.feed(b"xx")
    with pytest.raises(ValueError):
        decoder.flush()
    with pytest.raises(ValueError):
        decoder.feed("xx")
    assert decoder.pending == b"xx"


def test_PCMDecoder_invariants():
    from voxweave import audio

    x = np.linspace(-1, 0.8, 24).reshape(12, 2)
    payload = audio.pcm16_encode(x)
    for size in range(1, 12):
        decoder = m.PCMDecoder(2)
        result = audio.concatenate([decoder.feed(chunk) for chunk in m.byte_chunks(payload, size)])
        decoder.flush()
        np.testing.assert_array_equal(result, audio.pcm16_decode(payload, 2))


def test_AudioFramer_example():
    framer = m.AudioFramer(3)
    assert framer.feed([1, 2]) == []
    np.testing.assert_array_equal(framer.feed([3, 4])[0].ravel(), [1, 2, 3])
    np.testing.assert_array_equal(framer.flush()[0], [[4]])


def test_AudioFramer_boundaries():
    with pytest.raises(ValueError):
        m.AudioFramer(0)
    framer = m.AudioFramer(2, 2)
    with pytest.raises(ValueError):
        framer.feed([1])
    assert framer.emitted == 0 and len(framer.pending) == 0
    assert framer.flush() == []


def test_AudioFramer_invariants():
    from voxweave import audio

    x = np.arange(17)
    for packet in range(1, 8):
        framer, result = m.AudioFramer(4), []
        for i in range(0, len(x), packet):
            result.extend(framer.feed(x[i : i + packet]))
        result.extend(framer.flush())
        np.testing.assert_array_equal(audio.concatenate(result).ravel(), x)
        assert framer.emitted == len(x) and framer.flush() == []


def test_RingBuffer_example():
    buffer = m.RingBuffer(3)
    assert buffer.append([1, 2]) == 0
    np.testing.assert_array_equal(buffer.read(1), [[1]])
    assert len(buffer) == 1


def test_RingBuffer_boundaries():
    buffer = m.RingBuffer(2)
    buffer.append([1])
    with pytest.raises(BufferError):
        buffer.append([2, 3])
    with pytest.raises(ValueError):
        buffer.append([np.nan])
    with pytest.raises(ValueError):
        buffer.read(2)
    np.testing.assert_array_equal(buffer.read(1), [[1]])


def test_RingBuffer_invariants():
    buffer = m.RingBuffer(3, overflow="drop_oldest")
    assert buffer.append([1, 2]) == 0
    assert buffer.append([3, 4, 5]) == 2
    np.testing.assert_array_equal(buffer.read(3).ravel(), [3, 4, 5])
    assert len(buffer) == 0
    for count in range(4):
        buffer.append(np.arange(count))
        np.testing.assert_array_equal(buffer.read(count).ravel(), np.arange(count))


def test_SampleClock_example():
    clock = m.SampleClock(16000)
    assert clock.advance(320) == (0, 320)
    assert clock.seconds == 0.02


def test_SampleClock_boundaries():
    for rate in [0, -1, True]:
        with pytest.raises(ValueError):
            m.SampleClock(rate)
    clock = m.SampleClock(8000, 2**53 - 1)
    with pytest.raises(ValueError):
        clock.advance(1)
    assert clock.offset == 2**53 - 1


def test_SampleClock_invariants():
    clock = m.SampleClock(44100)
    for _ in range(1000):
        clock.advance(441)
    assert clock.offset == 441000 and clock.seconds == 10
    snapshot = clock.snapshot()
    assert set(snapshot) == {"sample_rate", "offset"}
    restored = m.SampleClock(snapshot["sample_rate"], snapshot["offset"])
    assert restored.seconds == clock.seconds
