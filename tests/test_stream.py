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
