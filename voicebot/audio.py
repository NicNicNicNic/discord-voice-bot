"""Audio helpers: resampling, WAV packing, and voice-activity detection."""
from __future__ import annotations

import io
import wave

import numpy as np

FRAME_MS = 20                      # discord delivers 20 ms frames
FRAMES_PER_S = 1000 // FRAME_MS    # 50
PREROLL_FRAMES = 10                # 200 ms kept before speech starts (avoids clipped first word)
STT_RATE = 16_000


def to_16k_mono(pcm48_stereo: bytes) -> np.ndarray:
    """Discord PCM (48 kHz, stereo, int16) -> 16 kHz mono int16.

    Averaging 3 neighbouring samples is a crude low-pass + decimate; it is
    plenty for speech recognition and costs almost nothing.
    """
    a = np.frombuffer(pcm48_stereo, dtype=np.int16)
    n = (a.size // 6) * 6
    if n == 0:
        return np.empty(0, dtype=np.int16)
    mono = a[:n].reshape(-1, 2).astype(np.float32).mean(axis=1)
    return mono.reshape(-1, 3).mean(axis=1).astype(np.int16)


def pcm_to_wav(pcm: bytes, rate: int = STT_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class EnergyVAD:
    """Minimal voice-activity detector based on frame loudness (RMS).

    Swappable: anything with `is_speech(samples: np.ndarray) -> bool` works,
    e.g. a Silero VAD wrapper (needs torch/onnxruntime - heavier on CPU/RAM).
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def is_speech(self, samples: np.ndarray) -> bool:
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        return rms >= self.threshold