"""Speech services: Deepgram STT + TTS over plain REST (no SDK, no version churn).

To swap providers (e.g. ElevenLabs for TTS, Whisper for STT), keep the same two
methods - `transcribe(pcm16k) -> str` and `synthesize(text) -> bytes` - and the
rest of the bot won't care.
"""
from __future__ import annotations

import aiohttp

from .audio import pcm_to_wav
from .config import Config


class Deepgram:
    STT_URL = "https://api.deepgram.com/v1/listen"
    TTS_URL = "https://api.deepgram.com/v1/speak"

    def __init__(self, http: aiohttp.ClientSession, cfg: Config) -> None:
        self.http = http
        self.cfg = cfg
        self._auth = {"Authorization": f"Token {cfg.deepgram_api_key}"}

    async def transcribe(self, pcm16k: bytes) -> str:
        """16 kHz mono int16 PCM -> transcript text ('' if nothing recognised)."""
        params = {
            "model": self.cfg.stt_model,
            "language": self.cfg.stt_language,
            "smart_format": "true",
        }
        headers = {**self._auth, "Content-Type": "audio/wav"}
        async with self.http.post(
            self.STT_URL, params=params, data=pcm_to_wav(pcm16k), headers=headers
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Deepgram STT {resp.status}: {(await resp.text())[:200]}")
            body = await resp.json()
        try:
            return body["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        except (KeyError, IndexError):
            return ""

    async def synthesize(self, text: str) -> bytes:
        """Return MP3 bytes for `text` (FFmpeg decodes it for Discord)."""
        params = {"model": self.cfg.tts_model, "encoding": "mp3"}
        async with self.http.post(
            self.TTS_URL, params=params, json={"text": text}, headers=self._auth
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Deepgram TTS {resp.status}: {(await resp.text())[:200]}")
            return await resp.read()