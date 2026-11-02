"""Configuration: persona text, tunables, and environment loading."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

DEFAULT_PERSONA = (
    "You are 'Glitch', a sarcastic but friendly gamer buddy hanging out in a "
    "Discord voice channel. You tease people lightly, give quick tactical "
    "advice when asked, and never get mean."
)

# Always appended to the persona so replies sound good when spoken aloud.
VOICE_RULES = (
    "You are speaking out loud in a voice channel. Reply in one to three short, "
    "natural spoken sentences. No markdown, lists, emojis or code. "
    "Each user message is prefixed with the speaker's name."
)


def _env(name: str, default: Optional[str] = None) -> str:
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


@dataclass
class Config:
    discord_token: str
    openai_api_key: str
    deepgram_api_key: str
    prefix: str = "!"
    persona: str = DEFAULT_PERSONA
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""         # optional: point at any OpenAI-compatible provider
    stt_model: str = "nova-3"
    stt_language: str = "en"
    tts_model: str = "aura-2-thalia-en"
    max_turns: int = 10               # sliding memory: last N user/assistant turns
    max_reply_tokens: int = 120       # keep spoken answers short (and fast)
    vad_threshold: float = 600.0      # RMS (int16) above which a frame counts as speech
    silence_s: float = 0.7            # silence that ends an utterance
    min_speech_s: float = 0.3         # ignore blips shorter than this (coughs, clicks)
    max_utterance_s: float = 20.0     # hard cap so a stuck mic can't grow forever
    post_speech_cooldown_s: float = 0.4  # keep ignoring input briefly after we talk
    wake_word: str = ""               # optional; if set, only respond when heard
    ffmpeg_path: str = "ffmpeg"

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            discord_token=_env("DISCORD_TOKEN"),
            openai_api_key=_env("OPENAI_API_KEY"),
            deepgram_api_key=_env("DEEPGRAM_API_KEY"),
            prefix=os.getenv("COMMAND_PREFIX", "!"),
            persona=os.getenv("BOT_PERSONA", DEFAULT_PERSONA),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
            stt_model=os.getenv("DEEPGRAM_STT_MODEL", "nova-3"),
            stt_language=os.getenv("DEEPGRAM_LANGUAGE", "en"),
            tts_model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en"),
            max_turns=int(os.getenv("MAX_TURNS", "10")),
            max_reply_tokens=int(os.getenv("MAX_REPLY_TOKENS", "120")),
            vad_threshold=float(os.getenv("VAD_THRESHOLD", "600")),
            silence_s=float(os.getenv("SILENCE_MS", "700")) / 1000,
            wake_word=os.getenv("WAKE_WORD", "").strip().lower(),
            ffmpeg_path=os.getenv("FFMPEG_PATH", "ffmpeg"),
        )