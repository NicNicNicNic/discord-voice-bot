"""LLM side: streamed OpenAI replies, sliding memory, and sentence splitting."""
from __future__ import annotations

import re
from collections import deque
from typing import AsyncIterator

from openai import AsyncOpenAI

from .config import VOICE_RULES, Config

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(buf: str, min_chars: int = 25) -> tuple[list[str], str]:
    """Pull complete sentences out of a streaming text buffer.

    Very short sentences ("Sure.") are merged with the next one so we do not
    fire a TTS request for every two words.
    Returns (finished_sentences, leftover_buffer).
    """
    out: list[str] = []
    start = 0
    for m in _SENT_RE.finditer(buf):
        if m.start() - start >= min_chars:
            sentence = buf[start:m.start()].strip()
            if sentence:
                out.append(sentence)
            start = m.end()
    return out, buf[start:]


class Brain:
    def __init__(self, client: AsyncOpenAI, cfg: Config, persona: str) -> None:
        self.client = client
        self.cfg = cfg
        self.persona = persona
        # 10 turns = 10 user messages + 10 assistant messages
        self.history: deque[dict] = deque(maxlen=cfg.max_turns * 2)

    def reset(self) -> None:
        self.history.clear()

    async def stream_reply(self, user_text: str) -> AsyncIterator[str]:
        """Yield the reply token-by-token; store the turn in memory when done."""
        messages = [{"role": "system", "content": f"{self.persona}\n\n{VOICE_RULES}"}]
        messages += list(self.history)
        messages.append({"role": "user", "content": user_text})

        parts: list[str] = []
        try:
            stream = await self.client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=messages,
                stream=True,
                max_tokens=self.cfg.max_reply_tokens,
                temperature=0.8,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    yield delta
        finally:
            if parts:
                self.history.append({"role": "user", "content": user_text})
                self.history.append({"role": "assistant", "content": "".join(parts)})