"""One voice session per guild: capture -> VAD -> STT -> LLM -> TTS -> playback.

While the bot is speaking (plus a short cooldown) incoming audio is dropped,
so it never answers itself or gets talked over mid-sentence.
"""
from __future__ import annotations

import asyncio
import io
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import discord
from discord.ext import voice_recv

from .audio import FRAMES_PER_S, PREROLL_FRAMES, EnergyVAD, to_16k_mono
from .brain import Brain, split_sentences
from .services import Deepgram

if TYPE_CHECKING:  # avoid a circular import at runtime
    from .bot import VoiceBot

log = logging.getLogger("voicebot.session")


@dataclass
class UserState:
    name: str
    active: bool = False
    last_voice: float = 0.0
    voiced_frames: int = 0
    chunks: list[bytes] = field(default_factory=list)
    preroll: deque = field(default_factory=lambda: deque(maxlen=PREROLL_FRAMES))


class VADSink(voice_recv.AudioSink):
    """Receives decoded PCM from discord-ext-voice-recv.

    IMPORTANT: `write` is called from a background thread, not the event loop,
    so we hop onto the loop with call_soon_threadsafe and do nothing heavy here.
    """

    def __init__(self, session: "VoiceSession") -> None:
        super().__init__()
        self.session = session

    def wants_opus(self) -> bool:
        return False  # we want decoded PCM

    def write(self, user, data: voice_recv.VoiceData) -> None:
        if user is None or getattr(user, "bot", False):
            return
        if self.session.deaf:  # bot is speaking -> drop audio cheaply, in-thread
            return
        self.session.loop.call_soon_threadsafe(
            self.session.on_frame, user.id, user.display_name, data.pcm
        )

    def cleanup(self) -> None:
        pass


class VoiceSession:
    def __init__(self, bot: "VoiceBot", vc: voice_recv.VoiceRecvClient,
                 text_channel: discord.abc.Messageable) -> None:
        self.bot = bot
        self.cfg = bot.cfg
        self.vc = vc
        self.text_channel = text_channel
        self.loop = asyncio.get_running_loop()
        self.dg = Deepgram(bot.http_session, bot.cfg)
        self.brain = Brain(bot.openai, bot.cfg, bot.persona)
        self.vad = EnergyVAD(bot.cfg.vad_threshold)

        self.users: dict[int, UserState] = {}
        self.utterances: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue(maxsize=3)
        self.speaking = False
        self._mute_until = 0.0
        self._last_report = 0.0
        self._tasks: list[asyncio.Task] = []

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self.vc.listen(VADSink(self))
        self._tasks = [
            asyncio.create_task(self._silence_watcher(), name="silence-watcher"),
            asyncio.create_task(self._worker(), name="pipeline-worker"),
        ]

    def cancel_tasks(self) -> None:
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    async def stop(self) -> None:
        self.cancel_tasks()
        try:
            self.vc.stop_listening()
        except Exception:
            pass
        try:
            if self.vc.is_playing():
                self.vc.stop()
            await self.vc.disconnect(force=True)
        except Exception:
            log.exception("error while disconnecting")

    @property
    def deaf(self) -> bool:
        """True while we should ignore incoming audio."""
        return self.speaking or self.loop.time() < self._mute_until

    # -- capture + VAD (runs on the event loop, one call per 20 ms frame) ----

    def on_frame(self, uid: int, name: str, pcm48: bytes) -> None:
        if self.deaf:
            return
        st = self.users.get(uid)
        if st is None:
            st = self.users[uid] = UserState(name=name)
        st.name = name

        samples = to_16k_mono(pcm48)
        if samples.size == 0:
            return
        frame = samples.tobytes()
        now = self.loop.time()

        if self.vad.is_speech(samples):
            st.last_voice = now
            st.voiced_frames += 1
            if not st.active:
                st.active = True
                st.chunks = list(st.preroll)  # include the lead-in so words aren't clipped
                st.preroll.clear()
        if st.active:
            st.chunks.append(frame)
        else:
            st.preroll.append(frame)

    async def _silence_watcher(self) -> None:
        """Discord sends no packets during silence, so end-of-speech needs a timer."""
        max_frames = int(self.cfg.max_utterance_s * FRAMES_PER_S)
        while True:
            await asyncio.sleep(0.1)
            now = self.loop.time()
            for st in list(self.users.values()):
                if not st.active:
                    continue
                if now - st.last_voice >= self.cfg.silence_s or len(st.chunks) >= max_frames:
                    self._finalize(st)

    def _finalize(self, st: UserState) -> None:
        pcm = b"".join(st.chunks)
        voiced_s = st.voiced_frames / FRAMES_PER_S
        st.active, st.chunks, st.voiced_frames = False, [], 0
        st.preroll.clear()
        if voiced_s < self.cfg.min_speech_s:
            return  # too short to be real speech
        try:
            self.utterances.put_nowait((st.name, pcm))
        except asyncio.QueueFull:
            log.warning("utterance queue full - dropping audio from %s", st.name)

    def _clear_buffers(self) -> None:
        for st in self.users.values():
            st.active, st.chunks, st.voiced_frames = False, [], 0
            st.preroll.clear()

    # -- STT -> LLM -> TTS -> playback --------------------------------------

    async def _worker(self) -> None:
        while True:
            name, pcm = await self.utterances.get()
            try:
                text = await self.dg.transcribe(pcm)
                if len(text) < 2:
                    continue
                log.info("%s: %s", name, text)
                
                # Clean up and check wake word
                wake_word = getattr(self.cfg, "wake_word", "").strip().lower().strip('"\'')
                if wake_word and wake_word not in text.lower():
                    continue
                    
                await self._respond(name, text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("pipeline error")
                await self._report(f"Voice pipeline error: `{exc}`")

    async def _respond(self, speaker: str, text: str) -> None:
        self.speaking = True           # stop listening
        self._clear_buffers()
        play_q: asyncio.Queue = asyncio.Queue()
        player = asyncio.create_task(self._player(play_q))
        try:
            buf = ""
            async for delta in self.brain.stream_reply(f"{speaker}: {text}"):
                buf += delta
                sentences, buf = split_sentences(buf)
                for s in sentences:
                    # Start TTS immediately; the player plays results in order.
                    await play_q.put(asyncio.create_task(self.dg.synthesize(s)))
            if buf.strip():
                await play_q.put(asyncio.create_task(self.dg.synthesize(buf.strip())))
        finally:
            await play_q.put(None)
            await player
            self._clear_buffers()
            self._mute_until = self.loop.time() + self.cfg.post_speech_cooldown_s
            self.speaking = False

    async def _player(self, q: asyncio.Queue) -> None:
        while True:
            item = await q.get()
            if item is None:
                return
            try:
                audio = await item
            except Exception as exc:
                log.error("TTS failed: %s", exc)
                continue
            await self._play(audio)

    async def _play(self, audio: bytes) -> None:
        if not self.vc.is_connected():
            return
        done: asyncio.Future = self.loop.create_future()

        def after(err: Optional[Exception]) -> None:  # runs in discord's player thread
            def _set() -> None:
                if not done.done():
                    done.set_result(err)
            self.loop.call_soon_threadsafe(_set)

        source = discord.FFmpegPCMAudio(
            io.BytesIO(audio), pipe=True, executable=self.cfg.ffmpeg_path, options="-vn"
        )
        self.vc.play(source, after=after)
        err = await done
        if err:
            log.error("playback error: %s", err)

    async def _report(self, message: str) -> None:
        """Post an error to the text channel, at most once every 30 s."""
        now = self.loop.time()
        if now - self._last_report < 30:
            return
        self._last_report = now
        try:
            await self.text_channel.send(message)
        except discord.HTTPException:
            pass