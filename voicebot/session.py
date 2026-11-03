"""One voice session per guild: capture -> VAD -> STT -> LLM -> TTS -> playback.

While the bot is speaking (plus a short cooldown) incoming audio is dropped,
so it never answers itself or gets talked over mid-sentence.
"""
from __future__ import annotations

import asyncio
import logging
import threading
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

SAMPLE_RATE = 48_000
FRAME_BYTES = SAMPLE_RATE // 50 * 2 * 2   # 20 ms of 16-bit stereo = 3840 bytes


class StreamSource(discord.AudioSource):
    """One continuous audio stream per reply.

    Sentences are appended as they finish synthesising. Until the stream is
    finished, an empty buffer produces silence instead of ending, so Discord never
    stops and restarts speaking between sentences - those restarts are what clip
    the first word of a sentence. A short silent lead-in is also added so the very
    first word isn't lost while Discord switches the "speaking" state on.
    """

    def __init__(self, lead_in_s: float = 0.3) -> None:
        self._buf = bytearray(b"\x00" * (round(lead_in_s * 50) * FRAME_BYTES))
        self._lock = threading.Lock()
        self._finished = False

    def feed(self, pcm: bytes) -> None:
        with self._lock:
            self._buf += pcm

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    def read(self) -> bytes:  # called every 20 ms from discord's audio thread
        with self._lock:
            if len(self._buf) >= FRAME_BYTES:
                frame = bytes(self._buf[:FRAME_BYTES])
                del self._buf[:FRAME_BYTES]
                return frame
            if self._finished:
                if not self._buf:
                    return b""  # end of stream
                frame = bytes(self._buf).ljust(FRAME_BYTES, b"\x00")  # pad the last partial frame
                self._buf.clear()
                return frame
            return b"\x00" * FRAME_BYTES  # waiting for the next sentence: stay alive with silence

    def is_opus(self) -> bool:
        return False



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

                # Only react if a wake word was said (WAKE_WORD may be a comma-separated list)
                if not self.cfg.heard_wake_word(text):
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
            reply = ""
            async for delta in self.brain.stream_reply(f"{speaker}: {text}"):
                reply += delta
                buf += delta
                sentences, buf = split_sentences(buf)
                for s in sentences:
                    # Start TTS immediately; the player plays results in order.
                    await play_q.put(asyncio.create_task(self.dg.synthesize(s)))
            if buf.strip():
                await play_q.put(asyncio.create_task(self.dg.synthesize(buf.strip())))
            if reply.strip():
                log.info("bot: %s", reply.strip())
        finally:
            await play_q.put(None)
            await player
            self._clear_buffers()
            self._mute_until = self.loop.time() + self.cfg.post_speech_cooldown_s
            self.speaking = False

    async def _player(self, q: asyncio.Queue) -> None:
        """Turn queued TTS results into ONE continuous Discord audio stream per reply."""
        source = StreamSource(self.cfg.playback_lead_in_s)
        done: Optional[asyncio.Future] = None
        started = False
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                try:
                    pcm = await self._decode(await item)
                except Exception as exc:
                    log.error("TTS/decode failed: %s", exc)
                    continue
                source.feed(pcm)
                if not started:  # start playing as soon as the first sentence is ready
                    started = True
                    done = self._start_playback(source)
        finally:
            source.finish()  # lets the stream end once its buffer has drained
        if done is not None:
            err = await done
            if err:
                log.error("playback error: %s", err)

    def _start_playback(self, source: StreamSource) -> Optional[asyncio.Future]:
        if not self.vc.is_connected():
            log.warning("not connected to voice - skipping playback")
            return None
        done: asyncio.Future = self.loop.create_future()

        def after(err: Optional[Exception]) -> None:  # runs in discord's player thread
            def _set() -> None:
                if not done.done():
                    done.set_result(err)
            self.loop.call_soon_threadsafe(_set)

        try:
            self.vc.play(source, after=after)
        except Exception:
            log.exception("could not start playback")
            return None
        log.info("playing reply")
        return done

    async def _decode(self, audio: bytes) -> bytes:
        """MP3/WAV bytes -> raw 48 kHz stereo 16-bit PCM (what Discord wants), via FFmpeg."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.cfg.ffmpeg_path, "-loglevel", "error", "-i", "pipe:0",
                "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise RuntimeError(f"FFmpeg not found at '{self.cfg.ffmpeg_path}'. Set FFMPEG_PATH in .env.")
        out, err = await proc.communicate(audio)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {err.decode(errors='ignore')[:200]}")
        return out

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