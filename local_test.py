"""local_test.py - exercise the voice pipeline WITHOUT Discord.

Uses your microphone and speakers instead of a voice channel, but runs the same
code for STT (Deepgram), the LLM (OpenAI), sentence splitting, TTS (Deepgram)
and FFmpeg decoding, so you can check keys, latency, persona and VAD tuning.

Run from the project root:

    python local_test.py --levels          # live mic meter, to tune VAD_THRESHOLD (no API keys needed)
    python local_test.py                   # talk to it: mic -> STT -> LLM -> TTS -> speakers
    python local_test.py --text            # type instead of speak (still answers out loud)
    python local_test.py --text --mute     # type, and save replies as reply_N.mp3 instead of playing
    python local_test.py --list-devices    # show audio devices (use with --input-device/--output-device)

Extra install for this script only:  pip install sounddevice
(needs PortAudio: bundled on Windows/macOS wheels; on Linux `sudo apt install libportaudio2`)

NOT covered here: the Discord connection, DAVE decryption and the voice-recv sink.
Those only get tested for real inside a voice channel.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import time
from collections import deque

import aiohttp
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from voicebot.audio import FRAMES_PER_S, PREROLL_FRAMES, EnergyVAD
from voicebot.brain import Brain, split_sentences
from voicebot.config import Config
from voicebot.services import Deepgram

MIC_RATE = 16_000                    # record straight at the rate STT wants
BLOCK = MIC_RATE // FRAMES_PER_S     # 320 samples = 20 ms, same frame size as Discord
PLAY_RATE = 48_000


# --------------------------------------------------------------------------- #
# Playback helpers
# --------------------------------------------------------------------------- #

async def decode_mp3(ffmpeg_path: str, mp3: bytes) -> np.ndarray:
    """MP3 bytes -> int16 mono samples via FFmpeg (this also tests your FFmpeg install)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            ffmpeg_path, "-loglevel", "error", "-i", "pipe:0",
            "-f", "s16le", "-ar", str(PLAY_RATE), "-ac", "1", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError(f"FFmpeg not found at '{ffmpeg_path}'. Install it or set FFMPEG_PATH in .env.")
    out, err = await proc.communicate(mp3)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {err.decode(errors='ignore')[:200]}")
    return np.frombuffer(out, dtype=np.int16)


class Speaker:
    """One persistent output stream per reply.

    Re-opening the audio device for every sentence can clip the start of each
    one (Bluetooth headsets are the worst for this), so we keep a single stream
    open, add a short silent lead-in to wake the device, and pad the tail.
    """

    PAD_S = 0.3

    def __init__(self, device) -> None:
        self.device = device
        self.stream = None

    def _silence(self) -> np.ndarray:
        return np.zeros((int(PLAY_RATE * self.PAD_S), 1), dtype=np.int16)

    def _open(self) -> None:
        import sounddevice as sd
        self.stream = sd.OutputStream(samplerate=PLAY_RATE, channels=1, dtype="int16", device=self.device)
        self.stream.start()
        self.stream.write(self._silence())  # lead-in so the device is awake before speech

    async def write(self, samples: np.ndarray) -> None:
        if self.stream is None:
            await asyncio.to_thread(self._open)
        # blocking write runs in a thread so the event loop stays free
        await asyncio.to_thread(self.stream.write, samples.reshape(-1, 1))

    async def close(self) -> None:
        if self.stream is None:
            return

        def _close() -> None:
            self.stream.write(self._silence())  # tail pad so the last word isn't cut
            self.stream.stop()                  # waits for buffered audio to finish
            self.stream.close()

        await asyncio.to_thread(_close)
        self.stream = None


# --------------------------------------------------------------------------- #
# LLM -> TTS -> speakers (same flow as VoiceSession._respond)
# --------------------------------------------------------------------------- #

async def respond(cfg: Config, brain: Brain, dg: Deepgram, args, counter, speaker: str, text: str) -> None:
    play_q: asyncio.Queue = asyncio.Queue()

    async def player() -> None:
        speaker = Speaker(args.output_device)
        try:
            while True:
                item = await play_q.get()
                if item is None:
                    return
                try:
                    mp3 = await item
                    if args.mute:
                        path = f"reply_{next(counter)}.mp3"
                        with open(path, "wb") as f:
                            f.write(mp3)
                        print(f"  [saved {path}]")
                    else:
                        await speaker.write(await decode_mp3(cfg.ffmpeg_path, mp3))
                except Exception as exc:
                    print(f"\n  [TTS/playback error] {exc}")
        finally:
            await speaker.close()

    player_task = asyncio.create_task(player())
    t0 = time.perf_counter()
    first_token = None
    buf = ""
    print("bot> ", end="", flush=True)
    try:
        async for delta in brain.stream_reply(f"{speaker}: {text}"):
            if first_token is None:
                first_token = time.perf_counter() - t0
            print(delta, end="", flush=True)
            buf += delta
            sentences, buf = split_sentences(buf)
            for s in sentences:
                await play_q.put(asyncio.create_task(dg.synthesize(s)))
        if buf.strip():
            await play_q.put(asyncio.create_task(dg.synthesize(buf.strip())))
    finally:
        await play_q.put(None)
        await player_task
    if first_token is not None:
        print(f"\n  [first LLM token after {first_token:.2f}s]")


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

async def run_text(cfg, brain, dg, args) -> None:
    counter = itertools.count(1)
    print("Type a message and press Enter. Empty line quits.")
    while True:
        try:
            line = await asyncio.to_thread(input, "you> ")
        except EOFError:
            break
        if not line.strip():
            break
        await respond(cfg, brain, dg, args, counter, "Tester", line.strip())


async def run_mic(cfg, brain, dg, args) -> None:
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    vad = EnergyVAD(cfg.vad_threshold)
    frames_q: asyncio.Queue = asyncio.Queue()
    counter = itertools.count(1)
    max_frames = int(cfg.max_utterance_s * FRAMES_PER_S)

    def callback(indata, frames, time_info, status) -> None:  # audio thread
        loop.call_soon_threadsafe(frames_q.put_nowait, indata[:, 0].copy())

    preroll: deque = deque(maxlen=PREROLL_FRAMES)
    chunks: list[bytes] = []
    active, voiced, last_voice = False, 0, 0.0

    print(f"Listening (VAD threshold {cfg.vad_threshold:.0f}, silence {cfg.silence_s * 1000:.0f} ms). "
          "Speak, then pause. Ctrl+C to quit.")
    if cfg.wake_word:
        print(f"Wake word is set: say '{cfg.wake_word}' in your sentence.")

    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="int16", blocksize=BLOCK,
                        device=args.input_device, callback=callback):
        while True:
            try:
                frame = await asyncio.wait_for(frames_q.get(), timeout=0.1)
            except asyncio.TimeoutError:
                frame = None
            now = loop.time()

            if frame is not None:
                if vad.is_speech(frame):
                    last_voice = now
                    voiced += 1
                    if not active:
                        active = True
                        chunks = list(preroll)
                        preroll.clear()
                if active:
                    chunks.append(frame.tobytes())
                else:
                    preroll.append(frame.tobytes())

            if active and (now - last_voice >= cfg.silence_s or len(chunks) >= max_frames):
                pcm, voiced_s = b"".join(chunks), voiced / FRAMES_PER_S
                active, chunks, voiced = False, [], 0
                preroll.clear()
                if voiced_s < cfg.min_speech_s:
                    continue  # blip, ignore
                try:
                    t = time.perf_counter()
                    text = await dg.transcribe(pcm)
                    print(f"you> {text}  [STT {time.perf_counter() - t:.2f}s]")
                    if len(text) < 2:
                        print("  (nothing recognised)")
                    elif cfg.wake_word and cfg.wake_word not in text.lower():
                        print("  (wake word not heard - ignoring)")
                    else:
                        await respond(cfg, brain, dg, args, counter, "Tester", text)
                except Exception as exc:
                    print(f"  [pipeline error] {exc}")
                # Throw away audio captured while we were talking, so it can't hear itself.
                while not frames_q.empty():
                    frames_q.get_nowait()
                preroll.clear()


async def run_levels(threshold: float, device) -> None:
    """Live RMS meter so you can pick a VAD_THRESHOLD. Needs no API keys."""
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def callback(indata, frames, time_info, status) -> None:
        loop.call_soon_threadsafe(q.put_nowait, indata[:, 0].copy())

    print(f"Current VAD_THRESHOLD = {threshold:.0f}. Talk normally, then stay quiet, and note the RMS "
          "values.\nSet the threshold between your background-noise level and your speaking level. "
          "Ctrl+C to stop.\n")
    window: list[np.ndarray] = []
    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="int16", blocksize=BLOCK,
                        device=device, callback=callback):
        while True:
            window.append(await q.get())
            if len(window) >= 5:  # update every 100 ms
                chunk = np.concatenate(window)
                window = []
                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
                bar = "#" * min(50, int(rms / 100))
                flag = "SPEECH" if rms >= threshold else "      "
                print(f"\rRMS {rms:6.0f} {bar:<50} {flag}", end="", flush=True)


async def run(cfg: Config, args) -> None:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as http:
        dg = Deepgram(http, cfg)
        client = AsyncOpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url or None)
        brain = Brain(client, cfg, cfg.persona)
        if args.text:
            await run_text(cfg, brain, dg, args)
        else:
            await run_mic(cfg, brain, dg, args)


def main() -> None:
    p = argparse.ArgumentParser(description="Test the voice pipeline without Discord.")
    p.add_argument("--text", action="store_true", help="type messages instead of using the microphone")
    p.add_argument("--mute", action="store_true", help="save replies as reply_N.mp3 instead of playing them")
    p.add_argument("--levels", action="store_true", help="show a live mic level meter for tuning VAD_THRESHOLD")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--input-device", type=int, default=None, help="microphone device number")
    p.add_argument("--output-device", type=int, default=None, help="speaker device number")
    args = p.parse_args()

    load_dotenv()
    try:
        if args.list_devices:
            import sounddevice as sd
            print(sd.query_devices())
            return
        if args.levels:
            asyncio.run(run_levels(float(os.getenv("VAD_THRESHOLD", "600")), args.input_device))
            return
        os.environ.setdefault("DISCORD_TOKEN", "unused-in-local-test")  # Config requires it; we don't
        asyncio.run(run(Config.from_env(), args))
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()