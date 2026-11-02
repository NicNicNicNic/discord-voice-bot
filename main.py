"""Entry point:  python main.py

Pipeline: Discord voice -> VAD -> Deepgram STT -> gpt-4o-mini -> Deepgram TTS -> FFmpeg -> Discord.
Config comes from environment variables / a .env file (see .env.example).
"""
import logging
import os

from voicebot.bot import VoiceBot
from voicebot.config import Config


def main() -> None:
    cfg = Config.from_env()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    bot = VoiceBot(cfg)
    bot.run(cfg.discord_token, log_handler=None)  # we configured logging ourselves


if __name__ == "__main__":
    main()