"""Discord bot object and chat commands (!join, !leave, !persona, !reset)."""
from __future__ import annotations

import asyncio
import logging

import aiohttp
import discord
from discord.ext import commands, voice_recv
from openai import AsyncOpenAI

from .config import Config
from .session import VoiceSession

log = logging.getLogger("voicebot.bot")


class VoiceBot(commands.Bot):
    def __init__(self, cfg: Config) -> None:
        intents = discord.Intents.default()      # includes guilds + voice_states
        intents.message_content = True           # privileged: enable in the Developer Portal
        super().__init__(command_prefix=cfg.prefix, intents=intents, help_command=None)
        self.cfg = cfg
        self.persona = cfg.persona
        self.sessions: dict[int, VoiceSession] = {}
        self.http_session: aiohttp.ClientSession  # created in setup_hook
        self.openai: AsyncOpenAI

    async def setup_hook(self) -> None:
        self.http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
        self.openai = AsyncOpenAI(api_key=self.cfg.openai_api_key)
        await self.add_cog(VoiceCog(self))

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id %s)", self.user, self.user.id)

    async def on_voice_state_update(self, member, before, after) -> None:
        # If the bot gets kicked/moved out, drop its session.
        if self.user and member.id == self.user.id and after.channel is None:
            session = self.sessions.pop(member.guild.id, None)
            if session:
                session.cancel_tasks()

    async def close(self) -> None:
        for session in list(self.sessions.values()):
            await session.stop()
        self.sessions.clear()
        await self.http_session.close()
        await super().close()


class VoiceCog(commands.Cog):
    def __init__(self, bot: VoiceBot) -> None:
        self.bot = bot

    @commands.command()
    @commands.guild_only()
    async def join(self, ctx: commands.Context) -> None:
        """Join the voice channel the caller is in."""
        voice = ctx.author.voice
        if not voice or not voice.channel:
            await ctx.send("Join a voice channel first, then run the command again.")
            return
        channel = voice.channel

        perms = channel.permissions_for(ctx.guild.me)
        if not (perms.connect and perms.speak):
            await ctx.send(f"I need **Connect** and **Speak** permissions in {channel.mention}.")
            return

        # Clean up any previous/stale session in this guild.
        old = self.bot.sessions.pop(ctx.guild.id, None)
        if old:
            await old.stop()
        elif ctx.guild.voice_client:
            await ctx.guild.voice_client.disconnect(force=True)

        try:
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient, timeout=20.0, reconnect=True)
        except asyncio.TimeoutError:
            await ctx.send("Timed out connecting to voice. Try again in a moment.")
            return
        except discord.ClientException as exc:
            await ctx.send(f"Couldn't connect: `{exc}`")
            return
        except Exception as exc:  # e.g. voice close code 4017 if DAVE isn't supported
            log.exception("voice connect failed")
            await ctx.send(f"Voice connection failed: `{exc}`. Check the console for details.")
            return

        session = VoiceSession(self.bot, vc, ctx.channel)
        session.start()
        self.bot.sessions[ctx.guild.id] = session
        wake = f" Say **{self.bot.cfg.wake_word}** to get my attention." if self.bot.cfg.wake_word else ""
        await ctx.send(f"Joined {channel.mention} - I'm listening.{wake}")

    @commands.command()
    @commands.guild_only()
    async def leave(self, ctx: commands.Context) -> None:
        """Leave the voice channel."""
        session = self.bot.sessions.pop(ctx.guild.id, None)
        if session:
            await session.stop()
            await ctx.send("Left the voice channel.")
        elif ctx.guild.voice_client:
            await ctx.guild.voice_client.disconnect(force=True)
            await ctx.send("Left the voice channel.")
        else:
            await ctx.send("I'm not in a voice channel.")

    @commands.command()
    @commands.guild_only()
    async def persona(self, ctx: commands.Context, *, text: str = "") -> None:
        """Show or change the persona: !persona You are a ruthless Valorant coach..."""
        if not text:
            await ctx.send(f"Current persona:\n> {self.bot.persona}")
            return
        self.bot.persona = text
        session = self.bot.sessions.get(ctx.guild.id)
        if session:
            session.brain.persona = text
        await ctx.send("Persona updated.")

    @commands.command()
    @commands.guild_only()
    async def reset(self, ctx: commands.Context) -> None:
        """Clear the conversation memory."""
        session = self.bot.sessions.get(ctx.guild.id)
        if session:
            session.brain.reset()
        await ctx.send("Memory cleared.")

    @commands.Cog.listener()
    async def on_command_error(self, ctx: commands.Context, error: Exception) -> None:
        if isinstance(error, commands.CommandNotFound):
            return
        if isinstance(error, commands.NoPrivateMessage):
            await ctx.send("Use this command in a server.")
            return
        log.error("command error: %r", error)
        await ctx.send(f"Error: `{error}`")