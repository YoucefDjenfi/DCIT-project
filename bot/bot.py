# bot/bot.py
import discord
from discord.ext import commands
import logging
from config import Config
from database.connection import db

logger = logging.getLogger(__name__)


class DiscordBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.members = True
        intents.message_content = True

        super().__init__(
            command_prefix=Config.COMMAND_PREFIX, intents=intents, help_command=None
        )

        # list of cogs to load
        self.initial_extensions = [
            "bot.cogs.help_commands",
            "bot.cogs.cyberfacts_commands",
            "bot.cogs.events_commands",
            "bot.cogs.command_management",
            "bot.cogs.error_handler",
            "bot.cogs.banned_words",
            "bot.cogs.quiz_commands",
            "bot.cogs.event_reminder",
            "bot.cogs.cyber_law_ai",
        ]

    async def setup_hook(self):

        # Connect to DB
        try:
            await db.get_connection()
            logger.info("✓ Database connected")
        except Exception as e:
            logger.error(f"✗ Database connection failed: {e}")

        # Load cogs
        for ext in self.initial_extensions:
            try:
                await self.load_extension(ext)
                logger.info(f"✓ Loaded cog: {ext}")
            except Exception as e:
                logger.error(f"✗ Failed to load {ext}: {e}")

    async def on_ready(self):
    logger.info(f"Bot ready! Logged in as {self.user} (ID: {self.user})")
    logger.info(f"Connected to {len(self.guilds)} guild(s)")
    try:
        synced = await self.tree.sync()
        logger.info(f"✓ Synced {len(synced)} slash command(s)")
    except Exception as e:
        logger.error(f"✗ Failed to sync commands: {e}")
    
    async def close(self):
        db.close_all_connections()
        await super().close()
