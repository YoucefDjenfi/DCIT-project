import asyncio
import logging
from config import Config
from bot.bot import DiscordBot
from utils.helpers import setup_logging


def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    # Create and run bot
    bot = DiscordBot()

    try:
        logger.info("Starting Discord bot...")
        bot.run(Config.DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

    finally:
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    main()
