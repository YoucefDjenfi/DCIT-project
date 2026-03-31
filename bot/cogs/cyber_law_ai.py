"""
bot/cogs/cyber_law_ai.py
────────────────────────
Discord Cog: DCIT Cyber-Law AI Assistant

Exposes one slash command:
    /ask-law <question>  — answers questions about Algerian cyber law using RAG

The RAG engine (rag_query.py) is imported from the project root and called
inside an asyncio thread so it doesn't block the bot's event loop.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

# Add project root to path so we can import rag_query from there
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# ── Lazy import of RAG engine ─────────────────────────────────────────────────
# We import rag_query lazily (on first use) rather than at module load time.
# This means if ChromaDB isn't built yet, the bot still starts — it just
# returns a helpful error message on the first /ask-law call instead of
# crashing on startup.

_rag_ready = False
_rag_error: str | None = None
_answer_question = None


def _load_rag_engine():
    global _rag_ready, _rag_error, _answer_question
    try:
        from rag_query import answer_question
        _answer_question = answer_question
        _rag_ready = True
        logger.info("✓ RAG engine loaded successfully")
    except FileNotFoundError:
        _rag_error = (
            "La base de données vectorielle n'existe pas encore. "
            "Lancez `python ingest.py` depuis la racine du projet pour la construire."
        )
        logger.warning(f"RAG engine not ready: {_rag_error}")
    except Exception as e:
        _rag_error = f"Erreur lors du chargement du moteur RAG : {e}"
        logger.error(f"RAG engine load failed: {e}")


# ── Discord character limit helper ────────────────────────────────────────────

MAX_DISCORD_LENGTH = 1900   # Discord limit is 2000; leave margin


def _truncate(text: str) -> str:
    if len(text) <= MAX_DISCORD_LENGTH:
        return text
    return (
        text[:MAX_DISCORD_LENGTH]
        + "\n\n*…(réponse tronquée — posez une question plus précise pour obtenir "
        "une réponse complète)*"
    )


# ── The Cog ───────────────────────────────────────────────────────────────────

class CyberLawAI(commands.Cog):
    """
    Cog that provides AI-powered answers about Algerian cyber law and
    digital citizenship, grounded in official legal texts via RAG.
    """

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # Try to load the RAG engine as soon as the cog is instantiated
        _load_rag_engine()

    # ── /ask-law ──────────────────────────────────────────────────────────────

    @app_commands.command(
        name="ask-law",
        description="Pose une question sur les lois algériennes liées au cyberespace et à la citoyenneté numérique",
    )
    @app_commands.describe(
        question="Ta question sur le droit algérien du numérique (en français)"
    )
    async def ask_law(self, interaction: discord.Interaction, question: str):
        """
        Answers a student's question using the RAG pipeline.

        Flow:
        1. defer()        — buys us 15 min instead of Discord's default 3-second timeout
        2. thread         — runs the blocking RAG call without freezing the event loop
        3. followup.send  — delivers the answer (or a clear error message)
        """

        # Immediately acknowledge to Discord (prevents "interaction failed" error)
        await interaction.response.defer()

        # Check if RAG engine is available
        if not _rag_ready:
            error_msg = _rag_error or "Le moteur RAG n'est pas disponible."
            await interaction.followup.send(
                f"⚠️ **Assistant non disponible**\n{error_msg}"
            )
            return

        # Log the query (helpful during demo / debugging)
        logger.info(
            f"[ask-law] User: {interaction.user} | Guild: {interaction.guild} | "
            f"Question: {question[:80]}{'…' if len(question) > 80 else ''}"
        )

        try:
            # Run the blocking RAG call in a thread so the bot stays responsive
            answer = await asyncio.to_thread(_answer_question, question)

            response = _truncate(
                f"**❓ Question :** {question}\n\n"
                f"**⚖️ Réponse :**\n{answer}"
            )
            await interaction.followup.send(response)

        except RuntimeError as e:
            # Known errors from rag_query (e.g. Groq API down)
            await interaction.followup.send(
                f"❌ **Erreur** : {e}\n"
                "Réessayez dans quelques instants ou contactez l'administrateur."
            )
        except Exception as e:
            logger.exception(f"Unexpected error in ask_law: {e}")
            await interaction.followup.send(
                "❌ Une erreur inattendue s'est produite. "
                "Veuillez réessayer ou signaler le problème."
            )

    # ── /law-help ─────────────────────────────────────────────────────────────

    @app_commands.command(
        name="law-help",
        description="Explique ce que fait le bot juridique et quelles lois il connaît",
    )
    async def law_help(self, interaction: discord.Interaction):
        """Static info command — no AI call, instant response."""

        status = "✅ Opérationnel" if _rag_ready else f"⚠️ Non disponible — {_rag_error}"

        embed = discord.Embed(
            title="⚖️ Assistant Juridique DCIT",
            description=(
                "Je réponds aux questions sur les lois algériennes liées au "
                "cyberespace et à la citoyenneté numérique, en me basant "
                "**uniquement** sur les textes officiels — pas d'invention."
            ),
            color=discord.Color.dark_blue(),
        )

        embed.add_field(
            name="📚 Textes couverts",
            value=(
                "• Loi 09-04 — Cybercriminalité (2009)\n"
                "• Loi 18-07 — Protection des données personnelles (2018)\n"
                "• Code pénal — Articles relatifs aux TIC\n"
                "• Décret 20-05 — Sécurité des systèmes d'information\n"
                "• Convention arabe sur la cybercriminalité (2010)\n"
                "• Loi 15-04 — Signature et certification électroniques\n"
                "• Et d'autres textes de support…"
            ),
            inline=False,
        )

        embed.add_field(
            name="💬 Comment utiliser",
            value=(
                "Tape `/ask-law` suivi de ta question en français.\n"
                "Exemple : `/ask-law Quelles sont les sanctions pour accès "
                "frauduleux à un système informatique ?`"
            ),
            inline=False,
        )

        embed.add_field(
            name="⚙️ Statut du moteur",
            value=status,
            inline=False,
        )

        embed.set_footer(
            text="ESI Alger · Module Citoyenneté Numérique et IA · Les réponses sont indicatives, pas un avis juridique."
        )

        await interaction.response.send_message(embed=embed)


# ── Required setup function ───────────────────────────────────────────────────

async def setup(bot: commands.Bot):
    await bot.add_cog(CyberLawAI(bot))
