"""
rag_query.py
────────────
The RAG (Retrieval-Augmented Generation) engine.

This module is imported by the Discord cog. It exposes one public function:
    answer_question(question: str) -> str

Everything is loaded at module import time (model, DB connection, Groq client)
so that per-query latency is minimal — no re-loading on every Discord command.
"""

import os
import sys
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

sys.path.insert(0, str(Path(__file__).parent))
from config import Config

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "cyber_law"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Retrieval settings
PRIORITY_1_MIN_RESULTS = 3      # minimum P1 results before broadening to P2
DISTANCE_THRESHOLD = 0.65       # cosine distance above which a chunk is "not relevant"
                                 # (lower = more similar; 0.0 = identical, 1.0 = unrelated)
MAX_CONTEXT_CHUNKS = 5           # total chunks sent to the LLM

GROQ_MODEL = "llama3-70b-8192"   # best quality on free tier

# ── Module-level initialisation (runs once at import) ────────────────────────

logger.info("Loading RAG components...")

_embedding_model = SentenceTransformer(EMBEDDING_MODEL)

_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
_collection = _chroma_client.get_collection(COLLECTION_NAME)

_groq_client = Groq(api_key=Config.GROQ_API_KEY)

logger.info("RAG components loaded.")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant juridique destiné aux étudiants de l'ESI Alger, spécialisé dans le droit algérien du numérique et de la cybercriminalité.

Règles strictes :
1. Tu réponds UNIQUEMENT en te basant sur les extraits de documents fournis dans le contexte.
2. Si la réponse n'est pas dans les extraits, dis-le clairement : "Je n'ai pas trouvé d'information sur ce point dans les textes juridiques disponibles."
3. Ne jamais inventer de lois, d'articles, ou de sanctions qui ne figurent pas dans le contexte.
4. Cite toujours la source (nom du fichier et article si disponible) dans ta réponse.
5. Réponds toujours en français, même si la question est posée en arabe ou en anglais.
6. Sois précis et concis. Évite le jargon inutile — l'étudiant n'est pas juriste.
7. Si une question touche plusieurs lois, structure ta réponse par loi."""


# ── Retrieval with priority fallback ─────────────────────────────────────────

def _retrieve_chunks(question_embedding: list[float]) -> list[dict]:
    """
    Tiered retrieval strategy:
    1. Query Priority 1 documents first.
    2. If we get fewer than PRIORITY_1_MIN_RESULTS chunks with distance
       below DISTANCE_THRESHOLD, broaden to Priority 1 + 2.
    3. If still insufficient, use all documents (P1 + P2 + P3).

    Returns a list of dicts: {text, source, page, priority, distance}
    """

    def query_with_filter(priority_filter: dict | None, n: int) -> list[dict]:
        kwargs = dict(
            query_embeddings=[question_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if priority_filter:
            kwargs["where"] = priority_filter

        results = _collection.query(**kwargs)

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "priority": meta["priority"],
                "distance": dist,
            })
        return chunks

    def relevant(chunks: list[dict]) -> list[dict]:
        """Filter to chunks below the distance threshold."""
        return [c for c in chunks if c["distance"] < DISTANCE_THRESHOLD]

    # Tier 1: Priority 1 only
    p1_chunks = query_with_filter({"priority": 1}, n=PRIORITY_1_MIN_RESULTS + 2)
    p1_relevant = relevant(p1_chunks)

    if len(p1_relevant) >= PRIORITY_1_MIN_RESULTS:
        # Priority 1 has enough good results — use only those
        return sorted(p1_relevant, key=lambda c: c["distance"])[:MAX_CONTEXT_CHUNKS]

    # Tier 2: Broaden to P1 + P2
    p1_p2_chunks = query_with_filter(
        {"priority": {"$in": [1, 2]}}, n=MAX_CONTEXT_CHUNKS + 2
    )
    p1_p2_relevant = relevant(p1_p2_chunks)

    if p1_p2_relevant:
        return sorted(p1_p2_relevant, key=lambda c: c["distance"])[:MAX_CONTEXT_CHUNKS]

    # Tier 3: No filter — use everything including P3 background documents
    all_chunks = query_with_filter(None, n=MAX_CONTEXT_CHUNKS)
    return sorted(all_chunks, key=lambda c: c["distance"])[:MAX_CONTEXT_CHUNKS]


# ── Answer generation ─────────────────────────────────────────────────────────

def answer_question(question: str) -> str:
    """
    Main public function.
    Takes a student's question (string), retrieves relevant law passages,
    calls Groq, and returns a cited answer (string).
    """
    if not question.strip():
        return "Veuillez poser une question."

    # 1. Embed the question
    question_embedding = _embedding_model.encode(question).tolist()

    # 2. Retrieve relevant chunks with priority fallback
    chunks = _retrieve_chunks(question_embedding)

    if not chunks:
        return (
            "Je n'ai pas trouvé d'information pertinente dans les textes juridiques "
            "disponibles pour répondre à cette question."
        )

    # 3. Build context block for the prompt
    context_parts = []
    sources_seen: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        source_label = f"{chunk['source']} (page {chunk['page']})"
        context_parts.append(
            f"--- Extrait {i} | Source : {source_label} | "
            f"Pertinence : {(1 - chunk['distance']) * 100:.0f}% ---\n{chunk['text']}"
        )
        if source_label not in sources_seen:
            sources_seen.append(source_label)

    context_block = "\n\n".join(context_parts)

    user_message = (
        f"Contexte juridique :\n\n{context_block}\n\n"
        f"Question de l'étudiant : {question}\n\n"
        "Réponds en te basant uniquement sur le contexte fourni. "
        "Cite les articles et lois pertinents."
    )

    # 4. Call Groq
    try:
        completion = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,       # factual, not creative
            max_tokens=1024,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise RuntimeError(f"Erreur lors de la génération de la réponse : {e}") from e

    # 5. Append source citations
    sources_text = "\n\n📚 **Sources consultées :**\n" + "\n".join(
        f"• {s}" for s in sources_seen
    )

    return answer + sources_text
