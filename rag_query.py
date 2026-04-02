"""
rag_query.py  v3
────────────────
Changes from v2:
- Cross-encoder reranker: initial retrieval fetches 20 candidates, then a
  cross-encoder scores each (question, chunk) pair semantically. This fixes
  the Nmap problem — the Penal Code chunk about "accès frauduleux" now ranks
  above the Loi 18-07 chunk about "réseaux publics" even though cosine
  similarity favoured 18-07.
- max_tokens reduced to 450 so responses fit Discord without truncation
- System prompt hardened: 4-sentence limit, no hedging unless genuinely absent
- Query expansion extended with article numbers as search terms
"""

import sys
import logging
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq

sys.path.insert(0, str(Path(__file__).parent))
from config import Config

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_DIR       = "./chroma_db"
COLLECTION_NAME     = "cyber_law"
EMBEDDING_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GROQ_MODEL          = "llama-3.3-70b-versatile"

# Retrieval: fetch wide, then rerank down to a small context
INITIAL_FETCH       = 20    # candidates pulled from ChromaDB before reranking
MAX_CONTEXT_CHUNKS  = 4     # chunks actually sent to the LLM after reranking
DISTANCE_THRESHOLD  = 0.80  # looser than before — reranker does precision work

# ── Query expansion ────────────────────────────────────────────────────────────

QUERY_EXPANSION: dict[str, list[str]] = {
    "nmap": [
        "accès frauduleux", "système de traitement automatisé de données",
        "intrusion réseau", "394 bis", "accès non autorisé système informatique",
    ],
    "scan": ["accès frauduleux", "système de traitement automatisé", "394 bis"],
    "scanning": ["accès frauduleux", "système de traitement automatisé", "394 bis"],
    "wireshark": ["interception", "écoute réseau", "accès frauduleux", "394 ter"],
    "sniffing": ["interception des communications", "écoute illégale", "accès frauduleux"],
    "ddos": ["atteinte au fonctionnement", "système de traitement automatisé", "394 quater"],
    "dos": ["atteinte au fonctionnement du système", "394 quater"],
    "hacking": ["accès frauduleux", "394 bis", "394 ter"],
    "hack": ["accès frauduleux", "394 bis", "système informatique"],
    "exploit": ["accès frauduleux", "vulnérabilité", "394 bis"],
    "bruteforce": ["accès frauduleux", "tentative accès non autorisé", "394 bis"],
    "phishing": ["fraude informatique", "usurpation d'identité", "escroquerie"],
    "malware": ["logiciel malveillant", "atteinte aux données", "394 quater"],
    "ransomware": ["atteinte aux données", "extorsion", "394 quater"],
    "virus": ["logiciel malveillant", "atteinte système informatique", "394 quater"],
    "keylogger": ["logiciel espion", "interception", "données à caractère personnel"],
    "spyware": ["logiciel espion", "surveillance illégale", "données personnelles"],
    "sql injection": ["accès frauduleux", "altération de données", "394 ter"],
    "injection": ["accès frauduleux", "altération de données", "394 ter"],
    "xss": ["atteinte aux données", "fraude informatique"],
    "tracking": ["données à caractère personnel", "vie privée", "loi 18-07"],
    "cookies": ["données à caractère personnel", "consentement", "loi 18-07"],
    "vpn": ["réseau privé virtuel", "anonymisation", "communications électroniques"],
    "tor": ["anonymisation", "accès réseau", "communications électroniques"],
    "cyberbullying": ["harcèlement en ligne", "cyberharcèlement", "atteinte à la dignité"],
    "harcèlement": ["cyberharcèlement", "harcèlement", "atteinte vie privée"],
    "fake news": ["désinformation", "fausses nouvelles", "atteinte ordre public"],
    "deepfake": ["usurpation d'identité", "atteinte à la dignité", "falsification"],
    "doxxing": ["divulgation données personnelles", "atteinte vie privée", "loi 18-07"],
    "wifi": ["système de traitement automatisé", "réseau communications électroniques", "394 bis"],
    "wi-fi": ["système de traitement automatisé", "accès non autorisé", "394 bis"],
    "réseau": ["système de traitement automatisé de données", "communications électroniques"],
    "serveur": ["système de traitement automatisé de données", "infrastructure"],
    "cryptage": ["chiffrement", "signature électronique", "loi 15-04"],
    "chiffrement": ["signature électronique", "loi 15-04", "certification électronique"],
    "mot de passe": ["accès frauduleux", "données d'authentification", "394 bis"],
    "password": ["accès frauduleux", "données d'authentification", "394 bis"],
}


def expand_query(query: str) -> str:
    query_lower = query.lower()
    extra: list[str] = []
    for term, expansions in QUERY_EXPANSION.items():
        if re.search(r"\b" + re.escape(term) + r"\b", query_lower):
            extra.extend(expansions)
    if extra:
        seen: set[str] = set()
        unique = [t for t in extra if not (t in seen or seen.add(t))]
        expanded = query + " | " + " | ".join(unique)
        logger.debug(f"Expanded: {expanded}")
        return expanded
    return query


# ── Module-level init ─────────────────────────────────────────────────────────

logger.info("Loading RAG components...")

_embed_model    = SentenceTransformer(EMBEDDING_MODEL)
_cross_encoder  = CrossEncoder(CROSS_ENCODER_MODEL)

_chroma         = chromadb.PersistentClient(path=CHROMA_DB_DIR)
_collection     = _chroma.get_collection(COLLECTION_NAME)
_groq           = Groq(api_key=Config.GROQ_API_KEY)

logger.info("RAG components loaded.")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant juridique pour étudiants de l'ESI Alger, spécialisé en droit algérien du numérique.

Règles STRICTES — respecte-les sans exception :
1. Réponds en MAXIMUM 4 phrases. Pas plus. Jamais.
2. Base-toi UNIQUEMENT sur les extraits fournis. N'invente rien.
3. Si la réponse est absente des extraits, réponds : "Cette question dépasse le contenu des textes juridiques disponibles." — puis arrête-toi.
4. Cite toujours la loi et l'article (ex : Art. 394 bis du Code pénal).
5. Va droit au but : commence par la réponse, pas par des généralités.
6. Si la question mentionne un outil technique (Nmap, DDoS, etc.), dis explicitement quelle infraction il constitue selon les extraits.
7. Réponds toujours en français."""


# ── Retrieval + reranking ─────────────────────────────────────────────────────

def _retrieve_and_rerank(question: str, expanded: str) -> list[dict]:
    """
    1. Embed the expanded query and fetch INITIAL_FETCH candidates from ChromaDB
       (no priority filter at this stage — cast a wide net).
    2. Rerank all candidates using a cross-encoder scoring (question, chunk_text).
    3. Return the top MAX_CONTEXT_CHUNKS by reranker score.
    """
    q_emb = _embed_model.encode(expanded).tolist()

    results = _collection.query(
        query_embeddings=[q_emb],
        n_results=min(INITIAL_FETCH, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    candidates = [
        {
            "text":     doc,
            "source":   meta["source"],
            "page":     meta["page"],
            "priority": meta["priority"],
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
        if dist < DISTANCE_THRESHOLD
    ]

    if not candidates:
        # Fallback: loosen threshold, take whatever we have
        candidates = [
            {"text": doc, "source": meta["source"], "page": meta["page"],
             "priority": meta["priority"], "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # Cross-encoder reranking: score each (original_question, chunk) pair
    pairs = [(question, c["text"]) for c in candidates]
    scores = _cross_encoder.predict(pairs)

    for i, score in enumerate(scores):
        candidates[i]["rerank_score"] = float(score)

    # Sort by reranker score descending (higher = more relevant)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    return candidates[:MAX_CONTEXT_CHUNKS]


# ── Main public function ──────────────────────────────────────────────────────

def answer_question(question: str) -> str:
    if not question.strip():
        return "Veuillez poser une question."

    expanded = expand_query(question)
    chunks   = _retrieve_and_rerank(question, expanded)

    if not chunks:
        return "Je n'ai pas trouvé d'information pertinente dans les textes juridiques disponibles."

    context_parts = []
    sources_seen: list[str] = []

    for i, c in enumerate(chunks, 1):
        label = f"{c['source']} (page {c['page']})"
        context_parts.append(
            f"--- Extrait {i} | {label} | "
            f"Score : {c.get('rerank_score', 0):.2f} ---\n{c['text']}"
        )
        if label not in sources_seen:
            sources_seen.append(label)

    context_block = "\n\n".join(context_parts)

    user_message = (
        f"Contexte juridique :\n\n{context_block}\n\n"
        f"Question : {question}\n\n"
        "Réponds en 4 phrases maximum. Cite la loi et l'article. "
        "Si un outil technique est mentionné, dis quelle infraction il constitue."
    )

    try:
        resp = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.05,
            max_tokens=450,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        raise RuntimeError(f"Erreur lors de la génération : {e}") from e

    sources_text = "\n\n📚 **Sources :**\n" + "\n".join(f"• {s}" for s in sources_seen)
    return answer + sources_text
