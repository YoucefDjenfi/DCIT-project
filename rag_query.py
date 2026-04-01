"""
rag_query.py
────────────
RAG engine with query expansion for technical cybersecurity terms.

Key improvements over v1:
- Query expansion: technical terms (Nmap, DDoS, phishing, etc.) are
  automatically enriched with their French legal equivalents before
  embedding, bridging the gap between student vocabulary and legal text
- Model updated to llama-3.3-70b-versatile (llama3-70b-8192 decommissioned)
- Distance threshold tuned for the cleaned, smaller corpus
"""

import os
import sys
import logging
import re
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

PRIORITY_1_MIN_RESULTS = 3
DISTANCE_THRESHOLD = 0.70
MAX_CONTEXT_CHUNKS = 5

GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Query expansion dictionary ────────────────────────────────────────────────
# Maps technical cybersecurity terms to French legal equivalents.
# When a user query contains a key, the corresponding legal terms are
# appended to the query BEFORE embedding, improving retrieval.

QUERY_EXPANSION: dict[str, list[str]] = {
    # Network scanning / reconnaissance
    "nmap": ["accès frauduleux", "système de traitement automatisé de données", "intrusion réseau", "scan réseau non autorisé"],
    "scan": ["accès frauduleux", "système de traitement automatisé", "intrusion"],
    "scanning": ["accès frauduleux", "système de traitement automatisé"],
    "reconnaissance": ["accès frauduleux", "système informatique", "données informatiques"],
    "wireshark": ["interception", "écoute réseau", "données informatiques", "accès frauduleux"],
    "sniffing": ["interception des communications", "écoute illégale", "accès frauduleux"],

    # Attacks
    "ddos": ["attaque informatique", "atteinte au fonctionnement", "système de traitement automatisé"],
    "dos": ["atteinte au fonctionnement du système", "sabotage informatique"],
    "hacking": ["accès frauduleux", "intrusion informatique", "394 bis"],
    "hack": ["accès frauduleux", "intrusion", "système informatique"],
    "exploit": ["accès frauduleux", "vulnérabilité informatique", "atteinte au système"],
    "bruteforce": ["accès frauduleux", "tentative d'accès non autorisé"],
    "phishing": ["fraude informatique", "usurpation d'identité", "escroquerie en ligne"],
    "malware": ["logiciel malveillant", "atteinte aux données", "sabotage informatique"],
    "ransomware": ["atteinte aux données", "extorsion", "chiffrement non autorisé"],
    "virus": ["logiciel malveillant", "atteinte au système informatique"],
    "keylogger": ["logiciel espion", "interception", "données à caractère personnel"],
    "spyware": ["logiciel espion", "surveillance illégale", "données personnelles"],
    "sql injection": ["accès frauduleux", "altération de données", "base de données"],
    "injection": ["accès frauduleux", "altération de données"],
    "xss": ["atteinte aux données", "fraude informatique"],

    # Privacy / data
    "tracking": ["traçage", "données à caractère personnel", "vie privée", "loi 18-07"],
    "cookies": ["données à caractère personnel", "consentement", "loi 18-07"],
    "données personnelles": ["données à caractère personnel", "loi 18-07", "traitement", "consentement"],
    "vpn": ["réseau privé virtuel", "anonymisation", "communications électroniques"],
    "tor": ["anonymisation", "accès au réseau", "communications électroniques"],

    # Social engineering / online offences
    "cyberbullying": ["harcèlement en ligne", "cyberharcèlement", "atteinte à la dignité"],
    "harcèlement": ["cyberharcèlement", "harcèlement en ligne", "atteinte à la vie privée"],
    "fake news": ["désinformation", "diffusion de fausses nouvelles", "atteinte à l'ordre public"],
    "deepfake": ["usurpation d'identité", "atteinte à la dignité", "falsification"],
    "doxxing": ["divulgation de données personnelles", "atteinte à la vie privée", "loi 18-07"],

    # Infrastructure
    "wifi": ["réseau de communications électroniques", "accès non autorisé", "système informatique"],
    "wi-fi": ["réseau de communications électroniques", "accès non autorisé"],
    "réseau": ["système de traitement automatisé de données", "communications électroniques"],
    "serveur": ["système de traitement automatisé de données", "infrastructure informatique"],
    "cryptage": ["chiffrement", "signature électronique", "loi 15-04"],
    "chiffrement": ["signature électronique", "loi 15-04", "certification électronique"],
    "password": ["mot de passe", "accès non autorisé", "données d'authentification"],
    "mot de passe": ["accès frauduleux", "données d'authentification", "sécurité informatique"],
}


def expand_query(query: str) -> str:
    """
    Appends French legal synonyms for any recognised technical terms in the query.
    The original query is preserved; legal terms are appended to improve retrieval.
    """
    query_lower = query.lower()
    extra_terms: list[str] = []

    for term, expansions in QUERY_EXPANSION.items():
        # Match whole word or phrase
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, query_lower):
            extra_terms.extend(expansions)

    if extra_terms:
        # Deduplicate while preserving order
        seen = set()
        unique = [t for t in extra_terms if not (t in seen or seen.add(t))]
        expanded = query + " | " + " | ".join(unique)
        logger.debug(f"Query expanded: '{query}' → '{expanded}'")
        return expanded

    return query


# ── Module-level initialisation ───────────────────────────────────────────────

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
4. Cite toujours la source (nom du fichier et numéro d'article si disponible).
5. Réponds toujours en français, même si la question est posée en anglais ou en arabe.
6. Sois précis et concis — l'étudiant n'est pas juriste.
7. Si la question utilise un terme technique (ex: Nmap, DDoS, phishing), relie-le explicitement aux infractions légales correspondantes mentionnées dans les extraits.
8. Si plusieurs lois s'appliquent, structure ta réponse par loi."""


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _retrieve_chunks(question_embedding: list[float]) -> list[dict]:
    def query_with_filter(where: dict | None, n: int) -> list[dict]:
        kwargs = dict(
            query_embeddings=[question_embedding],
            n_results=min(n, _collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        results = _collection.query(**kwargs)
        return [
            {"text": doc, "source": meta["source"], "page": meta["page"],
             "priority": meta["priority"], "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def relevant(chunks):
        return [c for c in chunks if c["distance"] < DISTANCE_THRESHOLD]

    # Tier 1: priority 1 only
    p1 = query_with_filter({"priority": 1}, PRIORITY_1_MIN_RESULTS + 2)
    p1_good = relevant(p1)
    if len(p1_good) >= PRIORITY_1_MIN_RESULTS:
        return sorted(p1_good, key=lambda c: c["distance"])[:MAX_CONTEXT_CHUNKS]

    # Tier 2: priority 1 + 2
    p1p2 = query_with_filter({"priority": {"$in": [1, 2]}}, MAX_CONTEXT_CHUNKS + 2)
    p1p2_good = relevant(p1p2)
    if p1p2_good:
        return sorted(p1p2_good, key=lambda c: c["distance"])[:MAX_CONTEXT_CHUNKS]

    # Tier 3: no filter
    return sorted(
        query_with_filter(None, MAX_CONTEXT_CHUNKS),
        key=lambda c: c["distance"]
    )[:MAX_CONTEXT_CHUNKS]


# ── Main public function ──────────────────────────────────────────────────────

def answer_question(question: str) -> str:
    if not question.strip():
        return "Veuillez poser une question."

    # Expand query with legal synonyms for any technical terms
    expanded = expand_query(question)

    # Embed the (possibly expanded) query
    question_embedding = _embedding_model.encode(expanded).tolist()

    chunks = _retrieve_chunks(question_embedding)

    if not chunks:
        return (
            "Je n'ai pas trouvé d'information pertinente dans les textes juridiques "
            "disponibles pour répondre à cette question."
        )

    context_parts = []
    sources_seen: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        label = f"{chunk['source']} (page {chunk['page']})"
        context_parts.append(
            f"--- Extrait {i} | Source : {label} | "
            f"Pertinence : {(1 - chunk['distance']) * 100:.0f}% ---\n{chunk['text']}"
        )
        if label not in sources_seen:
            sources_seen.append(label)

    context_block = "\n\n".join(context_parts)

    user_message = (
        f"Contexte juridique :\n\n{context_block}\n\n"
        f"Question de l'étudiant : {question}\n\n"
        "Réponds en te basant uniquement sur le contexte. "
        "Cite les articles et lois pertinents. "
        "Si la question mentionne un outil technique (Nmap, etc.), "
        "explique quelle infraction légale cela pourrait constituer selon les extraits."
    )

    try:
        completion = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise RuntimeError(f"Erreur lors de la génération de la réponse : {e}") from e

    sources_text = "\n\n📚 **Sources consultées :**\n" + "\n".join(f"• {s}" for s in sources_seen)
    return answer + sources_text
