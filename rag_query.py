"""
rag_query.py  v9 (final)
─────────────────────────
Hybrid BM25 + Cosine + Cross‑Encoder Reranker + Forced TIC Fetch + Reordering

What makes this version final:
  - BM25 keyword search (exact match on "394 bis", "accès frauduleux", etc.)
  - Cosine vector search (semantic similarity)
  - RRF fusion to combine both ranked lists
  - Forced TIC fetch for security queries – guarantees criminal law articles
  - Cross‑encoder reranking (semantic scoring)
  - Priority boost for core laws
  - Reordering: TIC chunks first for security questions
  - All in Claude's clean, modular style
"""

import sys
import logging
import re
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from groq import Groq

sys.path.insert(0, str(Path(__file__).parent))
from config import Config

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

CHROMA_DB_DIR       = "./chroma_db"
BM25_INDEX_PATH     = "./bm25_index.pkl"
COLLECTION_NAME     = "cyber_law"
EMBEDDING_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
GROQ_MODEL          = "llama-3.3-70b-versatile"

# Retrieval numbers
COSINE_FETCH        = 30
BM25_FETCH          = 30
FORCED_TIC_FETCH    = 10          # extra TIC chunks for security queries
RRF_K               = 60
FORCED_TIC_COUNT    = 2
MAX_CONTEXT_CHUNKS  = 4

PRIORITY_BOOST = {1: 2.0, 2: 0.5, 3: 0.0}

# Keywords that trigger forced TIC fetch and reordering
SECURITY_KEYWORDS = [
    "nmap", "scan", "scanning", "wireshark", "sniffing", "ddos", "dos",
    "hack", "hacking", "exploit", "bruteforce", "phishing", "malware",
    "ransomware", "virus", "keylogger", "spyware", "sql injection",
    "injection", "xss", "wifi", "wi-fi", "réseau", "network", "serveur",
    "mot de passe", "password", "unauthorized", "illegal", "charges",
    "penalty", "penalties", "criminal", "crime", "accès frauduleux",
    "394 bis", "394 ter", "394 quater", "intrusion", "cybercrime",
    "système informatique", "traitement automatisé",
]

# ── Query expansion ────────────────────────────────────────────────────────────

QUERY_EXPANSION: dict[str, list[str]] = {
    "nmap": [
        "accès frauduleux système traitement automatisé données",
        "394 bis code pénal",
        "intrusion réseau informatique non autorisé",
        "pénétration système informatique",
    ],
    "scan":            ["accès frauduleux système traitement automatisé", "394 bis", "intrusion réseau"],
    "scanning":        ["accès frauduleux système traitement automatisé", "394 bis"],
    "wireshark":       ["interception communications", "accès frauduleux", "394 ter"],
    "sniffing":        ["interception communications électroniques", "accès frauduleux"],
    "ddos":            ["atteinte fonctionnement système traitement automatisé", "394 quater"],
    "dos":             ["atteinte fonctionnement système", "394 quater"],
    "hacking":         ["accès frauduleux système informatique", "394 bis code pénal"],
    "hack":            ["accès frauduleux", "394 bis", "système informatique"],
    "exploit":         ["accès frauduleux vulnérabilité", "394 bis"],
    "bruteforce":      ["accès frauduleux tentative", "394 bis"],
    "phishing":        ["fraude informatique usurpation identité", "escroquerie"],
    "malware":         ["logiciel malveillant atteinte données", "394 quater"],
    "ransomware":      ["atteinte données extorsion", "394 quater"],
    "virus":           ["logiciel malveillant atteinte système", "394 quater"],
    "keylogger":       ["logiciel espion interception", "données caractère personnel"],
    "spyware":         ["logiciel espion surveillance illégale", "données personnelles"],
    "sql injection":   ["accès frauduleux altération données", "394 ter"],
    "injection":       ["accès frauduleux altération données", "394 ter"],
    "xss":             ["atteinte données fraude informatique"],
    "tracking":        ["données caractère personnel traçage", "loi 18-07"],
    "cookies":         ["données caractère personnel consentement", "loi 18-07"],
    "vpn":             ["réseau privé virtuel anonymisation", "communications électroniques"],
    "tor":             ["anonymisation accès réseau", "communications électroniques"],
    "cyberbullying":   ["harcèlement cyberharcèlement ligne", "atteinte dignité"],
    "harcèlement":     ["cyberharcèlement harcèlement ligne", "atteinte vie privée"],
    "fake news":       ["désinformation fausses nouvelles", "atteinte ordre public"],
    "deepfake":        ["usurpation identité atteinte dignité", "falsification"],
    "doxxing":         ["divulgation données personnelles", "atteinte vie privée loi 18-07"],
    "wifi":            ["système traitement automatisé réseau", "accès non autorisé réseau informatique", "394 bis"],
    "wi-fi":           ["système traitement automatisé", "accès non autorisé", "394 bis"],
    "réseau":          ["système traitement automatisé données", "communications électroniques"],
    "network":         ["système traitement automatisé", "réseau informatique", "394 bis"],
    "serveur":         ["système traitement automatisé infrastructure"],
    "cryptage":        ["chiffrement signature électronique", "loi 15-04"],
    "chiffrement":     ["signature électronique loi 15-04", "certification électronique"],
    "mot de passe":    ["accès frauduleux données authentification", "394 bis"],
    "password":        ["accès frauduleux données authentification", "394 bis"],
    "unauthorized":    ["accès frauduleux non autorisé", "394 bis"],
    "illegal":         ["infraction pénale", "accès frauduleux", "394 bis"],
    "charges":         ["sanctions pénales", "emprisonnement amende", "394 bis code pénal"],
    "penalty":         ["peine emprisonnement amende", "sanctions", "394 bis"],
    "penalties":       ["peines sanctions", "emprisonnement amende", "394 bis"],
    "criminal":        ["infraction pénale", "code pénal", "394 bis"],
    "crime":           ["infraction cybercriminalité", "loi 09-04", "394 bis"],
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
        logger.info(f"[expand] +{len(unique)} terms")
        return query + " " + " ".join(unique)
    return query


def is_security_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SECURITY_KEYWORDS)


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())


# ── Module-level init (runs once when bot starts) ────────────────────────────

logger.info("Loading RAG components...")

_embed_model   = SentenceTransformer(EMBEDDING_MODEL)
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

_chroma      = chromadb.PersistentClient(path=CHROMA_DB_DIR)
_collection  = _chroma.get_collection(COLLECTION_NAME)

# Load the BM25 index built by ingest.py
with open(BM25_INDEX_PATH, "rb") as _f:
    _bm25_payload = pickle.load(_f)
_bm25:   BM25Okapi   = _bm25_payload["bm25"]
_chunks: list[dict]  = _bm25_payload["chunks"]   # same order as BM25 corpus

_groq = Groq(api_key=Config.GROQ_API_KEY)

logger.info(f"RAG components loaded. BM25 corpus: {len(_chunks)} chunks.")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant juridique pour étudiants de l'ESI Alger, spécialisé en droit algérien du numérique.

RÈGLES ABSOLUES :
1. Réponds en MAXIMUM 4 phrases. Jamais plus.
2. Base-toi UNIQUEMENT sur les extraits fournis. N'invente rien.
3. Commence directement par la réponse — pas de préambule.
4. Cite toujours la loi et l'article (ex: Art. 394 bis du Code pénal).
5. Si un outil technique est mentionné (Nmap, DDoS, etc.), dis explicitement quelle infraction il constitue.
6. Si la réponse est absente des extraits : "Cette question n'est pas couverte par les textes disponibles." — puis stop.
7. Réponds en français même si la question est en anglais ou en arabe."""


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _cosine_search(query_embedding: list[float], n: int) -> list[dict]:
    """Returns top-n chunks from ChromaDB by cosine similarity."""
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
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
    ]


def _bm25_search(query: str, n: int) -> list[dict]:
    """Returns top-n chunks from the BM25 index by keyword relevance."""
    tokens = tokenize_for_bm25(query)
    scores = _bm25.get_scores(tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    results = []
    for idx in top_indices:
        if scores[idx] == 0:
            break
        c = _chunks[idx].copy()
        c["bm25_score"] = float(scores[idx])
        results.append(c)
    return results


def _forced_tic_fetch(query_embedding: list[float], n: int) -> list[dict]:
    """For security queries: fetch TIC_Articles.pdf chunks directly."""
    if not is_security_query(expand_query("")):  # placeholder, but we'll call only when needed
        return []
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n, _collection.count()),
        where={"source": "TIC_Articles.pdf"},
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text":     doc,
            "source":   meta["source"],
            "page":     meta["page"],
            "priority": meta["priority"],
            "distance": dist,
            "from_forced": True,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def _reciprocal_rank_fusion(
    cosine_results: list[dict],
    bm25_results:   list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """Merges two ranked lists using Reciprocal Rank Fusion."""
    rrf: dict[str, dict] = {}

    for rank, chunk in enumerate(cosine_results, start=1):
        key = chunk["text"]
        if key not in rrf:
            rrf[key] = {**chunk, "rrf_score": 0.0}
        rrf[key]["rrf_score"] += 1.0 / (k + rank)

    for rank, chunk in enumerate(bm25_results, start=1):
        key = chunk["text"]
        if key not in rrf:
            rrf[key] = {**chunk, "rrf_score": 0.0}
        rrf[key]["rrf_score"] += 1.0 / (k + rank)

    merged = list(rrf.values())
    merged.sort(key=lambda c: c["rrf_score"], reverse=True)
    return merged


def _retrieve_and_rerank(question: str, expanded: str) -> list[dict]:
    """
    Full hybrid retrieval pipeline with forced TIC fetch for security queries.
    """
    q_emb = _embed_model.encode(expanded).tolist()

    # ── 1. Cosine search ──────────────────────────────────────────────────────
    cosine_results = _cosine_search(q_emb, COSINE_FETCH)

    # ── 2. BM25 search ────────────────────────────────────────────────────────
    bm25_results = _bm25_search(expanded, BM25_FETCH)

    logger.info(
        f"[search] cosine={len(cosine_results)} candidates, "
        f"bm25={len(bm25_results)} candidates"
    )

    # ── 3. RRF merge ──────────────────────────────────────────────────────────
    candidates = _reciprocal_rank_fusion(cosine_results, bm25_results)
    logger.info(f"[rrf] merged pool: {len(candidates)} unique chunks")

    # ── 4. Forced TIC fetch (only for security queries) ───────────────────────
    if is_security_query(question):
        tic_forced = _forced_tic_fetch(q_emb, FORCED_TIC_FETCH)
        if tic_forced:
            # Merge into candidates, avoiding duplicates by text
            existing_texts = {c["text"] for c in candidates}
            for tc in tic_forced:
                if tc["text"] not in existing_texts:
                    candidates.append(tc)
                    existing_texts.add(tc["text"])
            logger.info(f"[forced] Added {len(tic_forced)} TIC chunks to pool")

    # ── 5. Cross‑encoder + priority boost ──────────────────────────────────────
    pairs = [(question, c["text"]) for c in candidates]
    scores = _cross_encoder.predict(pairs)

    for i, c in enumerate(candidates):
        boost = PRIORITY_BOOST.get(c["priority"], 0.0)
        c["rerank_score"] = float(scores[i]) + boost

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    # ── 6. Reorder: TIC chunks first for security queries ─────────────────────
    if is_security_query(question):
        tic_chunks = [c for c in candidates if c["source"] == "TIC_Articles.pdf"]
        other_chunks = [c for c in candidates if c["source"] != "TIC_Articles.pdf"]
        top_tic = tic_chunks[:FORCED_TIC_COUNT]
        remaining = other_chunks[:MAX_CONTEXT_CHUNKS - len(top_tic)]
        final = top_tic + remaining
        logger.info(f"[reorder] Placed {len(top_tic)} TIC chunks first in context")
    else:
        final = candidates[:MAX_CONTEXT_CHUNKS]

    # ── Debug log ─────────────────────────────────────────────────────────────
    logger.info("[final] Context order:")
    for i, c in enumerate(final):
        logger.info(
            f"  {i+1}. [{c['source'][:40]}] p{c['page']} "
            f"| rerank={c['rerank_score']:.3f} | P{c['priority']}"
        )
    return final


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
        context_parts.append(f"--- Extrait {i} | {label} ---\n{c['text']}")
        if label not in sources_seen:
            sources_seen.append(label)

    context_block = "\n\n".join(context_parts)

    user_message = (
        f"Contexte juridique :\n\n{context_block}\n\n"
        f"Question : {question}\n\n"
        "Réponds en 4 phrases MAXIMUM. Cite la loi et l'article. "
        "Si un outil technique est mentionné, nomme l'infraction légale qu'il constitue."
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
