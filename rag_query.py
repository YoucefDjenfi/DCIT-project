"""
rag_query.py  v11 (final – aggressive filter for security queries)
────────────────────────────────────────────────────────────────────
- For security queries: ONLY TIC_Articles.pdf chunks are sent to the LLM.
- No Loi 18-07, no other laws. This forces the correct answer.
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

COSINE_FETCH        = 30
BM25_FETCH          = 30
FORCED_TIC_FETCH    = 10
RRF_K               = 60
MAX_CONTEXT_CHUNKS  = 4

PRIORITY_BOOST = {1: 2.0, 2: 0.5, 3: 0.0}

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

QUERY_EXPANSION: dict[str, list[str]] = {
    "nmap": ["accès frauduleux système traitement automatisé données", "394 bis", "intrusion réseau"],
    "scan": ["accès frauduleux système traitement automatisé", "394 bis", "intrusion réseau"],
    "wifi": ["système traitement automatisé réseau", "accès non autorisé", "394 bis"],
    "wi-fi": ["système traitement automatisé", "accès non autorisé", "394 bis"],
    "hacking": ["accès frauduleux système informatique", "394 bis"],
    "phishing": ["fraude informatique usurpation identité", "escroquerie"],
    "malware": ["logiciel malveillant atteinte données", "394 quater"],
    "ddos": ["atteinte fonctionnement système traitement automatisé", "394 quater"],
    "charges": ["sanctions pénales", "emprisonnement amende", "394 bis"],
    "penalty": ["peine emprisonnement amende", "sanctions", "394 bis"],
    "unauthorized": ["accès frauduleux non autorisé", "394 bis"],
    "illegal": ["infraction pénale", "accès frauduleux", "394 bis"],
    "criminal": ["infraction pénale", "code pénal", "394 bis"],
    "394 bis": ["394 bis code pénal accès frauduleux"],
}


def expand_query(query: str) -> str:
    q_lower = query.lower()
    extra = []
    for term, expansions in QUERY_EXPANSION.items():
        if re.search(r"\b" + re.escape(term) + r"\b", q_lower):
            extra.extend(expansions)
    if extra:
        unique = list(dict.fromkeys(extra))
        logger.info(f"[expand] +{len(unique)} terms")
        return query + " " + " ".join(unique)
    return query


def is_security_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SECURITY_KEYWORDS)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())


# ── Module init ───────────────────────────────────────────────────────────────

logger.info("Loading RAG components...")
_embed_model = SentenceTransformer(EMBEDDING_MODEL)
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
_chroma = chromadb.PersistentClient(path=CHROMA_DB_DIR)
_collection = _chroma.get_collection(COLLECTION_NAME)
with open(BM25_INDEX_PATH, "rb") as f:
    bm25_data = pickle.load(f)
_bm25 = bm25_data["bm25"]
_bm25_chunks = bm25_data["chunks"]
_groq = Groq(api_key=Config.GROQ_API_KEY)
logger.info(f"RAG loaded. BM25 corpus: {len(_bm25_chunks)} chunks.")


SYSTEM_PROMPT = """Tu es un assistant juridique pour étudiants de l'ESI Alger.

RÈGLES ABSOLUES :
1. Réponds en MAXIMUM 4 phrases.
2. Base-toi UNIQUEMENT sur les extraits fournis. N'invente rien.
3. Si la question concerne le hacking, les scans de réseau, ou l'accès non autorisé à un système informatique, tu DOIS utiliser les articles du Code pénal (394 bis, etc.) qui sont dans TIC_Articles.pdf.
4. Ignore complètement la loi 18-07 (protection des données) pour ce type de questions – elle n'est pas pertinente.
5. Cite toujours la loi et l'article (ex: Art. 394 bis du Code pénal).
6. Si la réponse n'est pas dans les extraits : "Cette question n'est pas couverte par les textes disponibles."
7. Réponds en français."""


# ── Retrieval helpers ───────────────────────────────────────────────────────

def _cosine_search(query_emb, n):
    results = _collection.query(
        query_embeddings=[query_emb],
        n_results=min(n, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        out.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "priority": meta["priority"],
            "distance": dist,
        })
    return out


def _bm25_search(query, n):
    tokens = tokenize(query)
    scores = _bm25.get_scores(tokens)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    out = []
    for idx in indices:
        if scores[idx] == 0:
            break
        c = _bm25_chunks[idx].copy()
        c["bm25_score"] = float(scores[idx])
        out.append(c)
    return out


def _forced_tic_fetch(query_emb, n):
    results = _collection.query(
        query_embeddings=[query_emb],
        n_results=min(n, _collection.count()),
        where={"source": "TIC_Articles.pdf"},
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        out.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "priority": meta["priority"],
            "distance": dist,
        })
    return out


def _reciprocal_rank_fusion(cosine_list, bm25_list, k=RRF_K):
    rrf = {}
    for rank, c in enumerate(cosine_list, 1):
        key = c["text"]
        if key not in rrf:
            rrf[key] = c.copy()
            rrf[key]["rrf_score"] = 0.0
        rrf[key]["rrf_score"] += 1.0 / (k + rank)
    for rank, c in enumerate(bm25_list, 1):
        key = c["text"]
        if key not in rrf:
            rrf[key] = c.copy()
            rrf[key]["rrf_score"] = 0.0
        rrf[key]["rrf_score"] += 1.0 / (k + rank)
    merged = list(rrf.values())
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged


def _retrieve_and_rerank(question: str, expanded: str) -> list[dict]:
    q_emb = _embed_model.encode(expanded).tolist()

    cosine_res = _cosine_search(q_emb, COSINE_FETCH)
    bm25_res = _bm25_search(expanded, BM25_FETCH)
    logger.info(f"[search] cosine={len(cosine_res)}, bm25={len(bm25_res)}")

    candidates = _reciprocal_rank_fusion(cosine_res, bm25_res)
    logger.info(f"[rrf] merged={len(candidates)}")

    # Forced TIC fetch (only for security queries)
    if is_security_query(question):
        tic_forced = _forced_tic_fetch(q_emb, FORCED_TIC_FETCH)
        if tic_forced:
            existing = {c["text"] for c in candidates}
            for tc in tic_forced:
                if tc["text"] not in existing:
                    candidates.append(tc)
                    existing.add(tc["text"])
            logger.info(f"[forced] added {len(tic_forced)} TIC chunks")
        else:
            logger.info("[forced] no TIC chunks found")

    # Cross-encoder + priority boost
    if candidates:
        pairs = [(question, c["text"]) for c in candidates]
        scores = _cross_encoder.predict(pairs)
        for i, c in enumerate(candidates):
            boost = PRIORITY_BOOST.get(c["priority"], 0.0)
            c["rerank_score"] = float(scores[i]) + boost
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    # AGGRESSIVE FILTER for security queries: keep ONLY TIC chunks
    if is_security_query(question):
        tic_only = [c for c in candidates if c["source"] == "TIC_Articles.pdf"]
        if tic_only:
            final = tic_only[:MAX_CONTEXT_CHUNKS]
            logger.info(f"[filter] security query: using only {len(final)} TIC chunks (discarding others)")
        else:
            final = candidates[:MAX_CONTEXT_CHUNKS]
            logger.info("[filter] no TIC chunks found, using all")
    else:
        final = candidates[:MAX_CONTEXT_CHUNKS]

    logger.info("[final] Context order:")
    for i, c in enumerate(final):
        logger.info(f"  {i+1}. {c['source'][:40]} p{c['page']} | rerank={c.get('rerank_score', 0):.3f}")
    return final


def answer_question(question: str) -> str:
    if not question.strip():
        return "Veuillez poser une question."

    expanded = expand_query(question)
    chunks = _retrieve_and_rerank(question, expanded)

    if not chunks:
        return "Je n'ai pas trouvé d'information pertinente."

    context_parts = []
    sources = []
    for i, c in enumerate(chunks, 1):
        label = f"{c['source']} (page {c['page']})"
        context_parts.append(f"--- Extrait {i} | {label} ---\n{c['text']}")
        if label not in sources:
            sources.append(label)
    context_block = "\n\n".join(context_parts)

    user_msg = (
        f"Contexte juridique :\n\n{context_block}\n\n"
        f"Question : {question}\n\n"
        "Réponds en 4 phrases MAXIMUM. Cite la loi et l'article. "
        "Si la question concerne un scan réseau (Nmap, etc.), utilise impérativement les articles du Code pénal (394 bis, etc.) fournis dans TIC_Articles.pdf. Ignore la loi 18-07 pour ce type de question."
    )

    try:
        resp = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.05,
            max_tokens=450,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        raise RuntimeError(f"Erreur: {e}")

    sources_text = "\n\n📚 **Sources :**\n" + "\n".join(f"• {s}" for s in sources)
    return answer + sources_text
