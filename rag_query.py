"""
rag_query.py  v6
────────────────
For security queries: force at least 2 Penal Code chunks into the final context,
bypassing cross-encoder scores if necessary.
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
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
GROQ_MODEL          = "llama-3.3-70b-versatile"

INITIAL_FETCH       = 30          # general cosine candidates
PENAL_CODE_FETCH    = 10          # forced Penal Code candidates (if query is security-related)
FORCED_PENAL_COUNT  = 2           # number of Penal Code chunks to force into final context
MAX_CONTEXT_CHUNKS  = 4

PRIORITY_BOOST = {1: 2.0, 2: 0.5, 3: 0.0}

# Keywords that trigger forced Penal Code retrieval
SECURITY_KEYWORDS = [
    "nmap", "scan", "scanning", "wireshark", "sniffing", "ddos", "dos", "hack", "hacking",
    "exploit", "bruteforce", "phishing", "malware", "ransomware", "virus", "keylogger",
    "spyware", "sql injection", "injection", "xss", "wifi", "wi-fi", "réseau", "network",
    "serveur", "mot de passe", "password", "unauthorized", "illegal", "charges", "penalty",
    "penalties", "criminal", "crime", "accès frauduleux", "394 bis", "intrusion", "cybercrime"
]

# ── Query expansion ────────────────────────────────────────────────────────────

QUERY_EXPANSION: dict[str, list[str]] = {
    "nmap": [
        "accès frauduleux système traitement automatisé données",
        "394 bis code pénal",
        "intrusion réseau informatique non autorisé",
        "pénétration système informatique",
    ],
    "scan": [
        "accès frauduleux système traitement automatisé",
        "394 bis",
        "intrusion réseau",
    ],
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
    "wifi":            [
        "système traitement automatisé réseau",
        "accès non autorisé réseau informatique",
        "394 bis",
    ],
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
        logger.info(f"[expand] Added {len(unique)} terms")
        return query + " " + " ".join(unique)
    return query


def is_security_query(query: str) -> bool:
    q_lower = query.lower()
    return any(kw in q_lower for kw in SECURITY_KEYWORDS)


# ── Module-level init ─────────────────────────────────────────────────────────

logger.info("Loading RAG components...")
_embed_model   = SentenceTransformer(EMBEDDING_MODEL)
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
_chroma        = chromadb.PersistentClient(path=CHROMA_DB_DIR)
_collection    = _chroma.get_collection(COLLECTION_NAME)
_groq          = Groq(api_key=Config.GROQ_API_KEY)
logger.info("RAG components loaded.")

SYSTEM_PROMPT = """Tu es un assistant juridique pour étudiants de l'ESI Alger, spécialisé en droit algérien du numérique.

RÈGLES ABSOLUES :
1. Réponds en MAXIMUM 4 phrases. Jamais plus.
2. Base-toi UNIQUEMENT sur les extraits fournis. N'invente rien.
3. Commence directement par la réponse — pas de préambule, pas de "selon les extraits".
4. Cite toujours la loi et l'article (ex: Art. 394 bis du Code pénal).
5. Si un outil technique est mentionné (Nmap, DDoS, etc.), dis explicitement quelle infraction il constitue.
6. Si la réponse est absente des extraits : "Cette question n'est pas couverte par les textes disponibles." — puis stop.
7. Réponds en français même si la question est en anglais ou en arabe."""


def _retrieve_and_rerank(question: str, expanded: str) -> list[dict]:
    # 1. General cosine search
    q_emb = _embed_model.encode(expanded).tolist()
    general_results = _collection.query(
        query_embeddings=[q_emb],
        n_results=min(INITIAL_FETCH, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    penal_chunks = []  # store separately for forced inclusion

    for doc, meta, dist in zip(
        general_results["documents"][0],
        general_results["metadatas"][0],
        general_results["distances"][0],
    ):
        c = {
            "text":     doc,
            "source":   meta["source"],
            "page":     meta["page"],
            "priority": meta["priority"],
            "distance": dist,
        }
        candidates.append(c)
        if meta["source"] == "2016_Algeria_fr_Code Penal.pdf":
            penal_chunks.append(c)

    # 2. If security query, fetch additional Penal Code chunks (even if not in top general)
    if is_security_query(question):
        penal_results = _collection.query(
            query_embeddings=[q_emb],
            n_results=min(PENAL_CODE_FETCH, _collection.count()),
            where={"source": "2016_Algeria_fr_Code Penal.pdf"},
            include=["documents", "metadatas", "distances"],
        )
        existing_texts = {c["text"] for c in candidates}
        for doc, meta, dist in zip(
            penal_results["documents"][0],
            penal_results["metadatas"][0],
            penal_results["distances"][0],
        ):
            if doc not in existing_texts:
                pc = {
                    "text":     doc,
                    "source":   meta["source"],
                    "page":     meta["page"],
                    "priority": meta["priority"],
                    "distance": dist,
                }
                candidates.append(pc)
                penal_chunks.append(pc)
                existing_texts.add(doc)
        logger.info(f"[force] Added {len(penal_results['documents'][0])} Penal Code chunks (total candidates {len(candidates)})")

    # 3. Cross-encoder reranking
    pairs = [(question, c["text"]) for c in candidates]
    scores = _cross_encoder.predict(pairs)
    for i, c in enumerate(candidates):
        boost = PRIORITY_BOOST.get(c["priority"], 0.0)
        c["rerank_score"] = float(scores[i]) + boost

    # Sort by rerank score
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    # 4. FORCED INCLUSION: For security queries, ensure at least FORCED_PENAL_COUNT Penal Code chunks
    #    are in the final top MAX_CONTEXT_CHUNKS. If not, replace the lowest-ranked non-Penal Code chunks.
    if is_security_query(question):
        # Collect top MAX_CONTEXT_CHUNKS candidates
        final = candidates[:MAX_CONTEXT_CHUNKS]
        # Count how many Penal Code chunks are already in final
        penal_in_final = [c for c in final if c["source"] == "2016_Algeria_fr_Code Penal.pdf"]
        need = FORCED_PENAL_COUNT - len(penal_in_final)
        if need > 0:
            # Get the best Penal Code chunks not already in final, sorted by rerank score
            available_penal = [c for c in penal_chunks if c not in final]
            available_penal.sort(key=lambda c: c["rerank_score"], reverse=True)
            # Replace the lowest-scoring non-Penal Code chunks in final
            for i in range(min(need, len(available_penal))):
                # Find lowest-scoring non-Penal Code chunk in final
                for j in range(len(final)-1, -1, -1):
                    if final[j]["source"] != "2016_Algeria_fr_Code Penal.pdf":
                        final[j] = available_penal[i]
                        break
            logger.info(f"[force] Forced {need} Penal Code chunks into final context")
        candidates = final
    else:
        candidates = candidates[:MAX_CONTEXT_CHUNKS]

    # Debug log
    logger.info("[rerank] Top chunks after forcing:")
    for i, c in enumerate(candidates):
        logger.info(
            f"  {i+1}. [{c['source'][:40]}] p{c['page']} "
            f"| cosine={c['distance']:.3f} | rerank={c['rerank_score']:.3f} | P{c['priority']}"
        )
    return candidates


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
