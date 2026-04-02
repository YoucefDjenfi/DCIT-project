"""
ingest.py  v3
─────────────
Changes from v2:
- Penal Code page range removed. Instead, pages are filtered by content:
  only pages that contain TIC-related keywords are ingested. This is more
  reliable than hardcoded page indices which vary by PDF copy.
- SKIP_FILES now includes the translated Arab Convention duplicate check
- Priority mapping updated for Arab_Convention_Cybercrime_2010_FR.pdf.pdf
"""

import re
import sys
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from document_priorities import get_priority, should_skip

# ── Configuration ─────────────────────────────────────────────────────────────

KNOWLEDGE_BASE_DIR = "./knowledge_base"
CHROMA_DB_DIR      = "./chroma_db"
COLLECTION_NAME    = "cyber_law"
EMBEDDING_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 150

# For the Penal Code we do keyword-based page filtering instead of
# hardcoded page numbers (which differ across PDF versions).
PENAL_CODE_FILENAME = "2016_Algeria_fr_Code Penal.pdf"

# A page is kept from the Penal Code only if it contains at least one of
# these strings. This captures TIC articles + surrounding context.
PENAL_CODE_KEYWORDS = [
    "394 bis", "394 ter", "394 quater", "394 quinquies",
    "394 sexies", "394 septies", "394 octies", "394 nonies",
    "système de traitement automatisé",
    "accès frauduleux",
    "données informatiques",
    "cybercriminalité",
    "atteinte aux systèmes",
]

# Boilerplate to strip from Journal Officiel PDFs
BOILERPLATE_PATTERNS = [
    re.compile(r"JOURNAL\s+OFFICIEL\s+DE\s+LA\s+REPUBLIQUE\s+ALGERIENNE[^\n]*", re.IGNORECASE),
    re.compile(r"Joumada\s+El\s+(?:Oula|Ethania)[^\n]*\d{4}", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),
    re.compile(r"N°\s+\d+\s+/\s+\d+"),
    re.compile(r"République\s+Algérienne\s+Démocratique\s+et\s+Populaire", re.IGNORECASE),
    re.compile(r"Ministère\s+de\s+la\s+Justice[^\n]*", re.IGNORECASE),
]

ARTICLE_PATTERN = re.compile(
    r"(Art(?:icle|\.)?\s+\d+\s*"
    r"(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies)?"
    r"(?:\s*[\.:\-—])?|المادة\s+\d+)",
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    for p in BOILERPLATE_PATTERNS:
        text = p.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def page_is_relevant_for_penal_code(text: str) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in PENAL_CODE_KEYWORDS)


def extract_pages(pdf_path: str, keyword_filter: bool = False) -> list[tuple[str, int]]:
    pages = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            cleaned = clean_text(raw)
            if not cleaned:
                continue
            if keyword_filter and not page_is_relevant_for_penal_code(cleaned):
                continue
            pages.append((cleaned, i + 1))
    except Exception as e:
        print(f"  ⚠ Could not read {pdf_path}: {e}")
    return pages


def chunk_text(text: str, source: str, page: int, priority: int) -> list[dict]:
    chunks = []
    parts = ARTICLE_PATTERN.split(text)
    segments = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if ARTICLE_PATTERN.fullmatch(part.strip()):
            content = parts[i + 1] if i + 1 < len(parts) else ""
            segments.append((part + content).strip())
            i += 2
        else:
            if part.strip():
                segments.append(part.strip())
            i += 1
    if not segments:
        segments = [text]

    for seg in segments:
        if not seg.strip():
            continue
        if len(seg) <= CHUNK_SIZE:
            chunks.append({"text": seg, "source": source, "page": page, "priority": priority})
        else:
            start = 0
            while start < len(seg):
                chunk = seg[start: start + CHUNK_SIZE].strip()
                if chunk:
                    chunks.append({"text": chunk, "source": source, "page": page, "priority": priority})
                start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def build_database():
    pdf_dir = Path(KNOWLEDGE_BASE_DIR)
    if not pdf_dir.exists():
        print("✗ knowledge_base/ not found.")
        sys.exit(1)

    pdf_files = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
    print(f"Found {len(pdf_files)} PDF file(s)\n")

    all_chunks: list[dict] = []
    skipped = []

    for pdf_path in pdf_files:
        filename = pdf_path.name

        if should_skip(filename):
            print(f"  [SKIP] {filename}")
            skipped.append(filename)
            continue

        priority = get_priority(filename)
        is_penal_code = (filename == PENAL_CODE_FILENAME)

        print(f"  [P{priority}] {filename}" + (" (keyword-filtered)" if is_penal_code else ""))

        pages = extract_pages(str(pdf_path), keyword_filter=is_penal_code)

        if not pages:
            print(f"       ⚠ No extractable text. Skipping.")
            skipped.append(filename)
            continue

        file_chunks = []
        for page_text, page_num in pages:
            file_chunks.extend(chunk_text(page_text, filename, page_num, priority))

        print(f"       → {len(pages)} pages kept, {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    print(f"\nSkipped: {skipped}")
    print(f"Total chunks: {len(all_chunks)}\n")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    print("\nStoring in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 500
    for start in range(0, len(all_chunks), BATCH):
        batch     = all_chunks[start: start + BATCH]
        batch_emb = embeddings[start: start + BATCH]
        collection.add(
            ids=[str(start + i) for i in range(len(batch))],
            embeddings=[e.tolist() for e in batch_emb],
            documents=[c["text"] for c in batch],
            metadatas=[
                {"source": c["source"], "page": c["page"], "priority": c["priority"]}
                for c in batch
            ],
        )

    print(f"\n✅ Done. {len(all_chunks)} chunks in {CHROMA_DB_DIR}/")
    for tier in [1, 2, 3]:
        n = sum(1 for c in all_chunks if c["priority"] == tier)
        print(f"  Priority {tier}: {n} chunks")


if __name__ == "__main__":
    build_database()
