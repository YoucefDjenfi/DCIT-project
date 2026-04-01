"""
ingest.py
─────────
Run once to build the ChromaDB vector database from your PDFs.
Re-run whenever you add or update documents in knowledge_base/.

Usage:
    python ingest.py

Improvements over v1:
- Skips files listed in SKIP_FILES (duplicates, scanned, irrelevant)
- Extracts only TIC-relevant pages from the Penal Code (pages 110-135)
  to avoid 2500 chunks of unrelated criminal law drowning out results
- Strips Journal Officiel boilerplate (headers, page numbers, Republic headers)
- Article-aware chunking: splits on article boundaries first, then by char count
- Correct filename matching for priority tiers
"""

import os
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
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "cyber_law"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# The Penal Code is 362 pages. Only pages ~110-135 contain TIC articles
# (394 bis to 394 nonies). Ingesting the full code adds 2000+ irrelevant chunks.
# Adjust these if your PDF has different pagination.
PENAL_CODE_FILENAME = "2016_Algeria_fr_Code Penal.pdf"
PENAL_CODE_PAGE_RANGE = (108, 140)  # 0-indexed, covers art. 394 bis region + buffer

# Boilerplate patterns to strip from Journal Officiel PDFs
BOILERPLATE_PATTERNS = [
    re.compile(r"JOURNAL\s+OFFICIEL\s+DE\s+LA\s+REPUBLIQUE\s+ALGERIENNE[^\n]*", re.IGNORECASE),
    re.compile(r"Joumada\s+El\s+(?:Oula|Ethania)[^\n]*\d{4}", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),          # lone page numbers
    re.compile(r"N°\s+\d+\s+/\s+\d+"),                 # issue numbers like "N° 37 / 2009"
    re.compile(r"République\s+Algérienne\s+Démocratique\s+et\s+Populaire", re.IGNORECASE),
    re.compile(r"Ministère\s+de\s+la\s+Justice[^\n]*", re.IGNORECASE),
]

ARTICLE_PATTERN = re.compile(
    r"(Art(?:icle|\.)?\s+\d+\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies)?"
    r"(?:\s*[\.:\-—])?|"
    r"المادة\s+\d+)",
    re.IGNORECASE,
)


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove Journal Officiel boilerplate and normalise whitespace."""
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)
    # Collapse excessive whitespace/newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ── PDF extraction ─────────────────────────────────────────────────────────────

def extract_pages(pdf_path: str, page_range: tuple[int, int] | None = None) -> list[tuple[str, int]]:
    """
    Extracts (cleaned_text, page_number) tuples from a PDF.
    page_range: optional (start, end) 0-indexed page filter.
    """
    pages = []
    try:
        reader = PdfReader(pdf_path)
        total = len(reader.pages)
        indices = range(*page_range) if page_range else range(total)
        for i in indices:
            if i >= total:
                break
            raw = reader.pages[i].extract_text() or ""
            cleaned = clean_text(raw)
            if cleaned:
                pages.append((cleaned, i + 1))
    except Exception as e:
        print(f"  ⚠ Could not read {pdf_path}: {e}")
    return pages


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str, page: int, priority: int) -> list[dict]:
    """Article-boundary-aware chunking with character-level overlap fallback."""
    chunks = []

    # Split on article boundaries
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
        print(f"✗ knowledge_base/ not found. Create it and add PDFs.")
        sys.exit(1)

    pdf_files = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
    print(f"Found {len(pdf_files)} PDF file(s) in knowledge_base/\n")

    all_chunks: list[dict] = []
    skipped = []

    for pdf_path in pdf_files:
        filename = pdf_path.name

        if should_skip(filename):
            print(f"  [SKIP] {filename}")
            skipped.append(filename)
            continue

        priority = get_priority(filename)
        page_range = PENAL_CODE_PAGE_RANGE if filename == PENAL_CODE_FILENAME else None

        range_note = f" (pages {page_range[0]+1}–{page_range[1]})" if page_range else ""
        print(f"  [P{priority}] {filename}{range_note}")

        pages = extract_pages(str(pdf_path), page_range)
        if not pages:
            print(f"       ⚠ No extractable text — scanned PDF? Skipping.")
            skipped.append(filename)
            continue

        file_chunks = []
        for page_text, page_num in pages:
            file_chunks.extend(chunk_text(page_text, filename, page_num, priority))

        print(f"       → {len(pages)} pages, {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    print(f"\nSkipped {len(skipped)} file(s): {skipped}")
    print(f"Total chunks to embed: {len(all_chunks)}\n")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    print("\nStoring in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection (rebuilding from scratch)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 500
    for start in range(0, len(all_chunks), BATCH):
        batch = all_chunks[start: start + BATCH]
        batch_emb = embeddings[start: start + BATCH]
        collection.add(
            ids=[str(start + i) for i in range(len(batch))],
            embeddings=[e.tolist() for e in batch_emb],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "page": c["page"], "priority": c["priority"]} for c in batch],
        )

    print(f"\n✅ Done. {len(all_chunks)} chunks stored in {CHROMA_DB_DIR}/")
    print("\nPriority breakdown:")
    for tier in [1, 2, 3]:
        count = sum(1 for c in all_chunks if c["priority"] == tier)
        print(f"  Priority {tier}: {count} chunks")


if __name__ == "__main__":
    build_database()
