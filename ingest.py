"""
ingest.py
─────────
Run this script ONCE to build the ChromaDB vector database from your PDFs.
Re-run it only if you add new documents to knowledge_base/.

Usage:
    python ingest.py

What it does:
    1. Reads every PDF in knowledge_base/
    2. Splits each page into overlapping text chunks
    3. Tries to detect article boundaries (e.g. "Article 4") to keep legal
       units intact — if an article fits in one chunk, it won't be split
    4. Embeds each chunk using a multilingual sentence-transformers model
    5. Stores everything in chroma_db/ with metadata: source, page, priority
"""

import os
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# Import priorities from our mapping file
sys.path.insert(0, str(Path(__file__).parent))
from document_priorities import get_priority

# ── Configuration ─────────────────────────────────────────────────────────────

KNOWLEDGE_BASE_DIR = "./knowledge_base"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "cyber_law"

# paraphrase-multilingual-MiniLM-L12-v2 supports French, Arabic, English —
# appropriate since our corpus may include translated documents.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks

# Regex to detect the start of a legal article in French or Arabic transliteration
ARTICLE_PATTERN = re.compile(
    r"(Article\s+\d+[\s\S]{0,20}?[\.\:\-]|"   # "Article 4 :" or "Article 4."
    r"Art\.\s*\d+[\s\S]{0,10}?[\.\:\-]|"       # "Art. 4 —"
    r"المادة\s+\d+)",                           # Arabic: "المادة 4"
    re.IGNORECASE
)


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[tuple[str, int]]:
    """
    Returns a list of (page_text, page_number) tuples for a given PDF.
    Skips pages that yield no extractable text (e.g. scanned images).
    """
    pages = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append((text, i + 1))
    except Exception as e:
        print(f"  ⚠ Could not read {pdf_path}: {e}")
    return pages


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_into_chunks(text: str, source: str, page: int, priority: int) -> list[dict]:
    """
    Splits a page's text into overlapping chunks.

    Strategy:
    - First, try to split on article boundaries (keeps legal units whole).
    - If an article-level segment is still too long, fall back to character
      splitting with overlap.
    - Each chunk is a dict containing: text, source, page, priority, chunk_id.
    """
    chunks = []

    # Attempt article-boundary splitting first
    article_splits = ARTICLE_PATTERN.split(text)

    if len(article_splits) > 1:
        # Re-assemble: the regex split produces separators as elements
        # We pair each separator with the content that follows it
        segments = []
        i = 0
        while i < len(article_splits):
            if ARTICLE_PATTERN.match(article_splits[i]):
                # This element is an article header — join it with the next content block
                header = article_splits[i]
                content = article_splits[i + 1] if i + 1 < len(article_splits) else ""
                segments.append(header + content)
                i += 2
            else:
                if article_splits[i].strip():
                    segments.append(article_splits[i])
                i += 1
    else:
        segments = [text]

    # Now chunk each segment with overlap fallback
    for seg in segments:
        if len(seg) <= CHUNK_SIZE:
            if seg.strip():
                chunks.append(_make_chunk(seg.strip(), source, page, priority))
        else:
            # Character-level splitting with overlap
            start = 0
            while start < len(seg):
                end = start + CHUNK_SIZE
                chunk_text = seg[start:end].strip()
                if chunk_text:
                    chunks.append(_make_chunk(chunk_text, source, page, priority))
                start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def _make_chunk(text: str, source: str, page: int, priority: int) -> dict:
    return {
        "text": text,
        "source": source,
        "page": page,
        "priority": priority,
    }


# ── Main ingestion pipeline ───────────────────────────────────────────────────

def build_database():
    pdf_dir = Path(KNOWLEDGE_BASE_DIR)
    if not pdf_dir.exists():
        print(f"✗ knowledge_base/ folder not found at {KNOWLEDGE_BASE_DIR}")
        print("  Create it and add your PDFs, then re-run.")
        sys.exit(1)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("✗ No PDF files found in knowledge_base/")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) in knowledge_base/\n")

    # ── Step 1: Extract text from all PDFs ────────────────────────────────────
    all_chunks: list[dict] = []

    for pdf_path in sorted(pdf_files):
        filename = pdf_path.name
        priority = get_priority(filename)
        print(f"  [{priority}] Reading: {filename}")

        pages = extract_pages(str(pdf_path))
        if not pages:
            print(f"      ⚠ No extractable text — may be a scanned PDF. Skipping.")
            continue

        file_chunks = []
        for page_text, page_num in pages:
            file_chunks.extend(split_into_chunks(page_text, filename, page_num, priority))

        print(f"      → {len(pages)} pages, {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    # ── Step 2: Load embedding model ──────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~120MB — this is normal)\n")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Step 3: Embed all chunks ───────────────────────────────────────────────
    print("Embedding chunks (this may take 2–5 minutes)...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # ── Step 4: Store in ChromaDB ──────────────────────────────────────────────
    print("\nStoring in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Always start fresh so re-runs don't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Deleted existing collection (rebuilding from scratch)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # use cosine distance for similarity
    )

    # ChromaDB add() has a limit of ~5000 items per call — batch it
    BATCH_SIZE = 500
    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start: batch_start + BATCH_SIZE]
        batch_embeddings = embeddings[batch_start: batch_start + BATCH_SIZE]

        collection.add(
            ids=[str(batch_start + i) for i in range(len(batch))],
            embeddings=[e.tolist() for e in batch_embeddings],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "source": c["source"],
                    "page": c["page"],
                    "priority": c["priority"],
                }
                for c in batch
            ],
        )

    print(f"\n✅ Done. {len(all_chunks)} chunks stored in {CHROMA_DB_DIR}/")
    print("\nPriority breakdown:")
    for tier in [1, 2, 3]:
        count = sum(1 for c in all_chunks if c["priority"] == tier)
        print(f"  Priority {tier}: {count} chunks")


if __name__ == "__main__":
    build_database()
