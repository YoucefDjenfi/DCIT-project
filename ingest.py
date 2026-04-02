"""
ingest.py  v5
─────────────
Removed ALL page/keyword filtering from the Penal Code.
The full 362-page PDF is ingested. The cross-encoder handles precision.
18 chunks from 4 pages is not enough — we need all TIC articles.
"""

import re
import sys
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from document_priorities import get_priority, should_skip

KNOWLEDGE_BASE_DIR = "./knowledge_base"
CHROMA_DB_DIR      = "./chroma_db"
COLLECTION_NAME    = "cyber_law"
EMBEDDING_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 150

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


def clean_text(text: str) -> str:
    for p in BOILERPLATE_PATTERNS:
        text = p.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_pages(pdf_path: str) -> list[tuple[str, int]]:
    """Extract ALL pages — no filtering. Cross-encoder handles relevance."""
    pages = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            raw     = page.extract_text() or ""
            cleaned = clean_text(raw)
            if cleaned:
                pages.append((cleaned, i + 1))
    except Exception as e:
        print(f"  ⚠ Could not read {pdf_path}: {e}")
    return pages


def chunk_text(text: str, source: str, page: int, priority: int) -> list[dict]:
    chunks   = []
    parts    = ARTICLE_PATTERN.split(text)
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
        print(f"  [P{priority}] {filename}")

        pages = extract_pages(str(pdf_path))
        if not pages:
            print(f"       ⚠ No extractable text. Skipping.")
            skipped.append(filename)
            continue

        file_chunks = []
        for page_text, page_num in pages:
            file_chunks.extend(chunk_text(page_text, filename, page_num, priority))

        print(f"       → {len(pages)} pages, {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    print(f"\nSkipped: {skipped}")
    print(f"Total chunks: {len(all_chunks)}\n")

    print(f"Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding...")
    texts      = [c["text"] for c in all_chunks]
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
            ids       =[str(start + i) for i in range(len(batch))],
            embeddings=[e.tolist() for e in batch_emb],
            documents =[c["text"] for c in batch],
            metadatas =[
                {"source": c["source"], "page": c["page"], "priority": c["priority"]}
                for c in batch
            ],
        )

    print(f"\n✅ Done. {len(all_chunks)} chunks in {CHROMA_DB_DIR}/")
    for tier in [1, 2, 3]:
        n = sum(1 for c in all_chunks if c["priority"] == tier)
        print(f"  Priority {tier}: {n} chunks")
    print(f"\nPenal Code chunks: {sum(1 for c in all_chunks if '2016_Algeria_fr_Code Penal' in c['source'])}")


if __name__ == "__main__":
    build_database()
