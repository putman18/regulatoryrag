"""
ingest.py - PDF ingestion pipeline for RegulatoryRAG.

Parses a PDF, chunks with 15% overlap, embeds via sentence-transformers
(semantic embeddings — understands meaning, not just keywords),
saves index to tmp/regulatory_qa_index.json.

Run:
    python ingest.py
    python ingest.py --pdf path/to/custom.pdf
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

try:
    import fitz  # pymupdf
except ImportError:
    print("Missing pymupdf. Run: pip install pymupdf")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Missing anthropic. Run: pip install anthropic")
    sys.exit(1)

DEFAULT_PDF = PROJECT_ROOT / "data" / "fda_guidance.pdf"
INDEX_PATH = Path(tempfile.gettempdir()) / "regulatory_qa_index.json"

CHUNK_SIZE = 800
OVERLAP_PCT = 0.15
MIN_CHARS = 500


def _get_model():
    """Load sentence-transformers model (cached after first load)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Missing sentence-transformers. Run: pip install sentence-transformers")
        sys.exit(1)
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_pages(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap_pct: float = OVERLAP_PCT) -> list[dict]:
    chunks = []
    overlap = int(chunk_size * overlap_pct)
    step = chunk_size - overlap

    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:
                chunks.append({"page": page_num, "text": chunk_text})
            start += step

    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a)) or 1
    norm_b = math.sqrt(sum(x**2 for x in b)) or 1
    return dot / (norm_a * norm_b)


def build_index(pdf_path: Path) -> dict:
    """Full pipeline: extract → chunk → semantic embed → save index."""
    print(f"Loading: {pdf_path.name}")

    pages = extract_pages(pdf_path)
    total_text = " ".join(p["text"] for p in pages)

    if len(total_text) < MIN_CHARS:
        raise ValueError(
            f"Extracted only {len(total_text)} characters from PDF. "
            "Document may be scanned/image-based or corrupted. Cannot proceed."
        )

    print(f"Extracted {len(total_text):,} characters from {len(pages)} pages.")
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={int(OVERLAP_PCT*100)}%)")

    print("Loading embedding model (first run downloads ~90MB)...")
    model = _get_model()

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, show_progress_bar=True).tolist()

    index = {
        "pdf_name": pdf_path.name,
        "total_pages": max(p["page"] for p in pages),
        "chunk_count": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "chunks": [
            {"page": c["page"], "text": c["text"], "vector": v}
            for c, v in zip(chunks, vectors)
        ],
    }

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)

    print(f"Index saved to {INDEX_PATH}")
    return index


def retrieve(query: str, index: dict, top_k: int = 6) -> list[dict]:
    """Retrieve top_k semantically similar chunks for a query."""
    model = _get_model()
    query_vec = model.encode([query])[0].tolist()

    scored = [
        (cosine_similarity(query_vec, chunk["vector"]), chunk)
        for chunk in index["chunks"]
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def load_index() -> dict | None:
    if INDEX_PATH.exists():
        with open(INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    args = parser.parse_args()
    build_index(Path(args.pdf))
