"""
ingest.py
Standalone ingestion script. Run once before starting the server.

Usage:
    python ingest.py

What it does:
    1. Reads all PDFs from the DATA_DIR folder (set in .env or config.py).
    2. Splits each PDF into page-level chunks (1 chunk = 1 PDF page).
    3. Embeds all pages with the Gemini embedding model configured in config.py/.env.
    4. Persists the vector index to VECTOR_STORE_PATH (ChromaDB on disk).
    5. Builds and caches a BM25 keyword index alongside it.

Re-running this script will overwrite the existing indexes.
You only need to re-run if you add or change PDF documents.
"""

import sys
import time

import config
from services.ingestion import ingest


def main() -> None:
    print("=" * 55)
    print("  Aviation RAG — Ingestion Pipeline")
    print("=" * 55)
    print(f"  Data dir     : {config.DATA_DIR}")
    print(f"  Vector store : {config.VECTOR_STORE_PATH}")
    print(f"  Embedding    : {config.GEMINI_EMBEDDING_MODEL}")
    print(f"  LLM model    : {config.GEMINI_LLM_MODEL}")
    print("=" * 55 + "\n")

    if not config.GEMINI_API_KEY:
        print("❌  GEMINI_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    start = time.time()

    try:
        ingest(data_dir=config.DATA_DIR)
    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        print(f"    Place your PDF files in: {config.DATA_DIR}")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n✅  Ingestion complete in {elapsed:.1f}s")
    print("    You can now start the server with: python main.py")


if __name__ == "__main__":
    main()