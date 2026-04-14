"""
services/ingestion.py

Responsible for:
  1. Loading each PDF document page by page (1 chunk = 1 PDF page).
  2. Embedding the pages with Google's text-embedding-004 model.
  3. Persisting the vector store to disk (ChromaDB).
  4. Building an in-memory BM25 index for hybrid search.

Why page-level chunking?
  The evaluation metric is page-level retrieval accuracy. Using one chunk per
  page means the chunk boundary and the citation boundary are identical —
  retrieving the right chunk automatically gives the correct page number.
  Sub-page chunking would require extra logic to map chunks back to pages and
  could introduce off-by-one errors when a chunk spans a page break.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import pdfplumber
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

import config


# ── PDF loading ───────────────────────────────────────────────────────────────

def load_pdf_pages(pdf_path: str) -> List[Document]:
    """
    Load a PDF file and return one Document per page.

    Metadata stored per page:
      - source:           absolute file path
      - document_title:   filename (used in the API response references)
      - page_number:      1-based absolute PDF page index (what the evaluator expects)

    pdfplumber is used over PyPDF because it handles tables and mixed-layout
    pages (like the Boeing performance tables) more reliably.
    """
    filename = os.path.basename(pdf_path)
    pages: List[Document] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            # Extract text; pdfplumber returns None for image-only pages
            text = page.extract_text() or ""

            # For pages with tables, also extract structured table text so
            # numeric values are queryable (e.g. Boeing performance figures)
            tables = page.extract_tables()
            if tables:
                table_lines = []
                for table in tables:
                    for row in table:
                        # Filter None cells, join with tab for readability
                        clean_row = "\t".join(str(c) for c in row if c is not None)
                        if clean_row.strip():
                            table_lines.append(clean_row)
                if table_lines:
                    table_text = "\n".join(table_lines)
                    # Append table text if not already captured by extract_text()
                    if table_text not in text:
                        text = text + "\n" + table_text

            text = text.strip()

            # Skip completely blank pages (e.g. "Intentionally Blank" pages
            # in the Boeing manual) — they add noise without signal
            if not text:
                print(f"  ⚠  {filename} p{page_idx}: blank, skipping")
                continue

            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "document_title": filename,
                        "page_number": page_idx,  # 1-based absolute index
                    },
                )
            )

    print(f"  ✅ {filename}: {len(pages)} pages loaded")
    return pages


def load_all_pdfs(data_dir: str) -> List[Document]:
    """
    Walk data_dir and load all PDF files. Returns a flat list of page-level Documents.
    """
    all_pages: List[Document] = []
    pdf_files = sorted(Path(data_dir).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    print(f"\n📂 Loading {len(pdf_files)} PDF(s) from {data_dir}")
    for pdf_path in pdf_files:
        pages = load_pdf_pages(str(pdf_path))
        all_pages.extend(pages)

    print(f"\n📄 Total pages loaded: {len(all_pages)}\n")
    return all_pages


# ── Vector store ──────────────────────────────────────────────────────────────

def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Instantiate the Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=config.GEMINI_EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )


def build_vector_store(docs: List[Document]) -> Chroma:
    """
    Embed all documents and persist to ChromaDB.
    If the collection already exists it is replaced (re-ingest always wins).
    """
    print(f"🔢 Embedding {len(docs)} pages with {config.GEMINI_EMBEDDING_MODEL}...")

    embeddings = _get_embeddings()

    # persist_directory=None would be in-memory only;
    # we write to disk so the server can load without re-embedding on restart
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.VECTOR_STORE_PATH,
    )

    print(f"✅ Vector store persisted to {config.VECTOR_STORE_PATH}")
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing ChromaDB collection from disk (no re-embedding needed)."""
    embeddings = _get_embeddings()
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.VECTOR_STORE_PATH,
    )


# ── BM25 index ────────────────────────────────────────────────────────────────

_BM25_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "vector_store", "bm25_index.pkl"
)


def build_bm25_index(docs: List[Document]) -> BM25Retriever:
    """
    Build a BM25 retriever from the page-level documents and cache it to disk.

    BM25 is keyword-based: it excels at matching exact terms like aircraft model
    numbers ("737-600"), technical codes ("PD.10.1"), allergen names, or menu
    item names that a dense vector might miss due to low semantic signal.
    """
    retriever = BM25Retriever.from_documents(docs, k=config.RETRIEVAL_K)

    # Persist so the server can reload without keeping all raw text in memory
    os.makedirs(os.path.dirname(_BM25_CACHE_PATH), exist_ok=True)
    with open(_BM25_CACHE_PATH, "wb") as f:
        pickle.dump(retriever, f)

    print(f"✅ BM25 index cached to {_BM25_CACHE_PATH}")
    return retriever


def load_bm25_index() -> BM25Retriever:
    """Load the cached BM25 retriever from disk."""
    with open(_BM25_CACHE_PATH, "rb") as f:
        return pickle.load(f)


def bm25_index_exists() -> bool:
    return os.path.exists(_BM25_CACHE_PATH)


# ── Main ingest pipeline ──────────────────────────────────────────────────────

def ingest(data_dir: str = config.DATA_DIR) -> Tuple[Chroma, BM25Retriever]:
    """
    Full ingestion pipeline:
      load PDFs → embed → persist vector store → build BM25 index

    Returns (vector_store, bm25_retriever) ready to be used by the retrieval layer.
    Call this once before starting the server, via: python ingest.py
    """
    docs = load_all_pdfs(data_dir)
    vector_store = build_vector_store(docs)
    bm25_retriever = build_bm25_index(docs)
    return vector_store, bm25_retriever