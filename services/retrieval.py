"""
services/retrieval.py

Hybrid BM25 + vector retrieval with cross-encoder reranking
and context window expansion.

Full pipeline:
  1. EnsembleRetriever (BM25 + vector) retrieves RETRIEVAL_K candidate pages
  2. Cross-encoder reranks: scores each (question, page) pair independently
  3. Context expansion: adds ±1 neighbour pages to each reranked page
     so the LLM sees surrounding context (e.g. table header on p40 + data on p41)
  4. Generation: LLM receives expanded context and is asked to report
     which specific pages it used — those become the final citations

Why separate "context for understanding" from "pages for citing"?
  The reranker sometimes misses pages that are pure tables (low text signal)
  but are adjacent to high-scoring pages. Expanding ±1 gives the LLM the full
  picture. Asking the LLM to report evidence pages means citations reflect
  what actually supported the answer, not what the retriever happened to return.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Set, Tuple

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import config
from services.ingestion import (
    bm25_index_exists,
    load_bm25_index,
    load_vector_store,
)

# Cross-encoder model
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Pages kept after reranking (before expansion)
RERANK_TOP_K = 3


# ── Reranker ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    print(f"🔄 Loading reranker ({_RERANKER_MODEL})...")
    model = CrossEncoder(_RERANKER_MODEL)
    print("✅ Reranker ready")
    return model


def rerank(question: str, docs: List[Document]) -> List[Document]:
    """
    Score each (question, page) pair and keep top RERANK_TOP_K pages.
    Returns docs sorted by relevance score descending.
    """
    if not docs:
        return docs

    reranker = _get_reranker()
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:RERANK_TOP_K]]

    print(f"  Reranker: {len(docs)} → {len(top_docs)} pages kept")
    for score, doc in scored[:RERANK_TOP_K]:
        page = doc.metadata.get("page_number", "?")
        title = doc.metadata.get("document_title", "?")
        print(f"    score={score:.3f} | {title} p{page}")

    return top_docs


# ── Context window expansion ──────────────────────────────────────────────────

def expand_context(
    reranked_docs: List[Document],
    all_candidate_docs: List[Document],
) -> List[Document]:
    """
    For each reranked page, add its immediate neighbours (±1) from the
    same document if they exist in the original candidate pool.

    This solves the "split table" problem: when a table header is on page N
    and the data is on page N+1, the reranker may only pick page N because
    it has more keyword overlap. Expanding to N+1 gives the LLM the full table.

    The expanded pages are used ONLY as context for the LLM — citations
    are determined later by asking the LLM which pages it actually used.

    Args:
        reranked_docs:     Top-K pages selected by the reranker
        all_candidate_docs: All pages returned by the ensemble retriever

    Returns:
        Deduplicated list of pages: reranked + their neighbours, ordered
        by document and page number for coherent reading.
    """
    # Build lookup: (document_title, page_number) → Document
    candidate_index: Dict[Tuple[str, int], Document] = {}
    for doc in all_candidate_docs:
        meta = doc.metadata or {}
        title = meta.get("document_title", "")
        page = meta.get("page_number")
        if title and page is not None:
            candidate_index[(title, page)] = doc

    # Collect pages to include
    included: Dict[Tuple[str, int], Document] = {}

    for doc in reranked_docs:
        meta = doc.metadata or {}
        title = meta.get("document_title", "")
        page = meta.get("page_number")
        if not title or page is None:
            continue

        # Always include the reranked page itself
        included[(title, page)] = doc

        # Add neighbours if they exist in the candidate pool
        for neighbour_page in [page - 1, page + 1]:
            key = (title, neighbour_page)
            if key in candidate_index and key not in included:
                included[key] = candidate_index[key]

    # Sort by document title then page number for coherent context
    sorted_docs = sorted(
        included.values(),
        key=lambda d: (
            d.metadata.get("document_title", ""),
            d.metadata.get("page_number", 0),
        ),
    )

    expanded_pages = [
        f"{d.metadata.get('document_title','?')} p{d.metadata.get('page_number','?')}"
        for d in sorted_docs
    ]
    print(f"  Context expansion: {len(reranked_docs)} → {len(sorted_docs)} pages")
    print(f"    Pages sent to LLM: {expanded_pages}")

    return sorted_docs


# ── Retriever factory ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_retriever() -> EnsembleRetriever:
    """Build and cache the hybrid BM25 + vector retriever."""
    if not bm25_index_exists():
        raise RuntimeError(
            "BM25 index not found. Run 'python ingest.py' before starting the server."
        )

    print("🔄 Loading retrieval indexes...")

    vector_store = load_vector_store()
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": config.RETRIEVAL_K}
    )

    bm25_retriever = load_bm25_index()
    bm25_retriever.k = config.RETRIEVAL_K

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[config.BM25_WEIGHT, config.VECTOR_WEIGHT],
    )

    print("✅ Hybrid retriever ready")
    return retriever


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(docs: List[Document]) -> str:
    """Format expanded pages into a context string for the LLM prompt."""
    parts = []
    for doc in docs:
        meta = doc.metadata or {}
        title = meta.get("document_title", "unknown")
        page = meta.get("page_number", "?")
        parts.append(
            f"[Source: {title}, Page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)