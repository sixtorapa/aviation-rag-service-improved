"""
improvements/ingestion_with_vision.py
======================================
Improvement over services/ingestion.py: vision enrichment moved to ingestion time.

PROBLEM WITH CURRENT APPROACH
-------------------------------
In the production system, vision enrichment happens at query time:

    Query arrives → retrieval → reranking → expansion → vision fallback → generation

This means:
1. Every query that retrieves a diagram-heavy page pays the rendering + Vision API cost
2. The in-memory cache helps within a session, but is lost on server restart
3. Rendering the same page repeatedly wastes compute and API budget

SOLUTION
--------
Move vision enrichment to ingestion time. During load_pdf_pages(), every page
is scored for visual content. Pages above the VISUAL_SCORE_THRESHOLD are
rendered and interpreted by Gemini Vision once. The visual interpretation is
prepended to the page text before embedding, so the enriched content is stored
in ChromaDB from the start.

At query time, vision_fallback.py simply filters by the `needs_vision` metadata
field rather than running heuristics and making Vision API calls.

This trades higher ingestion cost (one-time) for lower query cost (per-request).

WHAT CHANGES
------------
- load_pdf_pages() calls compute_visual_score() and extract_vision_text() per page
- Pages with needs_vision=True get enriched text before embedding
- Metadata gains a `needs_vision` boolean field
- Two new helper functions: compute_visual_score(), extract_vision_text()

WHAT STAYS THE SAME
-------------------
- Page-level chunking (1 page = 1 chunk)
- Table extraction via extract_tables()
- Blank page filtering
- All other metadata fields (source, document_title, page_number)
- ChromaDB and BM25 build/load functions (unchanged)
"""

from __future__ import annotations

import base64
import io
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import pdfplumber
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.messages import HumanMessage

import config


# ── Vision configuration ──────────────────────────────────────────────────────

# Pages scoring at or above this threshold are enriched with Gemini Vision.
# Tune this value based on corpus characteristics.
VISUAL_SCORE_THRESHOLD = 6

# Keywords that strongly suggest diagram/visual content
STRONG_VISUAL_KEYWORDS = [
    "flow pattern", "responsibility", "pf responsibility", "cm1/cm2",
    "cockpit preparation", "diagram", "schema", "figure",
]

VISION_PROMPT = """You are analyzing a single page from an aviation technical document.

Extract ALL factual information that is conveyed visually on this page, including:
- Numbered sequences and what each step or item refers to
- Spatial relationships shown by arrows, lines, or colour coding
- Responsibility or role assignments (e.g. which pilot role handles which area)
- Legends, labels, callouts, and annotations
- Tabular structures presented as diagrams
- Warnings or caution indicators with associated conditions

Focus only on factual content. Ignore decorative elements.
Output plain text only. Be specific and structured.
"""


# ── Visual scoring ────────────────────────────────────────────────────────────

def compute_visual_score(page: pdfplumber.page.Page, text: str) -> int:
    """
    Score a PDF page for visual content richness.

    Combines semantic signals from extracted text with structural signals
    from the PDF geometry. Higher score = stronger candidate for vision enrichment.

    This runs at ingestion time so the computation is paid once,
    not on every query that retrieves the page.
    """
    score = 0
    text_lower = text.lower()

    # Semantic signals
    for kw in STRONG_VISUAL_KEYWORDS:
        if kw in text_lower:
            score += 4

    if len(text) < 350:
        score += 2  # sparse text often means diagram-heavy page

    # Structural PDF signals
    try:
        score += min(len(page.images or []) * 3, 6)      # embedded images
        score += min(len(page.rects or []) // 5, 4)      # rectangle density
        score += min(len(page.curves or []) // 10, 3)    # vector curves
        score += min(len(page.lines or []) // 15, 2)     # line density
    except Exception:
        pass  # structural inspection is best-effort

    return score


# ── Vision extraction ─────────────────────────────────────────────────────────

def extract_vision_text(pdf_path: str, page_number: int) -> str:
    """
    Render a PDF page as PNG and interpret it with Gemini Vision.

    Called at ingestion time for pages that score above VISUAL_SCORE_THRESHOLD.
    The result is stored in the ChromaDB document content so it is searchable
    and available at query time without any additional API calls.

    Args:
        pdf_path:    Absolute path to the source PDF.
        page_number: 1-based absolute page index.

    Returns:
        Vision interpretation as plain text, or empty string on failure.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            img = page.to_image(resolution=110)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
    except Exception as e:
        print(f"  ⚠ Vision render failed for {pdf_path} p{page_number}: {e}")
        return ""

    try:
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_LLM_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0.0,
        )
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": VISION_PROMPT},
        ])
        response = llm.invoke([message])
        return (response.content or "").strip()
    except Exception as e:
        print(f"  ⚠ Vision API failed for p{page_number}: {e}")
        return ""


# ── PDF loading (with vision enrichment) ──────────────────────────────────────

def load_pdf_pages(pdf_path: str) -> List[Document]:
    """
    Load a PDF file and return one Document per page.

    Improvement over production version: pages with high visual content scores
    are enriched with Gemini Vision interpretation at ingestion time.
    The enriched text is embedded into ChromaDB so it is available for
    semantic retrieval without any query-time Vision API calls.

    Metadata:
        source:           absolute file path
        document_title:   filename
        page_number:      1-based absolute index
        needs_vision:     True if the page was vision-enriched (for logging/audit)
        visual_score:     numeric score for transparency
    """
    filename = os.path.basename(pdf_path)
    pages: List[Document] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # Table extraction — preserves structure for BM25 matching
            tables = page.extract_tables()
            if tables:
                table_lines = []
                for table in tables:
                    for row in table:
                        clean_row = "\t".join(str(c) for c in row if c is not None)
                        if clean_row.strip():
                            table_lines.append(clean_row)
                if table_lines:
                    table_text = "\n".join(table_lines)
                    if table_text not in text:
                        text = text + "\n" + table_text

            text = text.strip()
            if not text:
                print(f"  ⚠  {filename} p{page_idx}: blank, skipping")
                continue

            # Visual scoring — computed once at ingestion time
            visual_score = compute_visual_score(page, text)
            needs_vision = visual_score >= VISUAL_SCORE_THRESHOLD

            combined_text = text

            if needs_vision:
                print(f"  🔍 Vision enrichment at ingestion: {filename} p{page_idx} (score={visual_score})")
                vision_text = extract_vision_text(pdf_path, page_idx)
                if vision_text:
                    combined_text = (
                        f"[VISUAL INTERPRETATION]\n{vision_text}\n\n"
                        f"[EXTRACTED TEXT]\n{text}"
                    )
                    print(f"    ✅ Vision text embedded for p{page_idx}")

            pages.append(Document(
                page_content=combined_text,
                metadata={
                    "source": pdf_path,
                    "document_title": filename,
                    "page_number": page_idx,
                    "needs_vision": needs_vision,
                    "visual_score": visual_score,
                },
            ))

    print(f"  ✅ {filename}: {len(pages)} pages loaded "
          f"({sum(1 for p in pages if p.metadata.get('needs_vision'))} vision-enriched)")
    return pages


def load_all_pdfs(data_dir: str) -> List[Document]:
    """Walk data_dir and load all PDFs. Same interface as production version."""
    all_pages: List[Document] = []
    pdf_files = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")
    print(f"\n📂 Loading {len(pdf_files)} PDF(s) from {data_dir}")
    for pdf_path in pdf_files:
        pages = load_pdf_pages(str(pdf_path))
        all_pages.extend(pages)
    print(f"\n📄 Total pages loaded: {len(all_pages)}")
    vision_count = sum(1 for p in all_pages if p.metadata.get("needs_vision"))
    print(f"   Vision-enriched pages: {vision_count}\n")
    return all_pages


# ── Vector store, BM25, and ingest pipeline ───────────────────────────────────
# These functions are identical to the production version.
# Reproduced here for completeness so this file is a self-contained drop-in.

def _get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=config.GEMINI_EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY,
    )


def build_vector_store(docs: List[Document]) -> Chroma:
    print(f"🔢 Embedding {len(docs)} pages with {config.GEMINI_EMBEDDING_MODEL}...")
    embeddings = _get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.VECTOR_STORE_PATH,
    )
    print(f"✅ Vector store persisted to {config.VECTOR_STORE_PATH}")
    return vector_store


def load_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=config.VECTOR_STORE_PATH,
    )


_BM25_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "vector_store", "bm25_index.pkl"
)


def build_bm25_index(docs: List[Document]) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(docs, k=config.RETRIEVAL_K)
    os.makedirs(os.path.dirname(_BM25_CACHE_PATH), exist_ok=True)
    with open(_BM25_CACHE_PATH, "wb") as f:
        pickle.dump(retriever, f)
    print(f"✅ BM25 index cached to {_BM25_CACHE_PATH}")
    return retriever


def load_bm25_index() -> BM25Retriever:
    with open(_BM25_CACHE_PATH, "rb") as f:
        return pickle.load(f)


def bm25_index_exists() -> bool:
    return os.path.exists(_BM25_CACHE_PATH)


def ingest(data_dir: str = config.DATA_DIR) -> Tuple[Chroma, BM25Retriever]:
    """Full ingestion pipeline with vision enrichment at ingestion time."""
    docs = load_all_pdfs(data_dir)
    vector_store = build_vector_store(docs)
    bm25_retriever = build_bm25_index(docs)
    return vector_store, bm25_retriever