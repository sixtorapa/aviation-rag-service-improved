"""
improvements/vision_fallback_improved.py
=========================================
Improved version of services/vision_fallback.py combining two changes:

CHANGE 1 — Deterministic detection via ingestion-time metadata
    Production version uses heuristics at query time (_is_visual_page).
    This version reads the `needs_vision` metadata field set at ingestion time
    by ingestion_with_vision.py. Detection is consistent, auditable, and free
    of runtime side effects.

CHANGE 2 — Generalised VISION_PROMPT
    Production prompt is biased toward cockpit flow diagrams (PF/CM1/CM2,
    numbered steps, cockpit areas). This version uses a general-purpose prompt
    that extracts visual information from any technical diagram type: schematics,
    performance charts, brake assembly diagrams, fuel system flows, etc.

WHAT STAYS THE SAME
-------------------
- enrich_visual_pages() public interface — drop-in replacement
- _render_page_as_image() — unchanged
- _get_or_compute_vision_text() — unchanged
- _pdf_cache and _vision_text_cache — unchanged
- MAX_VISION_PAGES cap — unchanged
- atexit cleanup — unchanged
"""

from __future__ import annotations

import atexit
import base64
import io
from typing import Dict, List, Optional, Tuple

import pdfplumber
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import config


# ── Configuration ─────────────────────────────────────────────────────────────

# Max pages enriched with Vision per query — controls latency and cost
MAX_VISION_PAGES = 2

# CHANGE 2: Generalised prompt — works for any aviation technical diagram,
# not just cockpit flow patterns.
#
# Production prompt was:
#   "Focus especially on: numbered steps, cockpit areas, pilot responsibility"
#
# Problem: biased toward A320 cockpit diagrams. A Boeing brake temperature
# diagram or a fuel system schematic would not benefit from cockpit-specific
# framing. The generalised version extracts the same information for cockpit
# diagrams but also works for any other visual content type.
VISION_PROMPT = """You are analyzing a single page from an aviation technical document.

This page may contain diagrams, annotated images, flow charts, schematics,
or tabular data presented visually. Extract ALL factual information that is
conveyed through visual means on this page.

Extract the following types of information if present:

1. SEQUENCES AND STEPS
   - Numbered items or ordered sequences shown visually
   - What each step or numbered item refers to or requires

2. SPATIAL RELATIONSHIPS
   - Arrows showing direction, flow, or association between elements
   - Colour coding and what each colour represents
   - Grouping of elements and what the grouping signifies

3. ASSIGNMENTS AND RESPONSIBILITIES
   - Which role, system, or component is associated with which item
   - Labels that connect an actor or system to an action or area

4. ANNOTATIONS AND LABELS
   - Text callouts on images or diagrams
   - Legend entries and their meanings
   - Unit labels attached to values or measurements

5. STRUCTURED DATA IN VISUAL FORM
   - Tables or grids presented as diagrams
   - Performance charts or limit envelopes with readable values
   - Threshold indicators with associated conditions

6. WARNINGS AND CAUTIONS
   - Warning or caution indicators and what conditions trigger them
   - Safety-critical values or limits shown visually

Rules:
- Extract only information explicitly visible on the page
- Do not add aviation knowledge not present in the image
- Ignore decorative elements, logos, page numbers, and headers
- Be specific: if an arrow points from label A to item B, state that explicitly
- Output plain text only, structured for clarity
"""


# ── Global clients / caches ───────────────────────────────────────────────────

llm_vision = ChatGoogleGenerativeAI(
    model=config.GEMINI_LLM_MODEL,
    google_api_key=config.GEMINI_API_KEY,
    temperature=0.0,
)

_pdf_cache: Dict[str, pdfplumber.PDF] = {}
_vision_text_cache: Dict[Tuple[str, int], str] = {}


def _get_pdf(source: str) -> Optional[pdfplumber.PDF]:
    if not source:
        return None
    if source not in _pdf_cache:
        try:
            _pdf_cache[source] = pdfplumber.open(source)
        except Exception as e:
            print(f"⚠ Vision: could not open {source}: {e}")
            return None
    return _pdf_cache[source]


def _close_pdf_cache() -> None:
    for pdf in _pdf_cache.values():
        try:
            pdf.close()
        except Exception:
            pass
    _pdf_cache.clear()


atexit.register(_close_pdf_cache)


# ── Detection — CHANGE 1: metadata-based instead of heuristic ────────────────

def _is_visual_page(doc: Document) -> bool:
    """
    Determine if a page needs visual enrichment.

    IMPROVEMENT over production version:
    Reads the `needs_vision` boolean set at ingestion time by
    ingestion_with_vision.py, rather than running keyword and PDF geometry
    heuristics at query time.

    Benefits:
    - Deterministic: same page always gets the same classification
    - Auditable: operators can inspect which pages are flagged before deployment
    - Fast: a dict lookup vs. opening the PDF and counting geometric objects

    Backward compatibility fallback:
    If `needs_vision` metadata is absent (document ingested with the original
    ingestion.py), falls back to keyword matching so the module works with
    existing indexes without requiring re-ingestion.
    """
    needs_vision = doc.metadata.get("needs_vision")
    if needs_vision is not None:
        return bool(needs_vision)

    # Fallback for documents ingested without vision metadata
    text = (doc.page_content or "").lower()
    fallback_keywords = [
        "flow pattern", "responsibility", "pf responsibility",
        "cm1/cm2", "cockpit preparation", "diagram",
    ]
    return any(kw in text for kw in fallback_keywords)


def _visual_score(doc: Document) -> int:
    """
    Score a page for ranking among vision candidates.

    Uses the pre-computed `visual_score` from ingestion metadata if available.
    Falls back to a text-length heuristic for backward compatibility.
    """
    stored_score = doc.metadata.get("visual_score")
    if stored_score is not None:
        return int(stored_score)

    # Fallback: shorter text = more likely visual
    return max(0, 10 - len(doc.page_content or "") // 100)


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render_page_as_image(doc: Document) -> Optional[bytes]:
    """Render a PDF page as PNG bytes. Unchanged from production version."""
    source = doc.metadata.get("source", "")
    page_number = doc.metadata.get("page_number")
    if not source or page_number is None:
        return None
    pdf = _get_pdf(source)
    if pdf is None:
        return None
    try:
        page = pdf.pages[page_number - 1]
        img = page.to_image(resolution=110)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"⚠ Vision: render failed p{page_number}: {e}")
        return None


# ── Gemini Vision ─────────────────────────────────────────────────────────────

def _interpret_with_vision(image_bytes: bytes) -> str:
    """Send rendered page to Gemini Vision. Uses the improved VISION_PROMPT."""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": VISION_PROMPT},
    ])
    response = llm_vision.invoke([message])
    content = response.content
    return content.strip() if isinstance(content, str) else str(content).strip()


def _get_or_compute_vision_text(doc: Document) -> str:
    """Return cached vision text or compute and cache it. Unchanged."""
    source = doc.metadata.get("source", "")
    page_number = doc.metadata.get("page_number")
    if not source or page_number is None:
        return ""
    cache_key = (source, int(page_number))
    if cache_key in _vision_text_cache:
        return _vision_text_cache[cache_key]
    image_bytes = _render_page_as_image(doc)
    if not image_bytes:
        return ""
    vision_text = _interpret_with_vision(image_bytes)
    if vision_text:
        _vision_text_cache[cache_key] = vision_text
    return vision_text


# ── Public API ────────────────────────────────────────────────────────────────

def enrich_visual_pages(docs: List[Document]) -> List[Document]:
    """
    Selectively enrich visual pages in the retrieved context.

    Drop-in replacement for the production enrich_visual_pages().

    Key differences from production:
    1. Detection uses needs_vision metadata (set at ingestion) rather than
       runtime heuristics — deterministic and consistent across queries.
    2. Vision prompt is generalised to work on any aviation diagram type,
       not just cockpit flow patterns.
    3. If a page was already enriched at ingestion time (ingestion_with_vision.py),
       the [VISUAL INTERPRETATION] block is already in page_content — no
       additional Vision API call is made.
    """
    if not docs:
        return docs

    visual_candidates = [doc for doc in docs if _is_visual_page(doc)]
    if not visual_candidates:
        return docs

    ranked = sorted(visual_candidates, key=_visual_score, reverse=True)
    selected = ranked[:MAX_VISION_PAGES]
    selected_ids = {id(doc) for doc in selected}

    enriched: List[Document] = []

    for doc in docs:
        if id(doc) not in selected_ids:
            enriched.append(doc)
            continue

        page = doc.metadata.get("page_number", "?")
        title = doc.metadata.get("document_title", "?")

        # Already enriched at ingestion time — no API call needed
        if "[VISUAL INTERPRETATION]" in doc.page_content:
            print(f"✅ Vision already embedded at ingestion: {title} p{page}")
            enriched.append(doc)
            continue

        score = _visual_score(doc)
        print(f"🔍 Vision fallback (query-time): {title} p{page} (score={score})")

        vision_text = _get_or_compute_vision_text(doc)
        if not vision_text:
            enriched.append(doc)
            continue

        enriched_content = (
            f"[VISUAL INTERPRETATION]\n{vision_text}\n\n"
            f"[ORIGINAL TEXT]\n{doc.page_content}"
        )
        enriched.append(Document(
            page_content=enriched_content,
            metadata=doc.metadata.copy(),
        ))
        print(f"  ✅ Vision applied for p{page}")

    return enriched