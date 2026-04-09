# """
# services/vision_fallback.py

# Fallback multimodal MUY acotado:
# - Solo se activa para la página 26 del documento
#   "A320_321 Flight Crew Training Manual - Normal Operations.pdf"
# - No usa heurísticas globales
# - Reutiliza el cliente Gemini Vision
# - Mantiene metadata intacta para citations
# """

# from __future__ import annotations

# import io
# import base64
# from typing import List, Optional

# import pdfplumber
# from langchain_core.documents import Document
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage

# import config


# # ---------------------------------------------------------------------------
# # CONFIG: solo activar vision en esta página concreta
# # ---------------------------------------------------------------------------

# SPECIAL_VISION_PAGES = {
#     ("A320_321 Flight Crew Training Manual - Normal Operations.pdf", 26),
# }

# VISION_PROMPT = """You are analyzing a page from an aviation flight crew training manual.

# This page may contain a diagram or annotated image with operational information.
# Extract only factual information that is explicitly visible on the page.

# Focus especially on:
# - numbered steps
# - cockpit areas associated with those steps
# - pilot responsibility assignments
# - visual relationships such as arrows, grouping, labels, and color-coded sections

# Be precise and concise.
# Output plain text only.
# """


# # ---------------------------------------------------------------------------
# # Reutilizamos el cliente para no recrearlo en cada página
# # ---------------------------------------------------------------------------

# llm_vision = ChatGoogleGenerativeAI(
#     model=config.GEMINI_LLM_MODEL,  # p.ej. gemini-2.5-flash
#     google_api_key=config.GEMINI_API_KEY,
#     temperature=0.0,
# )


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def _is_special_visual_page(doc: Document) -> bool:
#     """
#     Activa Vision solo para una página muy concreta.
#     """
#     title = doc.metadata.get("document_title")
#     page = doc.metadata.get("page_number")
#     return (title, page) in SPECIAL_VISION_PAGES


# def _render_page_as_image(doc: Document) -> Optional[bytes]:
#     """
#     Renderiza una página PDF como PNG.
#     Devuelve bytes PNG o None si falla.
#     """
#     source = doc.metadata.get("source", "")
#     page_number = doc.metadata.get("page_number")  # 1-based

#     if not source or page_number is None:
#         return None

#     try:
#         with pdfplumber.open(source) as pdf:
#             page = pdf.pages[page_number - 1]  # pdfplumber usa 0-based
#             img = page.to_image(resolution=100)  # 100 DPI suele bastar y va más rápido
#             buf = io.BytesIO()
#             img.save(buf, format="PNG")
#             return buf.getvalue()
#     except Exception as e:
#         print(f"  ⚠ Vision fallback: no se pudo renderizar p{page_number} de {source}: {e}")
#         return None


# def _interpret_with_gemini_vision(image_bytes: bytes) -> str:
#     """
#     Envía la imagen a Gemini Vision y devuelve el texto interpretado.
#     """
#     image_b64 = base64.b64encode(image_bytes).decode("utf-8")

#     message = HumanMessage(
#         content=[
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/png;base64,{image_b64}"
#                 },
#             },
#             {
#                 "type": "text",
#                 "text": VISION_PROMPT,
#             },
#         ]
#     )

#     response = llm_vision.invoke([message])
#     content = response.content

#     if isinstance(content, str):
#         return content.strip()

#     # Por compatibilidad defensiva, si LangChain devuelve algo raro
#     return str(content).strip() if content else ""


# # ---------------------------------------------------------------------------
# # API pública
# # ---------------------------------------------------------------------------

# def enrich_visual_pages(docs: List[Document]) -> List[Document]:
#     """
#     Enriquece solo la página 26 del manual A320 con interpretación visual.
#     El resto de páginas se devuelven sin cambios.

#     Args:
#         docs: lista de Document ya recuperados (normalmente tras retrieval/rerank)

#     Returns:
#         Lista de Document, con la página especial enriquecida si aplica.
#     """
#     enriched: List[Document] = []

#     for doc in docs:
#         if not _is_special_visual_page(doc):
#             enriched.append(doc)
#             continue

#         page = doc.metadata.get("page_number", "?")
#         title = doc.metadata.get("document_title", "?")
#         print(f"  🔍 Vision fallback (selectivo): {title} p{page}")

#         image_bytes = _render_page_as_image(doc)
#         if image_bytes is None:
#             enriched.append(doc)
#             continue

#         vision_text = _interpret_with_gemini_vision(image_bytes)
#         if not vision_text:
#             enriched.append(doc)
#             continue

#         enriched_content = (
#             f"[VISUAL INTERPRETATION OF PAGE {page}]\n"
#             f"{vision_text}\n\n"
#             f"[EXTRACTED TEXT]\n"
#             f"{doc.page_content}"
#         )

#         enriched_doc = Document(
#             page_content=enriched_content,
#             metadata=doc.metadata.copy(),  # importantísimo: preservar page_number/source/title
#         )

#         enriched.append(enriched_doc)
#         print(f"    ✅ Vision enrichment completo para p{page}")

#     return enriched




"""
services/vision_fallback.py

Selective multimodal fallback for visually-structured pages.

Goal
----
Keep the main RAG pipeline text-first, but selectively enrich retrieved pages
that likely contain important visual relationships (diagrams, flow patterns,
annotated cockpit images, responsibility maps, etc.).

Design
------
- Applies ONLY to already-retrieved pages, never to the full corpus
- Uses a hybrid detector:
    1) semantic signals from extracted text
    2) structural signals from the PDF page (images, rects, curves, etc.)
- Scores visual candidates and enriches only the top-N per query
- Reuses Gemini client for performance
- Caches opened PDFs and page-level vision outputs in memory
- Preserves original metadata for exact citations
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


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Max number of retrieved pages to enrich with Vision per query.
# Keep this small to control latency and cost.
MAX_VISION_PAGES = 2

# If extracted text is very short, the page may be diagram-heavy.
LOW_TEXT_THRESHOLD = 350

# Semantic signals suggesting visual / diagrammatic operational content.
STRONG_VISUAL_KEYWORDS = [
    "flow pattern",
    "responsibility",
    "pf responsibility",
    "cm1/cm2",
    "cockpit preparation",
    "diagram",
    "schema",
]

MEDIUM_VISUAL_KEYWORDS = [
    "figure",
    "cockpit",
    "step",
    "steps",
    "arrow",
    "sequence",
    "annotated",
    "layout",
]

VISION_PROMPT = """You are analyzing a single page from an aviation flight crew training manual.

Extract only factual information that is explicitly visible on the page.

Focus especially on:
- numbered steps
- cockpit areas associated with those steps
- pilot responsibility assignments for each specific step
- visual relationships such as arrows, grouping, labels, and color-coded sections

Rules:
- Do not speculate
- Do not add aviation knowledge not present on the page
- For each numbered step, assign only the most directly associated pilot role shown by the diagram
- Do not assign multiple roles to a step unless the diagram explicitly shows multiple roles for that same numbered step
- If a role applies to one side or area only, state that clearly
- Be precise and concise

Output plain text only.
"""


# ---------------------------------------------------------------------------
# GLOBAL REUSABLE CLIENTS / CACHES
# ---------------------------------------------------------------------------

# Reuse the same Gemini client instead of recreating it for every page.
llm_vision = ChatGoogleGenerativeAI(
    model=config.GEMINI_LLM_MODEL,
    google_api_key=config.GEMINI_API_KEY,
    temperature=0.0,
)

# Cache opened PDFs by path to avoid repeated pdfplumber.open(...)
_pdf_cache: Dict[str, pdfplumber.PDF] = {}

# Cache vision interpretations in-memory for this process.
# Key: (source, page_number)
_vision_text_cache: Dict[Tuple[str, int], str] = {}


# ---------------------------------------------------------------------------
# CACHE / RESOURCE HELPERS
# ---------------------------------------------------------------------------

def _get_pdf(source: str) -> Optional[pdfplumber.PDF]:
    """
    Return a cached pdfplumber PDF object for this source path.
    """
    if not source:
        return None

    if source not in _pdf_cache:
        try:
            _pdf_cache[source] = pdfplumber.open(source)
        except Exception as e:
            print(f"⚠ Vision fallback: could not open PDF {source}: {e}")
            return None

    return _pdf_cache[source]


def _close_pdf_cache() -> None:
    """
    Close cached PDFs on process exit.
    """
    for pdf in _pdf_cache.values():
        try:
            pdf.close()
        except Exception:
            pass
    _pdf_cache.clear()


atexit.register(_close_pdf_cache)


# ---------------------------------------------------------------------------
# DETECTION / SCORING
# ---------------------------------------------------------------------------

def _safe_page_content(doc: Document) -> str:
    return (doc.page_content or "").strip()


def _extract_page_structural_features(doc: Document) -> Dict[str, int]:
    """
    Read lightweight structural signals from the PDF page.

    Returns counts like:
    - images
    - rects
    - curves
    - lines

    If anything fails, returns zeros.
    """
    source = doc.metadata.get("source", "")
    page_number = doc.metadata.get("page_number")

    features = {
        "images": 0,
        "rects": 0,
        "curves": 0,
        "lines": 0,
    }

    if not source or page_number is None:
        return features

    pdf = _get_pdf(source)
    if pdf is None:
        return features

    try:
        page = pdf.pages[page_number - 1]  # 1-based -> 0-based
        features["images"] = len(page.images or [])
        features["rects"] = len(page.rects or [])
        features["curves"] = len(page.curves or [])
        features["lines"] = len(page.lines or [])
    except Exception as e:
        print(f"⚠ Vision fallback: could not inspect page {page_number} of {source}: {e}")

    return features


def _visual_score(doc: Document) -> int:
    """
    Score how likely a page is to need multimodal interpretation.

    Higher score = better candidate for Vision enrichment.
    """
    text = _safe_page_content(doc).lower()
    text_len = len(text)

    score = 0

    # Strong semantic signals
    for kw in STRONG_VISUAL_KEYWORDS:
        if kw in text:
            score += 4

    # Medium semantic signals
    for kw in MEDIUM_VISUAL_KEYWORDS:
        if kw in text:
            score += 1

    # Additional signal for numbered steps / list-like operational visuals
    if "1." in text:
        score += 2
    if "2." in text:
        score += 1
    if "step 1" in text:
        score += 2

    # Low text density is a useful clue, but not decisive on its own
    if text_len < LOW_TEXT_THRESHOLD:
        score += 2

    # Structural PDF signals
    features = _extract_page_structural_features(doc)
    if features["images"] > 0:
        score += 3
    if features["rects"] > 15:
        score += 2
    if features["curves"] > 20:
        score += 2
    if features["lines"] > 25:
        score += 1

    return score


def _is_visual_page(doc: Document) -> bool:
    """
    Decide whether a page is a candidate for visual enrichment.

    This is intentionally conservative: we only want pages that show real signs
    of diagrammatic / visual structure.
    """
    text = _safe_page_content(doc).lower()
    text_len = len(text)

    # Fast positive path: strong semantic evidence
    if any(kw in text for kw in STRONG_VISUAL_KEYWORDS):
        return True

    # Otherwise require BOTH:
    # - relatively low text density
    # - some visible PDF structure
    if text_len < LOW_TEXT_THRESHOLD:
        features = _extract_page_structural_features(doc)
        has_visual_structure = (
            features["images"] > 0
            or features["rects"] > 15
            or features["curves"] > 20
            or features["lines"] > 25
        )
        return has_visual_structure

    return False


# ---------------------------------------------------------------------------
# RENDERING
# ---------------------------------------------------------------------------

def _render_page_as_image(doc: Document) -> Optional[bytes]:
    """
    Render a PDF page as PNG bytes.
    """
    source = doc.metadata.get("source", "")
    page_number = doc.metadata.get("page_number")

    if not source or page_number is None:
        return None

    pdf = _get_pdf(source)
    if pdf is None:
        return None

    try:
        page = pdf.pages[page_number - 1]
        # 110 DPI is a decent balance between readability and speed.
        img = page.to_image(resolution=110)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"⚠ Vision fallback: could not render page {page_number} of {source}: {e}")
        return None


# ---------------------------------------------------------------------------
# GEMINI VISION
# ---------------------------------------------------------------------------

def _interpret_with_vision(image_bytes: bytes) -> str:
    """
    Send rendered page image to Gemini Vision and return extracted plain text.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}"
                },
            },
            {
                "type": "text",
                "text": VISION_PROMPT,
            },
        ]
    )

    response = llm_vision.invoke([message])
    content = response.content

    if isinstance(content, str):
        return content.strip()

    return str(content).strip() if content else ""


def _get_or_compute_vision_text(doc: Document) -> str:
    """
    Return cached vision text if available; otherwise compute and cache it.
    """
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


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def enrich_visual_pages(docs: List[Document]) -> List[Document]:
    """
    Selectively enrich a small number of retrieved pages with multimodal output.

    Algorithm:
    1) Inspect only the retrieved docs
    2) Mark pages that look visual-heavy
    3) Score those pages
    4) Keep only the top MAX_VISION_PAGES candidates
    5) Render and enrich only those pages
    6) Return all docs, preserving original metadata

    Notes:
    - If no page qualifies, the input docs are returned unchanged.
    - This does NOT touch documents outside the current retrieval context.
    """
    if not docs:
        return docs

    enriched_docs: List[Document] = []

    # 1) Candidate selection
    visual_candidates = [doc for doc in docs if _is_visual_page(doc)]

    if not visual_candidates:
        return docs

    # 2) Score + prioritize
    ranked_candidates = sorted(
        visual_candidates,
        key=_visual_score,
        reverse=True,
    )

    # 3) Cap how many pages per query get multimodal enrichment
    selected_candidates = ranked_candidates[:MAX_VISION_PAGES]

    # Use a stable identity for fast lookup
    selected_ids = {id(doc) for doc in selected_candidates}

    for doc in docs:
        if id(doc) not in selected_ids:
            enriched_docs.append(doc)
            continue

        page = doc.metadata.get("page_number", "?")
        title = doc.metadata.get("document_title", "?")
        score = _visual_score(doc)

        print(f"🔍 Vision fallback: {title} p{page} (score={score})")

        vision_text = _get_or_compute_vision_text(doc)
        if not vision_text:
            enriched_docs.append(doc)
            continue

        enriched_content = (
            f"[VISUAL INTERPRETATION]\n"
            f"{vision_text}\n\n"
            f"[ORIGINAL TEXT]\n"
            f"{doc.page_content}"
        )

        enriched_doc = Document(
            page_content=enriched_content,
            metadata=doc.metadata.copy(),  # preserve exact citation metadata
        )

        enriched_docs.append(enriched_doc)
        print(f"✅ Vision applied to p{page}")

    return enriched_docs