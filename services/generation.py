"""
improvements/generation_json.py
================================
Improvement over services/generation.py: structured JSON output + citation validation.

PROBLEM WITH CURRENT generation.py
------------------------------------
The current system instructs the LLM to append a PAGES_USED line:

    PAGES_USED: Boeing B737 Manual.pdf:15,16|BCN-FCO Menus.pdf:3

This is then parsed with string manipulation using rfind(":") and split("|").
Two failure modes exist:

1. FORMAT DRIFT: If the LLM does not follow the exact format (missing colon,
   extra spaces, different separator), the parser silently returns empty citations.
   This happens more often as models are upgraded or prompts evolve.

2. HALLUCINATED PAGES: The LLM might cite a page number that was not in the
   retrieved context — page 999 of a 70-page document. The current parser
   accepts any integer without validation.

SOLUTION
--------
Replace PAGES_USED with structured JSON output. The LLM returns its entire
response as a JSON object:

    {
      "answer": "The fuel required is 7.2 (1000 KG) and the time is 3:27.",
      "references": [
        {"document_title": "Boeing B737 Manual.pdf", "pages": [15]}
      ]
    }

A validation layer then cross-checks every cited page against the set of pages
that were actually present in the retrieval context. Pages that were not
retrieved cannot be cited — this eliminates hallucinated citations entirely.

WHAT CHANGES
------------
- SYSTEM_PROMPT: PAGES_USED instruction replaced with JSON schema instruction
- _parse_pages_used() replaced with _parse_json_response(raw, context_docs)
- generate_answer() passes context_docs to the parser for validation

WHAT STAYS THE SAME
-------------------
- LLM model and temperature
- All other system prompt rules (grounding, conciseness, no preamble)
- Return type: Tuple[str, List[Dict]]
"""

from __future__ import annotations

import json
from typing import Dict, List, Set, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

import config
from services.retrieval import build_context


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert aviation information assistant with access to:
- Boeing 737 Operations Manual (performance data, procedures, systems)
- Airbus A320/A321 Flight Crew Training Manual (normal operations)
- In-flight menus for routes BCN-FCO and VIE-LHR (food options, allergens)

STRICT RULES:
1. Answer ONLY using the provided context. If not found, say:
   "I could not find this information in the available documents."
2. Never fill gaps with training knowledge. Reproduce exact numbers — never approximate.
3. For allergen/menu data, be precise — passengers may have serious dietary requirements.
4. Be concise and direct, but adapt the level of detail to the question:
   - For simple factual questions (e.g. menu options, allergens, single values),
     provide a short, direct answer.
   - For procedural or multi-part questions (e.g. steps, components, conditions),
     include all necessary steps or elements clearly.

5. Answer ONLY the specific thing asked, but include all essential supporting
   details required for correctness (e.g. values, conditions, thresholds).
   Do not include unrelated information.

6. When the answer contains multiple items (e.g. procedures, components),
   structure it clearly using bullet points or short paragraphs.

OUTPUT FORMAT — mandatory, no exceptions:
Return a single valid JSON object with exactly this structure:

{
  "answer": "<your answer here>",
  "references": [
    {
      "document_title": "<exact filename from [Source: ...] tags>",
      "pages": [<integer page numbers where the evidence appears>]
    }
  ]
}

Rules for references:
- Only include pages where specific evidence for your answer appears.
- Do NOT include pages you read for context but did not use as evidence.
- Use the exact document title as it appears in the [Source: ...] tags.
- If evidence spans multiple documents, include one object per document.
- If no specific page contains the evidence, return an empty references array.

Return JSON only. No markdown fences. No text before or after the JSON object.
"""


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_LLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.0,
        max_output_tokens=2048,
    )


# ── Allowed pages index ───────────────────────────────────────────────────────

def _build_allowed_pages(context_docs: List[Document]) -> Dict[str, Set[int]]:
    """
    Build a lookup of {document_title: {page_numbers}} from the retrieved context.

    This is the validation allowlist — only pages that were actually retrieved
    can appear in the final citations. If the LLM cites a page that was not
    in the context, it is silently discarded.

    Why this matters: LLMs can occasionally hallucinate page numbers, especially
    when the context contains many source labels. Cross-referencing against the
    actual retrieved pages eliminates this failure mode entirely.
    """
    allowed: Dict[str, Set[int]] = {}
    for doc in context_docs:
        title = doc.metadata.get("document_title", "")
        page = doc.metadata.get("page_number")
        if title and isinstance(page, int):
            allowed.setdefault(title, set()).add(page)
    return allowed


# ── JSON response parser ──────────────────────────────────────────────────────

def _parse_json_response(
    raw_response: str,
    context_docs: List[Document],
) -> Tuple[str, List[Dict]]:
    """
    Parse the LLM's JSON response and validate citations against the context.

    Returns:
        answer:     The answer text extracted from the JSON.
        citations:  Validated list of { document_title, pages } dicts.
                    Pages not present in the retrieved context are discarded.

    Graceful degradation:
        If the LLM returns malformed JSON, the raw text is returned as the
        answer with empty citations rather than raising an exception.
    """
    allowed_pages = _build_allowed_pages(context_docs)

    # Strip markdown fences if the LLM wraps output despite instructions
    clean = raw_response.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse failed: {e}. Returning raw text.")
        return raw_response.strip(), []

    answer = str(data.get("answer", "")).strip()
    raw_refs = data.get("references", [])

    validated_refs: List[Dict] = []

    for ref in raw_refs:
        title = str(ref.get("document_title", "")).strip()
        pages_raw = ref.get("pages", [])

        # Validate each page: must be an integer present in the allowed set
        clean_pages = sorted({
            int(p)
            for p in pages_raw
            if isinstance(p, int) and p in allowed_pages.get(title, set())
        })

        if not clean_pages:
            # LLM cited a document+pages not in the retrieved context
            print(f"  ⚠ Citation discarded — '{title}' pages {pages_raw} not in context")
            continue

        validated_refs.append({
            "document_title": title,
            "pages": clean_pages,
        })

    print(f"  Validated citations: {validated_refs}")
    return answer, validated_refs


# ── Main generation function ───────────────────────────────────────────────────

def generate_answer(
    question: str,
    context_docs: List[Document],
) -> Tuple[str, List[Dict]]:
    """
    Generate a grounded answer with validated JSON citations.

    Identical signature to the production generate_answer() — drop-in replacement.

    Args:
        question:     The user's question.
        context_docs: Expanded page-level Documents (reranked + neighbours).

    Returns:
        answer:     Generated answer text.
        citations:  Validated evidence-based citations (pages cross-checked
                    against the actual retrieved context).
    """
    if not context_docs:
        return (
            "I could not find any relevant information in the available documents.",
            [],
        )

    context = build_context(context_docs)

    user_message = f"""Use the following document excerpts to answer the question.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Return your response as a JSON object following the schema in the system prompt.
Only cite pages where the specific evidence for your answer appears."""

    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw = (response.content or "").strip()

    answer, citations = _parse_json_response(raw, context_docs)
    return answer, citations