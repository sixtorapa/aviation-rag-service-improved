"""
services/generation.py

Generates grounded answers via Gemini and extracts evidence-based citations.

Key design: the LLM is asked to report which specific pages it used.
This separates "pages for context" (expanded neighbours) from
"pages for citing" (only those that actually supported the answer).

The spec penalises non-relevant pages in references — so citations
must reflect evidence, not retrieval artefacts.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

import config
from services.retrieval import build_context


# ── System prompt ─────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """You are an expert aviation information assistant with access to:
# - Boeing 737 Operations Manual (performance data, procedures, systems)
# - Airbus A320/A321 Flight Crew Training Manual (normal operations)
# - In-flight menus for routes BCN-FCO and VIE-LHR (food options, allergens)

# STRICT RULES:
# 1. Answer ONLY using the provided context. If not found, say:
#    "I could not find this information in the available documents."
# 2. Never fill gaps with training knowledge. Reproduce exact numbers — never approximate.
# 3. For allergen/menu data, be precise — passengers may have serious dietary requirements.
# 4. Keep answers concise and structured.

# CITATION FORMAT — this is mandatory:
# After your answer, on a new line write exactly:
# PAGES_USED: <document_title_1>:<page1>,<page2>|<document_title_2>:<page3>

# Rules for PAGES_USED:
# - Only include pages where the specific evidence for your answer appears.
# - Do NOT include pages you read for context but did not use.
# - If a number, procedure, or fact came from a page, cite it. If not, omit it.
# - Use the exact document titles as they appear in [Source: ...] tags.
# - If answer spans multiple documents, separate with |

# Example:
# PAGES_USED: Boeing B737 Manual.pdf:15,16|BCN-FCO Menus.pdf:3
# """


SYSTEM_PROMPT = """You are an expert aviation information assistant with access to:
- Boeing 737 Operations Manual (performance data, procedures, systems)
- Airbus A320/A321 Flight Crew Training Manual (normal operations)
- In-flight menus for routes BCN-FCO and VIE-LHR (food options, allergens)

STRICT RULES:
1. Answer ONLY using the provided context. If not found, say:
   "I could not find this information in the available documents."
2. Never fill gaps with training knowledge. Reproduce exact numbers — never approximate.
3. For allergen/menu data, be precise — passengers may have serious dietary requirements.
4. Use a strict answer format depending on the question type:

- If the answer is a list of options, steps, or procedures:
  Return a bullet list with one item per line.

- If the answer is a single fact (value, name, option, role):
  Return a single sentence with only that fact.

- If the answer compares items:
  Return one line per item, clearly labeled.

Do not mix formats. Do not add explanations unless strictly necessary.
5. Answer ONLY the specific thing asked in the question.
6. Do NOT describe unrelated parts of a diagram, table, procedure, or page.
7. If the question asks about one step, one item, one role, one page element, or one value, return only that specific step, item, role, page element, or value.
8. If multiple roles, areas, or sides appear in the context, select ONLY the one directly associated with the specific item asked about.
9. Prefer direct factual answers over broad summaries.
10. Start your answer immediately with the factual content. \
Never begin with phrases like "Based on the context", "According to the document", \
"The document states", or any preamble. Go straight to the answer.
11. Do NOT include any additional information beyond what is strictly required 
to answer the question. Extra details, even if correct, will be considered incorrect.
12. When multiple items are expected, label them clearly 
(e.g., "Primary Procedure:", "Alternate Procedure:").
13. Never combine paragraph explanations with lists or labeled items.
The output must strictly follow one format only.
14. The PAGES_USED line is mandatory. If missing or malformed, the answer is invalid.

CITATION FORMAT — this is mandatory:
After your answer, on a new line write exactly:
PAGES_USED: <document_title_1>:<page1>,<page2>|<document_title_2>:<page3>

Rules for PAGES_USED:
- Only include pages where the specific evidence for your answer appears.
- Do NOT include pages you read for context but did not use.
- If a number, procedure, or fact came from a page, cite it. If not, omit it.
- Use the exact document titles as they appear in [Source: ...] tags.
- If answer spans multiple documents, separate with |

Example:
PAGES_USED: Boeing B737 Manual.pdf:15,16|BCN-FCO Menus.pdf:3
"""


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_LLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.0,
        max_output_tokens=2048,
    )


# ── Citation parser ───────────────────────────────────────────────────────────

def _parse_pages_used(raw_response: str) -> Tuple[str, List[Dict]]:
    """
    Extract PAGES_USED from the LLM response and build citation objects.

    Returns:
        answer:    The answer text (without the PAGES_USED line)
        citations: List of { document_title, pages } dicts
    """
    # Split answer from pages line
    lines = raw_response.strip().split("\n")
    pages_line = ""
    answer_lines = []

    for line in lines:
        if line.strip().startswith("PAGES_USED:"):
            pages_line = line.strip()
        else:
            answer_lines.append(line)

    answer = "\n".join(answer_lines).strip()

    if not pages_line:
        # LLM didn't follow format — return empty citations
        print("  ⚠ No PAGES_USED line found in response")
        return answer, []

    # Parse: PAGES_USED: Doc1.pdf:1,2|Doc2.pdf:3
    citations = []
    try:
        raw = pages_line.replace("PAGES_USED:", "").strip()
        doc_blocks = raw.split("|")

        for block in doc_blocks:
            block = block.strip()
            if ":" not in block:
                continue
            # Split on last colon to handle filenames with colons
            last_colon = block.rfind(":")
            title = block[:last_colon].strip()
            pages_str = block[last_colon + 1:].strip()

            pages = []
            for p in pages_str.split(","):
                p = p.strip()
                if p.isdigit():
                    pages.append(int(p))

            if title and pages:
                citations.append({
                    "document_title": title,
                    "pages": sorted(pages),
                })

    except Exception as e:
        print(f"  ⚠ Error parsing PAGES_USED: {e}")
        return answer, []

    print(f"  Evidence pages cited by LLM: {citations}")
    return answer, citations


# ── Main generation function ───────────────────────────────────────────────────

def generate_answer(
    question: str,
    context_docs: List[Document],
) -> Tuple[str, List[Dict]]:
    """
    Generate a grounded answer from the expanded context docs.

    The LLM receives all expanded pages (reranked + neighbours) for
    full comprehension, but is instructed to cite only the pages where
    specific evidence appears.

    Args:
        question:     The user's question.
        context_docs: Expanded page-level Documents (reranked + neighbours).

    Returns:
        answer:     Generated answer text.
        citations:  Evidence-based citations from LLM self-reporting.
    """
    if not context_docs:
        return (
            "I could not find any relevant information in the available documents.",
            [],
        )

    context = build_context(context_docs)

#     user_message = f"""Use the following document excerpts to answer the question.

# --- CONTEXT START ---
# {context}
# --- CONTEXT END ---

# Question: {question}

# Remember: after your answer, include the PAGES_USED line with only the pages \
# that contain evidence for your specific answer."""

    user_message = f"""

The context may contain more information than needed.
Use ONLY the minimal subset required to answer the question.
Ignore all unrelated content.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Instructions for this answer:
- Answer only the specific question asked.
- Do not summarize the whole page or diagram.
- If the question refers to one step or one item, provide only the area/value/role directly associated with that step or item.
- Exclude unrelated roles, sides, or surrounding diagram details unless they are necessary to answer the question.

- Start directly with the answer — no preamble, no restatement of the question.
"""
# Remember: after your answer, include the PAGES_USED line with only the pages \
# that contain evidence for your specific answer."""

    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    raw = (response.content or "").strip()

    answer, citations = _parse_pages_used(raw)
    return answer, citations