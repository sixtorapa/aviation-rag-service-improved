"""
main.py
FastAPI entry point. Run with: python main.py

Single endpoint:
  POST /query
    Request:  { "question": "..." }
    Response: { "answer": "...", "references": [...] }
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.retrieval import get_retriever, rerank, expand_context
from services.vision_fallback import enrich_visual_pages
from services.generation import generate_answer

app = FastAPI(
    title="Aviation RAG Service",
    description="Retrieval-Augmented Generation over aviation manuals and in-flight menus.",
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


class Reference(BaseModel):
    document_title: str
    pages: list[int]


class QueryResponse(BaseModel):
    answer: str
    references: list[Reference]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick liveness check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main RAG endpoint.

    Pipeline:
      1. Hybrid BM25 + vector retrieval (candidate pages)
      2. Cross-encoder reranking (top-K most relevant pages)
      3. Context window expansion (add ±1 neighbour pages)
      4. Selective vision enrichment (Gemini Vision for diagram-heavy pages)
      5. Gemini generation with evidence-based citation reporting
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    retriever = get_retriever()

    # Step 1: retrieve candidates
    all_docs = retriever.invoke(request.question)

    # Step 2: rerank to top-K
    reranked_docs = rerank(request.question, all_docs)

    # Step 3: expand context with ±1 neighbours
    context_docs = expand_context(reranked_docs, all_docs)

    # Step 4: enrich diagram-heavy pages with Gemini Vision
    context_docs = enrich_visual_pages(context_docs)

    # Step 5: generate answer + evidence-based citations
    answer, references = generate_answer(request.question, context_docs)

    return QueryResponse(
        answer=answer,
        references=[Reference(**r) for r in references],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)