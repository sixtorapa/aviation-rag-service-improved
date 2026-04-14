"""
improvements/main_async.py
==========================
Improvement over main.py: async endpoint with run_in_threadpool.

PROBLEM WITH CURRENT main.py
------------------------------
The POST /query endpoint is a synchronous function. FastAPI runs sync endpoints
in a thread pool (default size: number of CPU cores). Under concurrent load,
the thread pool saturates — threads sit idle waiting for Gemini API responses,
while new requests queue up behind them.

For a single-user evaluation this is invisible. For a production deployment
serving multiple airline operators simultaneously, p99 latency degrades sharply
once the thread pool is full.

SOLUTION
--------
Wrap the pipeline in a synchronous helper (_run_query_pipeline) and call it
via FastAPI's run_in_threadpool. This keeps the endpoint async, so the event
loop is free to handle other requests while waiting for Gemini API calls.

The pipeline itself (retrieval, reranking, generation) remains synchronous
because the underlying libraries (LangChain, sentence-transformers, pdfplumber)
are not async-native. run_in_threadpool moves that work to a thread without
blocking the event loop.

WHAT CHANGES
------------
- `def query(...)` → `async def query(...)`
- Pipeline logic extracted to `_run_query_pipeline(question)`
- `await run_in_threadpool(...)` wraps the pipeline call

WHAT STAYS THE SAME
-------------------
- All five pipeline stages
- Request/response schemas
- Error handling
- /health endpoint
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from services.retrieval import get_retriever, rerank, expand_context
from services.vision_fallback import enrich_visual_pages
from services.generation import generate_answer

app = FastAPI(
    title="Aviation RAG Service",
    description="Retrieval-Augmented Generation over aviation manuals and in-flight menus.",
    version="2.0.0",
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


# ── Pipeline helper (sync — libraries are not async-native) ───────────────────

def _run_query_pipeline(question: str) -> tuple[str, list[dict]]:
    """
    Encapsulates the five-stage RAG pipeline as a synchronous function.

    This is called via run_in_threadpool so it runs in a thread without
    blocking the async event loop. All stages remain synchronous because
    LangChain, sentence-transformers, and pdfplumber are not async-native.
    """
    retriever = get_retriever()

    # Stage 1: hybrid BM25 + vector retrieval
    all_docs = retriever.invoke(question)

    # Stage 2: cross-encoder reranking
    reranked_docs = rerank(question, all_docs)

    # Stage 3: context window expansion (±1 neighbour pages)
    context_docs = expand_context(reranked_docs, all_docs)

    # Stage 4: selective vision enrichment for diagram-heavy pages
    context_docs = enrich_visual_pages(context_docs)

    # Stage 5: generate answer + LLM evidence citations
    return generate_answer(question, context_docs)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick liveness check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main RAG endpoint — async version.

    The pipeline runs in a thread pool via run_in_threadpool, freeing the
    event loop to handle other incoming requests while waiting for Gemini
    API responses. This makes the server non-blocking under concurrent load.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    answer, references = await run_in_threadpool(
        _run_query_pipeline,
        request.question,
    )

    return QueryResponse(
        answer=answer,
        references=[Reference(**r) for r in references],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)