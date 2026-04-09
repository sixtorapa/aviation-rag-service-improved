# Aviation RAG Service

A production-ready Retrieval-Augmented Generation API that answers questions about airline documentation — Boeing 737 and Airbus A320/A321 operations manuals, and in-flight menus — with **page-level source citations**.

Built for Overwatch AI's technical evaluation. Primary metric: retrieval score (page-level accuracy).

**Evaluation results on 10-question calibration set:**

| Metric | Score |
|---|---|
| Retrieval Precision (page-level) | **1.000** |
| Retrieval Recall (page-level) | **1.000** |
| Retrieval F1 (page-level) | **1.000** |
| Faithfulness (RAGAS) | **0.950** |
| Answer Relevancy (RAGAS) | **0.746** |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY

# 3. Place PDF documents in the data/ folder
mkdir data
# Add: Boeing_B737_Manual.pdf, A320_321_Flight_Crew_Training_Manual.pdf,
#      BCN-FCO_Menus.pdf, VIE-LHR_Menus.pdf

# 4. Run ingestion (once, or whenever documents change)
python ingest.py

# 5. Start the server
python main.py
```

Server available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## API Reference

### `POST /query`

**Request**
```json
{
  "question": "What are the allergens in the BCN to FCO business class menu?"
}
```

**Response**
```json
{
  "answer": "The Barcelona to Rome business class menu offers two options. Option A (Mediterranean Harvest) has no listed allergens. Option B (Tiber Seabass) contains Fish.",
  "references": [
    {
      "document_title": "BCN-FCO_Menus.pdf",
      "pages": [3]
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `answer` | string | Generated response grounded in retrieved documents |
| `references` | array | Source pages used to generate the answer |
| `references[].document_title` | string | Filename of the source document |
| `references[].pages` | array of integers | **1-based absolute PDF page indices** (not printed page numbers) |

### `GET /health`
```json
{ "status": "ok" }
```

---

## Architecture

The pipeline has five sequential stages:

```
POST /query
     │
     ▼  Stage 1: Hybrid Retrieval
     │  EnsembleRetriever (BM25 40% + ChromaDB vector 60%)
     │  → RETRIEVAL_K candidate pages
     │
     ▼  Stage 2: Cross-Encoder Reranking
     │  cross-encoder/ms-marco-MiniLM-L-12-v2
     │  Scores each (question, page) pair independently
     │  → Top RERANK_TOP_K pages
     │
     ▼  Stage 3: Context Window Expansion
     │  For each reranked page, add ±1 neighbour pages
     │  from the same document (if in candidate pool)
     │  → Expanded context set
     │
     ▼  Stage 4: Selective Vision Enrichment
     │  Detects diagram-heavy pages via text density +
     │  PDF structural signals (images, rects, curves)
     │  Sends candidates to Gemini Vision for interpretation
     │  → Enriched context set
     │
     ▼  Stage 5: Generation + Evidence Citations
        Gemini generates answer from enriched context
        LLM self-reports which pages it used (PAGES_USED)
        Citations = only pages with direct evidence
     │
     ▼
{ answer, references }
```

---

## Design Decisions

### 1. Page-level chunking (1 chunk = 1 PDF page)

The evaluation metric scores retrieval at page level. Using one chunk per page means the retrieval unit and the citation unit are identical — retrieving the right chunk automatically yields the correct page number with zero post-processing ambiguity.

Sub-page chunking would require mapping chunks back to pages, introducing off-by-one errors when content spans a page boundary. This is especially problematic in the Boeing manual where performance tables span entire pages and are self-contained units.

### 2. Hybrid BM25 + vector retrieval (40/60)

The document set contains two very different content types:

- **Boeing/A320 performance tables**: dense numeric data with low semantic variation. BM25 excels here — it matches exact tokens like `"737-600"`, `"CFM56-7B22"`, `"Flaps 5"`, `"MSN 1320-1637"` that dense vectors miss due to low semantic signal.
- **Menu and allergen pages**: short descriptive prose where semantic search handles paraphrased queries better (`"what can I eat if I have a fish allergy?"` → finds allergen table).

Combining both via `EnsembleRetriever` with Reciprocal Rank Fusion captures the strengths of each. The 40/60 split was tuned on the calibration set.

### 3. Cross-encoder reranking

The ensemble retriever ranks by similarity scores that are not directly comparable across BM25 and vector space. The cross-encoder evaluates the specific relevance of each (question, page) pair, producing a more accurate ranking that removes adjacent pages which share keywords with the query but do not actually answer it.

Model: `cross-encoder/ms-marco-MiniLM-L-12-v2` — free, local (no API calls), runs on CPU in ~100ms for 10 documents.

### 4. Context window expansion (±1 pages)

**Problem:** Aviation manuals often split related information across consecutive pages. For example, in the A320 manual, the 180° turn procedure is described on page 40 but the specific MSN variant figures are on page 41. The reranker scores page 40 high (it contains the keywords "180° turn", "Asymmetric Thrust") but page 41 is a pure table with low keyword overlap — so the reranker underscores it.

**Solution:** After reranking, for each selected page, add its immediate neighbours (±1) from the same document if they exist in the candidate pool. These expanded pages are passed to the LLM for comprehension but are not automatically cited — only the LLM-reported evidence pages appear in the response.

This solved the MSN comparison query (pages 40 + 41) without requiring any model changes.

### 5. LLM evidence citation (not retriever citation)

**Key insight:** separating "pages for context" from "pages for citing".

Standard RAG cites all retrieved pages in the response. This inflates the references array with pages that provided context but did not directly support the answer — penalising the retrieval score.

Instead, the generation prompt instructs Gemini to append a `PAGES_USED:` line reporting only the pages where specific evidence for the answer appears. The `_parse_pages_used()` function extracts this structured output and builds the citations array.

Result: for a query about fuel requirements at 1400 NM, the LLM receives pages 14, 15, 16 (context expansion) but cites only page 15 (where the exact values are).

### 6. Selective vision enrichment

**Problem:** Some pages convey critical information through visual elements — annotated cockpit photographs, flow diagrams with colour-coded arrows, numbered step diagrams — that pdfplumber cannot capture. For example, the A320 cockpit preparation flow pattern (page 26) shows PF responsibility via green arrows pointing to numbered steps; the extracted text only lists the step numbers and descriptions without the visual assignments.

**Solution:** A detection layer identifies diagram-heavy pages using two signals:
1. **Semantic**: extracted text contains visual-indicator keywords (`"flow pattern"`, `"responsibility"`, `"cockpit preparation"`)
2. **Structural**: PDF page contains embedded images, dense rectangle patterns, or vector curves (queried via pdfplumber's page object)

Pages meeting the threshold are rendered as PNG images (110 DPI) and sent to Gemini Vision for interpretation. The visual description is prepended to the extracted text before generation. Up to `MAX_VISION_PAGES=2` pages per query are enriched, and results are cached in-memory to avoid redundant API calls.

### 7. pdfplumber over PyPDF

The Boeing manual contains complex multi-column performance tables. pdfplumber's `extract_tables()` preserves row/column structure as tab-separated text, making numeric values queryable. PyPDF flattens these tables into unstructured strings that BM25 cannot match reliably.

### 8. Gemini for generation, Gemini embeddings for vectors

Single provider — no cross-provider latency or credential complexity. `gemini-2.5-flash` for generation (fast, multimodal, capable of reading rendered page images), `gemini-embedding-001` for embeddings (stable GA model, 3072 dimensions).

---

## Challenges and Solutions

### Tables in PDF pages

The Boeing 737 manual contains performance dispatch tables spanning entire pages — takeoff field corrections, climb limit weights, fuel reserves — with 10+ columns and 20+ rows. Standard text extraction flattens these into unreadable strings.

**Solution:** pdfplumber's `extract_tables()` is called on every page alongside `extract_text()`. Table rows are joined as tab-separated lines and appended to the page content, preserving the structure BM25 needs to match exact numeric values.

### Split information across pages

Technical tables that reference one another (header on page N, data on page N+1) caused the reranker to miss the data page due to low keyword overlap.

**Solution:** Context window expansion (±1 neighbours). The LLM receives the full context but only cites pages with direct evidence, keeping the references array clean.

### Diagram-only pages

Pages where critical information is encoded in visual relationships (arrows, colour coding, spatial position) rather than text cannot be answered by text-only RAG.

**Solution:** Selective Gemini Vision fallback for detected diagram-heavy pages. Applied only to pages that fail both a text-density threshold and contain PDF visual structure signals — not to the entire corpus.

### Model deprecations

Gemini model names changed during development (`gemini-2.0-flash` retired March 2026, `text-embedding-004` retired January 2026). All model names are configurable via `.env` with no code changes required.

---

## Evaluation

```bash
# Server must be running first
python main.py

# In a separate terminal
pip install "ragas>=0.2.0,<0.3.0" datasets
python evaluate.py
```

The evaluation script runs the 10-question calibration set against the live API and computes:
- **Page-level retrieval precision/recall/F1** — custom metric matching the evaluator's scoring approach
- **Faithfulness** — RAGAS metric: is the answer grounded in the context?
- **Answer Relevancy** — RAGAS metric: does the answer address the question?

---

## Project Structure

```
rag_service/
├── main.py                      # FastAPI app — POST /query endpoint
├── ingest.py                    # Standalone ingestion script
├── evaluate.py                  # RAGAS evaluation script
├── config.py                    # All configuration via environment variables
├── requirements.txt
├── .env.example
└── services/
    ├── ingestion.py             # PDF loading (page-level), ChromaDB, BM25 index
    ├── retrieval.py             # Hybrid retriever, cross-encoder reranker,
    │                            # context expansion, context builder
    ├── generation.py            # Gemini prompt, answer generation,
    │                            # PAGES_USED parser, evidence citations
    └── vision_fallback.py       # Diagram detection, PDF rendering,
                                 # Gemini Vision enrichment
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Google Gemini API key |
| `GEMINI_LLM_MODEL` | `gemini-2.5-flash` | Generation + Vision model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `DATA_DIR` | `./data` | PDF documents folder |
| `VECTOR_STORE_PATH` | `./vector_store` | ChromaDB persistence directory |
| `RETRIEVAL_K` | `4` | Candidate pages retrieved per query |
| `BM25_WEIGHT` | `0.4` | BM25 weight in hybrid search |
| `VECTOR_WEIGHT` | `0.6` | Vector weight in hybrid search |
| `MAX_REFERENCES` | `1` | Max documents in references array |

---

## Known Limitations

- **Diagram-only pages without text anchors**: the vision fallback is triggered by heuristics. Pages where visual content is embedded as vector graphics with no surrounding text may be missed by the detector.
- **Cross-page tables beyond ±1**: if a table spans 3+ pages and only the middle page is retrieved, the ±1 expansion may not capture all relevant data.
- **20-question evaluation set**: the calibration set has 10 questions covering all content types. The remaining 10 evaluator questions are unknown — performance on those depends on the generalisability of the pipeline, not tuning to specific questions.
