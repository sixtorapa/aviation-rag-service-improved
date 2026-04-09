"""
config.py
Central configuration — reads from environment variables.
All tuneable parameters live here so they can be adjusted without touching logic.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM & Embeddings ────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

# Generation model — gemini-2.0-flash balances speed and quality for Q&A
GEMINI_LLM_MODEL: str = os.environ.get("GEMINI_LLM_MODEL", "gemini-2.0-flash")


# Embedding model
# Current recommended Gemini embedding models:
# - gemini-embedding-001 (stable)
# - gemini-embedding-2-preview (preview)
GEMINI_EMBEDDING_MODEL: str = os.environ.get(
    "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
)

# ── Data paths ───────────────────────────────────────────────────────────────
# Directory where the 4 PDF documents are stored
DATA_DIR: str = os.environ.get(
    "DATA_DIR", os.path.join(os.path.dirname(__file__), "data")
)

# Directory where ChromaDB will persist the vector store
VECTOR_STORE_PATH: str = os.environ.get(
    "VECTOR_STORE_PATH",
    os.path.join(os.path.dirname(__file__), "vector_store"),
)

# ChromaDB collection name
COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "aviation_docs")

# ── Retrieval parameters ─────────────────────────────────────────────────────
# Total number of chunks (pages) to retrieve before building the answer.
# Higher k → better recall, slower generation.
RETRIEVAL_K: int = int(os.environ.get("RETRIEVAL_K", "10"))

# BM25 vs vector weight in the hybrid ensemble (must sum to 1.0).
# BM25 helps with exact keyword matches (e.g. "Flaps 5", "737-600", allergen names).
# Vector search handles semantic / paraphrased queries.
BM25_WEIGHT: float = float(os.environ.get("BM25_WEIGHT", "0.4"))
VECTOR_WEIGHT: float = float(os.environ.get("VECTOR_WEIGHT", "0.6"))

# Max distinct page references to include in the response.
# Fewer non-relevant pages = better eval score.
MAX_REFERENCES: int = int(os.environ.get("MAX_REFERENCES", "5"))