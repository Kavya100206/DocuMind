"""
Reranker Service — BAAI/bge-reranker-base

WHAT IS A RERANKER?
-------------------
FAISS + BM25 are *bi-encoders*:
  - Embed query separately → vector A
  - Embed chunk separately → vector B
  - Compare A and B → similarity score

A cross-encoder reranker works differently:
  - Feed BOTH query AND chunk together into one model
  - Full attention across both texts → cross-attention score
  - Much more accurate, because the model sees relationships between
    specific words in the query and specific words in the chunk

WHY NOT USE IT FOR ALL CHUNKS?
-------------------------------
Cross-encoders are slow: ~10–50ms per (query, chunk) pair.
On 10,000 chunks that's 100–500 seconds — unusable.

Solution = two-stage retrieval:
  Stage 1: FAISS + BM25 → top 20 candidates (fast, milliseconds)
  Stage 2: Reranker → top 5 of those 20 (slow but only 20 pairs)

Result: ~200ms extra latency, 20–40% better ranking accuracy.

MODEL: BAAI/bge-reranker-base
  - Size: ~300MB (downloaded once from HuggingFace, cached)
  - Type: CrossEncoder
  - Output: scalar relevance score per (query, chunk) pair
  - No prefix needed (unlike BGE bi-encoder)
"""

from typing import List, Dict, Any
import math
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Lazy model cache ──────────────────────────────────────────────────────────
# We do NOT load the reranker at module import time because:
#   1. It's ~300MB — adds ~2s to every uvicorn cold start
#   2. It's only used during search, not during upload
# Instead: load on first rerank() call, then cache in _model.
_model = None
# Lightweight cross-encoder (~67MB, ~4x faster than bge-reranker-base ~300MB)
# Nearly equivalent accuracy for short document chunks
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"


def _get_model():
    """Load the CrossEncoder model lazily and cache it."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker model: {_RERANKER_MODEL} (first call only)...")
        _model = CrossEncoder(_RERANKER_MODEL)
        logger.info("Reranker model loaded and cached.")
    return _model


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank chunks using cross-encoder attention scores.

    Called from retrieval_service.py AFTER FAISS + BM25 hybrid scoring.
    Takes up to 20 candidates and returns only the top N, sorted by
    reranker score (most relevant first).

    Each chunk gets a new key: "reranker_score" (float, higher = better).

    Args:
        query:  User's original question.
        chunks: List of chunk dicts from FAISS+BM25 (with hybrid_score).
        top_n:  How many to keep (default: settings.RERANKER_TOP_N).

    Returns:
        Top N chunks sorted by reranker_score descending.
    """
    if not chunks:
        return chunks

    model = _get_model()

    # Pre-sort by best available score and cap at 10 candidates
    # so the cross-encoder never evaluates more than 10 pairs per request.
    # This keeps inference fast regardless of how large the merged pool is.
    MAX_CANDIDATES = 10
    chunks = sorted(
        chunks,
        key=lambda c: c.get("hybrid_score") or c.get("similarity_score", 0),
        reverse=True
    )[:MAX_CANDIDATES]

    # Build (query, chunk_text) pairs for cross-encoder
    pairs = [(query, ch["text"]) for ch in chunks]

    # predict() returns raw logits (ms-marco outputs -∞ to +∞).
    # Apply sigmoid to normalize into (0, 1) so the confidence guard
    # and downstream scoring work consistently regardless of model.
    scores = model.predict(pairs)

    # Attach sigmoid-normalized reranker score to each chunk
    for chunk, score in zip(chunks, scores):
        chunk["reranker_score"] = round(1.0 / (1.0 + math.exp(-float(score))), 4)

    # Sort by reranker score descending, keep top N
    chunks.sort(key=lambda c: c["reranker_score"], reverse=True)
    top_chunks = chunks[:top_n]

    # Log the reranking results
    logger.info(f"Reranker: {len(chunks)} → {len(top_chunks)} chunks")
    for i, ch in enumerate(top_chunks, 1):
        hybrid = ch.get("hybrid_score", ch.get("similarity_score", 0))
        logger.info(
            f"  [{i}] reranker={ch['reranker_score']:.4f} "
            f"hybrid={hybrid:.4f} | "
            f"{ch.get('document_name','?')[:25]} | p{ch.get('page_number')} | "
            f"{ch['text'][:60].replace(chr(10), ' ')}..."
        )

    return top_chunks
