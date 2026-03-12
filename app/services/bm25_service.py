"""
BM25 Keyword Search Service

WHAT IS BM25?
-------------
BM25 (Best Match 25) is the gold-standard keyword retrieval algorithm
used by Elasticsearch, Solr, and classic search engines.

Unlike FAISS which works on MEANING (semantic vectors), BM25 works on
EXACT WORDS — it finds documents that literally contain the query terms,
weighted by how rare those terms are across the corpus.

WHY COMBINE WITH FAISS?
-----------------------
  FAISS alone misses exact-phrase matches when two concepts are
  semantically similar but lexically different ("team members" vs "team size").

  BM25 alone misses paraphrases ("neural network" vs "deep learning model").

  Hybrid = best of both worlds.

SCORING FORMULA:
  final_score = FAISS_WEIGHT * faiss_score + BM25_WEIGHT * bm25_score_normalised

  FAISS_WEIGHT = 0.65  (semantic — primary signal)
  BM25_WEIGHT  = 0.35  (keyword  — secondary signal)

HOW IT WORKS AT RUNTIME:
  1. The FAISS metadata list already holds all chunk texts in memory.
  2. We tokenise those texts → build a BM25 index in-place (~milliseconds).
  3. Score the query against every chunk via BM25.
  4. Normalise BM25 scores to [0, 1] so they're on the same scale as FAISS.
  5. Caller (retrieval_service) combines both scores.
"""

import re
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

# Hybrid weighting — these should sum to 1.0
FAISS_WEIGHT: float = 0.65
BM25_WEIGHT: float  = 0.35

# Stop-words to skip during tokenisation (same set used in lexical boost)
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "up", "about", "into", "through",
    "and", "or", "but", "not", "so", "yet", "both", "either", "neither",
    "that", "this", "these", "those", "it", "its",
}


def _tokenise(text: str) -> List[str]:
    """
    Split text into lower-case tokens, dropping stop-words and short tokens.
    Numbers and acronyms (e.g. "BM25", "LSTM") are kept as-is.
    """
    raw = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return [t for t in raw if t and t not in _STOPWORDS and len(t) > 1]


def rerank_with_bm25(
    query: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Re-rank a list of retrieved chunks by combining FAISS score with BM25.

    Called from retrieval_service.py AFTER the FAISS search + filtering step.
    Each chunk dict must already have a "similarity_score" key (from FAISS).

    Args:
        query:  The user's original question.
        chunks: List of chunk dicts from FAISS retrieval (with similarity_score).

    Returns:
        Same list, re-sorted by combined score (highest first).
        Each chunk gets two extra keys:
            "bm25_score"    — raw BM25 score for this chunk
            "hybrid_score"  — the final combined score used for ranking
    """
    if not chunks:
        return chunks

    # ── Build BM25 index from chunk texts ───────────────────────────────────
    corpus_tokens = [_tokenise(ch["text"]) for ch in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    # ── Score query against all chunks ──────────────────────────────────────
    query_tokens = _tokenise(query)
    raw_bm25_scores = bm25.get_scores(query_tokens)   # numpy array

    # ── Normalise BM25 scores to [0, 1] ─────────────────────────────────────
    max_bm25 = max(raw_bm25_scores) if max(raw_bm25_scores) > 0 else 1.0
    norm_bm25 = [float(s) / max_bm25 for s in raw_bm25_scores]

    # ── Combine scores ───────────────────────────────────────────────────────
    for i, chunk in enumerate(chunks):
        faiss_score = chunk.get("similarity_score", 0.0)
        bm25_score  = norm_bm25[i]
        hybrid      = FAISS_WEIGHT * faiss_score + BM25_WEIGHT * bm25_score
        chunk["bm25_score"]   = round(bm25_score, 4)
        chunk["hybrid_score"] = round(hybrid, 4)

    # ── Sort by hybrid score descending ─────────────────────────────────────
    chunks.sort(key=lambda c: c["hybrid_score"], reverse=True)
    return chunks
