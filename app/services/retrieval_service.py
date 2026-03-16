"""
Retrieval Service

WHAT DOES THIS FILE DO?
------------------------
This is the search brain of DocuMind.

When a user asks a question, this service:
  1. Converts the question into a vector (using embedding_service)
  2. Searches the FAISS index for the closest chunk vectors (using faiss_service)
  3. Applies lexical boost (token + bigram exact-match re-ranking)
  4. Applies BM25 hybrid scoring (65% FAISS + 35% BM25 keyword)
  5. Enriches results with document filenames from Postgres
  6. Returns a clean list of results with text, page number, and document name
"""

from typing import List, Dict, Any, Optional
import re
from sqlalchemy.orm import Session
from app.services import faiss_service
from app.services.embedding_service import get_embedding
from app.models.document import Document
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)



# Use the threshold from settings.py (single source of truth)
# Can be changed in settings.py without touching this file
SIMILARITY_THRESHOLD = settings.SIMILARITY_THRESHOLD

# Common English words that carry no useful search signal.
# We strip these when building query tokens so they don't generate
# false boosts (e.g. the word "is" appearing in every chunk).
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "what", "which", "who", "how", "many",
    "much", "this", "that", "these", "those", "and", "or", "but", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "each", "few",
    "list", "show", "tell", "give", "display", "all", "any", "find", "get", "inform", "information"
}


def _apply_lexical_boost(
    query: str,
    chunks: List[Dict[str, Any]],
    boost_per_token: float = 0.06,  # Increased from 0.04 to give resumes higher lexical weight
    max_boost: float = 0.15,        # Increased from 0.15
) -> List[Dict[str, Any]]:
    """
    Re-rank retrieved chunks by adding an exact-match score boost.

    WHY THIS IS NEEDED:
    -------------------
    Semantic embeddings encode *meaning*, not exact text. They treat
    "DSA I", "DSA II", "DSA III", "DSA IV" as almost identical vectors
    because they all mean the same kind of thing (Data Structures courses).

    Result: a question about "DSA III" retrieves the highest-semantic-score
    DSA chunk — which may be DSA IV instead of DSA III. The LLM then can't
    find the answer because the right chunk was never retrieved.

    The lexical boost fixes this by rewarding chunks that contain the exact
    words from the question, regardless of semantic similarity.

    HOW IT WORKS:
    -------------
    1. Tokenise the query into lowercase words (split on non-alphanumeric)
    2. Remove stopwords (common words with no search signal)
    3. For every chunk, count how many query tokens appear LITERALLY in text
    4. Add (count × boost_per_token) to the chunk's similarity_score, capped
    5. Re-sort all chunks by the new boosted score

    EXAMPLE:
    --------
    Query: "How many credits is DSA III?"
    Tokens after stopword removal: ["credits", "dsa", "iii"]

    Chunk A text contains "DSA III" and "3 credits":
      → matches: {dsa, iii, credits} → boost = 3 × 0.05 = +0.15

    Chunk B text contains "DSA IV" and "3 credits":
      → matches: {dsa, credits} → boost = 2 × 0.05 = +0.10

    If both had semantic score 0.20:
      Chunk A final score: 0.35  ← wins ✅
      Chunk B final score: 0.30

    GENERIC DESIGN:
    ---------------
    No domain-specific logic. No word lists. Works for any PDF.
    Numbers ("168", "3") and Roman numerals ("III", "IV") are kept
    as tokens — they are distinct and highly informative for structured docs.

    Args:
        query:           The user's original question
        chunks:          List of chunk dicts from FAISS (each has similarity_score)
        boost_per_token: Score added per matched unique query token (default 0.05)
        max_boost:       Maximum total boost regardless of matches (default 0.20)

    Returns:
        Same list re-sorted by (similarity_score + lexical_boost), descending
    """
    if not chunks or not query.strip():
        return chunks

    # Tokenise: lowercase, split on anything that isn't a letter or digit
    raw_tokens = re.split(r'[^a-zA-Z0-9]+', query.lower())

    # Keep tokens that are meaningful (not stopwords, not empty, len > 1)
    # We allow single-char tokens ONLY if they look like Roman numeral (I, V, X)
    # or a digit — those are highly significant in structured documents
    query_tokens = set()
    roman_single = {"i", "v", "x"}
    for tok in raw_tokens:
        if not tok:
            continue
        if tok in _STOPWORDS:
            continue
        if len(tok) == 1 and tok not in roman_single and not tok.isdigit():
            continue  # skip meaningless single letters like "s", "a"
        query_tokens.add(tok)

    if not query_tokens:
        return chunks  # no meaningful tokens → no boost, return as-is

    # Build ordered bigrams from the original query (before stopword removal)
    # so that "team members" stays as a phrase even if "the" sits between words.
    # We keep bigrams from the *filtered* token list so stopwords don't create
    # false phrases like "is members".
    token_list = [t for t in re.split(r'[^a-zA-Z0-9]+', query.lower()) if t and t not in _STOPWORDS]
    query_bigrams = {
        f"{token_list[i]} {token_list[i + 1]}"
        for i in range(len(token_list) - 1)
    } if len(token_list) >= 2 else set()

    # Bigram boost is stronger than per-token boost because an exact 2-word
    # phrase match is much more precise (e.g. "team members" vs "team size").
    BIGRAM_BOOST = 0.10

    print(f"  🔤 Lexical boost tokens: {sorted(query_tokens)} | bigrams: {sorted(query_bigrams)}")

    boosted = []
    for chunk in chunks:
        chunk_text_lower = chunk.get("text", "").lower()
        section_name_lower = str(chunk.get("section_name", "")).lower()

        # Single-token matches in body text
        matched_tokens = sum(1 for tok in query_tokens if tok in chunk_text_lower)

        # METADATA BOOST: Does the query mention the section name?
        # e.g. Query "list projects" matches section_name="PROJECTS"
        # This is a strong signal for all chunks in that section.
        section_match = any(tok in section_name_lower for tok in query_tokens)
        section_boost = 0.15 if section_match else 0.0

        # HEADER BOOST/PENALTY
        # is_pure_header: The chunk IS exactly the section name (e.g. "PROJECTS")
        is_pure_header = chunk_text_lower.strip() == section_name_lower.strip()
        
        # Detect if query has "list" or "show" intent
        query_lower = query.lower()
        has_list_intent = any(word in query_lower for word in ["list", "show", "tell", "give", "display", "all", "projects", "enumerate"])

        header_boost = 0.0
        if is_pure_header:
            # Deprioritize the header itself for "list" queries so we see items instead
            if has_list_intent:
                header_boost = -0.15 
        elif len(chunk_text_lower) < 50:
            # It's a short title but NOT the main section header (e.g. "AI Bot Project")
            if any(tok == chunk_text_lower.strip() for tok in query_tokens):
                header_boost = 0.12

        # Bigram phrase matches
        matched_bigrams = sum(1 for bg in query_bigrams if bg in chunk_text_lower)

        # ROMAN/NUMERIC BOOST: bonus for matching identifiers like "I", "II", "v1"
        roman_matches = sum(1 for tok in query_tokens if tok in roman_single and tok in chunk_text_lower)
        roman_boost = 0.05 if roman_matches > 0 else 0.0

        total_boost = min(
            matched_tokens * boost_per_token + matched_bigrams * BIGRAM_BOOST + section_boost + header_boost + roman_boost,
            max_boost + 0.25 
        )
        boosted_score = round(chunk["similarity_score"] + total_boost, 4)

        updated = dict(chunk)
        updated["similarity_score"] = boosted_score
        if total_boost != 0:
            msg = f"    ↑ Boost {total_boost:+.2f} (item:{'YES' if not is_pure_header else 'NO'}, tokens:{matched_tokens}, bg:{matched_bigrams}"
            if section_match: msg += ", section:YES"
            if header_boost != 0: msg += f", header_mod:{header_boost:+.2f}"
            msg += f"): {chunk.get('text','')[:50]}..."
            print(msg)
        boosted.append(updated)

    # Re-sort by boosted score — highest first
    boosted.sort(key=lambda c: c["similarity_score"], reverse=True)
    return boosted




def search_chunks(
    query: str,
    db: Session,
    k: int = 10,
    document_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for the most relevant chunks for a user's question.

    This is the main function used by the search controller in Phase 4.

    HOW IT WORKS:
    -------------
    1. Load the FAISS index from disk
    2. Embed the query → 384-dim vector
    3. Ask FAISS: "give me the k closest vectors"
    4. Filter out results below SIMILARITY_THRESHOLD
    5. Enrich each result with the document filename from Postgres
    6. Return the final list

    Args:
        query:       The user's question (plain English)
        db:          Database session (to look up document filenames)
        k:           Max number of results to return (default: 5)
        document_id: Optional — restrict search to one specific document.
                     If None, searches across ALL uploaded documents.

    Returns:
        List of result dicts, sorted by relevance (best first):
        [
            {
                "text": "Revenue grew 10% in Q3...",
                "page_number": 4,
                "document_id": "abc-123",
                "document_name": "annual_report.pdf",
                "chunk_index": 2,
                "similarity_score": 0.87
            },
            ...
        ]
        Returns empty list if no index exists or nothing is relevant enough.
    """

    if not query or not query.strip():
        return []

    # ------------------------------------------------------------------
    # STEP 1: Search FAISS
    # ------------------------------------------------------------------
    # faiss_service.search() returns the top-k results already sorted
    # We fetch a wide net of 25 candidates, then let BM25 + Reranker prune it down 
    # to the TOP_K_RESULTS (10) for the LLM.
    raw_results = faiss_service.search(query_text=query, k=40)

    if not raw_results:
        return []

    # ------------------------------------------------------------------
    # STEP 2: Filter by similarity threshold
    # ------------------------------------------------------------------
    # We only keep results where the meaning is close enough to the query
    # Low scores = the chunk is probably not actually relevant
    relevant = [
        r for r in raw_results
        if r["similarity_score"] >= SIMILARITY_THRESHOLD
    ]

    if not relevant:
        print(f"  ⚠️  No results above threshold {SIMILARITY_THRESHOLD} for: '{query}'")
        return []

    # ------------------------------------------------------------------
    # STEP 3: Optionally filter by document
    # ------------------------------------------------------------------
    # If the user wants to search only within a specific document,
    # filter out results from other documents.
    #
    # SINGLE-DOC SEARCH OPTIMISATION:
    # When a document_id is set, we re-run FAISS with a higher k and a
    # lower threshold. Why?
    #  - The pool is already restricted to one document's chunks, so a
    #    score of 0.10 in a focused search is meaningful (it means "this
    #    is the most relevant section in THIS document").
    #  - Using a broader search (k=20) ensures we don't miss sections
    #    whose terminology differs from the query
    #    (e.g. query: "machine learning technique" → chunk: "LSTM model").
    if document_id:
        # Re-fetch with higher k so we see more of the document's chunks
        if k <= 10:
            raw_results = faiss_service.search(query_text=query, k=25)

        single_doc_threshold = SIMILARITY_THRESHOLD

        # Try to filter to just this document's chunks
        filtered_for_doc = [
            r for r in raw_results
            if r.get("document_id") == document_id
            and r["similarity_score"] >= single_doc_threshold
        ]

        if filtered_for_doc:
            # Happy path: FAISS meta IDs match the requested document_id
            relevant = filtered_for_doc
        else:
            print(f"  ⚠️  No relevant chunks found in requested document '{document_id[:8]}...'.")
            return []

    # ------------------------------------------------------------------
    # STEP 3b: Per-document chunk cap (global search only)
    # ------------------------------------------------------------------
    # WHY THIS MATTERS:
    # When multiple documents are uploaded, one doc can dominate all TOP_K
    # slots if its vocabulary overlaps with the query (e.g. a resume with
    # generic tech terms answers a tech-stack question better than the
    # actual assessment doc).
    #
    # Fix: in global search (no doc_id filter), each document contributes
    # at most MAX_CHUNKS_PER_DOC chunks. This ensures fair representation.
    #
    # Example with 2 docs and MAX_CHUNKS_PER_DOC=3:
    #   Before: [resume×8, assessment×2]  ← resume dominates
    #   After:  [resume×3, assessment×3]  ← balanced
    #
    # We skip this when document_id is set — user intentionally wants one doc.
    if not document_id:
        doc_chunk_counts: Dict[str, int] = {}
        capped: list = []
        max_per_doc = settings.MAX_CHUNKS_PER_DOC
        for r in relevant:
            doc_id = r.get("document_id")
            count = doc_chunk_counts.get(doc_id, 0)
            if count < max_per_doc:
                capped.append(r)
                doc_chunk_counts[doc_id] = count + 1
        if len(capped) < len(relevant):
            print(f"  📊 Per-doc cap applied: {len(relevant)} → {len(capped)} chunks (max {max_per_doc}/doc)")
        relevant = capped

    # ------------------------------------------------------------------
    # STEP 3c: Lexical boost (exact-match re-ranking)
    # ------------------------------------------------------------------
    # WHY THIS EXISTS:
    # Semantic embeddings work on *meaning*, not exact text. They treat
    # "DSA I", "DSA II", "DSA III", "DSA IV" as almost identical because
    # they mean the same kind of thing. A question about "DSA III" gets
    # embeddings pointing at every DSA chunk equally.
    #
    # Lexical boost: if the exact words from the question appear literally
    # in a chunk, that chunk gets a small score bonus. This re-ranks precise
    # matches above semantically-similar-but-wrong chunks.
    #
    # EXAMPLE:
    #   Query: "How many credits is DSA III?"
    #   Tokens: ["credits", "dsa", "iii"]
    #
    #   Chunk A (DSA III): semantic=0.20, matches=[dsa, iii, credits] → +0.15 → 0.35
    #   Chunk B (DSA IV):  semantic=0.21, matches=[dsa, credits]      → +0.10 → 0.31
    #   → DSA III chunk now ranks above DSA IV  ✅
    #
    # GENERIC DESIGN:
    #   - Works for any document type (no domain-specific logic)
    #   - Boost is small (+0.05 per matched token, capped at +0.20)
    #   - Can never push an irrelevant chunk above the threshold
    #   - Roman numerals (I, II, III, IV) and numbers are kept as tokens
    relevant = _apply_lexical_boost(query, relevant)

    # ------------------------------------------------------------------
    # STEP 3d: BM25 hybrid re-ranking
    # ------------------------------------------------------------------
    # After lexical boost adjusts similarity_score, we run BM25 over the
    # shortlisted chunks.  BM25 scores exact keyword frequency across the
    # chunk corpus; the hybrid score combines both signals:
    #   hybrid = 0.65 * faiss_score + 0.35 * bm25_score_normalised
    from app.services import bm25_service
    relevant = bm25_service.rerank_with_bm25(query, relevant)

    # ------------------------------------------------------------------
    # STEP 3e: Cross-Encoder Reranking (bge-reranker-base)
    # ------------------------------------------------------------------
    # FAISS and BM25 are bi-encoders and keyword-based. 
    # The cross-encoder sees BOTH query and chunk text together, achieving
    # much higher accuracy but is too slow for all chunks. We run it ONLY
    # on the top candidates returned by the previous steps.
    from app.services import reranker_service
    relevant = reranker_service.rerank(
        query=query, 
        chunks=relevant, 
        top_n=settings.RERANKER_TOP_N
    )

    # ------------------------------------------------------------------
    # STEP 4: Enrich with document filename from Postgres
    # ------------------------------------------------------------------
    # FAISS metadata only has document_id (a UUID).
    # We look up the filename so the result says "annual_report.pdf"
    # instead of "abc-123-def-456".
    #
    # We cache lookups in a dict so we don't hit the DB for every result
    # if multiple chunks come from the same document.
    doc_cache: Dict[str, str] = {}

    enriched = []
    for result in relevant:
        doc_id = result.get("document_id")

        # Look up filename only if we haven't seen this doc_id before
        if doc_id and doc_id not in doc_cache:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            doc_cache[doc_id] = doc.filename if doc else "Unknown"

        enriched_chunk = {
            "text":             result["text"],
            "page_number":      result["page_number"],
            "document_id":      doc_id,
            "document_name":    doc_cache.get(doc_id, "Unknown"),
            "chunk_index":      result.get("chunk_index"),
            "similarity_score": round(result.get("similarity_score", 0), 4),
        }
        if "hybrid_score" in result:
            enriched_chunk["hybrid_score"] = result["hybrid_score"]
        if "reranker_score" in result:
            enriched_chunk["reranker_score"] = result["reranker_score"]
            
        enriched.append(enriched_chunk)

    # Structured retrieval log — visible in uvicorn console for debugging
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  RETRIEVAL [{len(enriched)} chunks] for: '{query[:55]}'")
    print(sep)
    for i, ch in enumerate(enriched, 1):
        preview = ch["text"][:80].replace("\n", " ")
        faiss_s = ch.get("similarity_score", 0)
        bm25_s  = ch.get("bm25_score", "n/a")
        hybr_s  = ch.get("hybrid_score", faiss_s)
        print(
            f"  [{i}] hybrid={hybr_s:.4f} faiss={faiss_s:.4f} bm25={bm25_s} | "
            f"{ch['document_name'][:22]} | p{ch['page_number']} | {preview}..."
        )
    print(f"{sep}\n")

    return enriched


def search_chunks_fast(
    query: str,
    db: Session,
    k: int = 10,
    document_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fast retrieval variant — FAISS + Lexical Boost + BM25, NO cross-encoder.

    Used by dual-query retrieval to gather candidates cheaply from two
    query formulations. The cross-encoder is then applied ONCE on the
    merged pool in qa_controller, halving reranker inference time.

    Returns chunks with hybrid_score but WITHOUT reranker_score.
    """
    if not query or not query.strip():
        return []

    # Default to finding 25 raw candidates if the caller just passes default `k=10`
    search_k = 40 if k <= 10 else k
    raw_results = faiss_service.search(query_text=query, k=search_k)
    
    if not raw_results:
        return []

    relevant = [r for r in raw_results if r["similarity_score"] >= SIMILARITY_THRESHOLD]
    if not relevant:
        return []

    # Document filter
    if document_id:
        if k <= 10:
            raw_results = faiss_service.search(query_text=query, k=20)
        filtered = [
            r for r in raw_results
            if r.get("document_id") == document_id and r["similarity_score"] >= SIMILARITY_THRESHOLD
        ]
        if filtered:
            relevant = filtered
        else:
            return []

    # Per-doc chunk cap (global search only)
    if not document_id:
        doc_chunk_counts: Dict[str, int] = {}
        capped: list = []
        for r in relevant:
            doc_id = r.get("document_id")
            count = doc_chunk_counts.get(doc_id, 0)
            if count < settings.MAX_CHUNKS_PER_DOC:
                capped.append(r)
                doc_chunk_counts[doc_id] = count + 1
        relevant = capped

    # Lexical boost + BM25 (fast steps)
    relevant = _apply_lexical_boost(query, relevant)
    from app.services import bm25_service
    relevant = bm25_service.rerank_with_bm25(query, relevant)

    # Enrich with document name (no reranker)
    doc_cache: Dict[str, str] = {}
    enriched = []
    for result in relevant:
        doc_id = result.get("document_id")
        if doc_id and doc_id not in doc_cache:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            doc_cache[doc_id] = doc.filename if doc else "Unknown"

        enriched_chunk = {
            "text":             result["text"],
            "page_number":      result["page_number"],
            "document_id":      doc_id,
            "document_name":    doc_cache.get(doc_id, "Unknown"),
            "chunk_index":      result.get("chunk_index"),
            "similarity_score": round(result.get("similarity_score", 0), 4),
        }
        if "hybrid_score" in result:
            enriched_chunk["hybrid_score"] = result["hybrid_score"]
        enriched.append(enriched_chunk)

    return enriched


def get_context_for_llm(
    query: str,
    db: Session,
    k: int = 5
) -> str:
    """
    Get formatted context string to feed into an LLM prompt.

    This is used in Phase 5 (Answer Generation).
    Instead of returning a list of dicts, it formats the results
    into a single string that can be pasted into a prompt like:

        "Answer the question based on the following context:
         [Page 4 - annual_report.pdf]: Revenue grew 10%...
         [Page 7 - annual_report.pdf]: Net profit was $5M..."

    Args:
        query: The user's question
        db:    Database session
        k:     Number of chunks to include as context

    Returns:
        A formatted string of context, or empty string if nothing found.
    """

    chunks = search_chunks(query=query, db=db, k=k)

    if not chunks:
        return ""

    # Format each chunk as "[Page N - filename]: text"
    # This tells the LLM exactly where each piece of information came from
    context_parts = []
    for chunk in chunks:
        header = f"[Page {chunk['page_number']} - {chunk['document_name']}]"
        context_parts.append(f"{header}: {chunk['text']}")

    return "\n\n".join(context_parts)
