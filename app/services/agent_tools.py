"""
Agent Tools

WHAT DOES THIS FILE DO?
------------------------
Defines the three tools the LangGraph agent can choose from
when deciding how to answer a question.

WHAT IS A "TOOL" IN THIS CONTEXT?
------------------------------------
A tool is just a plain Python function with a clear name and docstring.
The LangGraph router (which is an LLM call) reads the tool names and
descriptions and decides: "given this question, which function should
I call and with what arguments?"

The tools are THIN WRAPPERS — they don't contain any retrieval logic.
All the real work is still inside the existing services:
  - vector_search     → calls faiss_service + retrieval_service (unchanged)
  - keyword_search    → calls bm25_service directly (unchanged)
  - summarize_document→ calls llm_service.generate_answer (unchanged)

WHY KEEP THEM AS THIN WRAPPERS?
---------------------------------
Single Responsibility Principle:
  - retrieval_service.py = HOW to search (this is complex, tested, working)
  - agent_tools.py       = WHAT tools exist (this is simple, just glue)

If the retrieval logic ever changes, you change retrieval_service.py.
The agent tools stay the same.

THE THREE TOOLS AND WHEN THE AGENT PICKS EACH:
------------------------------------------------
1. vector_search(query)
   → Best for: conceptual questions, meaning-based lookups
   → "What is the project's main methodology?"
   → "What machine learning techniques are used?"
   → Uses: FAISS semantic search + BM25 + reranker

2. keyword_search(query)
   → Best for: exact term lookups, names, identifiers, codes
   → "Find all mentions of 'LSTM' in the documents"
   → "What pages mention 'Section 4.2'?"
   → Uses: BM25 exact keyword match directly

3. summarize_document(doc_id)
   → Best for: "summarize", "overview", "what is this document about"
   → "Give me a summary of the annual report"
   → Uses: retrieves top chunks across the whole doc + LLM
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.utils.logger import get_logger

logger = get_logger(__name__)


def vector_search(
    query: str,
    db: Session,
    document_id: Optional[str] = None,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Semantic vector search within a specific document's chunks.

    Args:
        query       : The search query (plain English)
        db          : Active database session
        document_id : REQUIRED — only search within this document
        k           : Number of candidates to retrieve (default: 10)
    """
    from app.services import retrieval_service

    # ── Strict PDF Isolation Guard ──
    if not document_id:
        raise ValueError("vector_search tool called without document_id. Select a document first.")

    logger.info(f"[Tool] vector_search called: query='{query[:50]}' doc_id={document_id}")

    results = retrieval_service.search_chunks_fast(
        query=query,
        db=db,
        k=k,
        document_id=document_id,
    )

    logger.info(f"[Tool] vector_search returned {len(results)} chunks")
    return results


def keyword_search(
    query: str,
    db: Session,
    document_id: Optional[str] = None,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Exact keyword search using BM25 scoring, scoped to one document.

    Args:
        query       : The keyword query
        db          : Active database session
        document_id : REQUIRED — only search within this document
        k           : Max results to return
    """
    from app.services import bm25_service
    from app.services import faiss_service
    from app.models.document import Document

    # ── Strict PDF Isolation Guard ──
    if not document_id:
        raise ValueError("keyword_search tool called without document_id. Select a document first.")

    logger.info(f"[Tool] keyword_search called: query='{query[:50]}' doc_id={document_id}")

    # Get all indexed chunks, then filter to this document only
    all_chunks = faiss_service.get_all_chunks_metadata()

    if not all_chunks:
        logger.warning("[Tool] keyword_search: no indexed chunks found")
        return []

    # ALWAYS filter to the requested document (not optional)
    all_chunks = [c for c in all_chunks if c.get("document_id") == document_id]

    if not all_chunks:
        logger.warning(f"[Tool] keyword_search: no chunks found for document {document_id[:8]}")
        return []

    ranked = bm25_service.rerank_with_bm25(query, all_chunks)
    top = ranked[:k]

    doc_cache: Dict[str, str] = {}
    for chunk in top:
        doc_id = chunk.get("document_id")
        if doc_id and doc_id not in doc_cache:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            doc_cache[doc_id] = doc.filename if doc else "Unknown"
        chunk["document_name"] = doc_cache.get(doc_id, "Unknown")

    logger.info(f"[Tool] keyword_search returned {len(top)} chunks")
    return top


def summarize_document(
    doc_id: str,
    db: Session,
    question: str = "Provide a comprehensive summary of this document.",
) -> List[Dict[str, Any]]:
    """
    Retrieve broad context across a full document for summarization.

    Rather than pinpointing one specific chunk, this tool fetches
    chunks spread across the whole document to give the LLM a wide
    view for generating a summary or overview answer.

    Strategy:
    - Runs a generic "summary" query against the specific document
    - Uses a higher k (20 chunks) to get broad coverage
    - No threshold filtering — we want breadth, not precision

    The agent picks this tool when the question includes words like:
    "summarize", "overview", "what is this document about",
    "describe the main topics", "give me a summary"

    Args:
        doc_id   : The document UUID to summarize
        db       : Active database session
        question : Optional custom question to guide retrieval
                   (defaults to a generic summary prompt)

    Returns:
        List of chunk dicts covering the document broadly
    """
    from app.services import faiss_service
    from app.services.embedding_service import get_embedding
    from app.models.document import Document
    from app.models.chunk import Chunk

    logger.info(f"[Tool] summarize_document called: doc_id={doc_id[:8]}")

    # Get all chunks for this document directly from Postgres
    # (more reliable than FAISS for full-doc coverage)
    db_chunks = (
        db.query(Chunk)
        .filter(Chunk.document_id == doc_id)
        .order_by(Chunk.page_number)
        .all()
    )

    if not db_chunks:
        logger.warning(f"[Tool] summarize_document: no chunks found for doc {doc_id[:8]}")
        return []

    # Fetch document name once
    doc = db.query(Document).filter(Document.id == doc_id).first()
    doc_name = doc.filename if doc else "Unknown"

    # ── Even sampling across the WHOLE document ──
    # The previous version did `chunks[::step][:max_chunks]` which had a bug:
    # for a 50-chunk doc with step=2 and max=20, it took chunks[::2][:20] which
    # is the first 20 of every-other chunk → only the first 80% of the doc made
    # it in. That caused summaries biased to early sections.
    #
    # Fix: pick max_chunks indices uniformly distributed across [0, n).
    # n=50, max=25 → indices [0, 2, 4, ..., 48] — covers the entire doc.
    # n=10, max=25 → all 10 chunks (no oversampling).
    max_chunks = 25
    n = len(db_chunks)
    if n <= max_chunks:
        sampled = list(db_chunks)
    else:
        indices = [int(i * n / max_chunks) for i in range(max_chunks)]
        sampled = [db_chunks[i] for i in indices]

    # ── Per-chunk truncation for balanced LLM coverage ──
    # llm_service packs chunks into a ~2500-token budget greedily. Without
    # truncation, the first 5–8 long chunks fill the budget and the rest get
    # dropped — re-creating the early-section bias even with good sampling.
    #
    # Truncating each chunk to ~400 chars (~100 tokens) lets all 25 sampled
    # chunks fit comfortably (25 × 100 ≈ 2500 tokens including header overhead).
    # 400 chars is enough for the LLM to grasp each section's topic and key
    # points without going deep — exactly what a summary needs.
    SUMMARY_CHUNK_CHAR_LIMIT = 400

    result_chunks = []
    for c in sampled:
        text = c.text or ""
        if len(text) > SUMMARY_CHUNK_CHAR_LIMIT:
            text = text[:SUMMARY_CHUNK_CHAR_LIMIT].rstrip() + "…"
        result_chunks.append({
            "text": text,
            "page_number": c.page_number,
            "document_id": c.document_id,
            "document_name": doc_name,
            "chunk_index": c.id,
            "similarity_score": 0.5,  # Neutral score — not a similarity search
        })

    logger.info(
        f"[Tool] summarize_document: returning {len(result_chunks)} chunks "
        f"(truncated to {SUMMARY_CHUNK_CHAR_LIMIT} chars each) from {n} total for '{doc_name}'"
    )
    return result_chunks
