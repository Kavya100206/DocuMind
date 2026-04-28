"""
QA Views (Response Schemas for Answer Generation)

These Pydantic schemas define the shape of the /api/ask response.
"""

from typing import List, Optional
from pydantic import BaseModel


class Citation(BaseModel):
    """
    One source citation — where a piece of the answer came from.

    document_name: Filename of the source document
    page_number:   Page where the relevant text was found
    text_snippet:  First 150 chars of the source chunk (preview)
    """
    document_name: str
    page_number: int
    text_snippet: str


class ChunkTrace(BaseModel):
    """
    Per-chunk debug trace (Phase 3.1 — Retrieval Trace Mode).

    Returned only when debug_mode=True on /api/ask. Lets a developer (or
    a demo audience) see EVERY retrieved chunk and the full breakdown of
    its scores: raw FAISS, BM25, the hybrid blend, and the cross-encoder
    reranker. Plus a `selected` flag indicating whether the chunk made it
    into the LLM's prompt or was dropped during context packing.

    This is the "killer feature" for transparency — it shows that nothing
    is hidden in the retrieval pipeline.
    """
    rank: int
    chunk_index: Optional[int] = None
    page: Optional[int] = None
    document_name: Optional[str] = None
    text_preview: str
    faiss_score: Optional[float] = None
    bm25_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    reranker_score: Optional[float] = None
    selected: bool


class RouterInfo(BaseModel):
    """
    Router's tool choice + reasoning (Phase 3.2).

    Captured from the LangGraph router_node — the LLM not only picks a
    tool but explains WHY. Surfaced in the observability panel so users
    can see the agent's decision-making.
    """
    tool_chosen: str
    reason: str


class RetrievalTrace(BaseModel):
    """
    Observability trace describing HOW the answer was produced (Phase 2.3).

    Rendered in the frontend's collapsible "🔍 Retrieval Info" panel.
    Lets the demo audience (and the developer debugging) see which
    retrieval tool ran, how confident the top chunk was, and how many
    iterations the agent took.

    Fields:
      tool_used:        Primary retrieval tool that produced the chunks.
                        One of: vector_search, keyword_search,
                        summarize_document, or "fallback (full pipeline)".
      tools_tried:      All tools the agent tried, in order. Useful when
                        the agent looped (e.g. ["vector_search", "keyword_search"]).
      top_score:        Best score among retrieved chunks, 0.0–1.0.
                        Uses reranker_score if available, else hybrid_score,
                        else similarity_score.
      chunks_retrieved: How many chunks were collected (across all tool calls).
      iterations:       How many router iterations the agent loop ran.
                        1 = router picked once and that was enough.
      fallback_used:    True if the agent couldn't produce a confident
                        answer and the original (pre-agent) pipeline ran.
    """
    tool_used: str
    tools_tried: List[str]
    top_score: float
    chunks_retrieved: int
    iterations: int
    fallback_used: bool


class QAResponse(BaseModel):
    """
    Full response for a question-answering request.

    Fields:
      question:        The original question (echoed back)
      answer:          The LLM-generated answer (grounded in documents)
      citations:       List of sources used to generate the answer
      confidence:      0.0–1.0 score based on retrieval similarity
      has_answer:      False if LLM said it doesn't have enough info
      model_used:      Which Groq model generated the answer
      rewritten_query: The keyword-rich query used for retrieval (if rewriting occurred)
      trace:           Observability trace — which tool, top score, etc. (Phase 2.3)
    """
    question: str
    answer: str
    citations: List[Citation]
    confidence: float
    has_answer: bool
    model_used: str
    rewritten_query: Optional[str] = None
    trace: Optional[RetrievalTrace] = None
    # Phase 3.1: Full per-chunk trace, only populated when debug_mode=True
    chunk_trace: Optional[List[ChunkTrace]] = None
    # Phase 3.2: Router's reasoning (tool choice + why)
    routing: Optional[RouterInfo] = None
