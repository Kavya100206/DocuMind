"""
QA Controller — Question Answering Endpoint with Query Rewriting + Memory + Dual Retrieval

Exposes: POST /api/ask

THE PIPELINE:
1. Load session history for this session_id
2. Rewrite the query using LLM (semantic broadening, history-aware)
3. Dual Retrieval: retrieve with BOTH rewritten AND original query, merge unique chunks
4. Generate answer with Groq using ORIGINAL question + merged chunks + history
5. Save (question, answer) to session history
6. Return structured response
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from app.database.postgres import get_db
from app.services import retrieval_service
from app.services import llm_service
from app.views.qa_views import QAResponse, Citation, RetrievalTrace, ChunkTrace, RouterInfo
from app.config.settings import settings

router = APIRouter(prefix="/api", tags=["Question Answering"])

# In-memory conversation store
# Maps session_id -> list of {role, content} dicts (OpenAI-style messages)
# Intentionally in-process memory (not Redis/DB) — cleared on server restart.
_sessions: dict = {}

MAX_HISTORY_PER_SESSION = 10  # keep last 10 messages (5 exchanges) per session


class AskRequest(BaseModel):
    """
    Request body for POST /api/ask

    question:    The user's question in plain English
    document_id: REQUIRED — the ID of the document to search within.
                 DocuMind is document-scoped: you must select a document first.
                 This prevents cross-document contamination.
    session_id:  Optional — enables conversation memory across turns
    k:           How many chunks to retrieve as context (default TOP_K_RESULTS)
    """
    question: str
    document_id: Optional[str] = None   # validated manually below for a clean 400 error
    session_id: Optional[str] = None
    k: int = settings.TOP_K_RESULTS
    # Phase 3.1: when true, the response includes a per-chunk score trace
    # for debugging / demoing the retrieval pipeline. Default False to keep
    # normal traffic light.
    debug_mode: bool = False


@router.post(
    "/ask",
    response_model=QAResponse,
    summary="Ask a question about your documents",
    description=(
        "Submit a question and get a grounded answer with citations. "
        "Pass a session_id to enable multi-turn conversation memory. "
        "The system rewrites your query for better retrieval and uses "
        "dual retrieval to maximize recall."
    )
)
def ask_question(
    request: AskRequest,
    db: Session = Depends(get_db)
):
    """
    Full RAG pipeline with Query Rewriting + Dual Retrieval + Conversation Memory.
    """

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not settings.GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY not configured. Add it to your .env file."
        )

    from app.services import faiss_service
    if faiss_service.IS_BUILDING:
        raise HTTPException(
            status_code=503,
            detail="The search index is currently being built in the background. Please wait a few moments and try again."
        )

    # ── Gate 2: document_id is required (Gate 1 is in the frontend) ──
    # DocuMind is document-scoped. Without a document_id we have no way
    # to know WHICH document to search, and searching all docs risks
    # cross-contamination (answer for doc A leaks into a query about doc B).
    #
    # WHY a manual check instead of making it required in Pydantic?
    # Pydantic's 422 error returns a wall of JSON the frontend can't cleanly
    # show to the user. Our 400 returns a simple "detail" string.
    if not request.document_id or not request.document_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Please select a document before asking a question. "
                   "DocuMind only answers based on an uploaded document's contents."
        )

    # Short query guard
    meaningful_words = [w for w in question.split() if len(w) > 1]
    if len(meaningful_words) < 3:
        return QAResponse(
            question=question,
            answer=(
                "Your question is too short or vague for me to search effectively. "
                "Please ask a full question — for example: "
                "\"What is the project title?\" or \"Which machine learning technique is used?\""
            ),
            citations=[],
            confidence=0.0,
            has_answer=False,
            model_used=settings.GROQ_MODEL,
            rewritten_query=None
        )

    print(f"\n Question: '{question}'")

    # STEP 1: Load conversation history
    session_id = request.session_id
    history: list = []
    if session_id:
        history = _sessions.get(session_id, [])
        if history:
            print(f"  Loaded {len(history)} history messages for session '{session_id[:8]}...'")

    # Pronoun ambiguity guard
    # If the question relies purely on a dangling pronoun ("Where did she work?") AND there is not enough history,
    # the LLM will assume context from the most relevant document — which is wrong.
    # Return a clarification request immediately instead of hallucinating a subject.
    # We restrict this by allowing the question IF it contains a Proper Noun (capitalized word)
    # anywhere after the first word, assuming it establishes its own subject (e.g. "Who is Kavya and what are her programs?").
    
    AMBIGUOUS_PRONOUNS = {"she", "he", "they", "her", "his", "their"}
    question_tokens = set(question.lower().split())
    
    # Check if there's a capitalized word (ignoring the first word of the sentence)
    words = question.split()
    has_proper_noun = any(w[0].isupper() for w in words[1:]) if len(words) > 1 else False
    
    if question_tokens & AMBIGUOUS_PRONOUNS and len(history) < 2 and not has_proper_noun:
        print("  Pronoun guard triggered: unresolved pronoun with empty history and no proper noun")
        return QAResponse(
            question=question,
            answer=(
                "I'm not sure who you're referring to. Could you clarify your question?"
            ),
            citations=[],
            confidence=0.0,
            has_answer=False,
            model_used=settings.GROQ_MODEL,
            rewritten_query=None
        )


    try:
        # ----------------------------------------------------------------
        # STEPS 2–4: Agentic Retrieval Loop (Phase 2)
        # ----------------------------------------------------------------
        # The agent:
        #   1. Rewrites the query (same as before)
        #   2. Lets the LLM pick which retrieval tool to run
        #   3. Checks confidence → loops or generates
        #   4. Falls back to the original pipeline if 3 attempts fail
        #
        # The agent is created fresh for this request and discarded
        # after run_agent() returns. No global agent. No shared state.
        from app.services import agent_service

        agent_result = agent_service.run_agent(
            query=question,
            db=db,
            document_id=request.document_id,
            history=history,
            debug_mode=request.debug_mode,
        )

        # Log which path was taken (visible in uvicorn console)
        fallback_tag = " [FALLBACK]" if agent_result.get("fallback_used") else " [AGENT]"
        print(
            f"{fallback_tag} tools={agent_result.get('tools_tried')}, "
            f"chunks={agent_result.get('chunks_used')}, "
            f"confidence={agent_result.get('confidence')}"
        )

        # STEP 5: Save to session history (unchanged)
        if session_id:
            if session_id not in _sessions:
                _sessions[session_id] = []

            _sessions[session_id].append({"role": "user",      "content": question})
            _sessions[session_id].append({"role": "assistant", "content": agent_result["answer"]})

            if len(_sessions[session_id]) > MAX_HISTORY_PER_SESSION:
                _sessions[session_id] = _sessions[session_id][-MAX_HISTORY_PER_SESSION:]

            print(f"  Session '{session_id[:8]}...' now has {len(_sessions[session_id])} messages")

        # STEP 6: Shape and return the response
        citations = [Citation(**c) for c in agent_result["citations"]]

        # Pass the observability trace through to the frontend (Phase 2.3).
        # Defensive: agent_result["trace"] should always exist now, but if
        # an older code path or a hard error returns without it, default to
        # None — the frontend already handles the missing-trace case.
        trace_dict = agent_result.get("trace")
        trace = RetrievalTrace(**trace_dict) if trace_dict else None

        # Phase 3.1 — full per-chunk trace (only when debug_mode=True)
        chunk_trace_list = agent_result.get("chunk_trace")
        chunk_trace = (
            [ChunkTrace(**c) for c in chunk_trace_list] if chunk_trace_list else None
        )

        # Phase 3.2 — router's tool choice + reasoning
        routing_dict = agent_result.get("routing")
        routing = RouterInfo(**routing_dict) if routing_dict else None

        return QAResponse(
            question=question,
            answer=agent_result["answer"],
            citations=citations,
            confidence=agent_result["confidence"],
            has_answer=agent_result["has_answer"],
            model_used=settings.GROQ_MODEL,
            rewritten_query=None,   # rewriting is handled inside agent_service
            trace=trace,
            chunk_trace=chunk_trace,
            routing=routing,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return QAResponse(
            question=question,
            answer="I'm sorry, an internal processing error occurred while analyzing your documents. Please try asking again.",
            citations=[],
            confidence=0.0,
            has_answer=False,
            model_used=settings.GROQ_MODEL,
            rewritten_query=None
        )


@router.delete(
    "/session/{session_id}",
    summary="Clear a conversation session",
    description="Clears the memory for a given session_id — starts a fresh conversation."
)
def clear_session(session_id: str):
    """Clear all history for a session (triggered by 'New Chat' button)."""
    if session_id in _sessions:
        del _sessions[session_id]
        print(f"  Cleared session '{session_id[:8]}...'")
        return {"cleared": True, "session_id": session_id}
    return {"cleared": False, "session_id": session_id}
