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
from app.views.qa_views import QAResponse, Citation
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
    document_id: Optional — restrict search to one document
    session_id:  Optional — enables conversation memory across turns
    k:           How many chunks to retrieve as context (default TOP_K_RESULTS)
    """
    question: str
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    k: int = settings.TOP_K_RESULTS


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
        # STEP 2: Query Rewriting
        # Semantically broadens the query (resolves pronouns, preserves intent).
        # The REWRITTEN query goes to FAISS; the ORIGINAL question goes to the LLM.
        rewritten_query = llm_service.rewrite_query(question=question, history=history)

        # STEP 3: Dual Query Retrieval (fast path)
        # Use search_chunks_fast() which skips the expensive cross-encoder.
        # We gather candidates from BOTH query formulations, merge unique chunks,
        # then run the cross-encoder ONCE on the merged pool.
        # Result: same quality as before but reranker inference happens only ONCE.
        from app.services import reranker_service
        chunks_rewritten = retrieval_service.search_chunks_fast(
            query=rewritten_query,
            db=db,
            k=request.k,
            document_id=request.document_id
        )

        chunks_original = []
        if rewritten_query != question:
            chunks_original = retrieval_service.search_chunks_fast(
                query=question,
                db=db,
                k=request.k,
                document_id=request.document_id
            )

        print(f"  [DEBUG] chunks_rewritten: {len(chunks_rewritten)} chunks from '{rewritten_query[:40]}'")
        print(f"  [DEBUG] chunks_original:  {len(chunks_original)} chunks from '{question[:40]}'")

        # Merge unique chunks (rewritten results have priority)
        seen_keys: set = set()
        merged = []
        for c in chunks_rewritten + chunks_original:
            key = (c.get("document_id"), c.get("chunk_index"))
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(c)

        print(f"  [DEBUG] merged: {len(merged)} unique candidates before reranker")
        if merged:
            top_score = max(c.get("hybrid_score") or c.get("similarity_score", 0) for c in merged)
            print(f"  [DEBUG] best hybrid/sim score in merged pool: {top_score:.4f}")

        # Apply cross-encoder ONCE on the merged candidate pool
        chunks = reranker_service.rerank(
            query=rewritten_query,
            chunks=merged,
            top_n=settings.RERANKER_TOP_N
        )
        print(f"  [DEBUG] after rerank: {len(chunks)} chunks selected")
        if chunks:
            print(f"  [DEBUG] reranker scores: {[c.get('reranker_score') for c in chunks]}")

        # STEP 4: Generate answer with Groq
        result = llm_service.generate_answer(
            question=question,       # original question as the final prompt
            context_chunks=chunks,
            history=history          # conversation context
        )

        # STEP 5: Save to session history
        if session_id:
            if session_id not in _sessions:
                _sessions[session_id] = []

            _sessions[session_id].append({"role": "user",      "content": question})
            _sessions[session_id].append({"role": "assistant", "content": result["answer"]})

            # Trim old messages to stay within memory limit
            if len(_sessions[session_id]) > MAX_HISTORY_PER_SESSION:
                _sessions[session_id] = _sessions[session_id][-MAX_HISTORY_PER_SESSION:]

            print(f"  Session '{session_id[:8]}...' now has {len(_sessions[session_id])} messages")

        # STEP 6: Shape and return the response
        citations = [Citation(**c) for c in result["citations"]]

        return QAResponse(
            question=question,
            answer=result["answer"],
            citations=citations,
            confidence=result["confidence"],
            has_answer=result["has_answer"],
            model_used=settings.GROQ_MODEL,
            rewritten_query=rewritten_query if rewritten_query != question else None,
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
