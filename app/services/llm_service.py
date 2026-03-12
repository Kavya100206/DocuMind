"""
LLM Service (Groq Integration)

WHAT DOES THIS FILE DO?
------------------------
This service connects to Groq's API and generates answers
based on retrieved document chunks.

WHAT IS GROQ?
-------------
Groq is a cloud AI service with a generous FREE tier.
It hosts open-source models (Llama 3, Mixtral) and serves them
extremely fast — often 10x faster than OpenAI.

We use: llama3-8b-8192
  - llama3 = Meta's Llama 3 model (very capable)
  - 8b     = 8 billion parameters (fast, fits in memory)
  - 8192   = 8192 token context window (plenty for our chunks)

FREE TIER LIMITS (as of 2025):
  - 30 requests/minute
  - 14,400 requests/day
  - More than enough for development + demos

HOW GROUNDED GENERATION WORKS:
--------------------------------
Normal LLM usage:
  User: "Who built the AI Code Review Bot?"
  LLM:  *uses training data* → might hallucinate

Grounded RAG usage:
  User: "Who built the AI Code Review Bot?"
  System: *retrieves chunks from resume* → sends to LLM with instruction:
          "Answer ONLY based on the following context"
  LLM:  *reads only those chunks* → cannot hallucinate

The key is the SYSTEM PROMPT — it tells the LLM:
  1. Who it is (a document assistant)
  2. What it HAS to do (answer from context only)
  3. What to do when it DOESN'T know ("I don't have enough information")

THE CONFIDENCE SCORE:
----------------------
We don't ask the LLM for a confidence score (it can't know).
Instead, we calculate it ourselves from FAISS similarity scores:

  confidence = average of top chunk similarity scores

  Example: top 3 chunks scored [0.32, 0.28, 0.21]
  confidence = (0.32 + 0.28 + 0.21) / 3 = 0.27

  We then map this to a 0–1 scale capped at 1.0
  and round to 2 decimal places.
"""

from groq import Groq
from typing import List, Dict, Any, Optional
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)



# Initialize the Groq client once at module load
# It reads GROQ_API_KEY automatically from environment
client = Groq(api_key=settings.GROQ_API_KEY)

# The system prompt is the most important part of grounded RAG.
# It defines the LLM's "rules of engagement" — what it can and cannot do.
SYSTEM_PROMPT = """You are DocuMind, a precise document assistant.

Your ONLY job is to extract and report information that is EXPLICITLY stated in the provided context.

RULES:
1. Answer using information explicitly present in the context.You may associate items with clearly labeled section headers if they appear directly under them.
2. If the answer is not explicitly stated, say: "I don't have enough information in the provided documents to answer this question."
3. NEVER perform calculations — if the document states a total, quote it directly; do not add up individual values
4. NEVER combine values from multiple sources to compute a new answer
5. If a direct answer exists in the context, quote it verbatim — do not rephrase or derive
6. Include specific names, numbers, and page references from the context
7. If multiple conflicting values appear, list all of them and note the conflict
8. Do NOT use your training knowledge — answer only from the provided context
9. Keep answers concise — one direct answer, then supporting details if needed
10. If you are unsure whether a value is explicitly stated or inferred, say so clearly
11. If a question asks for a total, summary, or overall value and it is NOT explicitly stated in the context, say you don't have that information — do NOT calculate or derive it from individual values"""

# If the best chunk scores below this, the context is too weak to be useful.
# We return "I don't know" immediately without wasting a Groq API call.
# IMPORTANT: must match SIMILARITY_THRESHOLD in settings.py (both = 0.10)
# If this is higher than SIMILARITY_THRESHOLD, chunks that pass FAISS still
# get silently rejected here — causing false "I don't know" responses.
LOW_CONFIDENCE_THRESHOLD = 0.10


def _deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate retrieved chunks before sending to Groq.

    WHY THIS MATTERS:
    -----------------
    FAISS can return the same chunk twice (different query angles hitting
    same vector) or very similar chunks from overlapping windows.
    Sending duplicates wastes tokens and confuses the LLM,
    causing it to list the same project multiple times.

    HOW WE DEDUPLICATE:
    -------------------
    1. By (document_id, chunk_index) — exact same chunk from DB
    2. By text prefix — chunks sharing the same first 80 chars
       are considered the same content

    We also sort by similarity_score descending so the best
    chunks appear first in the prompt.

    Args:
        chunks: Raw list from retrieval_service (may have duplicates)

    Returns:
        Deduplicated list, sorted best-first.
    """
    seen_ids: set = set()
    seen_text_prefixes: set = set()
    unique: List[Dict[str, Any]] = []

    # Sort best score first so when a duplicate is found, we keep the higher-scoring one
    sorted_chunks = sorted(chunks, key=lambda c: c.get("similarity_score", 0), reverse=True)

    for chunk in sorted_chunks:
        # Dedup by (document_id, chunk_index)
        chunk_id = (chunk.get("document_id"), chunk.get("chunk_index"))

        # Dedup by text content prefix (catches overlapping window duplicates)
        text_prefix = chunk.get("text", "")[:80].strip()

        if chunk_id in seen_ids or text_prefix in seen_text_prefixes:
            continue

        seen_ids.add(chunk_id)
        seen_text_prefixes.add(text_prefix)
        unique.append(chunk)

    removed = len(chunks) - len(unique)
    if removed > 0:
        print(f"  🧹 Deduplicated {removed} duplicate chunk(s). Sending {len(unique)} unique chunks to LLM.")

    return unique


def rewrite_query(question: str, history: list = []) -> str:
    """
    Rewrite the user's query into a semantically cleaner search query.

    Conservative by design:
    - If the query has unresolved pronouns (she/he/they/it) AND no history
      exists to resolve them, the original question is returned unchanged.
      This prevents the LLM from hallucinating a subject.
    - If history exists, pronouns are resolved using recent context.
    - Falls back to the original question on any API failure.
    """
    # GUARD: Skip rewriting if question has ambiguous pronouns and no history.
    # Without history the rewriter cannot resolve "she"/"he"/"they" and will
    # hallucinate a subject, causing semantic drift in retrieval.
    AMBIGUOUS_PRONOUNS = {"she", "he", "they", "her", "his", "their", "it", "its"}
    question_tokens = set(question.lower().split())
    has_unresolved_pronoun = bool(question_tokens & AMBIGUOUS_PRONOUNS)
    
    if has_unresolved_pronoun and not history:
        print(f"  ⏭️  Rewrite skipped: unresolved pronoun with no history — using original query")
        return question

    REWRITER_PROMPT = (
        "You are a query optimizer for a document retrieval system.\n"
        "Rewrite the user's question into a clear, standalone search query.\n"
        "Rules:\n"
        "- Resolve pronouns (she/he/they/it) using the conversation history ONLY\n"
        "- If a pronoun cannot be resolved from history, leave the question UNCHANGED\n"
        "- Do NOT guess or assume the subject if it is not in the chat history\n"
        "- Preserve the original intent and scope of the question\n"
        "- Do NOT list specific answers or assume facts\n"
        "- Keep it concise and natural — like a good search query\n"
        "- Return ONLY the rewritten query, nothing else"
    )

    # Build context string from last 3 turns of history
    history_str = ""
    if history:
        recent = history[-6:]  # last 3 exchanges (user + assistant each)
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:120]}" for m in recent
        )
        history_str = f"\nConversation so far:\n{history_str}\n"

    user_msg = f"{history_str}Question to rewrite: {question}"

    try:
        resp = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": REWRITER_PROMPT},
                {"role": "user",   "content": user_msg}
            ],
            max_tokens=60,
            temperature=0.0,
        )
        rewritten = resp.choices[0].message.content.strip()
        # Sanity check: blank or identical result → use original
        if rewritten and rewritten.lower() != question.lower():
            print(f"  ✏️  Query rewritten: '{question}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        print(f"  ⚠️  Query rewriting failed ({e}), using original question")

    return question


def build_prompt(question: str, context_chunks: List[Dict[str, Any]], history: list = []) -> str:
    """
    Build the full prompt to send to Groq.

    The prompt has two parts:
    1. The CONTEXT — formatted chunks from retrieved documents
    2. The QUESTION — the user's actual question

    Why format context this way?
    "[Page 1 - resume.pdf]: text..."
    → The LLM sees exactly which page and document each fact comes from
    → This lets it generate accurate citations in its answer

    Args:
        question:       The user's question
        context_chunks: List of retrieved chunks from retrieval_service

    Returns:
        A formatted string ready to send as the user message to Groq
    """

    if not context_chunks:
        return f"Question: {question}"

    # Include recent conversation history so the LLM knows the context
    history_block = ""
    if history:
        recent = history[-6:]  # last 3 exchanges (user + assistant each)
        history_lines = []
        for m in recent:
            role = "User" if m["role"] == "user" else "Assistant"
            snippet = m["content"][:200].replace("\n", " ")
            history_lines.append(f"{role}: {snippet}")
        history_block = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

    # Format each chunk as "[Page N - filename]: text"
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        page = chunk.get("page_number", "?")
        doc = chunk.get("document_name", "Unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Source {i} | Page {page} | {doc}]:\n{text}")

    context_str = "\n\n".join(context_parts)

    return f"""{history_block}Context from documents:
{context_str}

Question: {question}

IMPORTANT REMINDER: Answer using ONLY what is explicitly stated above. Do NOT calculate, infer, or add up values. If a total or summary value is stated in the context, quote it directly. If you cannot find an explicit answer, say so.

Answer:"""


def calculate_confidence(chunks: List[Dict[str, Any]]) -> float:
    """
    Calculate a confidence score based on retrieval similarity scores.

    HOW IT WORKS:
    -------------
    We take the average similarity score of retrieved chunks.
    Higher average = Groq had more relevant context to work with
                   = more confident in the answer.

    Score interpretation:
      0.00 – 0.20 = Low confidence (chunks barely matched the query)
      0.20 – 0.40 = Medium confidence (some relevant context found)
      0.40 – 0.60 = Good confidence (strong semantic match)
      0.60+       = High confidence (very relevant context)

    NOTE: We multiply by 2 to scale up (our model scores max ~0.5 for good matches,
    so multiplying by 2 brings 0.4 → 0.8, which is more intuitive).

    Args:
        chunks: List of retrieved chunk dicts with "similarity_score" key

    Returns:
        Confidence score between 0.0 and 1.0
    """

    if not chunks:
        return 0.0

    # Sort by score descending and take only the TOP 3
    # WHY: All retrieved chunks (including low-scoring ones near threshold 0.10)
    # drag the average down, even if the best chunks strongly matched the query.
    # Top-3 reflects the quality of evidence the LLM actually relied on most.
    top_scores = sorted(
        [c.get("similarity_score", 0.0) for c in chunks],
        reverse=True
    )[:3]

    avg_score = sum(top_scores) / len(top_scores)

    # Scale up by 3 (all-MiniLM-L6-v2 raw cosine scores are naturally low: ~0.15–0.30)
    # A "good match" score of ~0.21 × 3 = 0.63 — much more intuitive than 0.42
    # Cap at 0.9 — never return 1.0 (nothing is 100% certain in RAG)
    scaled = min(avg_score * 3, 0.9)

    return round(scaled, 2)


def _penalize_for_hedging(confidence: float, answer_text: str) -> float:
    """
    Halve confidence when the LLM hedges or admits it's guessing.

    WHY THIS MATTERS:
    -----------------
    Phrases like "not explicitly mentioned" or "implied" signal the LLM
    went OUTSIDE the retrieved context — which is when hallucination happens.
    If the LLM is guessing, the confidence should reflect that.

    Args:
        confidence: The raw calculated confidence score
        answer_text: The LLM's generated answer text

    Returns:
        Adjusted confidence (halved if hedging detected)
    """
    hedging_phrases = [
        "not explicitly mentioned",
        "not explicitly stated",
        "implied",
        "assumed",
        "likely based on",
        "probably",
        "i would assume",
        "it seems",
    ]

    answer_lower = answer_text.lower()
    if any(phrase in answer_lower for phrase in hedging_phrases):
        penalized = round(confidence * 0.5, 2)
        print(f"  ⚠️  Hedging detected — halving confidence ({confidence} → {penalized})")
        return penalized

    return confidence


def generate_answer(
    question: str,
    context_chunks: List[Dict[str, Any]],
    history: list = [],
    max_tokens: int = 1024   # Increased from 512 → prevents truncation when listing multiple items
) -> Dict[str, Any]:
    """
    Generate a grounded answer using Groq.

    This is the main function called by qa_controller.py.

    WHAT HAPPENS:
    1. Check if we have any context (no context → return "I don't know")
    2. Build the prompt (system + user message)
    3. Call Groq API (llama3-8b-8192 model)
    4. Extract the answer text from the response
    5. Calculate confidence from chunk scores
    6. Format citations from the chunks used
    7. Return everything as a dict

    WHAT IS max_tokens?
    -------------------
    Groq won't generate a response longer than this.
    512 tokens ≈ 350–400 words. Good for most answers.
    For very long summaries, you'd increase this.

    Args:
        question:       User's question
        context_chunks: Retrieved chunks from retrieval_service
        max_tokens:     Max length of the generated answer

    Returns:
        {
            "answer": "The candidate built...",
            "citations": [...],
            "confidence": 0.72,
            "has_answer": True
        }
    """

    # ------------------------------------------------------------------
    # STEP 1: Deduplicate chunks
    # ------------------------------------------------------------------
    # Remove any duplicate or near-duplicate chunks before doing anything else

    context_chunks = _deduplicate_chunks(context_chunks)

    # ------------------------------------------------------------------
    # STEP 1b: Token-based Context Packing
    # ------------------------------------------------------------------
    # We retrieve up to TOP_K_RESULTS candidates but only send the best chunks
    # to the LLM until we reach a ~2000 token limit. This packing avoids 
    # either wasting context (if chunks are small) or going over limit (if large).
    
    # Sort chunks by the best available score: reranker > hybrid > similarity
    context_chunks = sorted(
        context_chunks,
        key=lambda c: c.get("reranker_score") or c.get("hybrid_score") or c.get("similarity_score", 0),
        reverse=True
    )
    
    MAX_CONTEXT_TOKENS = 1400  # reduced from 2000 → faster LLM generation
    packed_chunks = []
    current_tokens = 0
    
    for chunk in context_chunks:
        # Estimate tokens: ~4 chars per token
        chunk_text = chunk.get("text", "")
        estimated_tokens = len(chunk_text) / 4
        
        if current_tokens + estimated_tokens > MAX_CONTEXT_TOKENS and packed_chunks:
            # Reached limit and we have at least one chunk
            break
            
        packed_chunks.append(chunk)
        current_tokens += estimated_tokens
        
    context_chunks = packed_chunks
    print(f"  🔍 Using {len(context_chunks)} chunks for LLM context (~{current_tokens:.0f}/{MAX_CONTEXT_TOKENS} tokens)")

    # ------------------------------------------------------------------
    # STEP 3: Confidence guard — "I don't know" for weak context
    # ------------------------------------------------------------------
    # If even the BEST chunk is below the low-confidence threshold,
    # the query is asking about something not in the documents.
    # Return immediately instead of letting Groq hallucinate.
    
    # Confidence guard — skip LLM if retrieval returned nothing
    # We trust the reranker to surface only relevant chunks.
    # Score-based thresholds are model-specific and error-prone; an empty
    # result list is an unambiguous signal that nothing matched.
    if not context_chunks:
        print("  No chunks retrieved — returning 'I don't know'.")
        return {
            "answer": "I don't have enough information in the provided documents to answer this question confidently.",
            "citations": [],
            "confidence": 0.0,
            "has_answer": False
        }

    # ------------------------------------------------------------------
    # STEP 2: Build the prompt
    # ------------------------------------------------------------------
    user_message = build_prompt(question, context_chunks, history=history)

    # ------------------------------------------------------------------
    # STEP 3: Call Groq API
    # ------------------------------------------------------------------
    # The Groq client sends a list of "messages" — same pattern as OpenAI.
    # "system" = the rules/persona for the LLM
    # "user"   = the actual input (context + question)
    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )

        # Extract the generated text from the response object
        answer_text = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  ❌ Groq API error: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "citations": [],
            "confidence": 0.0,
            "has_answer": False
        }

    # ------------------------------------------------------------------
    # STEP 4: Determine if the answer is meaningful
    # ------------------------------------------------------------------
    # If the LLM said it doesn't know, mark has_answer = False
    no_info_phrases = [
        "don't have enough information",
        "cannot answer",
        "not mentioned in",
        "not provided in",
        "no information"
    ]
    has_answer = not any(phrase in answer_text.lower() for phrase in no_info_phrases)

    # ------------------------------------------------------------------
    # STEP 5: Build citations from the chunks that were used
    # ------------------------------------------------------------------
    # Format citations cleanly for the frontend to display
    citations = []
    seen = set()  # avoid duplicate citations
    for chunk in context_chunks:
        key = (chunk.get("document_id"), chunk.get("page_number"))
        if key not in seen:
            seen.add(key)
            citations.append({
                "document_name": chunk.get("document_name", "Unknown"),
                "page_number":   chunk.get("page_number"),
                "text_snippet":  chunk.get("text", "")[:150] + "...",  # preview
            })

    # ------------------------------------------------------------------
    # STEP 6: Calculate confidence (+ penalize for hedging)
    # ------------------------------------------------------------------
    confidence = calculate_confidence(context_chunks)
    # If the LLM used hedging phrases ("not explicitly mentioned", "implied"...),
    # it means it went beyond the context → hallucination risk → halve confidence
    confidence = _penalize_for_hedging(confidence, answer_text)

    print(f"  🤖 Answer generated (confidence: {confidence}, has_answer: {has_answer})")

    return {
        "answer":     answer_text,
        "citations":  citations,
        "confidence": confidence,
        "has_answer": has_answer
    }
