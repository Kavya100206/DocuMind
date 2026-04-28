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

import re
from groq import Groq
from typing import List, Dict, Any, Optional
from app.config.settings import settings
from app.utils.logger import get_logger

# ── Language detection (Phase 2.1 — Multilingual Support) ──
# `langdetect` is a small pure-Python library that returns an ISO 639-1
# code ("en", "hi", "fr", ...) for a given string.
# Setting DetectorFactory.seed makes detection deterministic — without it,
# the same input can occasionally produce different codes due to internal
# random sampling.
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0

logger = get_logger(__name__)


# Map ISO codes to human-readable language names for use in the system prompt.
# We only include languages we actually expect — adding more is harmless but
# unused. Anything unmapped falls back to the ISO code itself.
_LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
}


def detect_language(text: str) -> str:
    """
    Detect the language of `text` and return a human-readable name.

    Public utility — also used by agent_service to translate refusal messages
    into the user's query language. Single source of truth for detection logic
    so the agent and the LLM call always agree on the language.

    Defaults to "English" when:
      - text is empty or shorter than 4 characters (langdetect is unreliable
        on very short input — a single word like "क्या?" can be misclassified)
      - langdetect raises LangDetectException (e.g. all-whitespace, all-symbols)

    The default is intentional: if we can't be confident, fall back to the
    most common case rather than asking the LLM to reply in a wrong language.
    """
    if not text or len(text.strip()) < 4:
        return "English"
    try:
        code = detect(text)
        return _LANGUAGE_NAMES.get(code, code)
    except LangDetectException:
        return "English"



# Initialize the Groq client once at module load
# It reads GROQ_API_KEY automatically from environment
client = Groq(api_key=settings.GROQ_API_KEY)

# The system prompt is the most important part of grounded RAG.
# It defines the LLM's "rules of engagement" — what it can and cannot do.
SYSTEM_PROMPT = """You are DocuMind, a precise document assistant capable of synthesizing information across multiple documents.

Your ONLY job is to extract and report information that is EXPLICITLY stated in the provided context. When asked a broad question, actively combine and synthesize information across all provided document sources to provide a complete answer.

RULES:
1. Answer using information explicitly present in the context. You MUST actively combine insights from ALL relevant documents if multiple sources contain pieces of the answer.
2. If the answer is not explicitly stated, say: "I don't have enough information in the provided documents to answer this question."
3. NEVER perform calculations — if the document states a total, quote it directly; do not add up individual values.
4. If a direct answer exists in the context, quote it verbatim — do not rephrase or derive.
5. Include specific names, numbers, and page references from the context.
6. If multiple conflicting values appear across different documents, list all of them and note the conflict.
7. Do NOT use your training knowledge — answer only from the provided context.
8. Keep answers concise — one direct answer, then supporting details if needed.
9. If you are unsure whether a value is explicitly stated or inferred, say so clearly."""

# If the best chunk scores below this, the context is too weak to be useful.
# We return "I don't know" immediately without wasting a Groq API call.
# IMPORTANT: must match SIMILARITY_THRESHOLD in settings.py (both = 0.10)
# If this is higher than SIMILARITY_THRESHOLD, chunks that pass FAISS still
# get silently rejected here — causing false "I don't know" responses.
LOW_CONFIDENCE_THRESHOLD = 0.05


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
        "- If the question requires multiple pieces of information (multi-hop), ensure ALL key concepts are included in your rewritten query.\n"
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


def build_prompt(
    question: str,
    context_chunks: List[Dict[str, Any]],
    history: list = [],
    response_language: str = "English",
) -> str:
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

    # Final nudge: the LAST thing the LLM sees before answering.
    # If the user asked in a non-English language, repeat the language
    # directive here so it dominates the immediate context window.
    if response_language.lower() != "english":
        language_reminder = (
            f"\n\nFINAL REMINDER: Your answer MUST be written entirely in "
            f"{response_language}. Translate the relevant facts from the "
            f"context above into {response_language}. Do NOT respond in English."
        )
    else:
        language_reminder = ""

    # Synthesis directive: questions that ask for reasoning, motivation, or
    # cause (V6.Q2: "Why might the Reserve Bank have made this decision based
    # on the broader economic context?") routinely get refused by an 8B model
    # that reads "Answer using ONLY what is explicitly stated" and bails on
    # anything analytical — even when the rationale IS stated briefly in the
    # chunks ("to ease potential liquidity stress…").
    #
    # Trigger on question shape (why/how/reasoning/rationale/...) OR on prior
    # history (a follow-up turn is often this shape after pronoun resolution).
    # The directive explicitly OVERRIDES the IMPORTANT REMINDER above so the
    # LLM doesn't keep reading "If you cannot find an explicit answer, say so"
    # and refusing.
    is_synthesis_question = bool(re.search(
        r"\b(?:why|how\s+come|rationale|reasoning|motivation|"
        r"based\s+on\s+(?:the|this|broader|wider|overall)|"
        r"for\s+what\s+reason|what\s+(?:was|is|were)\s+the\s+reason|"
        r"explain\s+(?:the|this|why))\b",
        question,
        re.IGNORECASE,
    ))
    if is_synthesis_question or history:
        followup_directive = (
            "\n\nSYNTHESIS DIRECTIVE (overrides the previous reminder for this "
            "question): This question asks for reasoning, motivation, or "
            "cause. You MAY and SHOULD combine explicitly-stated facts from "
            "the context to construct the answer. If the cause, rationale, or "
            "motivating conditions are stated anywhere in the context (even "
            "briefly, e.g. 'because…', 'in order to…', 'due to…', "
            "'caused by…', 'as a result of…'), quote those snippets and "
            "explain how they answer the question. Do NOT refuse with 'I "
            "don't have enough information' just because the question's "
            "framing sounds analytical or speculative. Refuse ONLY if NO "
            "related facts (no causes, no reasons, no rationale fragments) "
            "appear in the context at all."
        )
    else:
        followup_directive = ""

    return f"""{history_block}Context from documents:
{context_str}

Question: {question}

IMPORTANT REMINDER: Answer using ONLY what is explicitly stated above. Do NOT calculate, infer, or add up values. If a total or summary value is stated in the context, quote it directly. If you cannot find an explicit answer, say so.{followup_directive}{language_reminder}

Answer:"""


def calculate_confidence(chunks: List[Dict[str, Any]], answer_text: str = "") -> float:
    """
    Calculate a comprehensive confidence score based on retrieval signals and answer quality.

    HOW IT WORKS:
    -------------
    Picks the BEST available signal in order of accuracy:
      1. reranker_score  — cross-encoder sigmoid, already 0–1 calibrated
      2. hybrid_score    — FAISS+BM25 blend, ~0–0.5 typical
      3. similarity_score — raw FAISS cosine, ~0.05–0.30 typical

    Each scoring source needs different scaling: a reranker_score of 0.6 means
    "highly relevant" and shouldn't be multiplied by 3.2 (that would saturate
    instantly), while a raw cosine of 0.20 means the same thing and DOES need
    scaling to land near 0.65 confidence.

    We then apply small bonuses for multi-chunk support and small penalties
    for terse answers.
    """

    if not chunks:
        return 0.0

    # 1. Pick the highest-quality signal available across the chunks
    if any(c.get("reranker_score") is not None for c in chunks):
        scores = [c.get("reranker_score") or 0.0 for c in chunks]
        best_score = max(scores)
        # Reranker is already 0–1 sigmoid; use almost directly
        base_confidence = min(best_score * 1.05, 0.95)
        strong_threshold = 0.55
    elif any(c.get("hybrid_score") is not None for c in chunks):
        scores = [c.get("hybrid_score") or 0.0 for c in chunks]
        best_score = max(scores)
        # Hybrid blends FAISS+BM25, typically lower; scale up moderately
        base_confidence = min(best_score * 2.0, 0.90)
        strong_threshold = 0.30
    else:
        scores = [c.get("similarity_score", 0.0) for c in chunks]
        best_score = max(scores)
        # Raw FAISS cosine — original aggressive scaling
        base_confidence = min(best_score * 3.2, 0.85)
        strong_threshold = 0.15

    # 2. Volume bonus: multiple strong chunks → slightly more confident
    strong_chunks = sum(1 for s in scores if s > strong_threshold)
    volume_bonus = min((strong_chunks - 1) * 0.02, 0.10) if strong_chunks > 1 else 0.0

    # 3. Length penalty: a one-sentence answer to a complex question is suspect
    length_penalty = 0.0
    word_count = len(answer_text.split()) if answer_text else 0
    if 0 < word_count < 15:
        length_penalty = -0.15
    elif word_count == 0:
        length_penalty = -0.50

    total = base_confidence + volume_bonus + length_penalty
    return round(max(min(total, 0.95), 0.0), 2)


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
    max_tokens: int = 1024,   # Increased from 512 → prevents truncation when listing multiple items
    language: Optional[str] = None,
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
    
    MAX_CONTEXT_TOKENS = 2500  # increased to aggressively support multi-document cross-reasoning
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

    # ── Language resolution (Phase 2.1) ──
    # If the caller gave us a language (the agent does — it detected on the
    # ORIGINAL user query before query rewriting), trust it. Otherwise detect
    # locally on the question we received.
    #
    # WHY THIS MATTERS:
    # `question` here is often the rewritten query, which may have flipped
    # to a different language than what the user typed (e.g. the rewriter
    # pulled from English chat history when the current query was English
    # but a previous turn was Hindi). Detecting on the rewritten query
    # caused the bug where an English "Summarize the document" returned
    # a Hindi answer.
    if language:
        detected_language = language
        print(f"  🌐 Language: {detected_language} (from agent — original query)")
    else:
        detected_language = detect_language(question)
        print(f"  🌐 Language: {detected_language} (detected from question)")

    # ------------------------------------------------------------------
    # STEP 2: Build the prompt
    # ------------------------------------------------------------------
    user_message = build_prompt(
        question,
        context_chunks,
        history=history,
        response_language=detected_language,
    )

    # Build the per-call system prompt.
    # WHY ORDER MATTERS:
    #   LLMs weight earlier instructions more heavily. Rule 4 of SYSTEM_PROMPT
    #   says "quote verbatim — do not rephrase". For English queries on English
    #   chunks that's correct. For Hindi queries on English chunks, "verbatim"
    #   would keep the answer in English — which is exactly the bug we hit.
    #
    # FIX:
    #   When the query is non-English, PREPEND a categorical language override
    #   at the very top so the LLM reads "answer in Hindi" before it reads
    #   any rule about verbatim quoting. We also explicitly tell it that
    #   translating English context into Hindi is REQUIRED, not optional.
    if detected_language.lower() != "english":
        # Strong directive at the TOP of the system prompt so it dominates
        # the LLM's instruction-following. We can't just say "keep tech terms
        # in original language" — once an LLM is in Hindi mode it transliterates
        # "React" → "रिएक्ट" by default. We must explicitly forbid Devanagari
        # for these tokens and show a concrete correct/incorrect example,
        # which is the most reliable instruction-following signal for 8B models.
        language_directive = (
            f"⚠️ CRITICAL LANGUAGE INSTRUCTION ⚠️\n"
            f"YOUR ENTIRE RESPONSE MUST BE WRITTEN IN {detected_language.upper()}.\n"
            f"The source documents may be in English. You MUST translate the "
            f"information into {detected_language} when writing your answer.\n"
            f"This OVERRIDES any rule below about quoting verbatim — translate "
            f"the meaning faithfully into {detected_language}.\n\n"
            f"🚨 CRITICAL — DO NOT TRANSLITERATE THESE TO {detected_language.upper()} SCRIPT:\n"
            f"The following MUST appear in their ORIGINAL LATIN/ENGLISH SCRIPT, "
            f"never written in the {detected_language} alphabet:\n"
            f"  • Technical terms: React, Node.js, MongoDB, OpenAI, API, FAISS, "
            f"JavaScript, Python, SQL, ML, AI, etc.\n"
            f"  • Company / brand / product names: Google, Microsoft, Anthropic, "
            f"AWS, GitHub, etc.\n"
            f"  • Proper nouns: names of people, places, projects (unless they "
            f"have an established native spelling)\n"
            f"  • File names, page numbers, and any code/identifiers\n\n"
            f"EXAMPLE (Hindi):\n"
            f"  ✅ CORRECT:   \"उन्होंने React और Node.js का उपयोग करके MongoDB से जुड़ने वाला एक app बनाया।\"\n"
            f"  ❌ INCORRECT: \"उन्होंने रिएक्ट और नोड.जेएस का उपयोग करके मोंगोडीबी से जुड़ने वाला एक ऐप बनाया।\"\n\n"
            f"Maintain the citation format ([Page N - filename.pdf]) regardless of language.\n\n"
        )
        system_prompt_for_call = language_directive + SYSTEM_PROMPT
    else:
        # English query → no override needed, keep the original prompt clean
        system_prompt_for_call = SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # STEP 3: Call Groq API
    # ------------------------------------------------------------------
    # The Groq client sends a list of "messages" — same pattern as OpenAI.
    # "system" = the rules/persona for the LLM (now language-aware)
    # "user"   = the actual input (context + question)
    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt_for_call},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )

        # Extract the generated text from the response object
        answer_text = response.choices[0].message.content.strip()
        # TEMP DIAGNOSTIC — surface the opening so we can see why the hedging
        # detector is firing on follow-up questions. Remove once V6.Q2 stabilizes.
        print(f"  📝 LLM answer opening: {answer_text[:300]!r}")

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
    # If the LLM led with "I don't know", mark has_answer = False.
    # We only scan the OPENING (first ~200 chars) — a hedge sentence at the
    # end of a long, substantive answer is a caveat, not a refusal, and
    # killing it caused V6.Q2-style false negatives where the LLM gave the
    # right reasoning then added "I don't have enough information about the
    # broader economic context" as a final note.
    no_info_phrases = [
        "don't have enough information",
        "cannot answer",
        "not mentioned in",
        "not provided in",
        "no information"
    ]
    opening = answer_text[:200].lower()
    has_answer = not any(phrase in opening for phrase in no_info_phrases)

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
    confidence = calculate_confidence(context_chunks, answer_text)
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
