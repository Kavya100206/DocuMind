"""
Agent Service — LangGraph Decision Loop

WHAT DOES THIS FILE DO?
------------------------
This is the brain of Phase 2. It defines the agentic retrieval loop
using LangGraph — a framework for building stateful, multi-step LLM
workflows as a directed graph.

THE BIG PICTURE:
-----------------
Instead of always running the same retrieval steps, the LLM now DECIDES:
  1. Which retrieval tool best suits this question?
  2. After seeing results — is the confidence good enough?
  3. If not — try a different tool? Or fall back to the old pipeline?

This is called an "agent" because the system has agency — it can take
multiple actions, evaluate results, and change its approach.

HOW LANGGRAPH WORKS (the key concepts):
-----------------------------------------

STATE:
  A TypedDict (a typed Python dict) that flows through the entire graph.
  Every node reads from state and writes back to state.
  Think of it as a shared notepad all nodes can read and write to.

NODES:
  Plain Python functions. Each node:
    1. Receives the current state dict
    2. Does some work (LLM call, tool call, etc.)
    3. Returns a dict of UPDATES to the state
  LangGraph merges those updates into the state automatically.

EDGES:
  Connections between nodes. Two types:
  - Normal edge: always go from A to B
  - Conditional edge: call a function that returns which node to go to next

THE GRAPH STRUCTURE:
---------------------

    START
      │
      ▼
  [router_node]  ─── LLM picks a tool and writes it to state
      │
      ▼
  [tool_node]  ─── runs the chosen tool, writes chunks to state
      │
      ▼
  [confidence_check]  ─── conditional edge: evaluate the results
      │
      ├── "generate" ──→ [generation_node] ──→ END
      │                    (confident enough)
      │
      └── "router"   ──→ back to [router_node]  (try another tool)
           or
          "fallback"  ──→ [fallback_node] ──→ END
           (max iterations hit, or all tools tried)

MEMORY RULE:
-------------
The graph is constructed INSIDE run_agent() — a regular function.
Every time a request comes in, a new graph compile is created.
When the function returns, the graph object is garbage collected.
NO global graph. NO shared state between requests.
"""

import json
import re
from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.utils.logger import get_logger
from app.services import agent_tools

logger = get_logger(__name__)

# Maximum number of tool calls before we give up and fall back
MAX_ITERATIONS = 3

# Queries the router should send straight to summarize_document without
# burning an LLM router call. The 8B router has historically picked
# vector_search for these (V5 failure: "Summarize the main themes…" → vector
# search → audit-report chunks → confidence floor not met). Matching on
# clear summary intent avoids that.
_SUMMARY_INTENT_RE = re.compile(
    r"\b(?:summari[sz]e|summary|overview|main\s+themes?|key\s+themes?|"
    r"what\s+is\s+(?:this|the)\s+(?:document|report|paper|pdf)\s+about|"
    r"give\s+me\s+(?:a|an)\s+(?:summary|overview)|"
    r"describe\s+(?:this|the)\s+(?:document|report))\b",
    re.IGNORECASE,
)

# Confidence threshold: if the best chunk's hybrid_score is above this,
# we're satisfied with the retrieval and move to generation.
#
# Bumped 0.15 → 0.25 (Phase 4 hardening). At 0.15, off-topic queries that
# happened to share keywords with the document (e.g. "Prime Minister of
# India" hits governance chunks at score ~0.33) cleared the gate and reached
# the LLM, which sometimes wrote partial answers instead of refusing. 0.25
# is still well below the typical scores of valid factual queries (0.5+).
AGENT_CONFIDENCE_THRESHOLD = 0.25

# ── Hard refusal contract (Phase 1.2 + 2.1) ──
# Used by BOTH generation_node and fallback_node so the refusal shape
# is identical no matter which path ran. The frontend keys off has_answer=False
# and confidence==0.0 to render the red "Out of Scope" banner.
#
# WHY pre-translated and not LLM-translated?
#   The whole point of a hard refusal is to skip the LLM call (faster, no
#   token cost, deterministic). Asking Groq to translate the refusal would
#   defeat that. A small lookup table is the right tradeoff.
#
# WHY only English + Hindi?
#   Hindi is the demo target (multilingual bonus). For any other language
#   we fall back to English — a wrong machine-generated translation would
#   look worse than a correct English one. Easy to extend later.
REFUSAL_MESSAGES = {
    "English": (
        "This information is not present in the uploaded document. "
        "I can only answer based on its contents."
    ),
    "Hindi": (
        "यह जानकारी अपलोड किए गए दस्तावेज़ में मौजूद नहीं है। "
        "मैं केवल इसकी सामग्री के आधार पर ही उत्तर दे सकता हूँ।"
    ),
}


def _refusal_payload(language: str = "English") -> Dict[str, Any]:
    """
    Single source of truth for the hard-refusal response.
    Returns the refusal message in `language` if we have a pre-translated
    version, otherwise falls back to English.
    """
    msg = REFUSAL_MESSAGES.get(language, REFUSAL_MESSAGES["English"])
    return {
        "final_answer": msg,
        "citations": [],
        "confidence": 0.0,
        "has_answer": False,
        "fallback_used": False,
    }

# Phrases the LLM uses when it doesn't actually have the answer but
# tries to hedge instead of refusing cleanly. We treat any match as a
# refusal trigger and overwrite the LLM's hedged response.
#
# Two-layer detection (Phase 4 hardening):
#   1. Static substring list — catches the obvious phrases below verbatim.
#   2. Regex — catches structural variants like "does not specify",
#      "is not directly mentioned", "no specific information" that the
#      static list missed. These slipped through Phase 4 testing on the
#      "Prime Minister of India" query and produced false-positive answers.
HEDGING_PHRASES = [
    "not mentioned", "not present", "i don't know", "i do not know",
    "cannot find", "not found in", "not in the document",
    "not provided", "not available", "no information",
]

HEDGING_REGEX = re.compile(
    r"\b("
    # "I don't / cannot know|find|tell|determine|see"
    r"(?:i\s+)?(?:don'?t|do\s+not|cannot|can'?t)\s+(?:know|find|tell|determine|see)|"
    # "(is/are/was/were) not (directly|explicitly|specifically) mentioned/stated/..."
    r"(?:is\s+|are\s+|was\s+|were\s+)?not\s+(?:directly\s+|explicitly\s+|specifically\s+)?"
    r"(?:mentioned|stated|specified|named|provided|present|available|found|defined|discussed|addressed)|"
    # "does not mention/state/specify/define/indicate/discuss/address"
    r"does\s+not\s+(?:mention|state|specify|name|provide|define|indicate|address|discuss|tell)|"
    # "no specific|explicit|direct mention|reference|information|details"
    r"no\s+(?:specific|explicit|direct)\s+(?:mention|reference|information|details)|"
    # bare phrases
    r"cannot\s+find|"
    r"no\s+information"
    r")\b",
    re.IGNORECASE,
)


def _is_hedging(answer: str) -> bool:
    """
    Returns True if the LLM's answer contains hedging language indicating it
    doesn't actually have the answer grounded in the context.

    Scope: we only inspect the OPENING (first 250 chars) of the answer. A
    hedge in the lead means the whole answer is a refusal; a hedge late in
    a long answer is usually a caveat sitting alongside real content, and
    killing the entire response caused V6.Q2-style false negatives.

    Layer-1 checks HEDGING_PHRASES (verbatim), layer-2 runs HEDGING_REGEX
    (structural variants). Both run only on the opening window.
    """
    if not answer:
        return False
    # Window narrowed 250 → 180. V2's failure mode: real answer in sentence 1,
    # caveat "is not explicitly stated" tacked on at char ~215. Wider windows
    # caught the caveat and refused a correct answer.
    opening = answer[:180].lower()
    if any(p in opening for p in HEDGING_PHRASES):
        return True
    if HEDGING_REGEX.search(opening):
        return True
    return False


# ---------------------------------------------------------------------------
# GROUNDING GATE (lexical heuristics only)
# ---------------------------------------------------------------------------
#
# Catches confident-but-ungrounded answers that pass every other gate because
# the LLM speaks fluently using its training data. Three checks today:
#   1. Who-question evasion — "who is X?" answered with a title but no name.
#   2. Qualifier-distance — strong "explicitly stated" claim about a date
#      that is in the chunks but not associated with the question's
#      qualifying scope (I3 hallucination trap).
#   3. (Plus a non-Latin skip so Hindi answers aren't false-refused on
#      lexical overlap with English chunks.)
#
# Previously this had an LLM-as-judge layer (8B Llama) that fired on short
# factoid answers. Removed because it produced false negatives on partially-
# grounded answers (the headline fact was in chunks but auxiliary details
# like exact dates weren't, and the YES/NO contract couldn't express "mostly
# grounded"), and its rate-limit pressure on the Groq free tier caused 429
# retries that doubled latency. The qualifier-distance check covers the I3
# pattern that the judge was meant to catch, deterministically and for free.

# Generic proper-noun-shaped tokens we IGNORE during the injection check.
# These are titles, common places, months, and report-structure words that
# legitimately appear capitalized in answers without being names that need
# verification.
_GENERIC_PROPER = {
    # Articles / pronouns / determiners
    "i", "the", "a", "an", "and", "or", "for", "of", "to", "in", "on", "at",
    "by", "from", "with", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "he", "she", "they", "we", "you",
    # Honorifics
    "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "lord", "lady",
    "hon", "honble", "honorable", "shri", "smt",
    # Common org / title vocabulary in the RBI doc (and most reports)
    "india", "indian", "reserve", "bank", "rbi", "government", "republic",
    "ministry", "department", "annual", "report", "central", "monetary",
    "policy", "committee", "chairman", "president", "secretary",
    "prime", "minister", "deputy", "chief", "director", "general",
    "governor", "officer", "executive", "board",
    # Big Indian cities + generic place words
    "mumbai", "delhi", "kolkata", "chennai", "bengaluru", "bangalore",
    "hyderabad", "pune", "ahmedabad", "new", "national", "international",
    "european", "american", "asian", "global",
    # Months / days
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # Document-structure words
    "page", "source", "table", "chart", "figure", "section", "chapter",
    "part", "appendix", "annex", "schedule", "annexure",
    # Discourse markers that often start sentences capitalized
    "however", "therefore", "moreover", "furthermore", "additionally",
    "while", "during", "based",
}


def _grounding_gate(
    question: str,
    answer: str,
    chunks: List[Dict[str, Any]],
) -> bool:
    """
    Lexical grounding check. Returns True to keep the answer, False to refuse.
    See module-level comment above for design rationale.
    """
    if not answer or not chunks:
        return True  # other gates handle empty cases

    # ── Skip non-Latin answers (Hindi etc.) ──
    # Lexical overlap with English chunks is near zero by construction;
    # running this check would false-refuse every Hindi answer.
    non_latin = sum(1 for ch in answer if ord(ch) > 127 and ch.isalpha())
    if non_latin > len(answer) * 0.3:
        return True

    # NOTE: We previously had a proper-noun injection check — capitalized
    # words in the answer not appearing in chunks → refuse. We removed it
    # because LLMs capitalize discourse markers ("Note", "According"),
    # business terms ("Margin", "Reform"), and morphological variants
    # ("Reduction" vs "reduce") faster than _GENERIC_PROPER could ever
    # enumerate. The false-positive rate killed V3/V4. Hallucinated NAMES
    # (e.g. "Modi" injected by the LLM) are still caught by the who-evasion
    # check below.

    # ── Who-question evasion ──
    # If the user asked "who is/was X?", the answer must contain a
    # personal-name pattern. Two+ capitalized words in sequence where at
    # least one is NOT a generic title/place. Catches I1: the LLM said
    # 'Hon\'ble Prime Minister of India' (all generic words) without naming
    # the actual person.
    if re.search(r"\bwho\s+(?:is|was|are|were)\b", question.lower()):
        name_seqs = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", answer)
        has_personal_name = any(
            not all(tok.lower() in _GENERIC_PROPER for tok in seq.split())
            for seq in name_seqs
        )
        if not has_personal_name:
            logger.info(
                "[Grounding] REFUSE (who-question evasion): "
                "answer to 'who' question contains no personal-name pattern"
            )
            return False

    # ── Qualifier-distance check (I3 hallucination trap) ──
    if not _qualifier_distance_check(question, answer, chunks):
        return False

    return True


# ---------------------------------------------------------------------------
# QUALIFIER-DISTANCE CHECK (I3 hallucination trap)
# ---------------------------------------------------------------------------
#
# Specifically targets the I3 failure shape: the LLM gives a confident date
# answer with a strong "as explicitly stated"/"according to the report" claim,
# but the date in the chunks is associated with a different scope than what
# the question asked. Example from our test suite:
#
#   Q: "What is the official rollout date for e₹-Retail in rural India
#       announced in this report?"
#   Chunks: "Operationalisation of CBDC - Pilot for CBDC - retail (e₹-R)
#            was launched on December 1, 2022."
#   A: "The official rollout date for e₹-Retail in rural India is
#      December 1, 2022, as explicitly stated in the report on page 302..."
#
# The date is in the chunks, but it's not associated with "rural" — the
# question's distinguishing qualifier. A good heuristic must catch that.
#
# Why this is narrow on purpose:
#   - Triggers ONLY when the answer makes a strong grounding claim phrase.
#     Without that signal, V2-style multi-fact answers (which mention dates
#     incidentally) would be over-refused.
#   - Only checks dates within ±100 chars of the claim phrase. So an answer
#     that mentions one grounded date with "as stated" plus a separate
#     auxiliary date elsewhere only validates the asserted one.
#
# Caveats:
#   - English-only by token comparison. Skipped if the answer is Hindi
#     (handled by the non-Latin skip earlier in _grounding_gate).
#   - Date pattern is "Month DD, YYYY" — covers the common case in this
#     domain but not numeric percentages or year-only mentions.

# Phrases that signal the LLM is asserting strong grounding for a fact.
# Trigger this check only when one of these appears in the answer.
_GROUNDING_CLAIM_PHRASES = (
    "explicitly stated",
    "as stated in the report",
    "as stated in the document",
    "according to the report",
    "according to the document",
    "as mentioned in the report",
    "as mentioned in the document",
    "as noted in the report",
    "as noted in the document",
    "as per the report",
    "as per the document",
)

# Month-DD-YYYY (with or without comma): "December 1, 2022", "April 1 2025".
_FACTOID_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)


def _qualifier_distance_check(
    question: str,
    answer: str,
    chunks: List[Dict[str, Any]],
) -> bool:
    """
    Returns True (KEEP) by default; False (REFUSE) when the answer makes a
    strong grounding claim about a date that isn't actually associated with
    the question's qualifying scope in the chunks.
    """
    answer_lower = answer.lower()

    # Locate every claim-phrase occurrence (start, end) in the answer.
    claim_spans: List[tuple] = []
    for phrase in _GROUNDING_CLAIM_PHRASES:
        i = 0
        while True:
            idx = answer_lower.find(phrase, i)
            if idx == -1:
                break
            claim_spans.append((idx, idx + len(phrase)))
            i = idx + 1
    if not claim_spans:
        return True

    # Dates anywhere in the answer. We then keep only those within ±100 chars
    # of a claim phrase — those are the ones the LLM is asserting are grounded.
    PROXIMITY = 100
    suspect_dates: List[str] = []
    for m in _FACTOID_DATE_RE.finditer(answer):
        d_start, d_end = m.start(), m.end()
        for c_start, c_end in claim_spans:
            if d_end >= c_start - PROXIMITY and d_start <= c_end + PROXIMITY:
                suspect_dates.append(m.group())
                break
    if not suspect_dates:
        return True

    # Question content tokens (rough): drop short words, stopwords, generic
    # proper nouns, then keep what's left as candidate qualifiers.
    raw_q_tokens = re.split(r"\W+", question, flags=re.UNICODE)
    q_tokens = [
        t.lower() for t in raw_q_tokens
        if t and len(t) > 2
        and t.lower() not in _RELEVANCE_STOP
        and t.lower() not in _GENERIC_PROPER
    ]
    if len(q_tokens) < 2:
        return True

    chunk_text_lower = " ".join((c.get("text") or "") for c in chunks).lower()

    # Testable qualifiers: tokens that actually appear in the chunks. If a
    # qualifier is nowhere in the chunks we can't assess proximity for it,
    # so it's excluded from both the numerator and the denominator.
    testable = [t for t in q_tokens if t in chunk_text_lower]
    if not testable:
        return True

    WINDOW_CHARS = 100
    THRESHOLD = 0.66  # need ≥~2/3 of testable qualifiers within window

    for date in suspect_dates:
        date_lower = date.lower()
        # Find every occurrence of this date in the chunk text.
        occurrences: List[int] = []
        i = 0
        while True:
            idx = chunk_text_lower.find(date_lower, i)
            if idx == -1:
                break
            occurrences.append(idx)
            i = idx + 1

        if not occurrences:
            # Strong claim about a date that doesn't appear in the chunks
            # at all → fabrication.
            logger.info(
                f"[Grounding] REFUSE (qualifier-distance): claimed date "
                f"'{date}' not present in chunks despite grounding claim"
            )
            return False

        # Pick the occurrence with the best qualifier coverage.
        best_coverage = 0.0
        for occ in occurrences:
            window = chunk_text_lower[
                max(0, occ - WINDOW_CHARS) : occ + len(date) + WINDOW_CHARS
            ]
            matched = sum(1 for t in testable if t in window)
            best_coverage = max(best_coverage, matched / len(testable))

        if best_coverage < THRESHOLD:
            logger.info(
                f"[Grounding] REFUSE (qualifier-distance): claimed date "
                f"'{date}' not co-located with question qualifiers "
                f"(best coverage {best_coverage:.0%}, "
                f"testable={testable[:6]})"
            )
            return False

    return True


# Stopwords + question words filtered out of the query when computing topical
# relevance. We keep nouns, verbs of substance, and proper nouns.
_RELEVANCE_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "about", "into", "through", "during",
    "what", "which", "who", "whom", "whose", "how", "many", "much", "why",
    "when", "where", "this", "that", "these", "those", "and", "or", "but",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "few", "i", "you", "we", "they", "he", "she", "it", "its", "my", "your",
    "our", "their", "his", "her", "any", "all", "can", "could", "should",
    "would", "may", "might", "shall", "tell", "give", "show", "find", "get",
    "need", "want", "like", "say", "said",
}


def _has_topical_match(query: str, chunks: List[Dict[str, Any]]) -> bool:
    """
    Topical relevance gate (Phase 4 hardening, v2 — layered).

    Catches the "completely off-topic" failure mode while still allowing
    legitimate focused queries through. Uses two layers:

      LAYER A (strong) — any query content bigram appears ≥ 2 times in the
        normalized chunk text. Indicates the query's topic phrase is
        repeatedly discussed.
      LAYER B (fallback) — at least 60% of unique query content unigrams
        appear in chunks. Catches focused queries whose bigrams are rare by
        phrasing but whose topic vocabulary is well-represented (e.g.
        "change in CRR during 2024-25" has bigrams that may appear once,
        but the unigram "crr" appears many times in CRR-related chunks).

    PASS if either layer is satisfied. REFUSE only when BOTH fail —
    indicating the chunks share no meaningful vocabulary with the query
    (e.g. "Who is Albert Einstein?" against an RBI annual report).

    For queries where bigrams pass-through but the chunks only MENTION the
    topic in passing (e.g. "Prime Minister of India"), this gate is by
    design permissive — Layer B will pass on common doc terminology, and
    we then rely on the LLM hedging detection (HEDGING_REGEX in
    _is_hedging) to catch the LLM's "the document does not specify..."
    response.

    Unicode-aware (\\W+ split with re.UNICODE) so Hindi queries tokenize
    correctly.

    Returns:
        True  → chunks pass the relevance gate; proceed to LLM.
        False → chunks fail; caller should refuse.
    """
    if not chunks:
        return False

    tokens = [
        t.lower()
        for t in re.split(r"\W+", query, flags=re.UNICODE)
        if t and len(t) > 1 and t.lower() not in _RELEVANCE_STOP
    ]

    if len(tokens) < 2:
        return True  # too few content words to make a meaningful decision

    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    if not bigrams:
        return True

    # Normalize chunk text the same way we normalized the query — split on
    # \W+ then re-join with single spaces. This makes hyphenated tokens like
    # "2024-25" tokenize to ["2024", "25"] on BOTH sides, so a query bigram
    # "during 2024" matches a chunk containing "during 2024-25".
    raw = " ".join((c.get("text") or "") for c in chunks)
    chunk_tokens = [t.lower() for t in re.split(r"\W+", raw, flags=re.UNICODE) if t]
    chunks_norm = " ".join(chunk_tokens)
    chunk_token_set = set(chunk_tokens)

    # LAYER A: strong topical match — any bigram repeated ≥ 2 times
    if any(chunks_norm.count(bg) >= 2 for bg in bigrams):
        return True

    # LAYER B: broad keyword coverage — most unique unigrams present
    unique = set(tokens)
    coverage = sum(1 for t in unique if t in chunk_token_set) / len(unique)
    return coverage >= 0.6


def _best_score(chunks: List[Dict[str, Any]]) -> float:
    """Best score across chunks, preferring reranker > hybrid > similarity."""
    if not chunks:
        return 0.0
    return max(
        c.get("reranker_score") or c.get("hybrid_score") or c.get("similarity_score", 0.0)
        for c in chunks
    )


def _build_chunk_trace(
    chunks: List[Dict[str, Any]],
    selected_count: int,
) -> List[Dict[str, Any]]:
    """
    Build the per-chunk debug trace (Phase 3.1).

    Returns one entry per retrieved chunk with the FULL score breakdown
    (FAISS, BM25, hybrid, reranker) so the user can see *why* each chunk
    was ranked where it was.

    `selected_count` is how many top-ranked chunks made it into the LLM's
    prompt — the rest are marked selected=False. This approximates the
    behavior of llm_service's token-based context packing without
    duplicating the packing logic here.
    """
    sorted_chunks = sorted(
        chunks,
        key=lambda c: c.get("reranker_score") or c.get("hybrid_score") or c.get("similarity_score", 0),
        reverse=True,
    )
    trace = []
    for i, c in enumerate(sorted_chunks):
        text = c.get("text") or ""
        # chunk_index doubles as a dedup key elsewhere and may carry a UUID
        # string for chunks loaded from a rebuilt FAISS index (no original
        # int index is persisted on the Chunk DB model). The Pydantic
        # response field is Optional[int], so coerce non-ints to None here.
        ci = c.get("chunk_index")
        trace.append({
            "rank":           i + 1,
            "chunk_index":    ci if isinstance(ci, int) else None,
            "page":           c.get("page_number"),
            "document_name":  c.get("document_name"),
            "text_preview":   (text[:150] + "...") if len(text) > 150 else text,
            # similarity_score is the FAISS cosine after lexical boost — closest
            # thing we have to a raw "FAISS score" exposed to the user.
            "faiss_score":    c.get("similarity_score"),
            "bm25_score":     c.get("bm25_score"),
            "hybrid_score":   c.get("hybrid_score"),
            "reranker_score": c.get("reranker_score"),
            "selected":       i < selected_count,
        })
    return trace


def _build_trace(
    tool_calls_made: List[str],
    retrieved_chunks: List[Dict[str, Any]],
    iterations: int,
    fallback_used: bool,
) -> Dict[str, Any]:
    """
    Build the observability trace returned alongside the answer (Phase 2.3).

    The trace is what the frontend renders in the collapsible
    "🔍 Retrieval Info" panel — it tells the user (and the demo audience)
    HOW the answer was produced: which retrieval tool, how confident the
    top chunk was, how many chunks fed the LLM, and how many router
    iterations the agent burned.

    `tool_used` semantics:
      - If fallback ran, it dominates → "fallback (full pipeline)"
      - Otherwise, the LAST tool the agent tried is the one whose results
        cleared the confidence gate. Earlier tools, by definition, didn't.
      - If no tools ran at all (degenerate / error), report "none".
    """
    if fallback_used:
        tool_used = "fallback (full pipeline)"
    elif tool_calls_made:
        tool_used = tool_calls_made[-1]
    else:
        tool_used = "none"

    return {
        "tool_used":        tool_used,
        "tools_tried":      list(tool_calls_made),
        "top_score":        round(_best_score(retrieved_chunks), 4),
        "chunks_retrieved": len(retrieved_chunks),
        "iterations":       iterations,
        "fallback_used":    fallback_used,
    }


# ---------------------------------------------------------------------------
# STATE SCHEMA
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    The shared notepad that flows through every node in the graph.

    EVERY node receives a copy of this and returns a dict of updates.
    LangGraph automatically merges those updates.

    Fields:
    --------
    query          : The user's original question (never changes)
    document_id    : Optional — restrict search to one document
    tool_calls_made: List of tool names already tried this request
                     (so the router doesn't repeat itself)
    retrieved_chunks: The best chunks found so far
                     (Annotated[list, operator.add] means each node's
                      results are APPENDED, not replaced)
    final_answer   : Populated by generation_node, None until then
    citations      : Citation list for the final answer
    confidence     : Float 0–1 representing answer quality
    has_answer     : Whether a real answer was found
    iterations     : How many tool calls have been made so far
    chosen_tool    : Which tool the router picked this iteration
    chosen_args    : Arguments for that tool (e.g. {"query": "..."})
    fallback_used  : Whether we fell back to the old pipeline
    """
    query: str
    # Detected language of the user's query — used to translate refusal
    # messages so an out-of-scope Hindi query gets refused in Hindi.
    # Set once by run_agent() at the top, then read by generation_node /
    # fallback_node when constructing a refusal.
    query_language: str
    # Conversation history — list of {"role", "content"} dicts. Threaded
    # through to generation so multi-turn follow-ups can reference prior
    # turns when the LLM writes its answer.
    history: List[Dict[str, str]]
    document_id: Optional[str]
    tool_calls_made: List[str]
    # Annotated[list, operator.add] is a LangGraph reducer:
    # instead of replacing retrieved_chunks, new chunks get APPENDED.
    retrieved_chunks: Annotated[List[Dict[str, Any]], operator.add]
    final_answer: Optional[str]
    citations: List[Dict[str, Any]]
    confidence: float
    has_answer: bool
    iterations: int
    chosen_tool: Optional[str]
    chosen_args: Optional[Dict[str, Any]]
    fallback_used: bool
    # Phase 3.1: when True, generation/fallback nodes build per-chunk trace
    debug_mode: bool
    # Phase 3.2: router captures its tool-choice reasoning here
    routing: Optional[Dict[str, Any]]
    # Phase 3.1: populated by generation/fallback when debug_mode is on
    chunk_trace: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# NODE 1: ROUTER
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> Dict[str, Any]:
    """
    The LLM-powered decision maker.

    Looks at the question + which tools have already been tried,
    then picks the best next tool to call.

    HOW THE LLM MAKES THE DECISION:
    ---------------------------------
    We send a structured prompt with:
      - The question
      - Which tools have already been tried
      - Short description of each available tool
      - Instruction to respond with JSON: {"tool": "...", "args": {...}}

    The LLM responds with JSON telling us:
      {"tool": "vector_search", "args": {"query": "..."}}
      {"tool": "keyword_search", "args": {"query": "..."}}
      {"tool": "summarize_document", "args": {"doc_id": "..."}}
      {"tool": "fallback", "args": {}}   ← when it gives up

    WHY JSON OUTPUT?
    ----------------
    JSON is machine-readable and unambiguous. We parse it with
    json.loads() to get the tool name and arguments as Python objects.
    If the LLM produces malformed JSON, we catch it and fall back.
    """
    from groq import Groq

    query = state["query"]
    already_tried = state["tool_calls_made"]
    iterations = state["iterations"]
    doc_id = state.get("document_id")

    logger.info(f"[Router] Iteration {iterations + 1}/{MAX_ITERATIONS} | tried: {already_tried}")

    # Build the list of still-available tools
    all_tools = ["vector_search", "keyword_search", "summarize_document"]
    available = [t for t in all_tools if t not in already_tried]

    if not available or iterations >= MAX_ITERATIONS:
        logger.info("[Router] No tools left or max iterations — choosing fallback")
        return {
            "chosen_tool": "fallback",
            "chosen_args": {},
            "iterations": iterations + 1,
        }

    # ── Deterministic pre-route: summary intent → summarize_document ──
    # The 8B router LLM has been unreliable here (picks vector_search and
    # retrieves random pages). When the query has clear summary intent AND we
    # have a doc_id AND summarize_document is still available, skip the LLM
    # and route directly. Saves a Groq call too.
    if (
        doc_id
        and "summarize_document" in available
        and _SUMMARY_INTENT_RE.search(query)
    ):
        logger.info("[Router] Summary intent detected — routing to summarize_document (no LLM call)")
        update: Dict[str, Any] = {
            "chosen_tool": "summarize_document",
            "chosen_args": {"doc_id": doc_id},
            "iterations": iterations + 1,
        }
        if iterations == 0:
            update["routing"] = {
                "tool_chosen": "summarize_document",
                "reason": "summary intent matched (deterministic pre-route)",
            }
        return update

    # Build the summarize_document tool description with the doc_id if known
    summarize_note = (
        f'summarize_document — retrieve broad content from doc_id="{doc_id}" for summaries/overviews.'
        if doc_id
        else "summarize_document — retrieve broad content for summaries/overviews (requires a doc_id)."
    )

    router_prompt = f"""You are a retrieval router for a document Q&A system.
Given a question, pick the BEST next retrieval tool AND briefly explain why.

QUESTION: {query}

AVAILABLE TOOLS:
- vector_search — semantic FAISS search. Best for: conceptual, meaning-based questions.
  Args: {{"query": "<search query>"}}
- keyword_search — exact BM25 keyword match. Best for: specific names, codes, identifiers.
  Args: {{"query": "<keyword query>"}}
- {summarize_note}
  Args: {{"doc_id": "<document uuid>"}}

ALREADY TRIED (do NOT pick these again): {already_tried if already_tried else "none"}

Rules:
1. Pick exactly ONE tool from AVAILABLE TOOLS
2. DO NOT pick a tool that is in ALREADY TRIED
3. Return pure JSON only, no explanation outside the JSON:
   {{"tool": "<tool_name>", "args": {{...}}, "reason": "<one short sentence>"}}
4. The "reason" must be a single short sentence (under 15 words) explaining
   the choice — e.g. "query mentions a specific entity name" or
   "broad request needing semantic understanding"
5. If "summarize_document" is the best choice but no doc_id is known, pick "vector_search" instead
6. Default to "vector_search" if unsure"""

    client = Groq(api_key=settings.GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a routing assistant. Always respond with valid JSON only."},
                {"role": "user", "content": router_prompt},
            ],
            max_tokens=80,
            temperature=0.0,   # Zero temperature = deterministic, no creativity
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if the LLM wrapped in ```json ... ```
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        decision = json.loads(raw)
        tool_name = decision.get("tool", "vector_search")
        tool_args = decision.get("args", {})
        # Phase 3.2: capture the router's reasoning (one-line explanation)
        tool_reason = decision.get("reason", "").strip() or "no reason given"

        # Safety: if LLM hallucinated a tool name, default to first available
        if tool_name not in available and tool_name != "fallback":
            logger.warning(f"[Router] LLM picked unknown tool '{tool_name}' — defaulting to '{available[0]}'")
            tool_name = available[0]
            tool_args = {"query": query}
            tool_reason = "LLM picked an unknown tool — defaulted to vector_search"

        logger.info(f"[Router] Decision: tool={tool_name} | reason: {tool_reason}")

    except Exception as e:
        logger.warning(f"[Router] LLM routing failed ({e}) — defaulting to vector_search")
        tool_name = available[0]
        tool_args = {"query": query}
        tool_reason = f"router failed ({type(e).__name__}) — defaulted"

    # Only set `routing` on the FIRST iteration — that's the primary tool
    # decision the user sees in the UI. Later iterations (loop attempts)
    # are captured implicitly in `tools_tried`.
    update: Dict[str, Any] = {
        "chosen_tool": tool_name,
        "chosen_args": tool_args,
        "iterations": iterations + 1,
    }
    if iterations == 0:
        update["routing"] = {"tool_chosen": tool_name, "reason": tool_reason}
    return update


# ---------------------------------------------------------------------------
# NODE 2: TOOL EXECUTOR
# ---------------------------------------------------------------------------

def tool_node(state: AgentState, db: Session) -> Dict[str, Any]:
    """
    Executes the tool chosen by the router.

    Takes `chosen_tool` + `chosen_args` from state, calls the
    corresponding function in agent_tools.py, and writes the
    resulting chunks back to state.

    WHY A SEPARATE NODE?
    ---------------------
    Clean separation:
      - router_node   = WHAT to call (LLM decision)
      - tool_node     = HOW to call it (actual execution)

    This makes it easy to add new tools in the future — just
    add a new branch in the if/elif here and a function in agent_tools.py.

    MEMORY RULE:
    ------------
    The db session is passed in from the request handler.
    tool_node does NOT create or close sessions — that's the caller's job.
    """
    tool_name = state["chosen_tool"]
    tool_args = state.get("chosen_args") or {}
    query = state["query"]
    doc_id = state.get("document_id")

    logger.info(f"[Tool Executor] Running tool: {tool_name}")

    chunks: List[Dict[str, Any]] = []

    try:
        if tool_name == "vector_search":
            search_query = tool_args.get("query", query)
            chunks = agent_tools.vector_search(
                query=search_query,
                db=db,
                document_id=doc_id,
            )

        elif tool_name == "keyword_search":
            search_query = tool_args.get("query", query)
            chunks = agent_tools.keyword_search(
                query=search_query,
                db=db,
                document_id=doc_id,
            )

        elif tool_name == "summarize_document":
            # Use doc_id from args if provided, otherwise from state
            target_doc_id = tool_args.get("doc_id") or doc_id
            if not target_doc_id:
                logger.warning("[Tool Executor] summarize_document called without doc_id — skipping")
                chunks = []
            else:
                chunks = agent_tools.summarize_document(
                    doc_id=target_doc_id,
                    db=db,
                    question=query,
                )

        else:
            logger.warning(f"[Tool Executor] Unknown tool: {tool_name}")
            chunks = []

    except Exception as e:
        logger.error(f"[Tool Executor] Tool '{tool_name}' raised: {e}", exc_info=True)
        chunks = []

    logger.info(f"[Tool Executor] Tool '{tool_name}' returned {len(chunks)} chunks")

    return {
        # Annotated[list, operator.add] → these get APPENDED to existing chunks
        "retrieved_chunks": chunks,
        # Track which tools have been tried so router won't repeat
        "tool_calls_made": state["tool_calls_made"] + [tool_name],
    }


# ---------------------------------------------------------------------------
# CONDITIONAL EDGE: CONFIDENCE CHECK
# ---------------------------------------------------------------------------

def check_confidence(state: AgentState) -> Literal["generate", "router", "fallback"]:
    """
    Decides what to do after a tool call.

    This is a CONDITIONAL EDGE — not a node (it doesn't modify state).
    It just returns a string that tells LangGraph which node to go to next.

    DECISION LOGIC:
    ---------------
    "generate"  → We have enough good chunks → proceed to answer generation
    "router"    → Chunks are weak or missing → try another tool
    "fallback"  → Max iterations hit OR all tools tried → use old pipeline

    HOW CONFIDENCE IS MEASURED:
    ----------------------------
    We take the single best hybrid_score (or similarity_score) from ALL
    chunks collected so far across all tool calls.

    If that score is above AGENT_CONFIDENCE_THRESHOLD (0.15):
        → We're confident enough. Generate the answer.
    Else:
        → Try another tool, unless we've run out.
    """
    all_chunks = state["retrieved_chunks"]
    iterations = state["iterations"]
    tried = state["tool_calls_made"]

    logger.info(
        f"[Confidence Check] {len(all_chunks)} total chunks, "
        f"{iterations} iterations, tried: {tried}"
    )

    # No tools left to try
    all_tools = {"vector_search", "keyword_search", "summarize_document"}
    all_tried = all_tools.issubset(set(tried))

    if iterations >= MAX_ITERATIONS or all_tried:
        logger.info("[Confidence Check] → fallback (max iterations or all tools tried)")
        return "fallback"

    if not all_chunks:
        logger.info("[Confidence Check] → router (no chunks yet)")
        return "router"

    # Find the best score across all collected chunks
    best_score = max(
        c.get("hybrid_score") or c.get("similarity_score", 0.0)
        for c in all_chunks
    )

    logger.info(f"[Confidence Check] Best score across {len(all_chunks)} chunks: {best_score:.4f}")

    if best_score >= AGENT_CONFIDENCE_THRESHOLD:
        logger.info(f"[Confidence Check] → generate (score {best_score:.4f} ≥ threshold {AGENT_CONFIDENCE_THRESHOLD})")
        return "generate"
    else:
        logger.info(f"[Confidence Check] → router (score {best_score:.4f} < threshold {AGENT_CONFIDENCE_THRESHOLD})")
        return "router"


# ---------------------------------------------------------------------------
# NODE 3: GENERATION
# ---------------------------------------------------------------------------

def generation_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate the final answer using retrieved chunks.

    HARD REFUSAL LOGIC (Phase 1.2):
    --------------------------------
    Before calling the LLM, we check two conditions:
      1. Are there ANY chunks at all?
      2. Is the best chunk score above AGENT_CONFIDENCE_THRESHOLD (0.15)?
    If either fails → return a hardcoded refusal WITHOUT calling Groq.

    WHY hard-code the refusal instead of letting the LLM say it?
    - Saves a Groq API call (faster + no token cost)
    - LLM sometimes hedges ("I'm not sure...") instead of a clean refusal
    - Our structured refusal has has_answer=False, confidence=0.0
      which the frontend can check reliably to show the red banner
    """
    from app.services import llm_service

    query = state["query"]
    all_chunks = state["retrieved_chunks"]
    # Refusals must come back in the user's language (Phase 2.1)
    lang = state.get("query_language", "English")

    # Deduplicate by (document_id, chunk_index)
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        key = (c.get("document_id"), c.get("chunk_index"))
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    unique_chunks.sort(
        key=lambda c: c.get("reranker_score") or c.get("hybrid_score") or c.get("similarity_score", 0),
        reverse=True,
    )

    # ── Hard Refusal #1: No chunks found at all ──
    if not unique_chunks:
        logger.info("[Generation] Hard refusal: no chunks retrieved")
        return _refusal_payload(lang)

    # ── Hard Refusal #2: Best score below confidence threshold ──
    best_score = _best_score(unique_chunks)
    if best_score < AGENT_CONFIDENCE_THRESHOLD:
        logger.info(f"[Generation] Hard refusal: best score {best_score:.4f} < {AGENT_CONFIDENCE_THRESHOLD}")
        return _refusal_payload(lang)

    # ── Hard Refusal #2.5: Topical relevance check (Phase 4 hardening) ──
    # Even if best_score clears the threshold, retrieval may have surfaced
    # chunks that MENTION the query keywords without being ABOUT them.
    # Require at least one query bigram to appear repeatedly in the chunks.
    if not _has_topical_match(query, unique_chunks):
        logger.info("[Generation] Hard refusal: query bigrams not topical in retrieved chunks (mentioned-in-passing)")
        return _refusal_payload(lang)

    logger.info(f"[Generation] Generating answer from {len(unique_chunks)} unique chunks")

    result = llm_service.generate_answer(
        question=query,
        context_chunks=unique_chunks,
        history=state.get("history") or [],
        # Pass the language detected from the ORIGINAL user query.
        # `query` here is the rewritten effective_query, which can be in a
        # different language than the user actually typed (rewriter pulls
        # from English chat history etc.). Trusting state["query_language"]
        # ensures the answer matches what the user wrote.
        language=lang,
    )

    # Phase 3.1 — Build per-chunk debug trace if debug_mode is on.
    # We approximate "selected" as the top 8 by reranker rank (matches the
    # typical packed-context size of llm_service for our token budget).
    chunk_trace_update: Dict[str, Any] = {}
    if state.get("debug_mode"):
        chunk_trace_update["chunk_trace"] = _build_chunk_trace(unique_chunks, selected_count=8)

    # ── Hard Refusal #3: LLM hedged or returned has_answer=False ──
    if not result.get("has_answer") or _is_hedging(result.get("answer") or ""):
        logger.info("[Generation] Hard refusal: LLM hedged or returned has_answer=False")
        return {**_refusal_payload(lang), **chunk_trace_update}

    # ── Hard Refusal #4: Grounding gate (lexical heuristics) ──
    if not _grounding_gate(query, result["answer"], unique_chunks):
        return {**_refusal_payload(lang), **chunk_trace_update}

    return {
        "final_answer": result["answer"],
        "citations": result["citations"],
        "confidence": result["confidence"],
        "has_answer": result["has_answer"],
        "fallback_used": False,
        **chunk_trace_update,
    }


# ---------------------------------------------------------------------------
# NODE 4: FALLBACK
# ---------------------------------------------------------------------------

def fallback_node(state: AgentState, db: Session) -> Dict[str, Any]:
    """
    Safety net — runs the original pipeline when the agent fails.

    WHEN DOES THIS RUN?
    --------------------
    - Agent exhausted all 3 iterations without good results
    - All 3 tools were tried and none exceeded the confidence threshold
    - Router LLM call failed completely (network error, bad response)

    WHAT IT DOES:
    -------------
    Runs your existing search_chunks() → generate_answer() pipeline —
    exactly as qa_controller.py did before Phase 2.
    The response schema is identical, so the frontend never knows which
    path was taken.
    """
    from app.services import retrieval_service, llm_service

    query = state["query"]
    doc_id = state.get("document_id")
    # Refusals must come back in the user's language (Phase 2.1)
    lang = state.get("query_language", "English")

    logger.info(f"[Fallback] Running original pipeline for: '{query[:50]}'")

    # Use the full pipeline (with cross-encoder reranker)
    chunks = retrieval_service.search_chunks(
        query=query,
        db=db,
        document_id=doc_id,
    )

    # ── Hard Refusal #1: No chunks at all ──
    # The fallback path must enforce the same refusal contract as generation_node.
    # Without this, an off-topic query that exhausts the agent loop would slip
    # through with whatever the LLM hallucinates.
    if not chunks:
        logger.info("[Fallback] Hard refusal: no chunks retrieved")
        return {**_refusal_payload(lang), "fallback_used": True, "retrieved_chunks": []}

    # ── Hard Refusal #2: Best score below confidence threshold ──
    best_score = _best_score(chunks)
    if best_score < AGENT_CONFIDENCE_THRESHOLD:
        logger.info(f"[Fallback] Hard refusal: best score {best_score:.4f} < {AGENT_CONFIDENCE_THRESHOLD}")
        return {**_refusal_payload(lang), "fallback_used": True, "retrieved_chunks": chunks}

    # ── Hard Refusal #2.5: Topical relevance check (Phase 4 hardening) ──
    if not _has_topical_match(query, chunks):
        logger.info("[Fallback] Hard refusal: query bigrams not topical in retrieved chunks (mentioned-in-passing)")
        return {**_refusal_payload(lang), "fallback_used": True, "retrieved_chunks": chunks}

    result = llm_service.generate_answer(
        question=query,
        context_chunks=chunks,
        history=state.get("history") or [],
        language=lang,  # use the language from the original user query
    )

    # Phase 3.1 — Build per-chunk debug trace if debug_mode is on.
    chunk_trace_update: Dict[str, Any] = {}
    if state.get("debug_mode"):
        chunk_trace_update["chunk_trace"] = _build_chunk_trace(chunks, selected_count=8)

    # ── Hard Refusal #3: LLM hedged or returned has_answer=False ──
    if not result.get("has_answer") or _is_hedging(result.get("answer") or ""):
        logger.info("[Fallback] Hard refusal: LLM hedged or returned has_answer=False")
        return {**_refusal_payload(lang), "fallback_used": True, "retrieved_chunks": chunks, **chunk_trace_update}

    # ── Hard Refusal #4: Grounding gate (lexical heuristics) ──
    if not _grounding_gate(query, result["answer"], chunks):
        return {**_refusal_payload(lang), "fallback_used": True, "retrieved_chunks": chunks, **chunk_trace_update}

    logger.info(f"[Fallback] Generated answer via fallback pipeline")

    return {
        "final_answer": result["answer"],
        "citations": result["citations"],
        "confidence": result["confidence"],
        "has_answer": result["has_answer"],
        "fallback_used": True,
        # Override retrieved_chunks with fallback chunks for consistency
        "retrieved_chunks": chunks,
        **chunk_trace_update,
    }


# ---------------------------------------------------------------------------
# GRAPH BUILDER — called fresh per request
# ---------------------------------------------------------------------------

def _build_graph(db: Session):
    """
    Construct the LangGraph StateGraph for one request.

    WHY PER-REQUEST CONSTRUCTION?
    --------------------------------
    LangGraph graphs can hold state references. If we built the graph
    at module level, the `db` session reference would be shared across
    all requests — causing session conflicts and stale data.

    By building inside each request handler call, the graph and all
    its closures are garbage collected when run_agent() returns.

    The overhead of building the graph is minimal (~1ms) compared to
    the LLM API calls (500ms+) inside it.
    """

    # Wrap tool_node and fallback_node to inject `db` via closure.
    # LangGraph nodes receive only `state` — we can't pass extra args.
    # The closure captures `db` from the outer function scope.
    def _tool_node(state: AgentState) -> Dict[str, Any]:
        return tool_node(state, db)

    def _fallback_node(state: AgentState) -> Dict[str, Any]:
        return fallback_node(state, db)

    # Build the graph
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("tool_executor", _tool_node)
    graph.add_node("generation", generation_node)
    graph.add_node("fallback", _fallback_node)

    # Add edges
    # START → always go to router first
    graph.add_edge(START, "router")
    # router → always go to tool_executor
    graph.add_edge("router", "tool_executor")

    # tool_executor → conditional: check_confidence decides where to go
    graph.add_conditional_edges(
        "tool_executor",
        check_confidence,
        {
            "generate":  "generation",   # good enough → generate answer
            "router":    "router",        # not enough → try another tool
            "fallback":  "fallback",      # give up → use old pipeline
        }
    )

    # Both terminal nodes → END
    graph.add_edge("generation", END)
    graph.add_edge("fallback", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def run_agent(
    query: str,
    db: Session,
    document_id: Optional[str] = None,
    history: Optional[list] = None,
    debug_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run the agentic retrieval loop for one question.

    This is the ONLY function qa_controller.py needs to call.
    It replaces the multi-step retrieval block in the existing handler.

    LIFETIME OF THE AGENT:
    -----------------------
    Graph is built → initial state is created → graph runs to completion
    → result dict is returned → graph is garbage collected.
    No state survives beyond this function call.

    Args:
        query       : The user's question
        db          : Active database session from FastAPI dependency
        document_id : Optional — restrict search to one document
        history     : Conversation history (for query rewriting, passed through)

    Returns:
        dict with keys:
          answer     : str  — the generated answer
          citations  : list — source citations
          confidence : float
          has_answer : bool
          fallback_used: bool — True if the old pipeline was used
          chunks_used: int — how many chunks the LLM saw
          tools_tried: list — which tools the agent called
    """

    # Detect query language ONCE (Phase 2.1) — done on the ORIGINAL query,
    # not the rewritten one. Rewriting can shift tokens around and confuse
    # langdetect on borderline-short queries; the user's intent language
    # lives in what they actually typed.
    from app.services.llm_service import detect_language
    query_language = detect_language(query)
    logger.info(f"[Agent] Query language: {query_language}")

    # Optional: rewrite query before agent starts (same as before)
    effective_query = query
    if history:
        try:
            from app.services import llm_service
            rewritten = llm_service.rewrite_query(question=query, history=history)
            if rewritten and rewritten != query:
                logger.info(f"[Agent] Query rewritten: '{query[:40]}' → '{rewritten[:40]}'")
                effective_query = rewritten
        except Exception as e:
            logger.warning(f"[Agent] Query rewrite failed: {e} — using original")

    # Short follow-up augmentation: if the rewriter left us with a query that
    # is still pronoun-laden or very short, the retrieval signal is too weak
    # (e.g. "Why was it made?" embeds to almost nothing). Prepend the most
    # recent user turn so FAISS sees the topic. The LLM still gets the clean
    # original question via `history` for natural answer phrasing.
    _PRONOUN_TOKENS = {"it", "they", "them", "this", "that", "these", "those",
                       "he", "she", "him", "her"}
    eq_tokens = {t.lower().strip(".,?!:;") for t in effective_query.split()}
    if history and (eq_tokens & _PRONOUN_TOKENS or len(effective_query.split()) < 5):
        last_user_turn = next(
            (m.get("content") for m in reversed(history) if m.get("role") == "user"),
            None,
        )
        if last_user_turn and last_user_turn.strip().lower() != effective_query.strip().lower():
            augmented = f"{last_user_turn} — follow-up: {query}"
            logger.info(
                f"[Agent] Augmented short follow-up with prior turn: "
                f"'{effective_query[:40]}' → '{augmented[:60]}'"
            )
            effective_query = augmented

    # Initial state — all fields must be provided for TypedDict
    initial_state: AgentState = {
        "query": effective_query,
        "query_language": query_language,
        "history": history or [],
        "document_id": document_id,
        "tool_calls_made": [],
        "retrieved_chunks": [],      # starts empty; each tool appends to this
        "final_answer": None,
        "citations": [],
        "confidence": 0.0,
        "has_answer": False,
        "debug_mode": debug_mode,
        "routing": None,
        "chunk_trace": [],
        "iterations": 0,
        "chosen_tool": None,
        "chosen_args": None,
        "fallback_used": False,
    }

    # Build and run the graph — fresh per request
    compiled_graph = _build_graph(db)

    logger.info(f"[Agent] Starting run for query: '{effective_query[:60]}'")

    try:
        final_state = compiled_graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"[Agent] Graph execution failed: {e}", exc_info=True)
        # Hard fallback: use old pipeline directly
        from app.services import retrieval_service, llm_service
        chunks = retrieval_service.search_chunks(query=query, db=db, document_id=document_id)
        result = llm_service.generate_answer(question=query, context_chunks=chunks, history=history or [])
        # Even in the catastrophic-failure path we still return a well-formed
        # trace so the frontend's observability panel renders consistently.
        # iterations=0 because the graph crashed before any router iteration completed.
        crash_trace = _build_trace(
            tool_calls_made=[],
            retrieved_chunks=chunks,
            iterations=0,
            fallback_used=True,
        )
        return {
            **result,
            "fallback_used": True,
            "chunks_used":   len(chunks),
            "tools_tried":   [],
            "trace":         crash_trace,
            # Phase 3 fields — safe defaults on crash
            "chunk_trace":   None,
            "routing":       None,
        }

    logger.info(
        f"[Agent] Done. tools_tried={final_state['tool_calls_made']}, "
        f"chunks={len(final_state['retrieved_chunks'])}, "
        f"confidence={final_state['confidence']}, "
        f"fallback={final_state['fallback_used']}"
    )

    # Build the observability trace for the response (Phase 2.3)
    trace = _build_trace(
        tool_calls_made=final_state["tool_calls_made"],
        retrieved_chunks=final_state["retrieved_chunks"],
        iterations=final_state["iterations"],
        fallback_used=final_state["fallback_used"],
    )

    # Phase 3.1: chunk_trace is empty list when debug mode is off — return None
    # to make the frontend's "show debug panel" check trivial.
    raw_chunk_trace = final_state.get("chunk_trace") or []
    chunk_trace_payload = raw_chunk_trace if raw_chunk_trace else None

    return {
        "answer":       final_state["final_answer"] or "I couldn't find an answer in the provided documents.",
        "citations":    final_state["citations"],
        "confidence":   final_state["confidence"],
        "has_answer":   final_state["has_answer"],
        # Legacy top-level fields kept so qa_controller's logging keeps working
        "fallback_used": final_state["fallback_used"],
        "chunks_used":  len(final_state["retrieved_chunks"]),
        "tools_tried":  final_state["tool_calls_made"],
        # New: structured observability bundle for the API response
        "trace":        trace,
        # Phase 3.1: full per-chunk debug trace (only populated if debug_mode=True)
        "chunk_trace":  chunk_trace_payload,
        # Phase 3.2: router's first tool choice + reasoning
        "routing":      final_state.get("routing"),
    }
