# DocuMind — Technical Note

DocuMind is a PDF-constrained conversational agent. It answers questions strictly from a user-uploaded PDF and refuses cleanly when the answer isn't in there. This note walks through how it's put together, the calls I made along the way, and what I'd flag as the trade-offs behind each one.

---

## TL;DR

- **What it is** — a Q&A agent over a single PDF where every response is either grounded in that PDF or a structural refusal. No in-between, no soft fallbacks, no answers leaking from the LLM's pretrained knowledge.
- **The interesting part** — a *qualifier-distance grounding gate* that catches the realistic hallucination failure mode: queries whose vocabulary appears in the chunks but whose specific fact doesn't (e.g. the "rural rollout date" trap where the doc mentions e₹-Retail and rural India separately, just not together as the question implies).
- **How it works** — FAISS + BM25 hybrid retrieval → cross-encoder rerank → LangGraph agent loop that picks tools and validates results → 3-layer refusal gate (topical relevance → hedging detection → qualifier-distance grounding).
- **Multilingual** — English + Hindi, with localized refusals. The test suite asserts Devanagari script in the response, so a silent English fallback would fail.
- **Verified by** — 11 query specs in [`tests/test_cases.md`](../tests/test_cases.md) covering factual / numerical / section / synthesis / summary / multi-turn / outside-knowledge / off-topic / hallucination-trap / Hindi-valid / Hindi-refusal. Run `tests/test_suite.py` against a configured environment to reproduce; see [TESTING.md](TESTING.md).

---

## 1. Architecture

```
User Query
    │
    ▼
[ Language Detect ] ── EN / HI
    │
    ▼
[ Query Rewriter ] ── pronoun / context resolution from session memory
    │
    ▼
[ LangGraph Agent Loop ]
    │
    ├─▶ router_node ── LLM picks a tool (semantic / lexical / summarize)
    │
    ├─▶ tool_node ── runs hybrid retrieval + cross-encoder rerank
    │
    └─▶ evaluate_node ── confidence + refusal gates
            │
            ├── pass → generate answer + citations
            │
            └── fail → strict refusal (localized to user's language)

Hybrid Retrieval = FAISS (65%) ⊕ BM25 (35%)  →  Cross-Encoder Top-K
Refusal Gate     = Topical Relevance → Hedging Detection → Qualifier-Distance
```

### Components

- **Hybrid retrieval** — FAISS (semantic, 65% weight) ⊕ BM25 (lexical, 35%), normalized and merged, then reranked by a cross-encoder applied only to the top 10 candidates
- **LangGraph agent loop** — stateful directed graph; an LLM picks one of three retrieval tools (`vector_search`, `keyword_search`, `summarize_document`), then a downstream evaluator decides whether to answer, retry with a different tool, or refuse
- **Refusal gates** — three independent checks; any one failing triggers a hard refusal with `has_answer=False`, `confidence=0.0`, `citations=[]`
- **Multilingual** — `langdetect` on the query; answer language enforced via system-prompt directive; refusal messages localized

---

## 2. Design decisions

### A. Hard refusal contract — the load-bearing decision

This was the call I came back to the most while building. The API guarantees that every response is **either** strictly grounded in the user's selected PDF **or** a structural refusal. The `has_answer / confidence / citations` triple is contractually meaningful, not a UI hint. Two layers enforce it:
- **Controller**: missing or empty `document_id` is rejected with HTTP 400 before any retrieval runs
- **Agent**: any refusal-gate failure produces the canonical refusal payload

What I observed early on: without this contract, a "grounded" RAG can still draw on the LLM's pretrained knowledge whenever the chunks are weak — a silent failure mode that's invisible in confidence scores. Treating refusal as a first-class response shape (not a fallback string) is what makes the rest of the system trustable.

### B. LangGraph instead of a fixed pipeline

Different question types need different retrieval paths: a *summary* query should hit `summarize_document`, a *factual* query should hit `vector_search`, a *named-entity* query should hit `keyword_search`. Hard-coding these routes is brittle. LangGraph lets an LLM make the routing decision and a separate node evaluate the result, which generalizes to new query shapes and surfaces the routing decision in the observability trace for debugging.

### C. Qualifier-distance grounding gate

This turned out to be the trickiest piece to get right, and the one I'm most happy with. The realistic hallucination failure mode isn't an off-topic query (those are easy to refuse via topical-relevance gating) — it's a query whose vocabulary appears in the chunks but whose **specific fact** does not. Example: asking about the "rural rollout date" for e₹-Retail when the document mentions e₹-Retail and rural finance separately. A weakly grounded system retrieves the e₹-Retail chunk, sees that it's "relevant", and confabulates a date.

The gate fires when the answer asserts strong grounding ("explicitly stated", "according to the report") and verifies three things:
1. The cited date appears in the chunks at all (else: fabrication)
2. ≥ 66% of the question's distinctive qualifiers cluster within ~100 chars of that date in the chunks (else: the topic is in the doc but the specific fact isn't)
3. The grounding claim and the date sit in the same clause in the answer (within ~20 chars)

That third check came from a real over-firing I caught during deployment: a legitimate split-sentence answer like *"…April 1, 2025.\n\nThis information is explicitly stated…"* was being incorrectly refused because the heuristic looked at the claim phrase across a sentence boundary. Tightening the proximity to a single clause fixed it without weakening the I3 trap detection.

### D. Confidence as retrieval-evidence strength, not factual probability

Confidence measures how strongly the chunks support the answer, not the probability the answer is factually correct. We deliberately deviate from the spec's flat `confidence ≥ 0.6` floor — analysis and summary queries pull broad context with moderate per-chunk scores, so they legitimately score lower than direct factual extraction. Category-specific floors (0.5 factual, 0.4 analysis) reflect actual score distributions; factual correctness is enforced separately and more strictly via the refusal-gate stack.

### E. Section-aware chunking

Splitting on character count destroys document structure. The chunker detects section headers and preserves semantic blocks, which dramatically improves both citation precision and reranker behaviour on structured documents (annual reports, research papers, policy docs).

---

## 3. Trade-offs

| Decision | Chose | Alternative considered | Why we chose it |
|---|---|---|---|
| Retrieval strategy | Hybrid (FAISS + BM25) | Vector-only | Vector-only misses exact-match queries (e.g. "DSA I" vs "DSA II", named entities); BM25 fills that gap |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-2-v2` | Larger cross-encoder | Lazy-loaded and applied only to top 10 candidates; balances accuracy with ~120ms retrieval latency |
| Vector store | FAISS (in-process) | Pinecone / pgvector | Dependency-free, deterministic, rebuildable from PostgreSQL on cold start; no external service to manage |
| LLM provider | Groq (LLaMA 3.3 70B Versatile) | OpenAI GPT-4 | ~10× cheaper at similar quality on this workload; free-tier-friendly for development |
| Refusal trigger | Layered gates (topical / hedging / qualifier-distance) | Single confidence threshold | A flat threshold either over-refuses analysis queries or under-refuses subtle hallucinations; layered gates address each failure mode separately |
| State store | PostgreSQL (NeonDB) | SQLite | NeonDB free tier gives a real Postgres on Railway; SQLite would tie us to a single instance |
| Frontend | Vanilla JS SPA | React | No build step, single file, faster to ship for the assignment scope |
| Multilingual approach | Detect language → prompt-direct LLM → localize refusals | Translate queries to English internally | Translation introduces a second LLM call per query and loses subtlety; prompt-directed answer generation works well on LLaMA 3.3 70B |
| Agent framework | LangGraph | LangChain agents / hand-rolled | LangGraph's directed-graph model maps cleanly to "router → tool → evaluator → branch" without LangChain's heavier abstractions |

---

## 4. Known limitations

Honest list of things I'd flag, not oversights I'm hoping go unnoticed:

- **Embedding model is English-tuned** — `all-MiniLM-L6-v2` isn't multilingual. Hindi semantic search against English chunks works but is borderline; in practice it leans on the prompt-directive layer to translate at generation time. The clean fix is switching to `paraphrase-multilingual-MiniLM-L12-v2`.
- **Groq free-tier rate limits** — 6000 tokens/minute. Heavy queries (~3–4k tokens each) cap throughput at ~1.7 queries/min, which I hit during testing. Demos benefit from the Dev tier.
- **Drive integration is single-user** — `user_id` is hardcoded to `"default_user"` for the assignment scope. A multi-tenant rebuild would need a session/JWT layer.
- **FAISS cold-start latency** — the first query after a Railway deploy downloads the embedding model (~80MB) and rebuilds the FAISS index from Postgres. ~30s warmup, then instant.
- **Transient Groq errors look like refusals** — a 429 or upstream outage trips the hedging path and gets reported as "answer not in document", which is technically a refusal but semantically wrong. A future pass should distinguish "service unavailable, retry later" from a real grounding failure.

---

## 5. What's tested

11 query specifications in [`tests/test_cases.md`](../tests/test_cases.md):
- 6 valid English queries — factual / numerical / section / cross-section synthesis / summary / multi-turn with reasoning
- 3 invalid English queries — outside knowledge / off-topic entity / hallucination trap
- 2 Hindi queries — valid (must answer in Devanagari) / invalid (must refuse in Hindi)

Plus the strict PDF isolation contract (missing/empty `document_id` → 400) and a Devanagari assertion on Hindi responses (catches silent English fallbacks).

Reproducibility: see [TESTING.md](TESTING.md).
