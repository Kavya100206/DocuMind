# DocuMind — PDF-Constrained Conversational Agent

A grounded question-answering agent for PDF document repositories. DocuMind
combines a hybrid retrieval pipeline (FAISS + BM25 + cross-encoder reranking)
with a LangGraph agentic loop and a hard refusal contract that guarantees
the system never answers from outside the user's selected PDF. Multilingual
(English + Hindi), Drive-integrated, and instrumented end-to-end with an
observability panel and a retrieval trace mode.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Loop-1c3d5a?style=flat)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-f55036?style=flat)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-1abc9c?style=flat)](https://github.com/facebookresearch/faiss)
[![Multilingual](https://img.shields.io/badge/Multilingual-EN_%2B_HI-purple?style=flat)](https://pypi.org/project/langdetect/)
[![Google Drive](https://img.shields.io/badge/Google_Drive-Integrated-4285F4?style=flat&logo=googledrive)](https://developers.google.com/drive)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Live Demo

- Live: https://web-production-6f28c.up.railway.app/ui

## The Problem

Standard RAG applications fail in production. Naive setups hallucinate when
context is missing, rely on chunking strategies that destroy document
structure, and use vector-only similarity that misses exact keyword matches
("DSA I" vs "DSA II"). Worse, most basic implementations have **no hard
contract** preventing the LLM from drawing on its own training data when the
chunks don't contain the answer — which means a "grounded" RAG can still
fabricate a confident answer that has no source in the user's PDF.

## The Solution

DocuMind enforces a **PDF-constrained contract**: every response is either
strictly grounded in the user-selected document or is a refusal. The system
combines a 6-stage hybrid retrieval pipeline, a LangGraph agentic loop that
chooses retrieval tools and validates results, and a layered refusal gate
(topical relevance → hedging detection → qualifier-distance grounding) that
catches subtle hallucination traps where chunk vocabulary overlaps with the
question but the specific fact is not present.

---

## Features

### Refusal & Safety (the differentiator)
- **Hard Refusal Contract** — when grounding fails, the API returns `has_answer=false`, `confidence=0.0`, `citations=[]` with a localized refusal message. No soft fallbacks, no hedged guesses.
- **Qualifier-Distance Grounding Gate** — catches the realistic hallucination trap where chunk vocabulary matches the question but the specific fact doesn't (e.g. doc mentions e₹-Retail and rural India separately, but no rural-rollout date exists).
- **Strict PDF Isolation** — `/api/ask` rejects missing/empty `document_id` with 400 before any retrieval runs.
- **Topical relevance gate + hedging detection** — additional layered checks that convert weak responses into structured refusals.

### Retrieval & Reasoning
- **Hybrid Retrieval** — FAISS (65% semantic) + BM25 (35% lexical), reranked by a cross-encoder on the top 10. Lexical boosting for exact tokens, roman numerals, and section headers.
- **Section-Aware Chunking** — preserves document structure (section headers, semantic blocks) instead of splitting on character count.
- **LangGraph Agent Loop** — stateful directed graph; an LLM picks the retrieval tool (`vector_search` / `keyword_search` / `summarize_document`) and a downstream evaluator decides whether to answer, retry, or refuse.
- **Conversational Memory + Query Rewriting** — session history feeds query rewriting to resolve pronouns and implicit references before retrieval.

### Multilingual & Integrations
- **English + Hindi** — language detection via `langdetect`; the test suite asserts Devanagari-script responses to catch silent English fallbacks.
- **Google Drive Ingestion** — OAuth2 sync; Drive uploads share the same chunking/embedding pipeline as manual uploads. Webhook push notifications supported.

### UI & Observability
- **Refusal UI + Confidence Bar + Citation Accordions** — distinct visual treatment for refusals so a refusal can never be confused with a grounded answer.
- **Retrieval Trace Mode** — toggle that reveals retrieved chunks with full FAISS/BM25/hybrid/reranker score breakdown.
- **Observability Panel** — live view of agent tool selection, refusal-gate decisions, and confidence-score components.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla JavaScript, HTML, CSS |
| Backend | FastAPI (Python 3.11), SQLAlchemy |
| Database | PostgreSQL (NeonDB) |
| Agent Framework | LangGraph |
| AI / LLM | Groq API (LLaMA 3.3 70B Versatile) |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Lexical Search | rank_bm25 (BM25Okapi) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-2-v2 |
| Language Detection | langdetect |
| External Storage | Google Drive API (google-api-python-client + google-auth-oauthlib) |

---

## Architecture

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
    └─▶ evaluate_node ── confidence + grounding gates
            │
            ├── pass → generate answer + citations
            │
            └── fail → strict refusal (localized)

Hybrid Retrieval = FAISS (65%) ⊕ BM25 (35%)  →  Cross-Encoder Top-K
Refusal Gate     = Topical Relevance → Hedging Detection → Qualifier-Distance
```

### Key Architectural Decisions

- **LangGraph Agentic Loop** — Instead of a fixed pipeline, an LLM router picks the retrieval tool per query and a downstream evaluation node decides whether the chunks are good enough to answer or whether to retry / refuse. This keeps query-shape dependent logic (summary vs. factual vs. follow-up) out of hardcoded heuristics.
- **Hard Refusal Contract** — When any grounding gate trips, the response is structurally a refusal: `has_answer=false`, `confidence=0.0`, `citations=[]`, refusal message localized to the user's language. This is what prevents the system from fabricating answers when chunks are weak. See [docs/TECHNICAL.md §2.A](docs/TECHNICAL.md#a-hard-refusal-contract--the-load-bearing-decision) for the full rationale.
- **Qualifier-Distance Grounding** — The most realistic hallucination failure isn't off-topic queries; it's queries whose vocabulary appears in the chunks but whose specific fact does not (the I3 "rural rollout date" trap). A heuristic check fires when the answer asserts strong grounding ("explicitly stated", "according to the report") and verifies that ≥66% of question qualifiers cluster within ~100 chars of the cited date in the actual chunks. Failure → hard refusal.
- **Dual Retrieval Execution** — Searches run against both the user's raw query and the rewritten standalone query; results are merged for maximum coverage.
- **Weighted Hybridization** — Vector and lexical scores are normalized to a consistent scale and merged 65% semantic / 35% lexical.
- **Lazy-Loaded Cross-Encoders** — Cross-encoder models are loaded on demand and applied only to the top 10 merged candidates, keeping retrieval latency ~120ms.
- **Ephemeral FAISS Strategy** — The FAISS index is rebuilt from the persistent PostgreSQL chunk store on startup if missing, so vector state is always deterministically synchronized with relational metadata.

---

## API Endpoints

### Documents
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/documents/` | List all synced documents and status |
| POST | `/api/documents/upload` | Ingest and process a new PDF |
| GET | `/api/documents/:id` | Check processing status of a specific document |
| DELETE | `/api/documents/:id` | Delete document and cascade remove chunks |

### Question Answering
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/ask` | Submit a natural-language question (English or Hindi) and receive a grounded answer or a structured refusal |
| DELETE | `/api/session/:id` | Clear conversation memory for a session |

### Google Drive
| Method | Endpoint | Description |
|---|---|---|
| GET | `/auth/google` | Start the Google OAuth2 consent flow |
| GET | `/auth/google/callback` | OAuth2 callback — exchanges the code for tokens |
| GET | `/api/drive/files` | List the authenticated user's Drive files (PDFs + Docs) |
| POST | `/api/drive/ingest` | Ingest a Drive file through the standard pipeline |
| POST | `/api/drive/webhook` | Receive Drive change push notifications |

### System
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/system/stats` | Global statistics (documents, chunks, pages) |

---

## Test Suite

DocuMind ships with a behavioural test suite anchored to the RBI Annual
Report 2024-25. Each test category exercises a different contract: valid
queries must answer correctly with citations, invalid queries must refuse
cleanly under the strict contract, and Hindi queries verify multilingual
support end-to-end (including a Devanagari-character assertion that
prevents silent English fallbacks from passing).

| Category | Count | What it proves |
|---|---|---|
| Valid English queries (V1–V6) | 6 | System answers correctly when the answer is in the document — including factual extraction, numerical extraction, section-based, cross-section synthesis, summary, and multi-turn reasoning |
| Invalid English queries (I1–I3) | 3 | System refuses cleanly when the answer is NOT in the document, including the I3 hallucination trap (qualifier-distance grounding) |
| Hindi queries (H1–H2) | 2 | Multilingual support — both the answer path and the refusal path return Devanagari script |
| Phase 1 contract tests | (subset) | Strict PDF isolation gate — missing/empty `document_id` is rejected at the controller |

```bash
# Targeted V5 / V6 / I3 (~1 min, ~4 /api/ask calls)
.\venv\Scripts\python.exe tests\test_v5_v6.py

# Full suite (V1-V6 + I1-I3 + H1-H2 + isolation tests)
.\venv\Scripts\python.exe tests\test_suite.py
```

The full specification of every query, expected behaviour, and pass criteria
lives in [`tests/test_cases.md`](tests/test_cases.md).

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- PostgreSQL database
- Groq API Key

### Installation

1. Clone the repository and navigate to the project directory.
2. Create and activate a Python virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and configure your database and API keys.

### Running the Server

Start the application using Uvicorn:
`uvicorn app.main:app --reload --port 8000`

The REST API and frontend UI are available at `http://localhost:8000`.

### MCP Integration

The Model Context Protocol (MCP) server is exposed via SSE at `http://localhost:8000/mcp/sse`. To connect an MCP client (such as Claude Desktop or the MCP Inspector), configure the client to use the SSE transport and provide the Bearer token specified by your `MCP_TOKEN` environment variable.
