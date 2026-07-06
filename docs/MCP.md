# DocuMind MCP Integration

## ADR-001: MCP Server Architecture — Direct Import vs HTTP Self-Call

**Status:** Accepted  
**Date:** 2026-07-04  
**Deciders:** DocuMind engineering

---

## Context

We are adding an MCP (Model Context Protocol) server to DocuMind so that AI
clients (Claude Desktop, Cursor, etc.) can call two tools:

- `list_documents` — return the list of uploaded documents
- `ask_document`   — run the full RAG pipeline for a question scoped to one document

The MCP server needs to reach existing business logic. Two approaches were
evaluated.

---

## Options Considered

### Option A — Direct Import (chosen)

The MCP server is a standalone Python process that imports and calls service
modules directly:

```
AI Client ──→ MCP Server (Python process, stdio transport)
                  │
                  ├── from app.services import agent_service
                  ├── from app.models.document import Document
                  └── from app.database.postgres import SessionLocal
                        │
                        └── agent_service.run_agent(query, db, document_id, ...)
```

The MCP server owns its own DB session, created via `SessionLocal()` from
`app/database/postgres.py` — the same factory used by
`document_controller.process_document_bg()` (lines 472–481).

### Option B — HTTP Self-Call (rejected)

The MCP server acts as an HTTP client, forwarding tool calls to the running
FastAPI server:

```
AI Client ──→ MCP Server (Python process, stdio transport)
                  │
                  └── httpx.post("http://localhost:8000/api/ask", ...)
```

---

## Decision: Option A — Direct Import

### Rationale

**1. DB session management is already solved.**  
`run_agent()` in `app/services/agent_service.py` (line 1351) requires a
`db: Session` argument. The direct-import pattern replicates the same
`SessionLocal()` / `db.close()` lifecycle already used in
`document_controller.process_document_bg()`. This is 5 lines of boilerplate,
not a design problem.

**2. Memory: FAISS lives only in FastAPI.**  
`faiss_service` and `embedding_service` hold module-level globals: the FAISS
index and the sentence-transformers embedding model (~400 MB on CPU). A
direct-import MCP server would load its own copies — doubling RAM on a
free-tier deployment.

**Counter-argument:** This is resolved by the fact that for Phases 1–2 the
MCP server runs *locally only* (stdio transport, not deployed). The RAM concern
becomes a non-issue in Phase 3 when the MCP server is mounted inside the
FastAPI process itself and all globals are shared.

**3. Hard refusal contract is preserved by construction.**  
The contract — `has_answer=False`, `confidence=0.0`, canonical refusal message
from `REFUSAL_MESSAGES` in `agent_service.py` (lines 141–166) — lives inside
`run_agent()`. Both options call the same code path. Direct import calls it
without a serialization round-trip.

**4. Phase 3 migration path is clean.**  
Phase 3 mounts the MCP server on FastAPI via SSE transport. At that point the
MCP server and FastAPI share one process, so "direct import" is literally what
happens — there is no HTTP self-call to remove.

**5. Learning value (explicit project goal).**  
Direct import exposes the session lifecycle, the `SessionLocal` pattern, and
`run_agent()`'s return shape. HTTP self-call hides all of that behind JSON and
`httpx` plumbing.

---

## Consequences

### What this means for each phase

| Phase | Transport | Session source | Tool execution |
|-------|-----------|----------------|----------------|
| 1–2   | stdio (local) | `SessionLocal()` per tool call, closed in `finally` | Direct call to `agent_service.run_agent()` |
| 3     | SSE (mounted in FastAPI) | FastAPI `Depends(get_db)` or same `SessionLocal` pattern | Same direct call |

### Known trade-offs accepted

- The MCP server's `list_documents` tool replicates the 2-line DB query from
  `document_controller.list_documents()` (line 210) rather than calling a
  dedicated service function — because no such function exists in
  `app/services/document_service.py`. This is a minor violation of DRY that
  we accept for Phase 1. Phase 2 can extract a `document_service.list_all(db)`
  helper if needed.

- `run_agent()` returns a raw `dict`, not a Pydantic model. The MCP tool
  must read keys explicitly (`result["answer"]`, `result["has_answer"]`, etc.)
  rather than relying on schema validation. This is documented so Phase 2's
  contract-fidelity work knows where to add validation.

---

## Phase Plan

### Phase 0 — Decide the integration shape

Before writing code, one architectural call to make explicit: the MCP server
should import your existing services directly (`agent_service`,
`retrieval_service`, `document_controller` logic), not call your own `/api/ask`
over HTTP. Since it's the same codebase, a self-HTTP-call would duplicate
FastAPI's dependency injection, add a pointless network hop, and risk your MCP
responses drifting out of sync with your REST responses. Direct import keeps
`agent_service.py` and its grounding gates as the single source of truth.

**Practical implication:** MCP tool functions open a DB session manually (same
pattern `app/database` already uses, just without FastAPI's `Depends()`), since
they call service functions outside the request lifecycle.

**Completion criteria:** State in one sentence why MCP tools call
`agent_service` directly instead of hitting `/api/ask`.

> MCP tools import `agent_service.run_agent()` directly so the LangGraph
> decision loop, grounding gates, and hard-refusal contract are the single
> source of truth for all clients — no network hop, no schema drift, no
> duplicated dependency injection.

---

### Phase 1 — Minimal MCP server, local only (~1 day)

**Goal:** Prove a real MCP client can get a grounded answer end-to-end from
DocuMind.

**Build:**

New module: `app/mcp/server.py`, using the official MCP Python SDK, stdio
transport (for local Claude Desktop testing).

Two tools only:

- `list_documents()` → thin wrapper around your existing document-listing logic
  in `document_controller.py`. Returns `document_id` + title so a client can
  pick one.
- `ask_document(document_id: str, question: str)` → calls `agent_service`'s
  agent loop directly (the same LangGraph entry point `qa_controller.py`
  calls), reusing your existing Pydantic response schema from `app/views` as
  the tool's return type.

No new logic — this phase is pure wrapping.

**Expected output:** Running `mcp dev app/mcp/server.py` (or connecting via
Claude Desktop's config) lets you ask a real question against an
already-ingested PDF and get back a grounded answer with citations.

**Completion criteria:**

- One grounded query (a V-series case, e.g. V1) returns the correct answer
  with citations through the MCP tool.
- One refusal case (I3, the qualifier-distance trap) returns
  `has_answer=false`, `confidence=0.0`, `citations=[]` through the MCP tool —
  not just via the REST API. This is proof that the hard refusal contract
  survived the new interface.

---

### Phase 2 — Contract fidelity and error handling (~1 day)

**Goal:** Make MCP a first-class interface, not a fragile side door.

**Build:**

- Reuse your exact response schema (from `app/views`) for tool outputs — don't
  hand-roll a second shape. One contract, enforced everywhere.
- Handle bad input at the tool boundary the way `qa_controller` handles it at
  the API boundary: missing/invalid `document_id` → a clean structured tool
  error, not an unhandled exception (mirrors the existing 400-before-retrieval
  pattern).
- Wrap the Groq call with the same timeout expectations already present (~1.2s
  generation) — a slow/failed LLM call should surface as a clean error to the
  MCP client, not hang the connection.
- Route MCP tool invocations through the existing logging so they show up in
  the same observability trail as web-UI calls.

**Expected output:** MCP tool responses are structurally indistinguishable from
`/api/ask` responses; a forced failure (bad `document_id`, simulated Groq
timeout) degrades cleanly.

**Completion criteria:**

- Passing an invalid `document_id` through the MCP tool returns a clean error,
  matching the spirit of the controller's 400 behavior.
- A forced LLM failure doesn't crash the MCP server process.
- MCP calls appear in logs/observability with the same fields as
  UI-originated calls.

---

### Phase 3 — Deploy it, don't just demo it locally (~0.5–1 day)

**Goal:** The MCP server is reachable from the live Railway deployment, not
just localhost.

**Build:**

- Switch transport from stdio to SSE, mounted as a route on the existing
  FastAPI app (e.g. `/mcp`) rather than standing up a second Railway service.
  Reuses the already-running Postgres pool and loaded FAISS index — one running
  process, one source of truth.
- Add a lightweight bearer-token check on the `/mcp` route (an env var, not
  full OAuth — proportionate to a portfolio project, but signals deliberate
  thought about exposing document contents over a new interface).
- Update the README: add "MCP Clients" as a node in the architecture diagram,
  add an "MCP Integration" section next to the API Endpoints table, and record
  a short demo (Claude Desktop or the MCP inspector hitting the live Railway
  URL).

**Expected output:** A public MCP endpoint at
`https://web-production-44b3a.up.railway.app/mcp` (or similar),
token-protected, callable from any MCP client.

**Completion criteria:**

- Claude Desktop configured against the deployed URL (not localhost)
  successfully lists documents and answers a query.
- README updated with the new architecture node and a demo clip/gif.

---

### Phase 4 — Optional (out of scope for this sprint)

Exposing `upload_document` as a tool (file transfer over MCP) or making
`ask_document` session-aware (multi-turn memory across MCP calls) are real
extensions, but they add meaningfully more complexity for resume purposes
relative to the time cost. Skip unless this is being continued as a deeper
technical project.
