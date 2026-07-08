"""
MCP Server — DocuMind (Phase 2: Contract fidelity)

WHAT CHANGED IN PHASE 2 (vs Phase 1):
---------------------------------------
1. SCHEMA REUSE
   list_documents  → returns List[DocumentResponse] (app/views/document_views.py)
   ask_document    → returns QAResponse            (app/views/qa_views.py)
   Phase 1 returned raw dicts. Phase 2 runs the same Pydantic validation the
   REST layer runs, so MCP responses are structurally identical to /api/ask.

2. INPUT VALIDATION — mirrors qa_controller.ask_question() lines 76–122
   - Empty question      → ValueError (clean MCP tool error, not a crash)
   - Missing document_id → ValueError (same message as the 400 in qa_controller)
   - Short query (<3 meaningful words) → QAResponse(has_answer=False) — NOT a
     raise, because qa_controller returns a structured response here, not 400.
   - Document not found / not completed → ValueError (mirrors 404 spirit)
   These checks run BEFORE hitting the DB or calling run_agent().

3. LLM / TIMEOUT FAILURE → CLEAN ERROR
   run_agent() is wrapped in try/except inside _run_in_thread(). On any
   exception it returns QAResponse(has_answer=False) with a user-readable
   message — same recovery shape as qa_controller lines 239–250. The MCP
   server process does NOT crash.

4. STRUCTURED LOGGING
   Matches the fields qa_controller logs after every agent call (line 186–191):
   tools_tried, chunks_used, confidence, has_answer, fallback_used.

WHAT DID NOT CHANGE:
---------------------
- sys.path fix (lines 95–99)       — same
- stderr redirect (lines 101–122)  — same
- SessionLocal pattern             — same (db opened in thread, closed in finally)
- asyncio.to_thread() pattern      — same (avoids LangGraph/asyncio deadlock)
- stdio transport                  — same (Phase 3 switches to SSE)

HOW TO RUN (unchanged):
  mcp dev app/mcp/server.py

HOW IT FITS INTO THE ARCHITECTURE:
  AI Client (Claude Desktop)
        │ stdio (JSON-RPC)
        ▼
  app/mcp/server.py          ← YOU ARE HERE
        │
        ├── list_documents() ──▶ DB query → List[DocumentResponse]
        │
        └── ask_document()   ──▶ input guards
                                 → doc existence check (DB)
                                 → agent_service.run_agent()
                                 → QAResponse (same schema as /api/ask)

KEY DESIGN CHOICE — DB SESSIONS:
  MCP tools run outside the FastAPI request lifecycle (no Depends(get_db)).
  Sessions are managed manually:
    db = SessionLocal()
    try: ...do work... finally: db.close()
  Identical to document_controller.process_document_bg() (lines 472–481).

KEY DESIGN CHOICE — LOGGING TO STDERR:
  logger.py (line 85) installs StreamHandler(sys.stdout) by default.
  We install a stderr handler BEFORE importing any app module so logger.py's
  guard ("if root.handlers: return") skips stdout. Logs go to stderr;
  stdin/stdout stays clean for JSON-RPC framing.
"""

import os
import sys
import logging

# ── Add project root to sys.path ──────────────────────────────────────────
# mcp dev loads this file with importlib — does NOT add the project root to
# sys.path. We compute it from __file__ and insert it manually so that
# `from app.*` imports resolve correctly regardless of how the script is run.
#
#   __file__ (3 × dirname) → .../DocuMind   ← project root
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Fix Windows stdout encoding ───────────────────────────────────────────
# The MCP inspector launches this script as a subprocess. On Windows,
# subprocess stdout defaults to cp1252 which can't encode emoji.
# faiss_service.py:214 and llm_service.py:558 both use print() with emoji
# (📂 and 🔍). Without this fix they raise UnicodeEncodeError inside
# LangGraph nodes, returning 0 chunks and breaking the agent loop.
# reconfigure() changes the encoding in-place without replacing sys.stdout
# or its .buffer, so FastMCP's stdio transport (which uses sys.stdout.buffer)
# is unaffected.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Redirect logging to stderr BEFORE importing any app module ────────────
# Must be the first thing after stdlib imports. See module docstring for why.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Silence SQLAlchemy SQL echo ───────────────────────────────────────────
# If DEBUG=True, SQLAlchemy logs every SQL statement. Those lines begin with
# a timestamp ("2026-...") — the MCP inspector's JSON parser sees "2026" as
# a JSON number, hits "-" at position 4 → SyntaxError. Suppressing at WARNING
# stops this without losing any useful MCP-layer log output.
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)

# ── App imports (safe after stderr handler is installed) ──────────────────
from fastmcp import FastMCP
try:
    from mcp.server.transport_security import TransportSecuritySettings as _TransportSecuritySettings
    _NO_DNS_REBIND = _TransportSecuritySettings(enable_dns_rebinding_protection=False)
except ImportError:
    # mcp < 1.28 — TransportSecuritySettings doesn't exist yet; DNS-rebinding
    # protection is not present in these versions so nothing to disable.
    _TransportSecuritySettings = None  # type: ignore[assignment]
    _NO_DNS_REBIND = None

from app.config.settings import settings           # needed for GROQ_MODEL
from app.database.postgres import SessionLocal
from app.models.document import Document
from app.services import agent_service, faiss_service, embedding_service
from app.utils.logger import get_logger
from app.views.document_views import DocumentResponse
from app.views.qa_views import QAResponse, Citation, RetrievalTrace, RouterInfo

logger = get_logger(__name__)

# ── File log handler — survives mcp dev stderr capture ────────────────────
# mcp dev swallows the Python subprocess's stderr. A file handler lets us
# read exactly where in _run_in_thread the hang occurs without needing to
# see the terminal. Tail this file while the inspector is running:
#   Get-Content mcp_server.log -Wait   (PowerShell equivalent of tail -f)
_log_path = os.path.join(_PROJECT_ROOT, "mcp_server.log")
_fh = logging.FileHandler(_log_path, mode="a", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.getLogger().addHandler(_fh)
logger.info("[MCP] server.py loaded — log file: %s", _log_path)


# ---------------------------------------------------------------------------
# MCP SERVER INSTANCE
# ---------------------------------------------------------------------------
# "DocuMind" is the server name MCP clients display in their UI.
#
# transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
# FastMCP 1.28+ automatically enables DNS-rebinding protection when host is
# "127.0.0.1" (the default). In that mode TransportSecurityMiddleware rejects
# every request whose Host header is not localhost — which is ALL requests in
# production because Railway sets Host to the public Railway domain.
# We disable it here so the SSE and message endpoints are reachable in prod.
mcp = FastMCP(
    "DocuMind",
    **({"transport_security": _NO_DNS_REBIND} if _NO_DNS_REBIND is not None else {}),
)


# ---------------------------------------------------------------------------
# TOOL 1: list_documents
# ---------------------------------------------------------------------------

@mcp.tool()
def list_documents() -> list[DocumentResponse]:
    """
    List all documents uploaded to DocuMind.

    Returns a list of DocumentResponse objects, each containing:
      id         : document UUID — pass this to ask_document
      filename   : original PDF filename
      status     : "completed" | "processing" | "failed"
      page_count : number of pages (available after processing)
      created_at : upload timestamp

    Only documents with status="completed" are ready to query.

    PHASE 2 CHANGE (vs Phase 1):
    Returns List[DocumentResponse] instead of List[dict].
    The same Pydantic model used by GET /api/documents is now enforced here,
    so MCP clients see a schema-validated, consistent response shape.
    """
    logger.info("[MCP] list_documents called")

    db = SessionLocal()
    try:
        # Same query as document_controller.list_documents() line 210.
        docs = db.query(Document).order_by(Document.created_at.desc()).all()

        # Phase 2: build DocumentResponse from each ORM object.
        # DocumentResponse has model_config = {"from_attributes": True}
        # (document_views.py line 71), so model_validate() reads directly
        # from the SQLAlchemy model instance without manual field mapping.
        result = [DocumentResponse.model_validate(doc) for doc in docs]

        logger.info(
            f"[MCP] list_documents → {len(result)} document(s) | "
            f"completed={sum(1 for d in result if d.status == 'completed')}"
        )
        return result

    finally:
        # Always close — returns the connection to the SQLAlchemy pool.
        db.close()


# ---------------------------------------------------------------------------
# TOOL 2: ask_document
# ---------------------------------------------------------------------------

@mcp.tool()
async def ask_document(document_id: str, question: str) -> dict:
    """
    Ask a question about a specific document.

    Runs the full DocuMind agentic RAG pipeline — the same LangGraph loop
    that the /api/ask endpoint uses:
      1. Router node   : LLM picks vector_search / keyword_search / summarize_document
      2. Tool node     : retrieves chunks from FAISS + BM25 hybrid index
      3. Confidence check: loops or falls back if score is below threshold
      4. Grounding gate: lexical hallucination check before generation
      5. Hard refusal  : if out of scope → has_answer=False, confidence=0.0

    Args:
        document_id : UUID of the document (from list_documents).
                      Must exist in the DB and have status="completed".
        question    : Your question in plain text (must be at least 3 words).

    Returns a QAResponse matching the /api/ask response shape:
        answer      : grounded answer, or refusal message if out of scope
        citations   : [{document_name, page_number, text_snippet}, ...]
        confidence  : retrieval confidence (0.0–1.0)
        has_answer  : False means the question is out of scope
        model_used  : which Groq model generated the answer
        trace       : retrieval observability (tool used, top score, iterations)

    PHASE 2 CHANGES (vs Phase 1):
    - Returns QAResponse instead of a raw dict.
    - Input guards run before the DB / LLM are touched (see below).
    - LLM failures return a structured QAResponse, not a crash.
    - Log output matches the fields qa_controller emits.
    """

    # ── INPUT GUARD 1: empty question ─────────────────────────────────────
    # Mirrors qa_controller.ask_question() line 76–77.
    # raise ValueError → FastMCP converts to a clean tool error for the client.
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    # ── INPUT GUARD 2: missing document_id ────────────────────────────────
    # Mirrors qa_controller lines 100–105 (400 before retrieval).
    if not document_id or not document_id.strip():
        raise ValueError(
            "Please provide a document_id. "
            "Use list_documents to see available document IDs. "
            "DocuMind only answers based on a specific uploaded document."
        )

    # ── INPUT GUARD 3: short query ────────────────────────────────────────
    # Mirrors qa_controller lines 108–122.
    # NOTE: This returns a QAResponse, NOT a raise — same as the controller.
    # qa_controller returns a structured response here (not an HTTPException)
    # so the frontend can display the message cleanly. We match that behavior:
    # the MCP client receives a valid QAResponse, not an error.
    meaningful_words = [w for w in question.split() if len(w) > 1]
    if len(meaningful_words) < 3:
        logger.info(f"[MCP] ask_document: short query guard | question={question!r}")
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
        ).model_dump()

    logger.info(
        f"[MCP] ask_document called | "
        f"doc={document_id!r} | "
        f"question={question!r:.60}"
    )

    def _run_in_thread() -> QAResponse:
        """
        All blocking work (DB + LLM) runs here in a worker thread.

        WHY A THREAD?
        FastMCP runs on asyncio. Declaring the tool `async` and pushing
        blocking work into asyncio.to_thread() prevents the LangGraph event-
        loop conflict that causes a deadlock when a synchronous tool is called
        directly from the asyncio event loop. See Phase 1 server.py for the
        full explanation.

        WHY CREATE THE SESSION INSIDE THE THREAD?
        SQLAlchemy sessions are NOT thread-safe. Creating the session inside
        the thread (not in the async caller) ensures it is never shared across
        threads — the same discipline used in every FastAPI Depends(get_db)
        call (one session per request lifecycle).
        """
        # ── GIVE THIS THREAD A CLEAN EVENT LOOP ──────────────────────────
        # ROOT CAUSE OF THE HANG:
        # LangGraph 0.2.28's sync invoke() internally calls:
        #   asyncio.get_event_loop().run_until_complete(self.ainvoke(...))
        #
        # asyncio.to_thread() spawns a worker thread. In Python 3.10,
        # asyncio.get_event_loop() in a thread with no current loop set can
        # return the PARENT thread's loop — FastMCP's already-running loop.
        # Calling run_until_complete() on a running loop = deadlock.
        #
        # FIX: explicitly create and set a new event loop for this thread
        # before any LangGraph/Groq code runs. LangGraph then calls
        # run_until_complete() on THIS loop (not FastMCP's), runs the
        # coroutine to completion, and returns normally.
        # The loop is closed in the finally block so we don't leak resources.
        import asyncio as _asyncio
        _loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(_loop)

        db = SessionLocal()
        try:
            # ── DOCUMENT EXISTENCE CHECK ──────────────────────────────────
            # Mirrors the 404 spirit in the REST layer (document_controller
            # returns 404 when a document_id is not found).
            # We check BEFORE calling run_agent() so a bad ID fails fast with
            # a clean message instead of a cryptic FAISS / DB error inside
            # the agent loop.
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                raise ValueError(
                    f"Document '{document_id}' not found. "
                    "Use list_documents to see valid document IDs."
                )

            # ── STATUS CHECK ──────────────────────────────────────────────
            # An in-progress document has no chunks yet; running the agent
            # against it would return empty retrieval and likely a refusal.
            # Surfacing the real reason is more useful.
            if doc.status != "completed":
                raise ValueError(
                    f"Document '{doc.filename}' is not ready "
                    f"(status='{doc.status}'). "
                    "Wait for processing to complete before querying."
                )

            logger.debug("[MCP][thread] CHECKPOINT 1 — about to call run_agent()")
            try:
                result = agent_service.run_agent(
                    query=question,
                    db=db,
                    document_id=document_id,
                    history=[],
                    debug_mode=False,
                )
                logger.debug("[MCP][thread] CHECKPOINT 2 — run_agent() returned has_answer=%s", result.get("has_answer"))
            except Exception as exc:
                logger.error(
                    f"[MCP] ask_document: agent error | doc={document_id!r} | {exc}",
                    exc_info=True,
                )
                # Roll back the session so SQLAlchemy doesn't block further
                # use with PendingRollbackError. This matters when the SSL
                # connection drops during FAISS loading — the session is left
                # in an invalid state after the OperationalError, and any
                # subsequent query (e.g. summarize_document) would raise
                # "Can't reconnect until invalid transaction is rolled back."
                try:
                    db.rollback()
                except Exception:
                    pass  # rollback itself may fail if connection is gone
                return QAResponse(
                    question=question,
                    answer=(
                        "I'm sorry, an internal processing error occurred "
                        "while analyzing your documents. Please try asking again."
                    ),
                    citations=[],
                    confidence=0.0,
                    has_answer=False,
                    model_used=settings.GROQ_MODEL,
                ).model_dump()

            # ── STRUCTURED LOGGING ────────────────────────────────────────
            # Matches the fields qa_controller logs at lines 186–191,
            # using logger.info instead of print() because MCP logs must go
            # to stderr (logger is already configured to stderr at module top).
            fallback_tag = "[FALLBACK]" if result.get("fallback_used") else "[AGENT]"
            logger.info(
                f"[MCP] ask_document done | {fallback_tag} | "
                f"has_answer={result['has_answer']} | "
                f"confidence={result['confidence']:.3f} | "
                f"tools_tried={result.get('tools_tried')} | "
                f"chunks={result.get('chunks_used')}"
            )

            # ── BUILD CITATIONS ───────────────────────────────────────────
            # run_agent() returns citations as list[dict]. QAResponse expects
            # List[Citation] (Pydantic models). Same conversion as
            # qa_controller line 207: Citation(**c) for each dict.
            citations = [Citation(**c) for c in result["citations"]]

            # ── BUILD RETRIEVAL TRACE ─────────────────────────────────────
            # Defensive .get() — same pattern as qa_controller line 213–214.
            # run_agent() should always populate "trace" now, but older code
            # paths or hard errors can return without it.
            trace_dict = result.get("trace")
            trace = RetrievalTrace(**trace_dict) if trace_dict else None

            # ── BUILD ROUTING INFO ────────────────────────────────────────
            # Same pattern as qa_controller lines 223–224.
            routing_dict = result.get("routing")
            routing = RouterInfo(**routing_dict) if routing_dict else None

            # ── RETURN QAResponse ─────────────────────────────────────────
            # We construct QAResponse so Pydantic validates every field
            # (same guarantee the REST layer gets via response_model=QAResponse).
            # Then we call .model_dump() before returning from the thread.
            #
            # WHY model_dump() instead of returning the Pydantic object?
            # FastMCP serializes sync-tool returns (list[DocumentResponse])
            # correctly, but its async-tool serialization path expects either
            # a primitive, a dict, or a list — not a Pydantic model instance.
            # Returning a raw Pydantic object from an async tool causes FastMCP
            # to hang during response framing (the tool completes but the JSON-
            # RPC response is never sent). model_dump() converts to a plain dict
            # that FastMCP can JSON-encode without issue.
            #
            # The contract guarantee is still here — QAResponse.__init__ runs
            # full field validation before we discard the object wrapper.
            response = QAResponse(
                question=question,
                answer=result["answer"],
                citations=citations,
                confidence=result["confidence"],
                has_answer=result["has_answer"],
                model_used=settings.GROQ_MODEL,
                trace=trace,
                routing=routing,
            )
            return response.model_dump()

        finally:
            db.close()
            # Clean up the thread-local event loop we created above.
            # Not closing it leaks file descriptors (the loop's internal selector).
            _loop.close()
            _asyncio.set_event_loop(None)

    # ── WHY threading.Thread instead of asyncio.to_thread() ──────────────────
    # asyncio.to_thread() uses loop.run_in_executor() which copies the
    # asyncio context (contextvars + thread-local loop references) into the
    # worker thread. Sentence-transformers/PyTorch (used inside vector_search)
    # detect that asyncio context and interact with it — causing a deadlock
    # that shows up as the tool hanging at [Tool Executor] Running tool: vector_search.
    #
    # threading.Thread spawns a completely bare OS thread with no asyncio
    # context. PyTorch and FAISS can create their own thread pools freely.
    # We bridge back to the event loop using loop.call_soon_threadsafe() to
    # set an asyncio.Event once the thread finishes, then await that event.
    import asyncio
    import threading

    loop = asyncio.get_event_loop()
    done_event = asyncio.Event()
    result_holder: dict = {}

    def _thread_target():
        try:
            result_holder["value"] = _run_in_thread()
        except Exception as exc:
            result_holder["error"] = exc
        finally:
            # Signal the event loop that the thread is done.
            # call_soon_threadsafe is the correct way to set an asyncio.Event
            # from a non-asyncio thread — it schedules the set() on the loop.
            loop.call_soon_threadsafe(done_event.set)

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    await done_event.wait()
    t.join(timeout=0)  # thread already finished; join() just cleans up

    if "error" in result_holder:
        raise result_holder["error"]
    return result_holder["value"]


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------


