"""
FastAPI Application Entry Point

This is the main file that starts our DocuMind application.
It creates the FastAPI app and includes all routes.

What is FastAPI?
----------------
FastAPI is a modern, fast web framework for building APIs with Python.
It automatically generates API documentation and validates requests.

What happens here?
------------------
1. We create a FastAPI app instance
2. We configure CORS (Cross-Origin Resource Sharing) for frontend access
3. We include routers from controllers (API endpoints)
4. We define startup/shutdown events for database connections
"""

import os
import sys

# Force UTF-8 on stdout/stderr so emoji-laden print() / logger calls
# don't crash on Windows (default cp1252 codec can't encode \U0001f916 etc.).
# Without this, _get_model()'s "🤖 Loading embedding model..." print kills the
# embedding load and downstream retrieval returns zero chunks.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from app.config.settings import settings, BASE_DIR
from app.utils.logger import get_logger
from app.controllers import health_controller
from app.controllers import document_controller
from app.controllers import search_controller
from app.controllers import qa_controller
from app.controllers import system_controller
# Phase 1 — Google Drive Integration
# drive_controller exposes TWO routers:
#   auth_router  → /auth/google and /auth/google/callback (OAuth2 flow)
#   drive_router → /api/drive/... (file list, ingest, webhook)
from app.controllers.drive_controller import auth_router as drive_auth_router
from app.controllers.drive_controller import drive_router
from app.services import faiss_service

logger = get_logger(__name__)

# Create FastAPI application instance
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production RAG system for document Q&A with citations",
    debug=settings.DEBUG
)

# Configure CORS - allows frontend to communicate with backend
# In production, you'd restrict this to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ---------------------------------------------------------------------------
# PHASE 3 — MCP SERVER MOUNT (SSE Transport)
# ---------------------------------------------------------------------------
from starlette.middleware.base import BaseHTTPMiddleware


class MCPAuthMiddleware(BaseHTTPMiddleware):
    """
    Lightweight Bearer Token check for the MCP endpoints.
    FastAPI dependencies (Depends) do not apply to mounted ASGI apps,
    so we use a Starlette middleware directly on the mcp_app.

    Both GET /sse and POST /messages/ require a valid Bearer token.
    MCP Inspector sends the Authorization header on every request
    (GET and POST alike) when a token is configured in its settings.
    """
    async def dispatch(self, request: Request, call_next):
        expected_token = os.environ.get("MCP_TOKEN", "default-dev-token")
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {expected_token}":
            return JSONResponse(
                {"detail": "Unauthorized MCP access"},
                status_code=401
            )
        return await call_next(request)


# Import the MCP server instance
from app.mcp.server import mcp

# FASTMCP 3.4.3 — correct API for SSE transport
#
# fastmcp 3.4.3 (standalone package by Jeremiah Lowin) is a completely
# different package from mcp.server.fastmcp. Key differences:
#
#   • sse_app() does NOT exist — replaced by http_app(transport='sse')
#   • TransportSecuritySettings is NOT exposed — replaced by host_origin_protection=
#   • No mount_path parameter — Starlette strips the /mcp prefix automatically
#     when the sub-app is mounted, so the internal routes /sse and /messages/
#     are correct relative paths already.
#
# host_origin_protection=False:
#   Disables the Host/Origin header validation guard that fastmcp 3.4.3
#   enables by default. In production on Railway the Host header is the
#   public Railway domain (not localhost), which triggers the guard and
#   silently drops every request with 400. We disable it here and rely
#   on our own MCPAuthMiddleware for access control.
mcp_app = mcp.http_app(
    transport="sse",
    host_origin_protection=False,
)
# Auth middleware ONLY applies to the mounted mcp_app
mcp_app.add_middleware(MCPAuthMiddleware)

# Mount the fully configured MCP app under /mcp.
# Starlette strips the /mcp prefix before forwarding to mcp_app, so the
# internal routes /sse and /messages/ resolve to /mcp/sse and /mcp/messages/
# as seen by the client. No path rewriting is needed.
app.mount("/mcp", mcp_app)


# Include routers from controllers
app.include_router(health_controller.router)
app.include_router(document_controller.router)
app.include_router(search_controller.router)
app.include_router(qa_controller.router)
app.include_router(system_controller.router)
# Phase 1 — Google Drive routers
app.include_router(drive_auth_router)   # /auth/google, /auth/google/callback
app.include_router(drive_router)        # /api/drive/files, /ingest, /webhook

# ---------------------------------------------------------------------------
# GLOBAL EXCEPTION HANDLER
# ---------------------------------------------------------------------------
# This ensures that even "hard" crashes (500 errors) return JSON.
# It prevents the frontend from getting a 500 HTML page from Render/Proxy
# which causes the "Unexpected token '<'" JSON parsing error.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"GLOBAL ERROR: {exc}")
    import traceback
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal Server Error: {str(exc)}",
            "type": type(exc).__name__,
            "message": "The server encountered a problem. Check logs for details."
        }
    )

# ---------------------------------------------------------------------------
# DIAGNOSTIC ENDPOINT
# ---------------------------------------------------------------------------
@app.get("/api/system/diag", tags=["System"])
async def system_diagnostic():
    """Verify filesystem write permissions for Render persistence."""
    paths = {
        "uploads": settings.UPLOAD_DIR,
        "data": str(BASE_DIR / "data"),
    }
    
    results = {}
    for name, path in paths.items():
        exists = os.path.exists(path)
        writeable = os.access(path, os.W_OK) if exists else "Path not found"
        results[name] = {
            "path": path,
            "exists": exists,
            "writeable": writeable
        }
    
    return {
        "status": "diagnostic_complete",
        "filesystem": results,
        "memory_limit": "512MB (Render Free)",
        "embedding_model": settings.LOCAL_EMBEDDING_MODEL
    }



# UI endpoint — serves the single-page frontend dashboard
# Returning FileResponse directly avoids needing aiofiles / StaticFiles
@app.get("/ui", include_in_schema=False)
async def serve_ui():
    """
    Serve the DocuMind frontend dashboard.
    Open http://localhost:8000/ui in your browser.
    """
    return FileResponse(
        "frontend/index.html", 
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )


# Root endpoint - just to verify the server is running
@app.get("/")
async def root():
    """
    Root endpoint - returns a welcome message
    
    Why 'async'?
    ------------
    FastAPI supports async/await for better performance.
    Async functions can handle multiple requests concurrently.
    """
    return {
        "message": "Welcome to DocuMind API",
        "version": settings.APP_VERSION,
        "status": "running"
    }


# Health check endpoint - used to verify the service is healthy
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    What is this for?
    -----------------
    - Used by monitoring tools to check if the service is running
    - Returns basic system information
    - Later we'll add database connection checks here
    """
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "debug_mode": settings.DEBUG
    }


@app.on_event("startup")
async def startup_event():
    from app.database.postgres import init_db, engine
    from app.models import document, chunk
    from app.services.faiss_service import index_has_vectors
    import asyncio

    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    try:
        logger.info("Initializing database...")
        init_db()

        with engine.connect() as conn:
            logger.info("Database connection successful")

        # FAISS health check (lightweight, non-blocking)
        # start.sh already builds FAISS from stored embeddings before uvicorn.
        # This is a safety net for cases where start.sh was skipped (e.g. direct uvicorn).
        if not index_has_vectors():
            logger.warning("FAISS missing or empty — rebuilding in background")

            from scripts.rebuild_faiss import rebuild_faiss_index
            asyncio.create_task(asyncio.to_thread(rebuild_faiss_index))

            logger.info("FAISS rebuild started in background")
        else:
            logger.info("FAISS index valid")

    except Exception as e:
        logger.error(f"Startup error: {e}")
# Shutdown event - runs when the application stops
@app.on_event("shutdown")
async def shutdown_event():
    """
    Runs when the application shuts down
    
    What goes here?
    ---------------
    - Close database connections
    - Save state
    - Cleanup resources
    """
    logger.info(f"Shutting down {settings.APP_NAME}")
    # We'll add cleanup code here later
