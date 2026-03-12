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

# Include routers from controllers
app.include_router(health_controller.router)
app.include_router(document_controller.router)
app.include_router(search_controller.router)
app.include_router(qa_controller.router)
app.include_router(system_controller.router)

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


# Startup event - runs when the application starts
@app.on_event("startup")
async def startup_event():
    """
    Runs when the application starts
    
    What goes here?
    ---------------
    - Database connection initialization
    - Loading ML models
    - Cache warming
    - Any one-time setup tasks
    """
    from app.database.postgres import init_db, engine
    from app.models import document, chunk  # noqa: F401

    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    try:
        logger.info("Initializing database...")
        init_db()
        with engine.connect() as conn:
            logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.warning("Application will continue, but database features won't work")


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


# This allows running the app directly with: python app/main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes (development only)
    )
