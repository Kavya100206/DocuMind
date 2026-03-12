"""
Health Check Controller

This is our first controller (C in MVC)!

What is a Controller?
---------------------
A controller handles HTTP requests and returns responses.
It's the "C" in MVC - it coordinates between the Model (data) and View (response).

What does this controller do?
------------------------------
- Provides a health check endpoint
- Later we'll add database connection checks
- Returns system status information
"""

from fastapi import APIRouter
from app.config.settings import settings

# Create a router - this groups related endpoints together
# We'll include this router in main.py
router = APIRouter(
    prefix="/api",  # All routes will start with /api
    tags=["Health"]  # Groups endpoints in API documentation
)


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        dict: System health information
    
    Example response:
        {
            "status": "healthy",
            "app_name": "DocuMind",
            "version": "0.1.0"
        }
    """
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "debug_mode": settings.DEBUG
    }


@router.get("/info")
async def app_info():
    """
    Application information endpoint
    
    Returns:
        dict: Detailed application configuration (non-sensitive)
    """
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "embedding_model": settings.OPENAI_EMBEDDING_MODEL,
        "llm_model": settings.OPENAI_LLM_MODEL,
        "top_k_results": settings.TOP_K_RESULTS,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP
    }
