"""
Application Settings and Configuration
This file manages all environment variables and app settings
"""

from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

# Project root = 2 levels up from app/config/settings.py
# Path(__file__) = e:/SCALER/Projects/DocuMind/app/config/settings.py
# .parent = app/config
# .parent.parent = app/
# .parent.parent.parent = DocuMind/  ← this is our project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    
    What is this?
    -------------
    This class automatically loads configuration from your .env file.
    It provides type safety and validation for all settings.
    
    Why Pydantic Settings?
    ----------------------
    - Automatic environment variable loading
    - Type validation (ensures DATABASE_URL is a string, etc.)
    - Default values support
    - Easy to test and modify
    """
    
    # Application Settings
    APP_NAME: str = "DocuMind"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # Database Settings (PostgreSQL - NeonDB)
    DATABASE_URL: str  # This will come from your .env file
    
    # OpenAI Settings — now Optional (only used if you have credits)
    # We've switched to local embeddings + Groq for free usage
    OPENAI_API_KEY: Optional[str] = None

    # Groq Settings — free LLM API (sign up at console.groq.com)
    # Used in Phase 5 for answer generation
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"  # Upgraded: better reasoning over long documents

    # Google Drive OAuth2 (Phase 1 — Drive Integration)
    # Get these from Google Cloud Console → APIs & Services → Credentials
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/google/callback"

    # Embedding Settings — LOCAL model
    # Model: sentence-transformers/all-MiniLM-L6-v2 — ~80MB, perfect for Render Free
    LOCAL_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384  # same as MiniLM — no schema change needed

    # Vector Store Settings
    # str(BASE_DIR / "data" / "faiss_index") builds an absolute path:
    # e.g. E:/SCALER/Projects/DocuMind/data/faiss_index
    # This way it works regardless of where uvicorn is started from
    VECTOR_STORE_PATH: str = str(BASE_DIR / "data" / "faiss_index")

    # File Upload Settings
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")  # absolute path, same reason
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB in bytes
    ALLOWED_EXTENSIONS: list = [".pdf"]

    # Retrieval Settings
    TOP_K_RESULTS: int = 20       # Increased for cross-document synthesis (was 50/4)
    SIMILARITY_THRESHOLD: float = 0.05 # Relaxed threshold to allow more candidates through; was 0.02
    MAX_CHUNKS_PER_DOC: int = 30         # Max chunks any single doc contributes; was 20
    RERANKER_TOP_N: int = 8             # Sharper context for LLM; was 20

    # Chunking Settings
    # 600 chars ≈ 100 words — each chunk covers exactly one resume section or one document section.
    # Smaller chunks = more focused embeddings = better semantic match for specific queries.
    # (2500 was too large: mixed education + projects + experience into one diluted embedding)
    CHUNK_SIZE: int =  600 #600
    CHUNK_OVERLAP: int = 200 #80   # 80-char overlap: enough to preserve sentence continuity at boundaries

    # Pydantic v2 configuration
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True
    }


# Create a global settings instance
# This will be imported throughout the application
settings = Settings()
