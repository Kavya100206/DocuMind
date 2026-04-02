"""
Embedding Service (Local — sentence-transformers)

Model: sentence-transformers/all-MiniLM-L6-v2
- Size: ~80MB (downloaded once from HuggingFace, cached locally)
- Output: 384-dimensional vectors
- Speed: Fast on CPU

MEMORY OPTIMISATION (Railway / Render free tier):
- Model is LAZY-LOADED on first use, not at module import
- torch.no_grad() prevents gradient tracking memory
- gc.collect() releases tensor memory after batch operations
- torch threads limited to 1 to prevent thread-pool memory bloat
"""

import gc
from typing import List, Dict, Any
from app.config.settings import settings

# -----------------------------------------------------------------------
# LAZY MODEL LOADING
# -----------------------------------------------------------------------
# The model is NOT loaded at import time. It loads on first call to
# get_embedding() or embed_chunks(). This saves ~80MB during startup
# scripts that don't need embeddings (e.g. FAISS rebuild from stored vectors).
_model = None

# Batch size for processing multiple chunks at once
BATCH_SIZE = 16


def _get_model():
    """Load the SentenceTransformer model lazily and cache it."""
    global _model
    if _model is None:
        import torch
        torch.set_num_threads(1)  # Prevent thread-pool memory bloat on free tiers

        from sentence_transformers import SentenceTransformer
        print(f"🤖 Loading embedding model ({settings.LOCAL_EMBEDDING_MODEL})...")
        _model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
        print("✅ Model loaded")
    return _model


def get_embedding(text: str) -> List[float]:
    """
    Get the embedding vector for a single piece of text.

    Used when embedding a USER'S QUERY during retrieval.
    Returns a list of 384 floats (the vector).
    """
    import torch

    clean_text = text.replace("\n", " ").strip()

    if not clean_text:
        raise ValueError("Cannot embed empty text")

    # BGE-specific query prefix (only for BGE models)
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    if settings.LOCAL_EMBEDDING_MODEL.startswith("BAAI/bge"):
        clean_text = BGE_QUERY_PREFIX + clean_text

    model = _get_model()
    with torch.no_grad():
        embedding = model.encode(clean_text, normalize_embeddings=True)
    return embedding.tolist()


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a LIST of texts in one efficient batch.
    """
    import torch

    clean_texts = [t.replace("\n", " ").strip() for t in texts]

    model = _get_model()
    with torch.no_grad():
        embeddings = model.encode(
            clean_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    return embeddings.tolist()


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add embedding vectors to a list of chunk dicts.

    Called from document_service.py after chunking.
    Adds an "embedding" key to every chunk dict.
    """
    import torch

    if not chunks:
        return []

    print(f"  🔢 Generating embeddings for {len(chunks)} chunks (local model)...")

    texts = [chunk["text"] for chunk in chunks]

    model = _get_model()
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=BATCH_SIZE
        )

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    # Release tensor memory immediately
    del embeddings
    gc.collect()

    print(f"  ✅ Embeddings done! Each vector has {len(chunks[0]['embedding'])} dimensions.")
    return chunks
