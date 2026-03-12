"""
Embedding Service (Local — sentence-transformers)

Model: BAAI/bge-small-en-v1.5
- Size: ~130MB (downloaded once from HuggingFace, cached locally)
- Output: 384-dimensional vectors (same as old MiniLM)
- Speed: Fast on CPU
- Quality: ~10-15% better than all-MiniLM-L6-v2 on semantic retrieval benchmarks

BGE ASYMMETRIC ENCODING:
BGE models are trained with a query prefix for retrieval tasks:
  Queries  → prefix + text (tells the model: "optimise for finding passages")
  Passages → no prefix    (raw text, stored as-is in FAISS)
This asymmetry improves recall significantly vs symmetric encoding.
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.config.settings import settings

# -----------------------------------------------------------------------
# LOADING THE MODEL
# -----------------------------------------------------------------------
# SentenceTransformer() loads the model into memory.
# Using all-MiniLM-L6-v2: ~80MB, extremely memory efficient for 512MB RAM limits.
print("🤖 Loading local embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
print(f"✅ Embedding model loaded: {settings.LOCAL_EMBEDDING_MODEL}")

# Batch size for processing multiple chunks at once
BATCH_SIZE = 32  # Reduced batch size slightly for memory safety on Render Free


def get_embedding(text: str) -> List[float]:
    """
    Get the embedding vector for a single piece of text.

    This is used when embedding a USER'S QUERY during retrieval (Phase 4).

    HOW IT WORKS:
    - model.encode(text) runs the text through the local neural network
    - Returns a numpy array of shape (384,)  ← 384 numbers
    - .tolist() converts it to a regular Python list

    Args:
        text: Any string — a query, a sentence, etc.

    Returns:
        A list of 384 floats (the vector)

    Example:
        vec = get_embedding("What was the Q3 revenue?")
        print(len(vec))   # → 384
        print(vec[:3])    # → [0.023, -0.44, 0.81]
    """

    clean_text = text.replace("\n", " ").strip()

    if not clean_text:
        raise ValueError("Cannot embed empty text")

    # BGE-specific query prefix: instructs the model to produce a vector
    # optimised for RETRIEVAL (finding relevant passages) rather than
    # general-purpose similarity.  Only applied to queries, NOT to chunks
    # at index time (asymmetric encoding).
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    if settings.LOCAL_EMBEDDING_MODEL.startswith("BAAI/bge"):
        clean_text = BGE_QUERY_PREFIX + clean_text

    embedding = model.encode(clean_text, normalize_embeddings=True)
    return embedding.tolist()


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a LIST of texts in one efficient batch.

    sentence-transformers is built for this — it processes a list of
    texts in parallel on your CPU, much faster than one at a time.

    Args:
        texts: List of strings to embed

    Returns:
        List of 384-dim vectors, one per input text (same order)
    """

    clean_texts = [t.replace("\n", " ").strip() for t in texts]

    # encode() accepts a list directly — processes all at once
    # show_progress_bar=False → don't print a progress bar for each batch
    embeddings = model.encode(
        clean_texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # embeddings is a 2D numpy array of shape (len(texts), 384)
    # .tolist() converts it to a list of lists
    return embeddings.tolist()


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add embedding vectors to a list of chunk dicts.

    Called from document_service.py after chunking.
    Adds an "embedding" key to every chunk dict.

    BATCHING:
    We process all chunks in one shot because sentence-transformers
    handles batching internally. Much simpler than the OpenAI version!

    Args:
        chunks: List of chunk dicts from chunking_service.py

    Returns:
        Same list, each dict now has "embedding": [list of 384 floats]
    """

    if not chunks:
        return []

    print(f"  🔢 Generating embeddings for {len(chunks)} chunks (local model)...")

    # Extract all text at once
    texts = [chunk["text"] for chunk in chunks]

    # Embed them all in one call — sentence-transformers handles batching internally
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,   # shows a nice progress bar for large batches
        batch_size=BATCH_SIZE
    )

    # Attach each embedding back to its chunk
    # zip() pairs chunks[0] with embeddings[0], chunks[1] with embeddings[1], etc.
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    print(f"  ✅ Embeddings done! Each vector has {len(chunks[0]['embedding'])} dimensions.")
    return chunks
