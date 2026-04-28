"""
FAISS Vector Store Service

WHAT IS FAISS?
--------------
FAISS = Facebook AI Similarity Search.
It is a library that does ONE thing extremely well:

    "Given a query vector, find the N closest vectors in a large collection."

Think of it like a search index, but for numbers (vectors) instead of words.

WHY DO WE NEED IT?
------------------
After embedding, we have hundreds of chunk vectors.
When a user asks a question, we:
  1. Embed the question → get a query vector
  2. Compare it against ALL chunk vectors → find the closest ones
  3. Return those chunks as "relevant context"

Step 2 is the problem. If you have 10,000 chunks and compare naively
(one by one), that's slow. FAISS does this comparison near-instantly
even for millions of vectors, using smart indexing algorithms.

THE TWO FILES WE SAVE TO DISK:
--------------------------------
FAISS stores vectors, but NOT the text or metadata.
So we save TWO files side by side:

  data/
  ├── faiss_index.index    ← The FAISS index (just vectors + positions)
  └── faiss_index.meta     ← Our metadata (text, page_number, doc_id, etc.)

When searching, FAISS tells us "positions 3, 7, 42 are closest."
We look up positions 3, 7, 42 in the metadata file to get the actual text.

HOW THE INDEX WORKS (simplified):
-----------------------------------
Imagine each embedding is a point in 384-dimensional space.
FAISS organizes these points so it can quickly find nearby points.

  Query vector: [0.1, -0.4, 0.8, ...]     ← user's question, embedded
  Chunk A vec:  [0.1, -0.4, 0.8, ...]     ← very close → relevant
  Chunk B vec:  [-0.9, 0.2, -0.1, ...]    ← far away → not relevant

"Close" = similar meaning. This is called COSINE SIMILARITY.
(We use IndexFlatIP because our vectors are normalized — it's equivalent.)
"""

import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from app.config.settings import settings

# Global flag to prevent querying while background index build happens
IS_BUILDING = False

_cached_index = None
_cached_metadata = None

def build_and_save_index(chunks: List[Dict[str, Any]]) -> None:
    """
    Build a FAISS index from embedded chunks and save it to disk.

    Called from document_service.py after embedding is complete.

    What this does step by step:
    1. Extract all embedding vectors from the chunks
    2. Convert them to a numpy array (what FAISS needs)
    3. Create a FAISS index
    4. Add all vectors to the index
    5. Save the index to disk (faiss_index.index)
    6. Save the metadata (text, page, doc_id) to disk (faiss_index.meta)

    NOTE: If an index already exists (from previous uploads),
    we LOAD it and ADD to it — not replace it.
    This supports multiple document uploads.

    Args:
        chunks: List of chunk dicts, each must have an "embedding" key
    """
    global _cached_index, _cached_metadata

    # Filter out any chunks that somehow didn't get embedded
    embedded_chunks = [c for c in chunks if "embedding" in c]

    if not embedded_chunks:
        print("  ⚠️  No embedded chunks to index.")
        return

    print(f"  📦 Building FAISS index for {len(embedded_chunks)} chunks...")

    # ------------------------------------------------------------------
    # STEP 1: Extract vectors into a numpy array
    # ------------------------------------------------------------------
    # FAISS requires a 2D numpy array of shape (num_chunks, embedding_dim)
    # Example: (150, 384) for 150 chunks with 384-dim embeddings
    #
    # np.array([...]) converts our list of lists into a numpy matrix
    # .astype("float32") → FAISS requires 32-bit floats (not 64-bit)
    vectors = np.array(
        [chunk["embedding"] for chunk in embedded_chunks],
        dtype="float32"
    )

    embedding_dim = vectors.shape[1]  # Should be 384

    # ------------------------------------------------------------------
    # STEP 2: Create or load the FAISS index
    # ------------------------------------------------------------------
    index_path = settings.VECTOR_STORE_PATH + ".index"
    meta_path = settings.VECTOR_STORE_PATH + ".meta"

    # Make sure the data/ directory exists
    os.makedirs(os.path.dirname(settings.VECTOR_STORE_PATH), exist_ok=True)

    if os.path.exists(index_path):
        # Index already exists → load it and ADD to it
        # This handles the "upload a second PDF" case
        print(f"  📂 Found existing index — loading and appending...")
        index = faiss.read_index(index_path)

        # Load existing metadata too
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

    else:
        # No index yet → create a fresh one
        # IndexFlatIP = "Flat Inner Product" index
        # "Flat" = no compression, exact search (perfect for small-medium datasets)
        # "IP" = Inner Product similarity (= cosine similarity for normalized vectors)
        print(f"  🆕 Creating new FAISS index (dim={embedding_dim})...")
        index = faiss.IndexFlatIP(embedding_dim)
        metadata = []  # empty list, will hold our chunk metadata

    # ------------------------------------------------------------------
    # STEP 3: Normalise vectors, then add to the index
    # ------------------------------------------------------------------
    # embedding_service already calls normalize_embeddings=True, but we
    # explicitly normalise here too as a safety net.  For IndexFlatIP,
    # cosine similarity == inner product ONLY when vectors have unit length.
    # faiss.normalize_L2 rescales each row to length 1 in-place.
    faiss.normalize_L2(vectors)
    index.add(vectors)
    print(f"  Added {len(embedded_chunks)} vectors. Total in index: {index.ntotal}")

    # ------------------------------------------------------------------
    # STEP 4: Build metadata list (parallel to the vectors)
    # ------------------------------------------------------------------
    # For each chunk, save everything EXCEPT the embedding itself
    # (no point saving that again — it's already in the FAISS index)
    for chunk in embedded_chunks:
        metadata.append({
            "document_id":  chunk.get("document_id"),
            "page_number":   chunk.get("page_number"),
            "chunk_index":   chunk.get("chunk_index"),
            "text":          chunk.get("text"),
            "char_count":    chunk.get("char_count"),
            "section_name":  chunk.get("section_name"),
        })

    # ------------------------------------------------------------------
    # STEP 5: Save both files to disk
    # ------------------------------------------------------------------

    os.makedirs(os.path.dirname(settings.VECTOR_STORE_PATH), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"  💾 FAISS index saved to: {index_path}")

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"  💾 Metadata saved to: {meta_path}")

    # Force a cache reset so the next query re-reads the fresh index from disk
    _cached_index = None
    _cached_metadata = None


def load_index():
    """
    Load the FAISS index and metadata from disk, keeping it cached globally
    so it doesn't cause Out-Of-Memory (OOM) errors by reloading repeatedly.

    Returns:
        (index, metadata) tuple, or (None, None) if no index exists yet.

    What is the return type?
    - index: a faiss.Index object you can call .search() on
    - metadata: a list of dicts, one per chunk in the index
    """
    global _cached_index, _cached_metadata

    if _cached_index is not None and _cached_metadata is not None:
        return _cached_index, _cached_metadata

    # Prevent FAISS parallelism from exploding memory
    faiss.omp_set_num_threads(1)

    index_path = settings.VECTOR_STORE_PATH + ".index"
    meta_path = settings.VECTOR_STORE_PATH + ".meta"

    if not os.path.exists(index_path):
        print("  ⚠️  No FAISS index found. Upload some documents first.")
        return None, None

    _cached_index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        _cached_metadata = pickle.load(f)

    print(f"  📂 Loaded FAISS index into cache with {_cached_index.ntotal} vectors.")
    return _cached_index, _cached_metadata


def search(query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for the most relevant chunks for a query.

    This is the core retrieval function — used in Phase 4.

    HOW IT WORKS:
    1. Embed the query text → query vector (384 numbers)
    2. Ask FAISS: "find the k closest vectors to this query"
    3. FAISS returns: distances + positions
    4. Use positions to look up the actual chunk text from metadata
    5. Return the chunks with their similarity scores

    Args:
        query_text: The user's question (plain English)
        k: How many chunks to return (default: 5)
           → these become the "context" fed to the LLM

    Returns:
        List of chunk dicts, sorted by relevance (most relevant first).
        Each dict has: text, page_number, document_id, similarity_score

    Example:
        results = search("What was the Q3 revenue?", k=3)
        for r in results:
            print(f"[Page {r['page_number']}] (score: {r['similarity_score']:.3f})")
            print(r['text'][:200])
            print()
    """

    # Step 1: Load the index
    index, metadata = load_index()

    if index is None:
        return []

    # Step 2: Embed the query
    # Import here to avoid circular imports
    from app.services.embedding_service import get_embedding
    query_vector = get_embedding(query_text)

    # FAISS needs a 2D numpy array even for a single query
    # Shape: (1, 384) → "1 query, 384 dimensions"
    query_array = np.array([query_vector], dtype="float32")

    # Step 3: Search the index
    # index.search() returns two arrays:
    #   distances: shape (1, k) → similarity scores for each result
    #   indices:   shape (1, k) → positions in the metadata list
    #
    # We take [0] because we only have 1 query (the first and only row)
    distances, indices = index.search(query_array, k)
    distances = distances[0]   # shape (k,)
    indices = indices[0]        # shape (k,)

    # Step 4: Build results from metadata
    results = []
    for distance, idx in zip(distances, indices):
        # idx == -1 means FAISS didn't find enough results
        # (happens when k > total number of indexed chunks)
        if idx == -1:
            continue

        chunk_meta = metadata[idx]  # look up the chunk text, page, etc.

        results.append({
            "text":             chunk_meta["text"],
            "page_number":      chunk_meta["page_number"],
            "document_id":      chunk_meta["document_id"],
            "chunk_index":      chunk_meta["chunk_index"],
            "similarity_score": float(distance),  # higher = more relevant
            "section_name":     chunk_meta.get("section_name"),
        })
    return results

def index_has_vectors() -> bool:
    index_path = settings.VECTOR_STORE_PATH + ".index"

    if not os.path.exists(index_path):
        return False

    try:
        index = faiss.read_index(index_path)
        return index.ntotal > 0
    except:
        return False


def get_all_chunks_metadata() -> List[Dict[str, Any]]:
    """
    Return all chunk metadata currently stored in the FAISS index.

    WHY THIS EXISTS:
    ----------------
    The keyword_search agent tool needs access to ALL chunk texts
    so BM25 can score them by keyword frequency.

    FAISS's .meta file already holds this — we just expose it
    through a clean function instead of touching the global directly.

    Returns:
        List of chunk metadata dicts (text, page_number, document_id, etc.)
        Empty list if no index is loaded yet.
    """
    _, metadata = load_index()
    return metadata if metadata is not None else []