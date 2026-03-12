"""
System Controller (Phase 7)

Provides operational stats about the DocuMind system.

Endpoints:
    GET /api/system/stats  → document count, chunk count, FAISS vector count
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.postgres import get_db
from app.models.document import Document
from app.models.chunk import Chunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/system",
    tags=["System"]
)


@router.get("/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    """
    Return current system stats.

    WHAT THIS RETURNS:
    ------------------
    - total_documents: Number of PDFs uploaded (any status)
    - completed_documents: PDFs fully processed and indexed
    - processing_documents: PDFs still being ingested in background
    - failed_documents: PDFs that failed during ingestion
    - total_chunks: Total text chunks stored in PostgreSQL
    - faiss_vector_count: Number of vectors in the FAISS index on disk

    WHY IS faiss_vector_count USEFUL?
    -----------------------------------
    It tells you if the FAISS index is in sync with the database.
    If total_chunks != faiss_vector_count, the index may need rebuilding.

    Example response:
        {
            "total_documents": 3,
            "completed_documents": 2,
            "processing_documents": 1,
            "failed_documents": 0,
            "total_chunks": 47,
            "faiss_vector_count": 47,
            "index_in_sync": true,
            "status": "healthy"
        }
    """
    # ------------------------------------------------------------------
    # Document counts by status
    # ------------------------------------------------------------------
    all_docs = db.query(Document).all()
    total_documents = len(all_docs)
    completed = sum(1 for d in all_docs if d.status == "completed")
    processing = sum(1 for d in all_docs if d.status == "processing")
    failed = sum(1 for d in all_docs if d.status == "failed")

    # ------------------------------------------------------------------
    # Chunk count from PostgreSQL
    # ------------------------------------------------------------------
    total_chunks = db.query(Chunk).count()

    # ------------------------------------------------------------------
    # FAISS vector count from index on disk
    # ------------------------------------------------------------------
    # We load the FAISS index and ask its ntotal property.
    # ntotal = number of vectors currently indexed.
    # Returns 0 if the index file doesn't exist yet (no uploads yet).
    faiss_vector_count = _get_faiss_vector_count()

    index_in_sync = (total_chunks == faiss_vector_count)
    overall_status = "healthy" if index_in_sync else "index_out_of_sync"

    logger.info(
        f"System stats: docs={total_documents}, chunks={total_chunks}, "
        f"faiss_vectors={faiss_vector_count}, in_sync={index_in_sync}"
    )

    return {
        "total_documents": total_documents,
        "completed_documents": completed,
        "processing_documents": processing,
        "failed_documents": failed,
        "total_chunks": total_chunks,
        "faiss_vector_count": faiss_vector_count,
        "index_in_sync": index_in_sync,
        "status": overall_status
    }


def _get_faiss_vector_count() -> int:
    """
    Load the FAISS index from disk and return its vector count.

    Returns 0 if the index doesn't exist (no documents uploaded yet).

    HOW:
    ----
    faiss.read_index() loads the index file.
    index.ntotal = number of vectors stored in it.
    This is a fast O(1) read — no search happens.
    """
    try:
        import faiss
        from app.config.settings import settings
        import os

        index_path = os.path.join(settings.VECTOR_STORE_PATH, "faiss_index.index")
        if not os.path.exists(index_path):
            return 0

        index = faiss.read_index(index_path)
        return int(index.ntotal)

    except Exception as e:
        logger.warning(f"Could not read FAISS index for stats: {e}")
        return -1  # -1 signals "index unreadable" to the caller
