"""
Document Controller (Phase 7 — fully updated)

Endpoints:
    POST   /api/documents/upload    → Upload a PDF (async background ingestion)
    GET    /api/documents/          → List all documents
    GET    /api/documents/{id}      → Get single document by ID
    DELETE /api/documents/{id}      → Delete document + chunks + FAISS rebuild
"""

import os
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.database.postgres import get_db
from app.models.document import Document
from app.models.chunk import Chunk
from app.utils.file_validator import validate_pdf
from app.utils.logger import get_logger
from app.views.document_views import DocumentResponse, DocumentListResponse
from app.services import document_service as doc_svc

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"]
)


# ---------------------------------------------------------------------------
# POST /api/documents/upload  — Upload a PDF (async)
# ---------------------------------------------------------------------------
@router.post("/upload", response_model=DocumentResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF document.

    PHASE 7 CHANGE — ASYNC INGESTION:
    -----------------------------------
    Previously the endpoint waited for the full pipeline to finish
    (extract → chunk → embed → FAISS) before returning a response.
    That could take 10–30 seconds for large PDFs, which is terrible UX.

    Now we:
      1. Validate the file
      2. Save it to disk
      3. Create the DB record (status = "processing")
      4. Return 202 Accepted IMMEDIATELY
      5. Run the full pipeline in the BACKGROUND

    The client can poll GET /api/documents/{id} to check when
    status changes from "processing" → "completed" or "failed".

    Returns:
        202 Accepted with DocumentResponse (status = "processing")

    Raises:
        400: File validation failed (not a PDF, too large)
        500: Could not save file or create DB record
    """
    # ------------------------------------------------------------------
    # STEP 1: Validate
    # ------------------------------------------------------------------
    await validate_pdf(file)
    logger.info(f"File validated: {file.filename}")

    # ------------------------------------------------------------------
    # STEP 2: Prepare storage
    # ------------------------------------------------------------------
    doc_id = str(uuid.uuid4())

    file_content = await file.read()
    file_size = len(file_content)
    await file.seek(0)

    safe_filename = f"{doc_id}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

    # ------------------------------------------------------------------
    # STEP 3: Save file to disk
    # ------------------------------------------------------------------
    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to disk: {file_path} ({file_size} bytes)")

    except Exception as e:
        logger.error(f"Failed to save file to disk: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file to disk: {str(e)}"
        )

    # ------------------------------------------------------------------
    # STEP 4: Create DB record (status = "processing")
    # ------------------------------------------------------------------
    try:
        document = Document(
            id=doc_id,
            filename=file.filename,
            user_id="default_user",
            file_path=file_path,
            file_size=file_size,
            status="processing"
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        logger.info(f"Document record created: id={doc_id}, filename={file.filename}")

    except Exception as e:
        logger.error(f"DB record creation failed: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create document record: {str(e)}"
        )

    # ------------------------------------------------------------------
    # STEP 5: Queue background processing
    # ------------------------------------------------------------------
    # BackgroundTasks runs AFTER the response is sent to the client.
    # The client gets 202 immediately; processing happens in the background.
    #
    # WHY NOT async/await here?
    # BackgroundTasks is FastAPI's built-in mechanism for this.
    # For production scale, we'd use Celery + Redis, but BackgroundTasks
    # is perfect for a single-server setup.
    # background_tasks.add_task(
    #     doc_svc.process_document,
    #     document_id=doc_id,
    #     file_path=file_path,
    #     db=db
    # )
    process_document_bg(
    document_id=doc_id,
    file_path=file_path
)
    logger.info(f"Background ingestion queued for document: {doc_id}")

    # ------------------------------------------------------------------
    # STEP 6: Return 202 immediately
    # ------------------------------------------------------------------
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        file_size=document.file_size,
        page_count=document.page_count,
        status=document.status,
        created_at=document.created_at,
        message=f"'{file.filename}' received. Processing in background — poll GET /api/documents/{doc_id} for status."
    )


# ---------------------------------------------------------------------------
# GET /api/documents/  — List all documents
# ---------------------------------------------------------------------------
@router.get("/", response_model=DocumentListResponse)
async def list_documents(db: Session = Depends(get_db)):
    """
    List all uploaded documents, newest first.
    """
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    logger.info(f"Listed {len(documents)} documents")

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                file_size=doc.file_size,
                page_count=doc.page_count,
                status=doc.status,
                created_at=doc.created_at,
                message="OK"
            )
            for doc in documents
        ],
        total=len(documents)
    )


# ---------------------------------------------------------------------------
# GET /api/documents/{document_id}  — Single document metadata
# ---------------------------------------------------------------------------
@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, db: Session = Depends(get_db)):
    """
    Get metadata for a single document by its ID.

    USE CASE — Polling after async upload:
    ---------------------------------------
    After POST /upload returns 202, the client polls this endpoint
    until status changes from "processing" to "completed" or "failed".

    Returns:
        200: DocumentResponse with current status
        404: Document not found

    Raises:
        404: No document with this ID in the database
    """
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        logger.warning(f"Document not found: {document_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found"
        )

    logger.info(f"Fetched document: {document_id} (status={document.status})")
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        file_size=document.file_size,
        page_count=document.page_count,
        status=document.status,
        created_at=document.created_at,
        message=f"Status: {document.status}"
    )


# ---------------------------------------------------------------------------
# DELETE /api/documents/{document_id}
# ---------------------------------------------------------------------------
@router.delete("/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """
    Delete a document and all associated data.

    WHAT GETS DELETED:
    ------------------
    1. PDF file from disk (uploads/ directory)
    2. All Chunk rows for this document in PostgreSQL
    3. The Document row in PostgreSQL
    4. FAISS index is rebuilt from all remaining documents

    WHY REBUILD FAISS?
    ------------------
    FAISS doesn't support deleting individual vectors cleanly — you'd
    need to rebuild the entire index to remove them anyway.
    We rebuild from all remaining chunk texts (re-embedded).

    Returns:
        200: { "message": "...", "document_id": "...", "chunks_deleted": N }
        404: Document not found

    Raises:
        404: No document with this ID
        500: Unexpected error during deletion
    """
    # ------------------------------------------------------------------
    # STEP 1: Find the document
    # ------------------------------------------------------------------
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        logger.warning(f"Delete requested for non-existent document: {document_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found"
        )

    filename = document.filename
    file_path = document.file_path

    try:
        # ------------------------------------------------------------------
        # STEP 2: Delete PDF from disk
        # ------------------------------------------------------------------
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file from disk: {file_path}")
        else:
            logger.warning(f"File not found on disk (already deleted?): {file_path}")

        # ------------------------------------------------------------------
        # STEP 3: Delete chunks from PostgreSQL
        # ------------------------------------------------------------------
        chunks_deleted = db.query(Chunk).filter(Chunk.document_id == document_id).delete()
        logger.info(f"Deleted {chunks_deleted} chunks for document: {document_id}")

        # ------------------------------------------------------------------
        # STEP 4: Delete document from PostgreSQL
        # ------------------------------------------------------------------
        db.delete(document)
        db.commit()
        logger.info(f"Document record deleted: {document_id} ({filename})")

        # ------------------------------------------------------------------
        # STEP 5: Rebuild FAISS index from remaining documents
        # ------------------------------------------------------------------
        # After deleting, the old FAISS index still has the deleted doc's
        # vectors. We rebuild it clean from all remaining chunks in the DB.
        logger.info("Rebuilding FAISS index after deletion...")
        _rebuild_faiss_index(db)
        logger.info("FAISS index rebuilt successfully")

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

    return {
        "message": f"Document '{filename}' deleted successfully",
        "document_id": document_id,
        "chunks_deleted": chunks_deleted
    }


def _rebuild_faiss_index(db: Session) -> None:
    """
    Rebuild the FAISS index from all chunks currently in the database.

    Called after a document is deleted to ensure the FAISS index no longer
    contains vectors from the deleted document.

    HOW IT WORKS:
    -------------
    1. Query all remaining Chunk rows from PostgreSQL
    2. Re-embed their text using the embedding model
    3. Build a new FAISS index from these fresh embeddings
    4. Save to disk (overwrites the old index)

    NOTE: This is O(N) in total chunks — acceptable for MVP scale.
    For large deployments, we'd use a vector DB like Pinecone or Weaviate
    that supports single-vector deletion.
    """
    from app.models.chunk import Chunk as ChunkModel
    from app.services.embedding_service import embed_chunks
    from app.services import faiss_service

    # Fetch all remaining chunks
    remaining_chunks = db.query(ChunkModel).all()

    if not remaining_chunks:
        # No chunks left — delete the FAISS index files entirely
        import glob
        for f in glob.glob("data/faiss_index.*"):
            os.remove(f)
        logger.info("No chunks remaining — FAISS index files removed")
        return

    # Delete the existing FAISS index files so build_and_save_index()
    # creates a FRESH index (instead of appending to one that still
    # contains the deleted document's vectors).
    import glob
    from app.config.settings import settings as _settings
    for f in glob.glob(str(_settings.VECTOR_STORE_PATH) + ".*"):
        try:
            os.remove(f)
        except:
            pass

    # Build chunk dicts in the same format embed_chunks() expects
    chunk_dicts = [
        {
            "text": chunk.text,
            "page_number": chunk.page_number,
            "chunk_index": chunk.id,   # use DB id as chunk_index
            "document_id": chunk.document_id,
        }
        for chunk in remaining_chunks
    ]
    # Re-embed and rebuild
    embedded = embed_chunks(chunk_dicts)
    faiss_service.build_and_save_index(embedded)
    logger.info(f"FAISS index rebuilt with {len(embedded)} vectors from {len(remaining_chunks)} chunks")


# ---------------------------------------------------------------------------
# POST /api/documents/{id}/reprocess  — Re-chunk & re-embed with new settings
# ---------------------------------------------------------------------------
@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentResponse,
    status_code=202,
    summary="Reprocess a document with current chunking settings",
    description=(
        "Deletes the document's existing chunks and re-runs the full pipeline "
        "(extract → chunk → embed → FAISS) using the current CHUNK_SIZE/CHUNK_OVERLAP. "
        "Useful after changing chunk settings without needing to re-upload the file. "
        "Runs in the background — poll GET /api/documents/{id} for status."
    )
)
async def reprocess_document_endpoint(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Re-process an existing document with current chunking settings."""

    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    if document.status == "processing":
        raise HTTPException(
            status_code=409,
            detail="Document is already being processed. Wait for it to complete."
        )

    # Mark as processing immediately so the UI reflects the change
    document.status = "processing"
    db.commit()
    db.refresh(document)

    # Run the actual reprocess in the background
    background_tasks.add_task(
        doc_svc.reprocess_document,
        document_id=document_id,
        db=db
    )

    logger.info(f"Queued reprocess for document '{document.filename}'")
    return document
from app.database.postgres import SessionLocal

def process_document_bg(document_id: str, file_path: str):
    db = SessionLocal()
    try:
        doc_svc.process_document(
            document_id=document_id,
            file_path=file_path,
            db=db
        )
    finally:
        db.close()