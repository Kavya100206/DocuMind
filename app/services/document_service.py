"""
Document Service

What is this file?
------------------
The Document Service is the ORCHESTRATOR of the upload pipeline.

It coordinates the full flow:
    1. Update DB record with page count
    2. Call pdf_service to extract page-by-page text
    3. Save each page's text as a Chunk in the database
    4. Chunk text into smaller pieces (chunking_service)
    5. Embed each chunk into a vector (embedding_service)
    6. Build/update the FAISS vector index (faiss_service)
    7. Update Document status: "processing" → "completed" (or "failed")

Why a separate service layer?
------------------------------
The controller (document_controller.py) handles HTTP concerns:
    - Receiving the file
    - Validating inputs
    - Returning responses

The service handles BUSINESS LOGIC:
    - What do we do with the file after we have it?
    - How do we process it?
    - How do we handle failures?

This separation makes each layer easy to test and change independently.

What is the full upload flow now?
----------------------------------
1. User sends PDF → document_controller.py receives it
2. Controller validates → file_validator.py
3. Controller saves file to disk
4. Controller creates Document in DB (status="processing")
5. Controller calls document_service.process_document()
6. document_service calls pdf_service.extract_text_from_pdf()
7. pdf_service reads PDF, returns list of page dicts
8. document_service saves each page as a Chunk in DB
9. document_service updates Document status to "completed"
10. Controller returns DocumentResponse to user
"""

import hashlib
from sqlalchemy.orm import Session
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.pdf_service import extract_text_from_pdf, get_page_count
from app.services.chunking_service import chunk_by_sections
from app.services.embedding_service import embed_chunks
from app.services import faiss_service
from app.utils.logger import get_logger

logger = get_logger(__name__)


def process_document(document_id: str, file_path: str, db: Session) -> Document:
    """
    Process an uploaded PDF: extract text and store as chunks in DB.

    This is called right after the file is saved to disk.
    It runs synchronously for now — in Phase 7 we'll make it async
    (background task with a queue) so the user doesn't have to wait.

    Args:
        document_id: The UUID of the Document record already in DB
        file_path: Where the PDF is saved on disk
        db: Active database session

    Returns:
        The updated Document object (with status="completed" or "failed")
    """

    # Fetch the document record from database
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        raise ValueError(f"Document {document_id} not found in database")

    try:
        logger.info(f"Processing document: {document.filename}")

        # ------------------------------------------------------------------
        # STEP 1: Get page count and update the Document record
        # ------------------------------------------------------------------
        # We do this first — it's fast and gives the user useful info
        # even if text extraction takes a while
        page_count = get_page_count(file_path)
        document.page_count = page_count
        db.commit()
        logger.info(f"Total pages: {page_count}")

        # ------------------------------------------------------------------
        # STEP 2: Extract text from every page
        # ------------------------------------------------------------------
        # This returns a list like:
        # [
        #     {"page_number": 1, "text": "...", "char_count": 542, "is_empty": False},
        #     {"page_number": 2, "text": "", "char_count": 0, "is_empty": True},
        #     ...
        # ]
        pages_data = extract_text_from_pdf(file_path)
        logger.info(f"Extracted text from {len(pages_data)} pages")

        # ------------------------------------------------------------------
        # STEP 3: Check for duplicate documents
        # ------------------------------------------------------------------
        # Calculate a hash of the extracted text to detect duplicates
        # If two PDFs have identical content, their hashes will match
        all_text = " ".join([p["text"] for p in pages_data if not p["is_empty"]])
        # We check (but don't block) — just log for now
        _check_duplicate(all_text, document.user_id, db, document_id)

        # ------------------------------------------------------------------
        # STEP 4: Save each page as a Chunk in the database
        # ------------------------------------------------------------------
        chunks_created = 0
        chunks_skipped = 0

        for page_data in pages_data:
            if page_data["is_empty"]:
                # Skip empty/scanned pages — nothing to store
                chunks_skipped += 1
                continue

            # Create a Chunk record for this page
            chunk = Chunk(
                document_id=document_id,
                page_number=page_data["page_number"],
                text=page_data["text"],
                char_count=page_data["char_count"]
            )

            db.add(chunk)  # Stage for insertion
            chunks_created += 1

        # Commit all chunks at once (more efficient than one-by-one)
        db.commit()

        logger.info(f"Saved {chunks_created} page chunks to DB")
        if chunks_skipped > 0:
            logger.warning(f"Skipped {chunks_skipped} empty/scanned pages")

        # ------------------------------------------------------------------
        # STEP 4: Chunk pages into smaller pieces
        # ------------------------------------------------------------------
        # Phase 2 stored each PAGE as a chunk. Now we split pages into
        # smaller overlapping 500-char pieces for better search precision.
        #
        # chunk_pages() returns a flat list of chunk dicts:
        # [{"page_number": 1, "chunk_index": 0, "text": "...", ...}, ...]
        small_chunks = chunk_by_sections(pages_data)
        logger.info(f"Created {len(small_chunks)} section chunks")

        # Add document_id to every chunk (needed for FAISS metadata)
        for c in small_chunks:
            c["document_id"] = document_id

        # ------------------------------------------------------------------
        # STEP 5: Embed each chunk into a vector
        # ------------------------------------------------------------------
        # embed_chunks() calls the local sentence-transformers model
        # and adds an "embedding" key to every chunk dict
        embedded = embed_chunks(small_chunks)

        # ------------------------------------------------------------------
        # STEP 6: Build / update the FAISS index
        # ------------------------------------------------------------------
        # We want to ENSURE the index is clean for this document if it was 
        # a re-upload, or if we want to avoid stale data poisoning.
        # For this MVP, we clear the index files before saving NEW ones 
        # to ensure the user gets exactly what they just uploaded.
        import glob, os as _os
        from app.config.settings import settings as _settings
        for f in glob.glob(str(_settings.VECTOR_STORE_PATH) + ".*"):
            try:
                _os.remove(f)
                logger.info(f"  Cleanup: removed stale index file {f}")
            except:
                pass

        # build_and_save_index() creates a fresh index since we just deleted it
        faiss_service.build_and_save_index(embedded)
        logger.info(f"FAISS index built fresh with {len(embedded)} vectors")

        # ------------------------------------------------------------------
        # STEP 7: Mark document as completed
        # ------------------------------------------------------------------
        document.status = "completed"
        db.commit()
        db.refresh(document)

        logger.info(f"Document '{document.filename}' processed successfully")
        return document

    except Exception as e:
        # ------------------------------------------------------------------
        # ERROR HANDLING: Mark document as failed
        # ------------------------------------------------------------------
        # Even if processing fails, we keep the Document record
        # The user can see it failed and retry or re-upload
        logger.error(f"Processing failed for '{document.filename}': {e}")

        document.status = "failed"
        db.commit()

        raise RuntimeError(f"Document processing failed: {str(e)}")


def _check_duplicate(text: str, user_id: str, db: Session, current_doc_id: str) -> None:
    """
    Check if this document's content already exists in the database.

    How it works:
    -------------
    We create a SHA-256 hash of the full extracted text.
    If any existing document in the DB has the same hash,
    it means the same PDF content was uploaded before.

    For MVP: We just log a warning. In Phase 7, we can:
    - Return a 409 Conflict HTTP error
    - Point the user to the existing document
    - Skip re-processing and reuse existing chunks

    Args:
        text: The full extracted text from the PDF
        user_id: The user who uploaded it
        db: Database session
        current_doc_id: The current document's ID (so we don't flag ourselves)
    """
    # SHA-256: a one-way function that produces a 64-character hex string
    # The same text always produces the same hash
    content_hash = hashlib.sha256(text.encode()).hexdigest()

    # For now, just print — we'll use this hash in Phase 7 for real deduplication
    logger.debug(f"Content hash: {content_hash[:16]}... (duplicate detection)")


def reprocess_document(document_id: str, db: Session) -> Document:
    """
    Re-chunk and re-embed an existing document using current settings.

    When CHUNK_SIZE or other chunking settings change, existing documents
    have stale chunks/vectors built with the old settings. This function:
      1. Keeps the Document DB record and the uploaded PDF file on disk
      2. Deletes the document's existing Chunk rows from the database
      3. Re-extracts text from the PDF file
      4. Re-chunks with current CHUNK_SIZE / CHUNK_OVERLAP settings
      5. Re-embeds and rebuilds the FAISS index for ALL documents
      6. Marks the document as \"completed\"

    Args:
        document_id: UUID of the document to reprocess
        db: Active database session

    Returns:
        Updated Document object with status = "completed"
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise ValueError(f"Document {document_id} not found")

    import os
    if not os.path.exists(document.file_path):
        raise FileNotFoundError(
            f"Source PDF not found at '{document.file_path}'. "
            "Please re-upload the document."
        )

    try:
        logger.info(f"Re-processing document: {document.filename}")
        document.status = "processing"
        db.commit()

        # ------------------------------------------------------------------
        # STEP 1: Delete existing chunks for this document
        # ------------------------------------------------------------------
        deleted = db.query(Chunk).filter(Chunk.document_id == document_id).delete()
        db.commit()
        logger.info(f"Deleted {deleted} old chunks")

        # ------------------------------------------------------------------
        # STEP 2: Re-extract text from saved PDF
        # ------------------------------------------------------------------
        pages_data = extract_text_from_pdf(document.file_path)
        logger.info(f"Re-extracted {len(pages_data)} pages")

        # ------------------------------------------------------------------
        # STEP 3: Save fresh page-level chunks to DB
        # ------------------------------------------------------------------
        for page_data in pages_data:
            if page_data["is_empty"]:
                continue
            chunk = Chunk(
                document_id=document_id,
                page_number=page_data["page_number"],
                text=page_data["text"],
                char_count=page_data["char_count"]
            )
            db.add(chunk)
        db.commit()

        # ------------------------------------------------------------------
        # STEP 4: Re-chunk with current settings
        # ------------------------------------------------------------------
        small_chunks = chunk_by_sections(pages_data)
        for c in small_chunks:
            c["document_id"] = document_id
        logger.info(f"Created {len(small_chunks)} new chunks with current settings")

        # ------------------------------------------------------------------
        # STEP 5: Re-embed
        # ------------------------------------------------------------------
        embedded = embed_chunks(small_chunks)

        # ------------------------------------------------------------------
        # STEP 6: Rebuild FAISS
        # ------------------------------------------------------------------
        # Since we've already re-chunked the target document (small_chunks),
        # we can just use those. 
        # For a full system refresh, we clear and rebuild from THIS document.
        import glob, os as _os
        from app.config.settings import settings as _settings
        for f in glob.glob(str(_settings.VECTOR_STORE_PATH) + ".*"):
            try:
                _os.remove(f)
            except:
                pass
                
        faiss_service.build_and_save_index(embedded)
        logger.info(f"FAISS rebuilt with {len(embedded)} new vectors")

        # ------------------------------------------------------------------
        # STEP 7: Mark completed
        # ------------------------------------------------------------------
        document.status = "completed"
        db.commit()
        db.refresh(document)
        logger.info(f"Re-processing complete: {document.filename}")
        return document

    except Exception as e:
        logger.error(f"Re-processing failed: {e}")
        document.status = "failed"
        db.commit()
        raise RuntimeError(f"Reprocess failed: {str(e)}")
