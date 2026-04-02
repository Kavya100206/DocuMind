"""
Rebuild FAISS index — MEMORY-OPTIMISED version for Railway/Render free tier.

KEY CHANGE (memory fix):
- OLD: Loaded all chunks from DB → re-embedded every one (loaded PyTorch + model)
- NEW: Reads stored embeddings from DB → builds FAISS directly (NO model needed)

This eliminates the ~200MB memory spike from loading the embedding model during
startup. The embedding model is only loaded lazily when a user query comes in.

Safe for:
- manual run
- FastAPI startup import
- Railway / Render deployment
"""

import json
import os
import sys
import glob

# Ensure project root is on the path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.config.settings import settings
from app.database.postgres import SessionLocal
from app.models.document import Document
from app.models.chunk import Chunk
from app.services import faiss_service


def rebuild_faiss_index():
    from app.utils.logger import get_logger
    logger = get_logger(__name__)

    faiss_service.IS_BUILDING = True
    logger.info("Setting IS_BUILDING = True. Starting background FAISS rebuild.")

    db = SessionLocal()

    try:
        docs = db.query(Document).filter(Document.status == "completed").all()
        logger.info(f"Found {len(docs)} completed documents in DB")

        all_chunk_dicts = []

        for doc in docs:
            page_chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).all()

            if not page_chunks:
                logger.info(f"SKIP {doc.filename} — no chunks")
                continue

            # Try to use stored embeddings first (no model needed!)
            chunks_with_embeddings = [c for c in page_chunks if c.embedding]
            chunks_without_embeddings = [c for c in page_chunks if not c.embedding]

            if chunks_with_embeddings:
                logger.info(
                    f"{doc.filename}: {len(chunks_with_embeddings)} chunks with stored embeddings"
                )
                for c in chunks_with_embeddings:
                    all_chunk_dicts.append({
                        "embedding": json.loads(c.embedding),
                        "document_id": str(c.document_id),
                        "page_number": c.page_number,
                        "chunk_index": c.id,
                        "text": c.text,
                        "char_count": c.char_count or len(c.text),
                    })

            if chunks_without_embeddings:
                # Fallback: re-embed chunks that don't have stored embeddings
                # This path is only hit for old data uploaded before this change
                logger.warning(
                    f"{doc.filename}: {len(chunks_without_embeddings)} chunks need re-embedding (no stored vectors)"
                )
                from app.services.chunking_service import chunk_by_sections
                from app.services.embedding_service import embed_chunks

                pages_data = [
                    {
                        "page_number": pc.page_number,
                        "text": pc.text,
                        "char_count": pc.char_count,
                        "is_empty": not pc.text or len(pc.text.strip()) == 0,
                    }
                    for pc in chunks_without_embeddings
                ]

                section_chunks = chunk_by_sections(pages_data)
                for c in section_chunks:
                    c["document_id"] = doc.id

                embedded = embed_chunks(section_chunks)

                # Save embeddings to DB for next restart
                for emb_chunk in embedded:
                    embedding_vec = emb_chunk.get("embedding")
                    if embedding_vec:
                        db_chunk = db.query(Chunk).filter(
                            Chunk.document_id == doc.id,
                            Chunk.text.ilike(f"{emb_chunk['text'][:80]}%")
                        ).first()
                        if db_chunk:
                            db_chunk.embedding = json.dumps(embedding_vec)

                db.commit()
                all_chunk_dicts.extend(embedded)

        logger.info(f"Total chunks to index: {len(all_chunk_dicts)}")

        if not all_chunk_dicts:
            logger.info("No data to index")
            return

        # delete old index
        for f in glob.glob(settings.VECTOR_STORE_PATH + ".*"):
            os.remove(f)
            logger.info(f"Deleted old index file: {f}")

        faiss_service.build_and_save_index(all_chunk_dicts)

        # Force the main FastAPI search thread to reload the new index from disk
        faiss_service._cached_index = None
        faiss_service._cached_metadata = None

        logger.info(
            f"SUCCESS: FAISS rebuilt with {len(all_chunk_dicts)} vectors and global cache flushed."
        )

    except Exception as e:
        import traceback
        logger.error(f"CRITICAL ERROR during FAISS rebuild: {e}")
        logger.error(traceback.format_exc())

    finally:
        faiss_service.IS_BUILDING = False
        logger.info("Setting IS_BUILDING = False. Background rebuild finished or crashed.")
        db.close()

if __name__ == "__main__":
    rebuild_faiss_index()