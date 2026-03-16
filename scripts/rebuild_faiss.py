"""
Rebuild FAISS index properly — runs FULL pipeline from DB chunks.

Safe for:
- manual run
- FastAPI startup import
- Railway deployment
"""

import os
import glob
from dotenv import load_dotenv
load_dotenv()

from app.config.settings import settings
from app.database.postgres import SessionLocal
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.chunking_service import chunk_by_sections
from app.services.embedding_service import embed_chunks
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

        all_embedded = []

        for doc in docs:
            page_chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).all()

            if not page_chunks:
                logger.info(f"SKIP {doc.filename} — no chunks")
                continue

            pages_data = [
                {
                    "page_number": pc.page_number,
                    "text": pc.text,
                    "char_count": pc.char_count,
                    "is_empty": not pc.text or len(pc.text.strip()) == 0,
                }
                for pc in page_chunks
            ]

            section_chunks = chunk_by_sections(pages_data)

            for c in section_chunks:
                c["document_id"] = doc.id

            logger.info(
                f"{doc.filename}: {len(page_chunks)} page → {len(section_chunks)} chunks"
            )

            embedded = embed_chunks(section_chunks)
            all_embedded.extend(embedded)

        logger.info(f"Total chunks to index: {len(all_embedded)}")

        if not all_embedded:
            logger.info("No data to index")
            return

        # delete old index
        for f in glob.glob(settings.VECTOR_STORE_PATH + ".*"):
            os.remove(f)
            logger.info(f"Deleted old index file: {f}")

        faiss_service.build_and_save_index(all_embedded)

        # 🚨 CRITICAL FAISS CACHE CLEAR 🚨
        # Force the main FastAPI search thread to reload the new index from disk
        faiss_service._cached_index = None
        faiss_service._cached_metadata = None

        logger.info(
            f"SUCCESS: FAISS rebuilt with {len(all_embedded)} vectors and global cache flushed."
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