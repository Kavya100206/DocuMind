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
    db = SessionLocal()

    try:
        docs = db.query(Document).filter(Document.status == "completed").all()
        print(f"Found {len(docs)} completed documents in DB\n")

        all_embedded = []

        for doc in docs:
            page_chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).all()

            if not page_chunks:
                print(f"SKIP {doc.filename} — no chunks")
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

            print(
                f"{doc.filename}: {len(page_chunks)} page → {len(section_chunks)} chunks"
            )

            embedded = embed_chunks(section_chunks)
            all_embedded.extend(embedded)

        print(f"\nTotal chunks to index: {len(all_embedded)}")

        if not all_embedded:
            print("No data to index")
            return

        # delete old index
        for f in glob.glob(settings.VECTOR_STORE_PATH + ".*"):
            os.remove(f)
            print(f"Deleted: {f}")

        faiss_service.build_and_save_index(all_embedded)

        print(
            f"\nSUCCESS: FAISS rebuilt with {len(all_embedded)} vectors"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    rebuild_faiss_index()