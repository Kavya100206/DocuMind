"""
Rebuild FAISS index properly — runs the FULL pipeline on each doc.

Difference from naive rebuild:
  Naive:  DB_page_chunks → embed → FAISS   (1 blob per page, bad)
  This:   DB_page_chunks → chunk_by_sections → embed → FAISS
              (5-10 focused 600-char chunks per page, good)

Run from project root with venv active:
    venv\Scripts\python rebuild_faiss.py
"""
import sys, os, glob
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()

from app.config.settings import settings
from app.database.postgres import SessionLocal
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.chunking_service import chunk_by_sections
from app.services.embedding_service import embed_chunks
from app.services import faiss_service

db = SessionLocal()
try:
    docs = db.query(Document).filter(Document.status == "completed").all()
    print(f"Found {len(docs)} completed documents in DB\n")

    all_embedded = []

    for doc in docs:
        # Load this doc's page-level chunks from DB
        page_chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).all()
        if not page_chunks:
            print(f"  SKIP {doc.filename} — no page chunks in DB")
            continue

        # Reconstruct the pages_data format that chunk_by_sections expects
        pages_data = [
            {
                "page_number": pc.page_number,
                "text":        pc.text,
                "char_count":  pc.char_count,
                "is_empty":    not pc.text or len(pc.text.strip()) == 0,
            }
            for pc in page_chunks
        ]

        # Run section chunking (600-char pieces) — the same as upload pipeline
        section_chunks = chunk_by_sections(pages_data)
        for c in section_chunks:
            c["document_id"] = doc.id

        print(f"  {doc.filename}: {len(page_chunks)} page(s) → {len(section_chunks)} section chunks")

        # Embed section chunks
        embedded = embed_chunks(section_chunks)
        all_embedded.extend(embedded)

    print(f"\nTotal section chunks to index: {len(all_embedded)}")

    if not all_embedded:
        print("Nothing to index — upload some documents first.")
        sys.exit(0)

    # Delete old FAISS files
    for f in glob.glob(settings.VECTOR_STORE_PATH + ".*"):
        os.remove(f)
        print(f"Deleted: {f}")

    # Build fresh index
    faiss_service.build_and_save_index(all_embedded)
    print(f"\nSUCCESS: FAISS rebuilt with {len(all_embedded)} focused section vectors.")
    print("All document IDs match the current DB.")

except Exception as e:
    import traceback; traceback.print_exc()
    sys.exit(1)
finally:
    db.close()
