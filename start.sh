#!/bin/bash
# start.sh — Railway / Render startup script
#
# MEMORY-OPTIMISED: Rebuilds FAISS from stored embeddings in the database.
# Does NOT re-embed chunks (no model loading needed at startup).
# The embedding model loads lazily on the first user query instead.

set -e

echo "=== DocuMind Startup ==="
echo "PORT: ${PORT:-8000}"

echo "Rebuilding FAISS index from stored embeddings in database..."
python -c "
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from app.database.postgres import SessionLocal, init_db
    from app.models.chunk import Chunk
    from app.services import faiss_service
    import json

    init_db()
    db = SessionLocal()
    chunks = db.query(Chunk).all()
    db.close()

    if not chunks:
        print('No chunks in database — FAISS will be empty until documents are uploaded.')
    else:
        # Count chunks with stored embeddings vs those needing re-embedding
        with_emb = [c for c in chunks if c.embedding]
        without_emb = [c for c in chunks if not c.embedding]

        if with_emb:
            print(f'Found {len(with_emb)} chunks with stored embeddings. Building FAISS (no model needed)...')
            chunk_dicts = []
            for c in with_emb:
                chunk_dicts.append({
                    'embedding':   json.loads(c.embedding),
                    'document_id': str(c.document_id),
                    'page_number': c.page_number,
                    'chunk_index': c.id,
                    'text':        c.text,
                    'char_count':  c.char_count or len(c.text),
                })
            faiss_service.build_and_save_index(chunk_dicts)
            print(f'FAISS index rebuilt with {len(chunk_dicts)} vectors (from stored embeddings).')

        if without_emb:
            print(f'Warning: {len(without_emb)} chunks have no stored embeddings.')
            print('These will be re-embedded in the background after server starts.')

        if not with_emb and without_emb:
            print('No stored embeddings found. FAISS rebuild deferred to background task.')

except Exception as e:
    print(f'FAISS rebuild warning: {e}')
    import traceback; traceback.print_exc()
    print('Continuing startup — uploads will rebuild index automatically.')
"

echo "Starting uvicorn on port ${PORT:-8000}..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --log-level info
