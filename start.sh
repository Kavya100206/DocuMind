#!/bin/bash
# start.sh â€” Render startup script
# Re-embeds all chunks from PostgreSQL and rebuilds the FAISS index before
# launching uvicorn. Render's filesystem is ephemeral so we must do this
# on every restart.

set -e

echo "=== DocuMind Startup ==="
echo "PORT: ${PORT:-8000}"

echo "Rebuilding FAISS index from database (re-embedding chunks)..."
python -c "
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from app.database.postgres import SessionLocal, init_db
    from app.models.chunk import Chunk
    from app.services import faiss_service
    from app.services.embedding_service import get_embedding

    init_db()
    db = SessionLocal()
    chunks = db.query(Chunk).all()
    db.close()

    if not chunks:
        print('No chunks in database â€” FAISS will be empty until documents are uploaded.')
    else:
        print(f'Found {len(chunks)} chunks. Re-embedding and rebuilding FAISS index...')
        chunk_dicts = []
        for c in chunks:
            emb = get_embedding(c.text)
            chunk_dicts.append({
                'embedding':   emb,
                'document_id': str(c.document_id),
                'page_number': c.page_number,
                'chunk_index': c.chunk_index,
                'text':        c.text,
                'char_count':  c.char_count or len(c.text),
            })

        faiss_service.build_and_save_index(chunk_dicts)
        print(f'FAISS index rebuilt with {len(chunk_dicts)} vectors.')

except Exception as e:
    print(f'FAISS rebuild warning: {e}')
    import traceback; traceback.print_exc()
    print('Continuing startup â€” uploads will rebuild index automatically.')
"

echo "Starting uvicorn on port ${PORT:-8000}..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --log-level info
