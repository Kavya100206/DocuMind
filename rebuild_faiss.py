import json
import os
import faiss
from sqlalchemy.orm import Session
from app.database.postgres import SessionLocal
from app.models.document import Document
from app.models.chunk import Chunk
from app.services import chunking_service
from app.services import faiss_service
from app.services.embedding_service import embed_chunks
from app.config.settings import settings

def rebuild_index():
    print(f"Starting FULL FAISS rebuild with chunk size: {settings.CHUNK_SIZE}")
    db = SessionLocal()
    try:
        # 1. Delete existing FAISS files
        faiss_index_path = settings.VECTOR_STORE_PATH + ".index"
        faiss_meta_path = settings.VECTOR_STORE_PATH + ".meta"
        
        if os.path.exists(faiss_index_path):
            os.remove(faiss_index_path)
            print(f"Deleted existing FAISS index: {faiss_index_path}")
            
        if os.path.exists(faiss_meta_path):
            os.remove(faiss_meta_path)
            print(f"Deleted existing FAISS metadata: {faiss_meta_path}")

        # 2. Get completed documents
        documents = db.query(Document).filter(Document.status == "completed").all()
        print(f"Found {len(documents)} completed documents.")
        
        if not documents:
            print("No completed documents found. Exiting.")
            return

        all_chunks = []
        for doc in documents:
            print(f"\nProcessing document: {doc.filename} (ID: {doc.id})")
            
            # Fetch existing page-level chunks (we rebuild from these rather than re-extracting PDF)
            page_chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).all()
            
            # Reconstruct pages_data format expected by chunking_service
            pages_data = []
            for pc in page_chunks:
                pages_data.append({
                    "page_number": pc.page_number,
                    "text": pc.text,
                    "char_count": pc.char_count,
                    "is_empty": False
                })
                
            print(f"  Loaded {len(pages_data)} pages from DB.")
            
            # 3. Re-chunk with new settings
            small_chunks = chunking_service.chunk_by_sections(pages_data)
            print(f"  Created {len(small_chunks)} small chunks.")
            
            # Add document_id
            for c in small_chunks:
                c["document_id"] = doc.id
                all_chunks.extend(small_chunks)
                
        print(f"\nTotal chunks to embed: {len(all_chunks)}")
        
        # 4. Embed
        embedded_chunks = embed_chunks(all_chunks)
        print(f"Embedded {len(embedded_chunks)} chunks.")
        
        # 5. Build and save index
        faiss_service.build_and_save_index(embedded_chunks)
        print("\nRebuild complete!")
        
        # Verify
        if os.path.exists(faiss_index_path):
             index = faiss.read_index(faiss_index_path)
             print(f"Verification: FAISS index has {index.ntotal} vectors.")
             
    except Exception as e:
        print(f"Error during rebuild: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    rebuild_index()
