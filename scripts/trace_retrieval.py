import sys
from app.database.postgres import SessionLocal
from app.services.retrieval_service import _search_faiss_with_filters, _apply_lexical_boost
from app.services import bm25_service
from app.services import reranker_service
from app.models.document import Document
from app.models.chunk import Chunk

def trace_retrieval(query: str):
    print(f"\n{'='*60}\n TRACING QUERY: '{query}'\n{'='*60}")
    db = SessionLocal()
    try:
        # Step 1: FAISS
        faiss_chunks = _search_faiss_with_filters(query=query, db=db, doc_ids=None, k=20)
        print(f"\n>>> FAISS RETURNED {len(faiss_chunks)} CHUNKS:")
        title_found_faiss = False
        for i, c in enumerate(faiss_chunks[:10]):
            text = c['text'].lower()
            is_target = "team member" in text or "orbitmind" in text
            if is_target: title_found_faiss = True
            mark = "*TARGET*" if is_target else "        "
            print(f"[{i+1:<2}] {mark} (score: {c['similarity_score']:.3f}) {c['text'][:80].replace(chr(10), ' ')}")
            
        if not title_found_faiss:
            print(f"\n❌ THE TARGET CHUNK IS MISSING FROM FAISS OUTPUT COMPLETELY!")
        
        # Step 2: Lexical Boost
        boosted_chunks = _apply_lexical_boost(query, faiss_chunks)

        # Step 3: BM25
        bm25_chunks = bm25_service.rerank_with_bm25(query, boosted_chunks)
        print(f"\n>>> AFTER BM25 HYBRID:")
        for i, c in enumerate(bm25_chunks[:10]):
            text = c['text'].lower()
            is_target = "team member" in text or "orbitmind" in text
            mark = "*TARGET*" if is_target else "        "
            print(f"[{i+1:<2}] {mark} (hyb: {c.get('hybrid_score',0):.3f}) {c['text'][:80].replace(chr(10), ' ')}")

        # Step 4: Cross encoder
        final_chunks = reranker_service.rerank(query, bm25_chunks, top_n=5)
        print(f"\n>>> AFTER CROSS ENCODER (Final 5):")
        for i, c in enumerate(final_chunks):
            text = c['text'].lower()
            is_target = "team member" in text or "orbitmind" in text
            mark = "*TARGET*" if is_target else "        "
            print(f"[{i+1:<2}] {mark} (rr: {c.get('reranker_score',0):.3f}) {c['text'][:80].replace(chr(10), ' ')}")
            
    finally:
        db.close()

if __name__ == "__main__":
    trace_retrieval("what is the project title and team member names?")
