import sys
from app.database.postgres import SessionLocal
from app.services.retrieval_service import search_chunks

def test_retrieval(query: str):
    print(f"\n--- Testing Query: '{query}' ---")
    db = SessionLocal()
    try:
        results = search_chunks(query, db=db)
        
        print(f"\n[ Final Top {len(results)} Chunks Sent to LLM ]")
        for i, c in enumerate(results, 1):
            faiss_score = c.get('similarity_score', 0)
            hybrid_score = c.get('hybrid_score', 0)
            reranker_score = c.get('reranker_score', 0)
            print(f"{i}. [RR: {reranker_score:.3f} | Hyb: {hybrid_score:.3f} | FAISS: {faiss_score:.3f}] Doc: {c.get('document_name')} (Page {c.get('page_number')})")
            print(f"   Text: {c.get('text')[:150].replace(chr(10), ' ')}...")
    finally:
        db.close()

if __name__ == "__main__":
    test_retrieval("What is the project title mentioned in the OrbitMind Advanced Machine Learning project proposal?")
    test_retrieval("Who are the members of the OrbitMind team in the proposal?")
