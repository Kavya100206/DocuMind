
import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config.settings import settings
from app.services import retrieval_service
from app.database.postgres import get_db

def test_full_retrieval():
    # Setup DB session
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    query = "list all projects"
    print(f"\n--- Testing Full Retrieval for: '{query}' ---")
    
    try:
        # We manually call search_chunks to see the logs
        # The service itself prints the "RETRIEVAL" table if logger is configured or print is used
        results = retrieval_service.search_chunks(query=query, db=db, k=10)
        
        print(f"\nResults count: {len(results)}")
        for i, res in enumerate(results, 1):
            print(f"  [{i}] doc: {res.get('document_name')} | score: {res.get('hybrid_score', res.get('similarity_score'))}")
            print(f"      section: {res.get('section_name', 'N/A')}")
            print(f"      text: {res.get('text')[:150]}...")
            print("-" * 20)
            
    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    test_full_retrieval()
