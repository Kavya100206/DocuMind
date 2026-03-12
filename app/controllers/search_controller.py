"""
Search Controller

WHAT DOES THIS FILE DO?
------------------------
This is the HTTP layer for search.
It exposes one endpoint:

    GET /api/search?q=your+question&k=5

When someone hits this URL, this controller:
  1. Reads the query from the URL parameters
  2. Calls retrieval_service.search_chunks()
  3. Shapes the response using SearchResponse

WHY GET AND NOT POST?
----------------------
Search is a READ operation — we're not creating or changing anything.
GET is the correct HTTP method for reads.
The query goes in the URL as a query parameter: ?q=...

This also means you can test it directly in your browser:
  http://localhost:8000/api/search?q=what+are+the+responsibilities

QUERY PARAMETERS EXPLAINED:
-----------------------------
URL: /api/search?q=your+question&k=5&document_id=abc-123

  q           = the search query (required)
  k           = how many results to return (optional, default 5)
  document_id = only search this one document (optional)

FastAPI reads these automatically from the URL using Query().
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from app.database.postgres import get_db
from app.services import retrieval_service
from app.views.search_views import SearchResponse, ChunkResult
from app.config.settings import settings
from typing import Optional

router = APIRouter(prefix="/api", tags=["Search"])


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search documents",
    description="Search across all uploaded documents using natural language."
)
def search_documents(
    q: str = Query(
        ...,  # "..." means REQUIRED — no default, must be provided
        min_length=1,
        max_length=500,
        description="Your search query or question",
        example="What are the key responsibilities?"
    ),
    k: int = Query(
        default=settings.TOP_K_RESULTS,  # uses value from settings.py (currently 10)
        ge=1,
        le=20,
        description="Number of results to return"
    ),
    document_id: Optional[str] = Query(
        default=None,
        description="Optional: restrict search to a specific document ID"
    ),
    db: Session = Depends(get_db)
):
    """
    Search your uploaded documents with a natural language question.

    HOW FASTAPI READS THE URL:
    ---------------------------
    When you call: GET /api/search?q=revenue+growth&k=3

    FastAPI automatically:
      - Sets q = "revenue growth"
      - Sets k = 3
      - Calls get_db() and injects the db session

    You don't write any parsing code — FastAPI handles it all.

    WHAT HAPPENS STEP BY STEP:
    ---------------------------
    1. FastAPI validates q (not empty, under 500 chars)
    2. FastAPI validates k (between 1 and 20)
    3. retrieval_service.search_chunks() is called
    4. Results are shaped into SearchResponse
    5. JSON is returned

    Args:
        q:           Search query string from URL param ?q=
        k:           Number of results from URL param &k=
        document_id: Optional document filter from &document_id=
        db:          Injected database session (from Depends(get_db))
    """

    print(f"\n🔍 Search request: '{q}' (k={k})")

    # Check if any documents have been uploaded + indexed
    # If no FAISS index exists yet, give a helpful error
    import os
    from app.config.settings import settings
    index_path = settings.VECTOR_STORE_PATH + ".index"
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail=(
                "No documents have been indexed yet. "
                "Please upload at least one PDF first using "
                "POST /api/documents/upload"
            )
        )

    # Call the retrieval service
    results = retrieval_service.search_chunks(
        query=q,
        db=db,
        k=k,
        document_id=document_id
    )

    # Build human-readable message based on result count
    if len(results) == 0:
        message = (
            f"No relevant results found for '{q}'. "
            "Try rephrasing your question or upload more documents."
        )
    elif len(results) == 1:
        message = f"Found 1 relevant result for your query."
    else:
        message = f"Found {len(results)} relevant results for your query."

    # Convert each result dict into a ChunkResult Pydantic object
    # This ensures the response matches our defined schema exactly
    chunk_results = [ChunkResult(**r) for r in results]

    return SearchResponse(
        query=q,
        total_results=len(chunk_results),
        results=chunk_results,
        message=message
    )
