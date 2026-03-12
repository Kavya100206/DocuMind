"""
Search Views (Response Schemas)

WHAT IS THIS FILE?
------------------
Same idea as document_views.py — these are Pydantic models that define
the SHAPE of the JSON our search endpoint sends back.

Why define the shape upfront?
  1. FastAPI uses these to auto-validate — if a field is missing or
     the wrong type, FastAPI throws a clear error instead of crashing.
  2. Auto-documentation — the /docs page will show exactly what fields
     the search response contains.
  3. Clean API contract — whoever builds the frontend knows exactly
     what to expect.

THE RESPONSE STRUCTURE:
------------------------
A search call returns:
  {
    "query": "What are the key responsibilities?",
    "total_results": 3,
    "results": [
      {
        "text": "Key responsibilities include...",
        "page_number": 4,
        "document_id": "abc-123",
        "document_name": "job_description.pdf",
        "chunk_index": 2,
        "similarity_score": 0.87
      },
      ...
    ]
  }

Each item in "results" is a ChunkResult.
The whole response is a SearchResponse.
"""

from typing import List, Optional
from pydantic import BaseModel


class ChunkResult(BaseModel):
    """
    Represents ONE matching chunk from the search results.

    Each field:
      text:             The actual text content of the chunk
                        (the relevant paragraph you asked about)

      page_number:      Which page this chunk came from
                        (used for citations — "this is on page 4")

      document_id:      The UUID of the document in our database
                        (useful if the frontend wants to link to the doc)

      document_name:    The original filename e.g. "annual_report.pdf"
                        (human-readable — shown in the UI)

      chunk_index:      Position of this chunk within its page (0, 1, 2...)
                        (useful for ordering or debugging)

      similarity_score: How relevant this chunk is to the query
                        Range: 0.0 (irrelevant) to 1.0 (perfect match)
                        Anything above 0.3 is considered relevant.
    """

    text: str
    page_number: int
    document_id: Optional[str] = None
    document_name: str
    chunk_index: Optional[int] = None
    similarity_score: float

    # This allows Pydantic to read data from SQLAlchemy model objects
    # (same setting we used in DocumentResponse)
    model_config = {"from_attributes": True}


class SearchResponse(BaseModel):
    """
    The full response for a search request.

    Fields:
      query:         The original question the user asked
                     (echoed back so the client knows what was searched)

      total_results: How many chunks were found above the threshold
                     (could be 0 if nothing was relevant)

      results:       The list of matching chunks (ChunkResult objects)
                     Sorted by similarity_score descending (best first)

      message:       Human-readable summary of the search outcome
                     e.g. "Found 3 relevant results for your query."
    """

    query: str
    total_results: int
    results: List[ChunkResult]
    message: str


class SearchRequest(BaseModel):
    """
    OPTIONAL: For POST-based search (we'll use GET query params instead).
    Kept here for reference if we ever want to accept JSON body requests.

    Fields:
      query:       The user's question
      k:           How many results to return (default: 5)
      document_id: Optional — limit search to one specific document
    """

    query: str
    k: int = 5
    document_id: Optional[str] = None
