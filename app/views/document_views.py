"""
Document Views (Response Schemas)

What is this file?
------------------
This file defines the *shape* of JSON data that our API returns.
These are called "schemas" or "views" - they sit between your
internal data (database models) and what the outside world sees.

Why not just return the database model directly?
-------------------------------------------------
1. Security: You control exactly what fields are exposed
2. Flexibility: You can rename, combine, or format fields
3. Validation: Pydantic ensures the data is correct before sending
4. Docs: FastAPI auto-generates API docs from these schemas

Example Flow:
    1. User uploads PDF → controller receives it
    2. Controller creates a Document in the DB
    3. Controller returns a DocumentResponse (this file)
    4. FastAPI validates and serializes it to JSON
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DocumentResponse(BaseModel):
    """
    Returned after a successful document upload.

    This is what the caller (e.g., frontend) receives when
    they POST a PDF to /api/documents/upload

    Fields:
    -------
    - id: The unique UUID for this document (auto-generated)
    - filename: Original filename of the uploaded PDF
    - file_size: Size in bytes (useful for display in UI)
    - page_count: How many pages the PDF has (filled after processing)
    - status: Current processing state
        - "processing" = just uploaded, text extraction pending
        - "completed"  = text extracted and stored in DB
        - "failed"     = something went wrong during processing
    - created_at: When it was uploaded
    - message: A human-readable message (e.g., "Upload successful")

    Example JSON response:
        {
            "id": "abc-123",
            "filename": "report.pdf",
            "file_size": 204800,
            "page_count": null,
            "status": "processing",
            "created_at": "2024-01-01T12:00:00",
            "message": "Document uploaded successfully"
        }
    """

    id: str
    filename: str
    file_size: Optional[int] = None      # bytes, None until we read the file
    page_count: Optional[int] = None     # None until PDF is parsed
    status: str
    created_at: datetime
    message: str = "Document uploaded successfully"

    # This tells Pydantic to read data from ORM objects (SQLAlchemy models)
    # Without this, Pydantic can't read from SQLAlchemy model instances
    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """
    Returned when listing all documents.

    Used by: GET /api/documents
    Returns a list of documents + total count.

    Why include total?
    ------------------
    For pagination later — the UI may show "Showing 10 of 47 documents"
    """

    documents: list[DocumentResponse]
    total: int


class ErrorResponse(BaseModel):
    """
    Returned when something goes wrong.

    Example use cases:
    - File is not a PDF → 400 Bad Request
    - File is too large → 400 Bad Request
    - Document not found → 404 Not Found
    - Server crash → 500 Internal Server Error

    Example JSON:
        {
            "error": "Invalid file type",
            "detail": "Only PDF files are accepted. Got: .docx",
            "status_code": 400
        }
    """

    error: str               # Short error name (e.g., "Invalid file type")
    detail: str              # Longer explanation
    status_code: int         # HTTP status code (400, 404, 500, etc.)
