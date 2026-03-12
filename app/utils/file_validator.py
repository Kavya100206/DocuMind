"""
File Validator Utility

What is this file?
------------------
A utility module that validates uploaded files BEFORE we do
anything with them. This is the "guard at the door" of our system.

Why validate early?
-------------------
- Security: Reject non-PDF files immediately
- Performance: Don't waste time processing invalid files
- User experience: Return clear, helpful error messages

What we check:
--------------
1. File extension must be .pdf
2. MIME type must be "application/pdf" (double-checks the real type)
3. File size must be under MAX_UPLOAD_SIZE (10MB by default)
4. File must not be empty (0 bytes)
"""

from fastapi import UploadFile, HTTPException
from app.config.settings import settings
import os


async def validate_pdf(file: UploadFile) -> None:
    """
    Validate an uploaded file to ensure it's a valid PDF within size limits.

    What is UploadFile?
    -------------------
    FastAPI's built-in class for handling file uploads.
    It gives us access to:
        - file.filename  → original filename (e.g. "report.pdf")
        - file.content_type → MIME type (e.g. "application/pdf")
        - file.read()    → actual file bytes

    What is HTTPException?
    ----------------------
    FastAPI's way to return HTTP error responses.
    Raising HTTPException(status_code=400, detail="...") automatically
    sends a 400 Bad Request JSON response to the caller.

    Args:
        file: The uploaded file from the HTTP request

    Raises:
        HTTPException 400: If file is not PDF, too large, or empty
    """

    # ---------------------------------------------------------
    # Check 1: File extension must be .pdf
    # ---------------------------------------------------------
    # os.path.splitext("report.pdf") → ("report", ".pdf")
    # We take [1] to get just the extension, then lowercase it
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided. Please upload a PDF file."
        )

    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type '{file_extension}'. "
                f"Only PDF files are accepted (.pdf). "
                f"Got: {file.filename}"
            )
        )

    # ---------------------------------------------------------
    # Check 2: MIME type must be application/pdf
    # ---------------------------------------------------------
    # Someone could rename "malware.exe" to "malware.pdf" to fool
    # the extension check. MIME type is a second layer of defense.
    # The browser/client sets this based on the actual file content.
    ALLOWED_CONTENT_TYPES = ["application/pdf", "application/x-pdf"]

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid content type '{file.content_type}'. "
                f"Expected 'application/pdf'. "
                f"Make sure you're uploading a real PDF file."
            )
        )

    # ---------------------------------------------------------
    # Check 3: File size must be under MAX_UPLOAD_SIZE
    # ---------------------------------------------------------
    # We read the entire file into memory to check its size.
    # Then we "seek" back to the beginning so we can read it again later.
    #
    # Why seek back?
    # After read(), the file "cursor" is at the end.
    # If we don't seek(0), the next read() will return empty bytes.
    file_content = await file.read()
    file_size = len(file_content)

    # Check if empty
    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty (0 bytes). Please upload a valid PDF."
        )

    # Check if too large
    if file_size > settings.MAX_UPLOAD_SIZE:
        max_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)  # Convert bytes to MB
        actual_mb = file_size / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=(
                f"File too large ({actual_mb:.1f}MB). "
                f"Maximum allowed size is {max_mb:.0f}MB."
            )
        )

    # IMPORTANT: Seek back to start so the file can be read again later
    # (by the service that actually saves/processes it)
    await file.seek(0)
