"""
PDF Service

What is this file?
------------------
The PDF Service is responsible for one thing: reading a PDF file
and extracting structured text from it, page by page.

This is the "reading" step of the pipeline:
    PDF file on disk → Python reads it → returns structured text data

Why pdfplumber?
---------------
We have two PDF libraries installed: PyPDF2 and pdfplumber.
pdfplumber is more powerful because it:
    - Handles text layout better (preserves reading order)
    - Can extract tables as structured data
    - Better handles PDFs with complex formatting
    - Gives us page-level metadata (dimensions, etc.)

We use PyPDF2 as a fallback for edge cases.

What does "structured text" mean?
----------------------------------
Instead of dumping all text into one big string, we return:
    [
        { "page_number": 1, "text": "...", "char_count": 542 },
        { "page_number": 2, "text": "...", "char_count": 318 },
        ...
    ]

This structure is important for citations later (Phase 6).
"""

import pdfplumber
from PyPDF2 import PdfReader
from typing import List, Dict, Any
import re


def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file, page by page.

    How it works:
    -------------
    1. Open the PDF with pdfplumber (primary)
    2. For each page, extract the text
    3. Clean the text (remove weird whitespace, fix line breaks)
    4. Handle edge cases:
       - Empty pages (no text) → mark as empty, skip storing later
       - Scanned pages (images, no text) → detected by empty text
    5. Return list of page dicts

    Args:
        file_path: Absolute path to the PDF file on disk

    Returns:
        List of dicts, one per page:
        [
            {
                "page_number": 1,          # 1-indexed (like real page numbers)
                "text": "extracted text",   # cleaned text content
                "char_count": 542,          # number of characters
                "is_empty": False           # True if no text found
            },
            ...
        ]

    Raises:
        ValueError: If the file cannot be read as a PDF
        FileNotFoundError: If the file path doesn't exist
    """

    pages_data: List[Dict[str, Any]] = []

    try:
        # pdfplumber.open() opens the PDF and gives us a context
        # The 'with' block ensures the file is closed properly after
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1  # Convert to 1-indexed

                # Extract raw text from this page
                # extract_text() returns a string or None if page has no text
                raw_text = page.extract_text()

                if raw_text is None or raw_text.strip() == "":
                    # This page has no extractable text
                    # Could be:
                    # 1. A blank page
                    # 2. A scanned image (OCR would be needed — Phase 2 stretch goal)
                    # 3. A page with only images/diagrams
                    pages_data.append({
                        "page_number": page_number,
                        "text": "",
                        "char_count": 0,
                        "is_empty": True,
                        "note": "No extractable text — may be scanned or blank"
                    })
                    print(f"  ⚠️  Page {page_number}/{total_pages}: No text found (blank or scanned)")
                else:
                    # Extract plain text and clean it
                    cleaned_text = _clean_text(raw_text)

                    # Also extract any tables on this page and append as readable text
                    # WHY: pdfplumber's extract_text() garbles table data — column
                    # values get mixed with headers in random order. A table row
                    # like "Total | 168" becomes "IV Total 168 Credits Title..."
                    # which the embedding model can't match to "total credit requirement".
                    # extract_tables() gives us clean row-column structure.
                    table_text = _extract_tables_as_text(page)
                    if table_text:
                        cleaned_text = cleaned_text + "\n\n" + table_text
                        print(f"      + Table data appended ({len(table_text)} chars)")

                    pages_data.append({
                        "page_number": page_number,
                        "text": cleaned_text,
                        "char_count": len(cleaned_text),
                        "is_empty": False
                    })
                    print(f"  ✅ Page {page_number}/{total_pages}: {len(cleaned_text)} characters extracted")

    except Exception as e:
        # If pdfplumber fails, try PyPDF2 as a fallback
        print(f"  ⚠️  pdfplumber failed: {e}. Trying PyPDF2 fallback...")
        pages_data = _extract_with_pypdf2(file_path)

    return pages_data


def _extract_tables_as_text(page) -> str:
    """
    Extract tables from a PDF page and convert them to readable plain text.

    WHY THIS EXISTS:
    ----------------
    pdfplumber.extract_text() reads PDF content in a left-to-right, top-to-bottom
    scan. For a multi-column table like:

        Title            | Credits
        Theory of Comp.  | 3
        Advanced Java    | 2
        Total            | 168

    ...it produces garbage like:
        "Theory of Comp. 3 Advanced Java 2 Title Credits Total 168"

    pdfplumber.extract_tables() understands table structure and returns:
        [["Title", "Credits"], ["Theory of Comp.", "3"], ["Total", "168"]]

    We then format each row as "Title: Credits" or "col1 | col2" so the
    embedding model can understand it as natural language.

    EXAMPLE OUTPUT:
    ---------------
        [TABLE]
        Title | Credits
        Theory of Computing | 3
        Advanced Java | 2
        Total | 168
        [END TABLE]

    Now "Total 168" is a complete readable line that semantically matches
    queries like "what is the total credit requirement?"

    Args:
        page: A pdfplumber Page object

    Returns:
        Formatted table string, or empty string if no tables found.
    """
    try:
        tables = page.extract_tables()
    except Exception:
        return ""

    if not tables:
        return ""

    table_texts = []

    for table in tables:
        if not table:
            continue

        rows = []
        for row in table:
            if not row:
                continue

            # Each cell may be None (empty cell in PDF table)
            # Replace None with empty string, strip whitespace
            cells = [str(cell).strip() if cell is not None else "" for cell in row]

            # Skip rows that are entirely empty
            if not any(cells):
                continue

            # Join cells with " | " to create a readable row
            # e.g. ["Total", "168"] → "Total | 168"
            rows.append(" | ".join(cells))

        if rows:
            # Wrap with clear markers so it's identifiable in chunk text
            table_texts.append("[TABLE]\n" + "\n".join(rows) + "\n[END TABLE]")

    return "\n\n".join(table_texts)


def _clean_text(raw_text: str) -> str:
    """
    Clean extracted text from common PDF artifacts.

    What are PDF artifacts?
    -----------------------
    PDFs often have:
    - Multiple spaces between words (from column layouts)
    - Weird line breaks in the middle of sentences
    - Special characters that don't render well
    - Trailing whitespace on every line

    This function normalizes all of that.

    Args:
        raw_text: Raw text string from pdfplumber

    Returns:
        Cleaned text string
    """

    # Step 1: Remove CID artifacts — e.g. (cid:127), (cid:32)
    # WHY: pdfplumber can't always decode special characters (bullet points,
    # arrows, ligatures) in some PDFs. Instead of the actual character,
    # they show up as "(cid:NUMBER)". These are meaningless noise that
    # confuses the LLM and clutters chunk text.
    text = re.sub(r'\(cid:\d+\)', '', raw_text)

    # Step 2: Normalize Windows-style line endings (\r\n) to Unix (\n)
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Step 3: Replace multiple spaces with a single space
    # \s+ means "one or more whitespace characters (but not newlines)"
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Step 4: Remove spaces at the start and end of each line
    lines = [line.strip() for line in text.split('\n')]

    # Step 5: Collapse 3+ consecutive newlines to max 2 (paragraph break)
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Step 6: Final strip of leading/trailing whitespace
    return text.strip()


def _extract_with_pypdf2(file_path: str) -> List[Dict[str, Any]]:
    """
    Fallback PDF text extraction using PyPDF2.

    When is this used?
    ------------------
    Only if pdfplumber fails (e.g., encrypted PDF, corrupted file, etc.)
    PyPDF2 is faster but less accurate for complex layouts.

    Args:
        file_path: Absolute path to the PDF file

    Returns:
        Same structure as extract_text_from_pdf()
    """

    pages_data = []

    try:
        reader = PdfReader(file_path)

        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1
            raw_text = page.extract_text()

            if raw_text and raw_text.strip():
                cleaned_text = _clean_text(raw_text)
                pages_data.append({
                    "page_number": page_number,
                    "text": cleaned_text,
                    "char_count": len(cleaned_text),
                    "is_empty": False
                })
            else:
                pages_data.append({
                    "page_number": page_number,
                    "text": "",
                    "char_count": 0,
                    "is_empty": True,
                    "note": "PyPDF2 fallback: no text found"
                })

    except Exception as e:
        raise ValueError(f"Could not extract text from PDF: {str(e)}")

    return pages_data


def get_page_count(file_path: str) -> int:
    """
    Get the total number of pages in a PDF without extracting text.

    Why this function?
    ------------------
    We call this right after upload to update the page_count field
    on the Document. It's faster than full text extraction.

    Args:
        file_path: Path to the PDF file

    Returns:
        Number of pages as an integer
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            return len(pdf.pages)
    except Exception:
        # Fallback to PyPDF2
        reader = PdfReader(file_path)
        return len(reader.pages)
