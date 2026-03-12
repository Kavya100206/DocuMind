"""
Chunking Service

WHY DOES THIS FILE EXIST?
--------------------------
In Phase 2, we saved each PAGE of a PDF as one row in the chunks table.
One page can have 800-1500 words. That's too big for good semantic search.

Imagine asking: "What was the Q3 revenue?"
If we search across full pages, we might retrieve a 1000-word page where
the actual answer is just one sentence buried in the middle.

This service SPLITS each page into SMALLER pieces — around 500 characters each.
Each small piece = one "chunk" we can precisely retrieve.

WHAT IS OVERLAPPING?
--------------------
Look at this example text (simplified):

  "Revenue grew 10% in Q3. Costs also rose. Net profit was $5M in Q3."

If we split at exactly 500 chars, we might cut like this:

  Chunk 1: "Revenue grew 10% in Q3. Costs also rose. Net profit was $5M"
  Chunk 2: "in Q3."  ← the sentence "Net profit was $5M in Q3" is now SPLIT

To fix this, we OVERLAP — the end of chunk 1 is repeated at the start of chunk 2:

  Chunk 1: "Revenue grew 10% in Q3. Costs also rose. Net profit was $5M"
  Chunk 2: "Net profit was $5M in Q3."  ← full sentence preserved!

The overlap is controlled by CHUNK_OVERLAP in settings.py (default: 50 chars).

WHAT DOES THIS RETURN?
-----------------------
A list of dicts, one per chunk:
[
    {
        "text": "Revenue grew 10% in...",
        "chunk_index": 0,       ← position within the page (0, 1, 2...)
        "start_char": 0,        ← where in the page text this chunk starts
        "end_char": 500,        ← where it ends
        "char_count": 500
    },
    {
        "text": "Net profit was $5M...",
        "chunk_index": 1,
        "start_char": 450,      ← overlaps with previous chunk's end (500 - 50 overlap)
        "end_char": 950,
        "char_count": 500
    },
    ...
]
"""

from typing import List, Dict, Any
from app.config.settings import settings
import re


def chunk_text(text: str) -> List[Dict[str, Any]]:
    """
    Split a piece of text into overlapping chunks.

    HOW IT WORKS (step by step):
    -----------------------------
    1. Start at position 0 in the text
    2. Take the next CHUNK_SIZE characters → that's one chunk
    3. Move forward by (CHUNK_SIZE - CHUNK_OVERLAP) characters
       (i.e., step back a little to create overlap)
    4. Repeat until we reach the end of the text

    Args:
        text: The full text to split (usually one page from a PDF)

    Returns:
        List of chunk dicts. Empty list if text is empty.
    """

    # Don't process empty or whitespace-only text
    if not text or not text.strip():
        return []

    chunks = []

    # Pull chunk size and overlap from our settings
    # (These are defined in settings.py and can be changed in .env)
    chunk_size = settings.CHUNK_SIZE      # default: 500 characters
    chunk_overlap = settings.CHUNK_OVERLAP  # default: 50 characters

    # ---------------------------------------------------------------
    # THE SLIDING WINDOW LOOP
    # ---------------------------------------------------------------
    # 'start' is where our current chunk begins in the full text
    start = 0
    chunk_index = 0  # counter: which chunk number is this? (0, 1, 2...)

    while start < len(text):

        # Calculate where this chunk ends
        # min() makes sure we don't go past the end of the text
        end = min(start + chunk_size, len(text))

        # Slice the text to get this chunk's content
        chunk_text_content = text[start:end]

        # ---------------------------------------------------------------
        # SMART BOUNDARY: Don't cut in the middle of a word
        # ---------------------------------------------------------------
        # If we're NOT at the last chunk, try to end at a word boundary.
        # Example: "...revenue grew 10" → bad, cuts "10" from "10%"
        # Better:  "...revenue grew"    → ends at a space
        if end < len(text):
            # Find the last space before the end of this chunk
            last_space = chunk_text_content.rfind(' ')
            if last_space > chunk_size // 2:
                # Only adjust if the space is in the second half
                # (avoids making chunks way too short)
                chunk_text_content = chunk_text_content[:last_space]
                end = start + last_space

        # Save this chunk
        chunks.append({
            "text": chunk_text_content.strip(),
            "chunk_index": chunk_index,
            "start_char": start,
            "end_char": end,
            "char_count": len(chunk_text_content.strip())
        })

        # ---------------------------------------------------------------
        # ADVANCE THE WINDOW
        # ---------------------------------------------------------------
        # Move forward by (chunk_size - overlap)
        # Example: chunk_size=500, overlap=50 → advance by 450
        # The next chunk starts 450 chars later (not 500),
        # so the last 50 chars of this chunk appear again in the next one.
        start += (chunk_size - chunk_overlap)
        chunk_index += 1

    return chunks


def chunk_pages(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk all pages from a PDF into smaller pieces.

    This is the function called from document_service.py.
    It loops over all pages and chunks each one.

    Args:
        pages_data: List of page dicts from pdf_service.py
                    Each dict has: page_number, text, is_empty, etc.

    Returns:
        List of chunk dicts, each enriched with page_number so we
        know WHICH PAGE a chunk came from (needed for citations later).

    Example output:
        [
            {
                "page_number": 1,
                "chunk_index": 0,
                "text": "Introduction: This report...",
                "start_char": 0,
                "end_char": 487,
                "char_count": 487
            },
            {
                "page_number": 1,
                "chunk_index": 1,
                "text": "report covers the key metrics...",
                "start_char": 437,
                "end_char": 924,
                ...
            },
            {
                "page_number": 2,
                "chunk_index": 0,   ← resets to 0 for each new page
                ...
            },
        ]
    """

    all_chunks = []

    for page in pages_data:
        # Skip empty or scanned pages (no text to chunk)
        if page.get("is_empty") or not page.get("text"):
            continue

        # Split this page's text into chunks
        page_chunks = chunk_text(page["text"])

        # Add page_number to each chunk so we know where it came from
        for chunk in page_chunks:
            chunk["page_number"] = page["page_number"]
            all_chunks.append(chunk)

    return all_chunks


# ---------------------------------------------------------------------------
# SECTION-AWARE CHUNKING  (Phase 6)
# ---------------------------------------------------------------------------
#
# DESIGN PRINCIPLE: Zero word lists. Purely structural detection.
# ---------------------------------------------------------------
# We detect section headers by HOW they look, not WHAT they say.
# This means the code works identically for:
#   - A resume       ("PROJECTS", "Education:")
#   - A novel        ("CHAPTER ONE", "Prologue:")
#   - A tech spec    ("EVALUATION CRITERIA", "Requirements:")
#   - A legal doc    ("TERMS AND CONDITIONS", "Definitions:")
#   - A research paper ("ABSTRACT", "Methodology:")
#   - Any other PDF  (as long as headers are visually distinct)
#
# Two structural rules:
#
# Rule 1 — ALL CAPS short line
#   A line written entirely in UPPERCASE that is short (2–60 chars).
#   PDFs use caps formatting to make section titles stand out visually.
#   Examples:  "PROJECTS"  "EVALUATION CRITERIA"  "CHAPTER ONE"
#
# Rule 2 — Short line ending with a colon
#   A line that ends with ":" and is short enough to be a heading.
#   Authors write headers as "Education:" or "Task Requirements:" to
#   signal "a list/explanation follows this label".
#   Examples:  "Education:"  "Task Requirements:"  "Overview:"
#
# What is NOT a header (guard conditions):
#   - Line ends with "." or "?" → it's a sentence, not a label
#   - Line is longer than 60 chars → probably a sentence starting with caps
#   - Line has no letters → "-------" or "======" dividers
#   - Colon rule: line must be under 60 chars AND not end with "." or "?"

# Rule 1: All-caps line (2–60 chars, no sentence-ending punctuation)
# Breakdown of the regex:
#   ^              → must start at beginning of a line
#   (?=.*[A-Z])    → lookahead: must contain at least one letter (not "---")
#   [A-Z0-9]      → must start with a capital letter or digit
#   [A-Z0-9\s\-]  → body: only uppercase letters, digits, spaces, hyphens
#   {0,58}         → 0 to 58 more chars (so total max = 60)
#   [A-Z0-9]?     → optional last char (handles 1-word headers like "SKILLS")
#   [:\s]*$        → optional trailing colon/space at end of line
ALLCAPS_LINE = re.compile(
    r'^(?=.*[A-Z])[A-Z0-9][A-Z0-9\s\-]{0,58}[:\s]*$',
    re.MULTILINE
)

# Rule 2: Short colon-terminated line (catches mixed-case headers like "Education:")
# Breakdown:
#   ^          → start of line
#   .{2,59}    → 2 to 59 any characters (the header label)
#   :          → must end with a colon (the defining feature)
#   \s*$       → optional whitespace after colon
#   (?<!\.\s:) → NOT a sentence ending (guard against "See note 1:")
COLON_HEADER_LINE = re.compile(
    r'^[^\.\?]{2,59}:\s*$',
    re.MULTILINE
)


def _is_likely_header(line: str) -> bool:
    """
    Single-line check: is this line a section header?

    Called by _detect_sections() for every line that matches one of
    the two structural patterns above. Applies a few extra sanity checks
    to reduce false positives.

    Args:
        line: A single stripped line from the document

    Returns:
        True if the line looks like a section header, False otherwise
    """
    stripped = line.strip()

    # Must have at least one letter (filters out "---", "===", "123")
    if not any(c.isalpha() for c in stripped):
        return False

    # Must not end with sentence punctuation (filters out "See below.")
    if stripped.rstrip(": \t").endswith(('.', '?', '!')):
        return False

    # Must not be multiple sentences (a real header never has ". " in the middle)
    if '. ' in stripped or '? ' in stripped:
        return False

    return True


def _detect_sections(text: str) -> List[Dict[str, Any]]:
    """
    Find all section headers in a document using pure structural rules.

    NO WORD LISTS — works for any document type:
    ----------------------------------------------
    Rule 1:  ALL CAPS short line    → "PROJECTS", "CHAPTER ONE", "EVALUATION CRITERIA"
    Rule 2:  Short colon-ended line → "Education:", "Task Requirements:", "Overview:"

    Args:
        text: Full document text (all pages joined)

    Returns:
        List of section dicts sorted by position:
        [
            {"name": "EDUCATION", "start": 0},
            {"name": "PROJECTS",  "start": 542},
            ...
        ]
        Returns empty list if no headers found → fallback to sliding-window.
    """
    seen_starts: set = set()
    sections: List[Dict[str, Any]] = []

    # Rule 1: ALL-CAPS lines
    for match in ALLCAPS_LINE.finditer(text):
        line = match.group()
        if _is_likely_header(line) and match.start() not in seen_starts:
            seen_starts.add(match.start())
            sections.append({
                "name":  line.strip().rstrip(":").strip(),
                "start": match.start()
            })

    # Rule 2: Colon-terminated short lines (e.g. "Education:", "Overview:")
    for match in COLON_HEADER_LINE.finditer(text):
        line = match.group()
        if _is_likely_header(line) and match.start() not in seen_starts:
            seen_starts.add(match.start())
            sections.append({
                "name":  line.strip().rstrip(":").strip(),
                "start": match.start()
            })

    # Sort by position (both rules scan independently)
    sections.sort(key=lambda s: s["start"])
    return sections


def chunk_by_sections(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split a document into chunks using SECTION BOUNDARIES instead of
    a fixed character window.

    WHY IS THIS BETTER THAN chunk_pages()?
    ---------------------------------------
    chunk_pages() doesn't know about document structure. It slices at
    every 1200 characters, often cutting right through a project entry
    or an education record — leaving one section split across two chunks.

    chunk_by_sections() reads the MEANING first:
      1. Detect all section headers (PROJECTS, EDUCATION, ...)
      2. Slice the text at those natural breaks
      3. One chunk = one complete section (Projects, Education, etc.)

    If a section is very long (> CHUNK_SIZE chars), we fall back to the
    existing sliding-window chunk_text() so we don't create monster chunks.

    FALLBACK BEHAVIOUR:
    -------------------
    If no section headers are detected (e.g. a plain-text report with no
    headings), this function falls back to chunk_pages() automatically.
    Nothing breaks — you just get the old behaviour.

    RETURN FORMAT:
    --------------
    Identical to chunk_pages() so document_service.py needs only a
    1-line import change — all downstream code stays the same:
    [
        {
            "page_number":  1,
            "chunk_index":  0,
            "section_name": "PROJECTS",   ← NEW — which section this came from
            "text":         "PROJECTS\nAI Code Review Bot...",
            "start_char":   0,
            "end_char":     843,
            "char_count":   843
        },
        ...
    ]

    Args:
        pages_data: Same list of page dicts from pdf_service

    Returns:
        List of chunk dicts (one per section, or sub-chunks if section is large)
    """

    # ------------------------------------------------------------------
    # STEP 1: Merge all pages into one big text block
    # ------------------------------------------------------------------
    # We join pages with TWO newlines so sections that span a page break
    # are still connected. We also track which page each character falls on.
    #
    # Why join? A resume is 1 page, but a report might span 10 pages with
    # "PROJECTS" starting on page 2.  If we chunk page-by-page, we'd never
    # find the section header in the right context.
    full_text = ""
    page_boundaries: List[Dict[str, Any]] = []   # [{start_char, end_char, page_number}, ...]

    for page in pages_data:
        if page.get("is_empty") or not page.get("text"):
            continue
        page_start = len(full_text)
        full_text += page["text"] + "\n\n"
        page_end   = len(full_text)
        page_boundaries.append({
            "page_number": page["page_number"],
            "start_char":  page_start,
            "end_char":    page_end
        })

    if not full_text.strip():
        return []

    # ------------------------------------------------------------------
    # STEP 2: Detect section headers
    # ------------------------------------------------------------------
    sections = _detect_sections(full_text)

    # ------------------------------------------------------------------
    # FALLBACK: No headers found → use original sliding-window chunker
    # ------------------------------------------------------------------
    # This handles plain documents (raw reports, articles) with no
    # structured headings.
    if not sections:
        print("  ⚠️  No section headers detected — falling back to sliding-window chunking.")
        return chunk_pages(pages_data)

    print(f"  🗂️  Detected {len(sections)} sections: {[s['name'] for s in sections]}")

    # ------------------------------------------------------------------
    # STEP 3: Slice text into sections
    # ------------------------------------------------------------------
    # For each detected section, the section text runs from its own
    # start position to the start of the NEXT section (or end of document).
    #
    # Example:
    #   full_text = "SKILLS\njava python...\nPROJECTS\nAI Bot...EDUCATION\n..."
    #   sections  = [{name:SKILLS, start:0}, {name:PROJECTS, start:120}, ...]
    #
    #   section_text for SKILLS   = full_text[0:120]   = "SKILLS\njava python..."
    #   section_text for PROJECTS = full_text[120:480] = "PROJECTS\nAI Bot..."
    # ------------------------------------------------------------------
    # MIN_SECTION_CHARS: stub-header threshold
    # ------------------------------------------------------------------
    # Some documents use short colon-labeled lines as inline value labels:
    #   "Project Title:"
    #   "OrbitMind"           ← this is the VALUE, not a new section
    #   "Team Size:"
    #   "5"                   ← same problem
    #
    # The section detector fires on "Project Title:" and creates a split
    # boundary, producing a tiny chunk of just 1-2 lines.  The actual value
    # ends up in the NEXT section's chunk, disconnected from its label.
    #
    # Fix: if the section content (header + body) is shorter than this
    # threshold it is treated as a stub — it is NOT emitted as its own chunk.
    # The text will naturally be included in the previous section's span
    # (since we only split at sections we DO emit).
    MIN_SECTION_CHARS = 20

    all_chunks: List[Dict[str, Any]] = []
    global_chunk_index = 0   # unique counter across all sections

    for i, section in enumerate(sections):
        section_start = section["start"]
        # The section ends where the next section begins (or end of document)
        section_end   = sections[i + 1]["start"] if i + 1 < len(sections) else len(full_text)
        section_text  = full_text[section_start:section_end].strip()
        section_name  = section["name"]

        if not section_text:
            continue

        # Skip stub sections — they are just label lines with no real content.
        # Their text will be absorbed into the chunk built from the previous
        # section's span, keeping the label and value together.
        if len(section_text) < MIN_SECTION_CHARS:
            print(f"    ⏭  Skipping stub section '{section_name}' ({len(section_text)} chars) — absorbed into surrounding chunk")
            continue

        # Figure out which page this section STARTS on
        # (for the citation page_number field)
        page_number = 1  # default
        for boundary in page_boundaries:
            if boundary["start_char"] <= section_start < boundary["end_char"]:
                page_number = boundary["page_number"]
                break

        # ------------------------------------------------------------------
        # STEP 4: Chunk the section
        # ------------------------------------------------------------------
        # If the section fits in one chunk → keep it whole (ideal case)
        # If the section is too large     → split it with the sliding window
        if len(section_text) <= settings.CHUNK_SIZE:
            # Section fits as a single chunk — perfect semantic unit
            all_chunks.append({
                "page_number":  page_number,
                "chunk_index":  global_chunk_index,
                "section_name": section_name,
                "text":         section_text,
                "start_char":   section_start,
                "end_char":     section_end,
                "char_count":   len(section_text)
            })
            global_chunk_index += 1
            print(f"    ✓ Section '{section_name}': {len(section_text)} chars → 1 chunk")

        else:
            # Section is too large — apply sliding window within this section
            sub_chunks = chunk_text(section_text)
            for sub in sub_chunks:
                all_chunks.append({
                    "page_number":  page_number,
                    "chunk_index":  global_chunk_index,
                    "section_name": section_name,
                    "text":         sub["text"],
                    "start_char":   section_start + sub["start_char"],
                    "end_char":     section_start + sub["end_char"],
                    "char_count":   sub["char_count"]
                })
                global_chunk_index += 1
            print(f"    ✓ Section '{section_name}': {len(section_text)} chars → {len(sub_chunks)} sub-chunks (too large for one chunk)")

    print(f"  ✂️  Section-aware chunking: {len(all_chunks)} total chunks from {len(sections)} sections")
    return all_chunks

