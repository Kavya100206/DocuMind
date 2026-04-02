"""
Chunk Model

What is a Chunk?
----------------
A "chunk" is a piece of text extracted from a PDF.
For now (Phase 2), each chunk = one page of a PDF.

In Phase 3, we'll break these page-chunks into smaller
overlapping pieces for better semantic search.

Why store chunks in the database?
----------------------------------
1. We parse the PDF once, store the text forever
2. No need to re-read the PDF file again later
3. We can query chunks by document, page number, etc.
4. Chunks carry metadata (doc_id, page number) that
   becomes crucial for citations in Phase 6

Relationship:
    Document (1) ──── has many ──── Chunk (many)
    One PDF upload → many chunks (one per page)
"""

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
import json
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.database.postgres import Base


class Chunk(Base):
    """
    Chunk model - stores extracted text from each page of a PDF

    Table name: chunks

    Columns:
    --------
    - id: Unique identifier (UUID)
    - document_id: Which document this chunk belongs to (Foreign Key)
    - page_number: Which page this text came from (1-indexed)
    - text: The actual extracted text content
    - char_count: How many characters in this chunk (useful for analytics)
    - created_at: When it was extracted

    What is a Foreign Key?
    ----------------------
    A foreign key links this table to another table.
    document_id → documents.id means:
        "This chunk belongs to the document with this ID"
    
    If you delete a Document, its Chunks should also be deleted.
    That's what "CASCADE" means below.
    """

    __tablename__ = "chunks"

    # Primary key
    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True
    )

    # Foreign key → links to the documents table
    # ondelete="CASCADE" means: if the Document is deleted,
    # all its Chunks are automatically deleted too
    document_id = Column(
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True   # Index for fast lookups like: "give me all chunks for doc X"
    )

    # Which page this text came from (1-indexed, like how humans count pages)
    page_number = Column(Integer, nullable=False)

    # The actual extracted text from this page
    # Text type = unlimited length (vs String which has a limit)
    text = Column(Text, nullable=False)

    # Character count — quick way to know how much text is on this page
    # Useful for filtering out near-empty pages
    char_count = Column(Integer, default=0)

    # Embedding vector stored as JSON string (list of 384 floats).
    # Persisting this avoids re-running the embedding model on every
    # deployment restart — saves ~200MB peak memory on free tiers.
    embedding = Column(Text, nullable=True)

    # When this chunk was created (auto-set by DB server)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # SQLAlchemy relationship — lets us navigate from Chunk → Document in Python
    # Usage: chunk.document.filename  (instead of a separate DB query)
    # back_populates="chunks" means Document model will have a .chunks attribute too
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return (
            f"<Chunk(id={self.id[:8]}..., "
            f"doc={self.document_id[:8]}..., "
            f"page={self.page_number}, "
            f"chars={self.char_count})>"
        )
