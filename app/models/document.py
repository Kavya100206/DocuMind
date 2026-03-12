"""
Document Model

This is our first database model (M in MVC)!

What is a Model?
----------------
A model represents a database table as a Python class.
Each instance of the class = one row in the table.

Example:
    # Create a new document
    doc = Document(
        filename="report.pdf",
        user_id="user123"
    )
    # Save to database
    db.add(doc)
    db.commit()
"""

from sqlalchemy import Column, String, Integer, DateTime, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database.postgres import Base
import uuid


class Document(Base):
    """
    Document model - stores metadata about uploaded PDFs
    
    Table name: documents
    
    Columns:
    --------
    - id: Unique identifier (UUID)
    - filename: Original filename
    - user_id: Who uploaded it (for multi-user support later)
    - file_path: Where the file is stored
    - file_size: Size in bytes
    - page_count: Number of pages in PDF
    - status: processing/completed/failed
    - created_at: When it was uploaded
    - updated_at: Last modification time
    """
    
    __tablename__ = "documents"
    
    # Primary key - unique identifier
    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),  # Auto-generate UUID
        index=True
    )
    
    # Document metadata
    filename = Column(String, nullable=False)
    user_id = Column(String, nullable=False, index=True)  # Index for fast lookups
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)  # Size in bytes
    page_count = Column(Integer)
    
    # Processing status
    status = Column(
        String,
        default="processing",  # processing, completed, failed
        nullable=False
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),  # Auto-set to current time
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),  # Auto-update when row changes
        nullable=False
    )
    
    # Relationship to Chunk model
    # Lets you do: document.chunks  → list of all Chunk objects
    # cascade="all, delete-orphan" → if Document is deleted, delete its Chunks too
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        """
        String representation of the object
        Useful for debugging
        """
        return f"<Document(id={self.id}, filename={self.filename})>"
