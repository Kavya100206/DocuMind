"""
Drive File Model

WHAT DOES THIS FILE DO?
------------------------
This table is the "bridge" between Google Drive files and our
internal documents table.

WHY DO WE NEED A SEPARATE TABLE?
---------------------------------
When a user ingests a file from Drive, we run it through our existing
pipeline (extract → chunk → embed → FAISS) and create a row in the
`documents` table. That's identical to a manual upload.

BUT we need to remember:
  - Which Google Drive file_id corresponds to which document_id
  - When the Drive file was last modified (to detect stale data)
  - Whether we have a "watch" active on it (push notifications)
  - When that watch expires (Drive watches last max 7 days)

We can't add these Drive-specific columns to the existing `documents`
table because that table should stay generic — it doesn't know or care
whether a document came from a PDF upload or from Google Drive.

HOW DOES THE WATCH SYSTEM WORK?
---------------------------------
1. After ingesting a file, we call Drive API's `files.watch()`
2. Google gives us a "channel_id" — a unique webhook subscription ID
3. Google sets an expiry (max 7 days from now)
4. Whenever the Drive file changes, Google POSTs to our webhook URL
5. Our webhook reads the channel_id from the request header
6. We look up this table by channel_id → find the document_id
7. We delete old chunks for that document and re-ingest the file

TABLE: drive_files
"""

from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.sql import func
from app.database.postgres import Base
import uuid


class DriveFile(Base):
    """
    Links a Google Drive file to our internal document record.

    One row per ingested Drive file.

    Columns:
    --------
    - id               : UUID primary key (our internal ID)
    - user_id          : Who ingested this file
    - drive_file_id    : Google's identifier for the file (e.g. "1BxiMV...")
                         This is how we call Drive API to re-fetch the file.
    - document_id      : FK to our `documents` table.
                         This is how we find and delete old chunks on update.
    - filename         : Human-readable display name (e.g. "Q3_Report.pdf")
    - mime_type        : "application/pdf" or "application/vnd.google-apps.document"
                         We need this because Google Docs need to be exported
                         as PDF before we can process them.
    - last_modified    : The modifiedTime from Drive when we last ingested.
                         We can compare this to detect staleness without a watch.
    - watch_channel_id : The channel ID Google gave us for the push notification.
                         Drive sends this in the X-Goog-Channel-ID request header
                         when it notifies us of a change.
    - watch_expiry     : When the Drive watch subscription expires.
                         We must re-register before this time or we'll miss updates.
    - created_at       : When this record was created (first ingestion)
    - updated_at       : Last time we re-ingested this file
    """

    __tablename__ = "drive_files"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True
    )

    user_id = Column(String, nullable=False, index=True)

    # Google's own ID for the file.
    # This never changes even if the user renames the file on Drive.
    # Indexed because we look this up when we get a webhook notification.
    drive_file_id = Column(String, nullable=False, unique=True, index=True)

    # Our internal document ID — links to the `documents` table.
    # When we re-ingest, we delete all Chunk rows for this document_id
    # and re-run the pipeline, creating fresh chunks.
    document_id = Column(String, nullable=False, index=True)

    # Display info
    filename = Column(String, nullable=False)
    mime_type = Column(String, nullable=True)

    # Drive modification timestamp.
    # Format: ISO 8601 string from Drive API, stored as DateTime.
    # We can poll this to detect changes even without a watch.
    last_modified = Column(DateTime(timezone=True), nullable=True)

    # Watch / webhook subscription info.
    # watch_channel_id: the unique ID Google assigned to this subscription.
    # We receive it in the X-Goog-Channel-ID header of webhook POSTs.
    watch_channel_id = Column(String, nullable=True, index=True)
    watch_expiry = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    def __repr__(self):
        return (
            f"<DriveFile(drive_file_id={self.drive_file_id}, "
            f"document_id={self.document_id}, filename={self.filename})>"
        )
