"""
Drive Service

WHAT DOES THIS FILE DO?
------------------------
This is the orchestrator for all Google Drive interactions.

It has 6 responsibilities:
  1. Build the OAuth2 consent URL (send user to Google login)
  2. Exchange the one-time code for tokens, save tokens to DB
  3. Auto-refresh the access_token before every API call
  4. List the user's Drive files (PDFs and Google Docs)
  5. Fetch a Drive file, run it through our existing pipeline, delete temp file
  6. Register a Drive watch (webhook) so we're notified of changes

MEMORY RULES (strictly enforced):
-----------------------------------
- Tokens are written to DB immediately. No global token variables.
- Drive file bytes are written to a temp file on disk immediately.
  The temp file is deleted in a `finally:` block — runs even if
  the embedding step crashes.
- The Google API client (`build(...)`) is constructed inside each
  function and goes out of scope when the function returns.
  No global client object.

HOW DOES OAUTH2 WORK? (simplified)
-------------------------------------
OAuth2 is a protocol that lets users grant your app access to their
Google data WITHOUT giving you their Google password.

Step 1: Your app → "Here's my client_id. User, please approve these scopes."
Step 2: User → logs in to Google, sees what permissions your app wants
Step 3: Google → "User approved. Here's a one-time `code`."
Step 4: Your app exchanges `code` → access_token + refresh_token
Step 5: Your app uses access_token to call Drive API
Step 6: access_token expires → use refresh_token silently → new access_token
"""

import os
import uuid
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from app.config.settings import settings
from app.models.drive_token import DriveToken
from app.models.drive_file import DriveFile
from app.utils.logger import get_logger

logger = get_logger(__name__)

# The Drive API scope we request.
# "drive.readonly" = read files the user explicitly selects (least privilege).
# We never request write access — we only read files, we never modify Drive.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# ---------------------------------------------------------------------------
# 1. Build OAuth2 Consent URL
# ---------------------------------------------------------------------------

def get_oauth_url() -> str:
    """
    Build the Google OAuth2 consent URL.

    HOW THIS WORKS:
    ---------------
    We create a Flow object using our client credentials.
    The flow builds a URL like:
        https://accounts.google.com/o/oauth2/auth
            ?client_id=YOUR_ID
            &redirect_uri=http://localhost:8000/auth/google/callback
            &scope=https://www.googleapis.com/auth/drive.readonly
            &response_type=code
            &access_type=offline    ← this is what gets us a refresh_token
            &prompt=consent         ← forces Google to include refresh_token every time

    The user visits this URL, logs in, sees the permission screen, approves.
    Google then redirects to our callback URL with a `code` query parameter.

    Args:
        None

    Returns:
        str: The full Google consent URL to redirect the user to

    Raises:
        ValueError: If GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET are not set
    """
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
        raise ValueError(
            "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in .env "
            "before using Google Drive integration."
        )

    # Build the OAuth2 flow.
    # Flow handles the URL construction and code exchange for us.
    flow = Flow.from_client_config(
        client_config={
            "web": {
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.GOOGLE_REDIRECT_URI],
            }
        },
        scopes=SCOPES,
        redirect_uri=settings.GOOGLE_REDIRECT_URI,
    )

    # authorization_url: the URL we redirect the user to
    # state: a random string to protect against CSRF attacks
    #        (we're not using it for now but it's good practice to include)
    authorization_url, state = flow.authorization_url(
        access_type="offline",   # Gets us a refresh_token
        prompt="consent",        # Forces Google to always include refresh_token
        include_granted_scopes="true",
    )

    return authorization_url


# ---------------------------------------------------------------------------
# 2. Exchange Code for Tokens → Save to DB
# ---------------------------------------------------------------------------

def exchange_code_for_tokens(code: str, user_id: str, db: Session) -> dict:
    """
    Exchange the one-time authorization code for access + refresh tokens.

    This is called once after the user approves on Google's consent screen.
    Google sends `code` to our /auth/google/callback URL.

    HOW TOKEN EXCHANGE WORKS:
    --------------------------
    The `code` is a one-time, short-lived string (valid for ~10 minutes).
    We send it (along with our client_secret) to Google's token endpoint.
    Google verifies everything and responds with:
      - access_token: use this for API calls (~1 hour lifetime)
      - refresh_token: use this to get new access_tokens (permanent)
      - expires_in: seconds until access_token expires (usually 3600)
      - token_type: "Bearer"

    MEMORY RULE:
    -------------
    The tokens from Google are in `credentials` (a local variable).
    We immediately write them to PostgreSQL.
    After this function returns, `credentials` goes out of scope.
    Tokens are never held in a Python dict or global between requests.

    Args:
        code     :  The authorization code from Google's redirect
        user_id  :  The user to link these tokens to
        db       :  Active database session

    Returns:
        dict with keys: user_id, has_refresh_token, token_expiry
        (We never return the raw token strings — callers don't need them)
    """
    flow = Flow.from_client_config(
        client_config={
            "web": {
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.GOOGLE_REDIRECT_URI],
            }
        },
        scopes=SCOPES,
        redirect_uri=settings.GOOGLE_REDIRECT_URI,
    )

    # fetch_token() sends the code to Google's token endpoint and
    # populates the flow's credentials object with the response.
    flow.fetch_token(code=code)
    credentials = flow.credentials

    # Calculate the exact expiry datetime.
    # credentials.expiry is already a datetime if available; fall back to +1 hour
    token_expiry = credentials.expiry or (datetime.now(timezone.utc) + timedelta(hours=1))

    # Check if a token row already exists for this user (re-authentication case)
    existing = db.query(DriveToken).filter(DriveToken.user_id == user_id).first()

    if existing:
        # UPDATE the existing row in place — don't create a duplicate
        existing.access_token = credentials.token
        # Only update refresh_token if Google sent a new one
        # (Google doesn't always re-send it on subsequent auth)
        if credentials.refresh_token:
            existing.refresh_token = credentials.refresh_token
        existing.token_expiry = token_expiry
        logger.info(f"Updated Drive tokens for user: {user_id}")
    else:
        # INSERT new row — first time this user authenticates
        token_row = DriveToken(
            user_id=user_id,
            access_token=credentials.token,
            refresh_token=credentials.refresh_token,
            token_expiry=token_expiry,
        )
        db.add(token_row)
        logger.info(f"Saved new Drive tokens for user: {user_id}")

    db.commit()

    # Return only safe metadata — never the raw token strings
    return {
        "user_id": user_id,
        "has_refresh_token": bool(credentials.refresh_token or (existing and existing.refresh_token)),
        "token_expiry": token_expiry.isoformat(),
    }


# ---------------------------------------------------------------------------
# 3. Get a Valid Access Token (auto-refresh if expired)
# ---------------------------------------------------------------------------

def _get_valid_credentials(user_id: str, db: Session) -> Credentials:
    """
    Retrieve a valid Google Credentials object for this user.

    This is an INTERNAL helper used by all Drive API functions.
    Callers don't deal with tokens — they just call this and get
    a ready-to-use Credentials object.

    AUTO-REFRESH LOGIC:
    --------------------
    Google access_tokens expire in ~1 hour.
    We check if the token is within 5 minutes of expiry.
    If so, we call `credentials.refresh(Request())` which:
      1. Sends the refresh_token to Google's token endpoint
      2. Gets a new access_token + new expiry
      3. Updates the credentials object in memory

    We then immediately write the new access_token + expiry back to DB.

    Args:
        user_id: The user whose credentials to fetch
        db:      Active database session

    Returns:
        google.oauth2.credentials.Credentials — ready for use

    Raises:
        ValueError: If no token exists for this user (not authenticated)
        RuntimeError: If the refresh fails (e.g. user revoked access)
    """
    token_row = db.query(DriveToken).filter(DriveToken.user_id == user_id).first()

    if not token_row:
        raise ValueError(
            f"No Google token found for user '{user_id}'. "
            "Please authenticate at GET /auth/google first."
        )

    # Build a Credentials object from our stored values
    credentials = Credentials(
        token=token_row.access_token,
        refresh_token=token_row.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
        scopes=SCOPES,
    )

    # Set the expiry on the credentials object so it knows when to refresh
    if token_row.token_expiry:
        credentials.expiry = token_row.token_expiry.replace(tzinfo=None)  # Google expects naive UTC

    # Check if the token is expired or within 5 minutes of expiry
    needs_refresh = (
        credentials.expired or
        (token_row.token_expiry and
         token_row.token_expiry <= datetime.now(timezone.utc) + timedelta(minutes=5))
    )

    if needs_refresh:
        logger.info(f"Access token expiring soon for user {user_id} — refreshing...")

        if not token_row.refresh_token:
            raise RuntimeError(
                "Access token expired and no refresh token is available. "
                "User must re-authenticate at GET /auth/google."
            )

        # This makes a real HTTP request to Google's token endpoint
        credentials.refresh(Request())

        # Write updated token back to DB immediately
        token_row.access_token = credentials.token
        token_row.token_expiry = credentials.expiry.replace(tzinfo=timezone.utc) if credentials.expiry else None
        db.commit()

        logger.info(f"Access token refreshed and saved for user {user_id}")

    return credentials


# ---------------------------------------------------------------------------
# 4. List Drive Files
# ---------------------------------------------------------------------------

def list_drive_files(user_id: str, db: Session) -> list:
    """
    List the user's Google Drive files filtered to PDFs and Google Docs.

    WHY ONLY THESE TWO MIME TYPES?
    --------------------------------
    - application/pdf                        → regular PDF files
    - application/vnd.google-apps.document   → Google Docs (like Word docs online)

    Google Docs can be exported as PDFs via the Drive API using
    `files.export()`. We handle that in fetch_and_ingest_file().

    Other file types (Sheets, Slides, etc.) don't make sense for
    document Q&A, so we filter them out.

    Args:
        user_id: The user whose Drive files to list
        db:      Active database session

    Returns:
        List of dicts:
        [
            {
                "id": "1BxiMV...",            ← Google's file ID
                "name": "Q3_Report.pdf",
                "mimeType": "application/pdf",
                "modifiedTime": "2024-01-15T10:30:00.000Z",
                "size": "245891",
                "already_ingested": True/False  ← whether we've already processed it
            },
            ...
        ]
    """
    credentials = _get_valid_credentials(user_id=user_id, db=db)

    # build() creates a Drive API v3 client.
    # This is a local variable — it goes out of scope when this function returns.
    # No global Drive client.
    drive_client = build("drive", "v3", credentials=credentials)

    # Drive API files.list() with a mimeType filter.
    # The query syntax is a subset of SQL — only "=" and "or" are supported.
    mime_filter = (
        "mimeType = 'application/pdf' or "
        "mimeType = 'application/vnd.google-apps.document'"
    )

    result = drive_client.files().list(
        q=mime_filter,
        fields="files(id, name, mimeType, modifiedTime, size)",
        orderBy="modifiedTime desc",   # Most recently modified first
        pageSize=50,                   # Max files to return (increase if needed)
    ).execute()

    files = result.get("files", [])

    # Check which files are already ingested (in our drive_files table)
    ingested_ids = {
        row.drive_file_id
        for row in db.query(DriveFile).filter(DriveFile.user_id == user_id).all()
    }

    # Add an "already_ingested" flag to each file so the frontend
    # can show a visual indicator and avoid duplicate ingestion
    for f in files:
        f["already_ingested"] = f["id"] in ingested_ids

    logger.info(f"Listed {len(files)} Drive files for user {user_id}")
    return files


# ---------------------------------------------------------------------------
# 5. Fetch Drive File → Run Existing Pipeline → Delete Temp File
# ---------------------------------------------------------------------------

def fetch_and_ingest_file(drive_file_id: str, user_id: str, db: Session) -> dict:
    """
    Download a Drive file and run it through the existing ingestion pipeline.

    FLOW:
    ------
    1. Call Drive API to download file bytes
    2. Write bytes to a temporary file on disk (uploads/temp_{uuid}.pdf)
    3. Call document_service.process_document() — UNCHANGED existing pipeline:
         extract text → chunk → embed → save to FAISS + PostgreSQL
    4. Delete the temp file IMMEDIATELY in a `finally:` block
       (runs even if step 3 crashes)
    5. Save a DriveFile record linking drive_file_id → document_id

    MEMORY RULE:
    -------------
    We never hold the file bytes in a Python variable beyond the
    file.write() call. The bytes go from Google's response stream
    directly to disk using `MediaIoBaseDownload` (streaming, not
    all-at-once). The temp file is immediately deleted after embedding.

    GOOGLE DOCS SPECIAL CASE:
    --------------------------
    Google Docs (vnd.google-apps.document) can't be downloaded directly.
    They must be "exported" as PDF using files.export().
    We detect the mimeType and use the correct API call.

    Args:
        drive_file_id : Google's file ID (from list_drive_files)
        user_id       : Who is ingesting this file
        db            : Active database session

    Returns:
        dict with document_id, filename, status, chunks_created
    """
    import io
    from googleapiclient.http import MediaIoBaseDownload
    from app.services import document_service as doc_svc
    from app.models.document import Document
    from app.models.chunk import Chunk

    # Check for duplicate — don't ingest the same Drive file twice
    existing_drive_file = db.query(DriveFile).filter(
        DriveFile.drive_file_id == drive_file_id
    ).first()
    if existing_drive_file:
        raise ValueError(
            f"Drive file '{drive_file_id}' is already ingested as "
            f"document '{existing_drive_file.document_id}'. "
            "Use /api/drive/refresh to re-ingest."
        )

    credentials = _get_valid_credentials(user_id=user_id, db=db)
    drive_client = build("drive", "v3", credentials=credentials)

    # Get file metadata first (name, mimeType, modifiedTime)
    file_meta = drive_client.files().get(
        fileId=drive_file_id,
        fields="id, name, mimeType, modifiedTime"
    ).execute()

    filename = file_meta.get("name", "unknown_drive_file")
    mime_type = file_meta.get("mimeType", "")
    modified_time_str = file_meta.get("modifiedTime")

    logger.info(f"Fetching Drive file: '{filename}' ({mime_type})")

    # Generate a unique temporary file path
    doc_id = str(uuid.uuid4())
    temp_filename = f"drive_{doc_id}_{filename}"
    # Ensure the filename ends with .pdf
    if not temp_filename.lower().endswith(".pdf"):
        temp_filename += ".pdf"
    temp_path = os.path.join(settings.UPLOAD_DIR, temp_filename)

    # Make sure the uploads directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    try:
        # ----------------------------------------------------------------
        # STEP 1: Download file bytes to temp file on disk
        # ----------------------------------------------------------------
        with open(temp_path, "wb") as temp_file:
            if mime_type == "application/vnd.google-apps.document":
                # Google Docs must be exported as PDF
                # export_media() streams the PDF export
                request = drive_client.files().export_media(
                    fileId=drive_file_id,
                    mimeType="application/pdf"
                )
            else:
                # Regular PDF — download directly
                request = drive_client.files().get_media(fileId=drive_file_id)

            # MediaIoBaseDownload streams the file in chunks (8MB default)
            # This avoids loading the entire file into memory at once.
            downloader = MediaIoBaseDownload(temp_file, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%")

        file_size = os.path.getsize(temp_path)
        logger.info(f"Temp file written: {temp_path} ({file_size} bytes)")

        # ----------------------------------------------------------------
        # STEP 2: Create a Document DB record for this Drive file
        # ----------------------------------------------------------------
        # We insert it manually (instead of through the upload endpoint)
        # so we can set our own doc_id and the correct filename.
        document = Document(
            id=doc_id,
            filename=filename,
            user_id=user_id,
            file_path=temp_path,        # Points to the temp file (will be deleted shortly)
            file_size=file_size,
            status="processing",
        )
        db.add(document)
        db.commit()

        # ----------------------------------------------------------------
        # STEP 3: Run the EXISTING ingestion pipeline (unchanged)
        # ----------------------------------------------------------------
        # process_document() does: extract → chunk → embed → FAISS + DB
        doc_svc.process_document(
            document_id=doc_id,
            file_path=temp_path,
            db=db,
        )

        # ----------------------------------------------------------------
        # STEP 4: Save DriveFile record linking drive_file_id → document_id
        # ----------------------------------------------------------------
        last_modified = None
        if modified_time_str:
            try:
                last_modified = datetime.fromisoformat(modified_time_str.replace("Z", "+00:00"))
            except Exception:
                pass

        drive_file_record = DriveFile(
            user_id=user_id,
            drive_file_id=drive_file_id,
            document_id=doc_id,
            filename=filename,
            mime_type=mime_type,
            last_modified=last_modified,
        )
        db.add(drive_file_record)
        db.commit()

        logger.info(f"Drive file ingested: '{filename}' → document_id={doc_id}")

        # Count chunks created for the response
        chunks_count = db.query(Chunk).filter(Chunk.document_id == doc_id).count()

        return {
            "document_id": doc_id,
            "filename": filename,
            "status": "completed",
            "chunks_created": chunks_count,
        }

    finally:
        # ----------------------------------------------------------------
        # ALWAYS delete the temp file — even if an exception occurred
        # ----------------------------------------------------------------
        # `finally` runs no matter what: success, exception, even SystemExit.
        # This guarantees we never leave Drive file bytes sitting on disk.
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Temp file deleted: {temp_path}")


# ---------------------------------------------------------------------------
# 6. Register Drive Watch (Webhook for Change Notifications)
# ---------------------------------------------------------------------------

def register_drive_watch(drive_file_id: str, user_id: str, webhook_url: str, db: Session) -> dict:
    """
    Register a Drive "push notification" watch for a file.

    HOW DRIVE WATCH WORKS:
    ----------------------
    Drive API lets you say: "Watch this file. When it changes,
    call my webhook URL."

    We call files.watch() with:
      - A unique channel_id (we generate a UUID)
      - Our webhook URL (must be HTTPS and publicly reachable)
      - Type = "web_hook"

    Google stores this subscription and remembers to notify us.
    The subscription lives for up to 7 days. After that, Drive stops
    sending notifications. We'd need to re-register to continue watching.

    When Google detects a change to the watched file, it sends a POST
    request to our webhook URL with these headers:
      X-Goog-Channel-ID: <our channel_id>
      X-Goog-Resource-State: change / sync / etc.

    Our webhook handler (/api/drive/webhook) reads X-Goog-Channel-ID,
    looks up the document_id from drive_files table, and re-ingests.

    Args:
        drive_file_id : The Drive file to watch
        user_id       : The user who owns this file
        webhook_url   : The publicly reachable URL (e.g. https://your-app.onrender.com/api/drive/webhook)
        db            : Active database session

    Returns:
        dict with channel_id, watch_expiry

    NOTE ON LOCAL DEVELOPMENT:
    --------------------------
    Google cannot call localhost:8000. For local testing, use ngrok:
        ngrok http 8000
    Then pass the ngrok URL as webhook_url.
    """
    credentials = _get_valid_credentials(user_id=user_id, db=db)
    drive_client = build("drive", "v3", credentials=credentials)

    # Unique channel ID — must be unique per watch registration
    channel_id = str(uuid.uuid4())

    watch_body = {
        "id": channel_id,
        "type": "web_hook",
        "address": webhook_url,   # Must be HTTPS
    }

    response = drive_client.files().watch(
        fileId=drive_file_id,
        body=watch_body
    ).execute()

    # Google responds with expiration as milliseconds since epoch
    expiry_ms = response.get("expiration")
    watch_expiry = None
    if expiry_ms:
        watch_expiry = datetime.fromtimestamp(int(expiry_ms) / 1000, tz=timezone.utc)

    # Save channel_id and expiry to drive_files table
    drive_file_record = db.query(DriveFile).filter(
        DriveFile.drive_file_id == drive_file_id,
        DriveFile.user_id == user_id,
    ).first()

    if drive_file_record:
        drive_file_record.watch_channel_id = channel_id
        drive_file_record.watch_expiry = watch_expiry
        db.commit()
        logger.info(
            f"Watch registered for file '{drive_file_record.filename}': "
            f"channel_id={channel_id}, expires={watch_expiry}"
        )

    return {
        "channel_id": channel_id,
        "watch_expiry": watch_expiry.isoformat() if watch_expiry else None,
        "drive_file_id": drive_file_id,
    }


# ---------------------------------------------------------------------------
# 7. Handle Webhook — Re-ingest on Change
# ---------------------------------------------------------------------------

def handle_webhook_change(channel_id: str, resource_state: str, db: Session) -> dict:
    """
    Handle a Drive change notification from Google.

    Called by POST /api/drive/webhook when Google detects a file changed.

    HOW WE KNOW WHAT CHANGED:
    --------------------------
    Google sends a POST with:
      Header: X-Goog-Channel-ID  = <our channel_id>
      Header: X-Goog-Resource-State = "change" or "sync" or "remove"

    We look up the channel_id in drive_files → get document_id.
    Then we:
      1. Delete all Chunk rows for that document_id (stale data)
      2. Re-download the file from Drive
      3. Re-run the full ingestion pipeline
      4. The document_id stays the same — FAISS gets updated vectors

    "sync" is a test notification Google sends right after watch registration.
    We skip reprocessing for "sync" — it's just a handshake.

    Args:
        channel_id     : From X-Goog-Channel-ID header
        resource_state : From X-Goog-Resource-State header
        db             : Active database session

    Returns:
        dict with status and which document was updated
    """
    from app.models.chunk import Chunk

    # Handshake notification — Google sends this immediately after registration
    # to verify our webhook URL is reachable. No action needed.
    if resource_state == "sync":
        logger.info(f"Drive watch sync confirmation received for channel: {channel_id}")
        return {"status": "sync_acknowledged", "channel_id": channel_id}

    # Look up which file this channel belongs to
    drive_file_record = db.query(DriveFile).filter(
        DriveFile.watch_channel_id == channel_id
    ).first()

    if not drive_file_record:
        logger.warning(f"Received webhook for unknown channel_id: {channel_id}")
        return {"status": "unknown_channel", "channel_id": channel_id}

    logger.info(
        f"Drive change detected for '{drive_file_record.filename}' "
        f"(document_id={drive_file_record.document_id})"
    )

    # Delete old chunks for this document (stale data from before the change)
    deleted = db.query(Chunk).filter(
        Chunk.document_id == drive_file_record.document_id
    ).delete()
    db.commit()
    logger.info(f"Deleted {deleted} stale chunks for document {drive_file_record.document_id}")

    # Remove the DriveFile record so fetch_and_ingest_file() doesn't
    # think this is a duplicate and rejects it
    old_document_id = drive_file_record.document_id
    old_drive_file_id = drive_file_record.drive_file_id
    old_channel_id = drive_file_record.watch_channel_id
    old_watch_expiry = drive_file_record.watch_expiry
    user_id = drive_file_record.user_id

    db.delete(drive_file_record)
    db.commit()

    # Re-ingest the file (download fresh copy from Drive + full pipeline)
    result = fetch_and_ingest_file(
        drive_file_id=old_drive_file_id,
        user_id=user_id,
        db=db,
    )

    # Restore the watch channel info on the new DriveFile record
    new_drive_file = db.query(DriveFile).filter(
        DriveFile.drive_file_id == old_drive_file_id
    ).first()
    if new_drive_file:
        new_drive_file.watch_channel_id = old_channel_id
        new_drive_file.watch_expiry = old_watch_expiry
        db.commit()

    logger.info(f"Re-ingestion complete: {result}")
    return {
        "status": "reingested",
        "old_document_id": old_document_id,
        "new_document_id": result["document_id"],
        "filename": result["filename"],
        "chunks_created": result["chunks_created"],
    }
