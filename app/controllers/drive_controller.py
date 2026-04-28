"""
Drive Controller

WHAT DOES THIS FILE DO?
------------------------
This is the HTTP layer for Phase 1 — Google Drive Integration.
It defines 5 endpoints that expose drive_service.py to the outside world.

ENDPOINTS:
-----------
  GET  /auth/google                → Redirect user to Google consent screen
  GET  /auth/google/callback       → Google sends back the code; exchange for tokens
  GET  /api/drive/files            → List user's Drive files (PDFs + Docs)
  POST /api/drive/ingest           → Fetch a Drive file + run existing pipeline
  POST /api/drive/webhook          → Receive Drive change push notifications

RELATIONSHIP TO EXISTING CODE:
--------------------------------
- Manual upload  → POST /api/documents/upload  (UNCHANGED, still works)
- Drive upload   → POST /api/drive/ingest      (NEW, same pipeline internally)
Both paths end up in document_service.process_document() → same FAISS + DB.

MEMORY RULE (controller layer):
---------------------------------
This controller never touches tokens directly.
All token handling is inside drive_service.py.
Endpoints only pass user_id + db → service does the rest.
"""

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from app.database.postgres import get_db
from app.services import drive_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# We use TWO routers:
#   auth_router  → prefix /auth  (OAuth2 flow, no /api prefix)
#   drive_router → prefix /api/drive  (Drive file operations)
#
# WHY SEPARATE ROUTERS?
# ----------------------
# The OAuth2 callback URL (/auth/google/callback) must exactly match
# what you registered in Google Cloud Console.
# If you registered "http://localhost:8000/auth/google/callback",
# having it under /api/auth/... would break the OAuth flow.
# So we keep /auth/... clean and separate from the API routes.
# ---------------------------------------------------------------------------
auth_router = APIRouter(prefix="/auth", tags=["Google Auth"])
drive_router = APIRouter(prefix="/api/drive", tags=["Google Drive"])

# Hardcoded for MVP  — same pattern as existing code (user_id = "default_user")
# In a real multi-user app, this would come from a session cookie or JWT token
DEFAULT_USER_ID = "default_user"


# ---------------------------------------------------------------------------
# GET /auth/google
# ---------------------------------------------------------------------------
@auth_router.get(
    "/google",
    summary="Start Google OAuth2 flow",
    description="Redirects the user to Google's consent screen to authorize Drive access.",
)
async def google_auth_start():
    """
    Redirect the user to Google's OAuth2 consent screen.

    HOW IT WORKS:
    -------------
    1. We call drive_service.get_oauth_url() which builds a Google URL
       like: https://accounts.google.com/o/oauth2/auth?client_id=...
    2. We return a 302 redirect — the browser follows it automatically
    3. User sees Google's "DocuMind wants to access your Drive" screen
    4. User clicks "Allow"
    5. Google redirects back to /auth/google/callback?code=...

    WHAT COULD GO WRONG:
    --------------------
    - GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET not set in .env
      → 400 error with a helpful message
    """
    try:
        url = drive_service.get_oauth_url()
        # 302 redirect — browser follows automatically
        return RedirectResponse(url=url)
    except ValueError as e:
        # Missing credentials in .env
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# GET /auth/google/callback
# ---------------------------------------------------------------------------
@auth_router.get(
    "/google/callback",
    summary="Handle Google OAuth2 callback",
    description="Exchanges the authorization code for tokens and saves them to DB.",
)
async def google_auth_callback(
    code: str,          # Google sends this as a query param: ?code=...
    db: Session = Depends(get_db),
    error: Optional[str] = None,  # Google sends ?error=access_denied if user clicks "Deny"
):
    """
    Handle the OAuth2 callback from Google.

    HOW IT WORKS:
    -------------
    1. Google redirects here with ?code=<one-time-code>
    2. We pass the code to drive_service.exchange_code_for_tokens()
    3. The service exchanges it for access_token + refresh_token
    4. Tokens are written to PostgreSQL — NEVER returned in the response
    5. We redirect the user to /ui so they can start using Drive features

    WHY REDIRECT TO /ui AFTER?
    ---------------------------
    After OAuth completes, the user is "logged in with Google".
    We send them back to the frontend so they can immediately
    click "Browse Drive Files" without any manual navigation.

    WHAT COULD GO WRONG:
    --------------------
    - User clicked "Deny" → Google sends ?error=access_denied
    - Code already used (each code is one-time only)
    - Clock skew between server and Google (rare)
    """
    # Handle user clicking "Deny" on Google's consent screen
    if error:
        logger.warning(f"OAuth2 error from Google: {error}")
        raise HTTPException(
            status_code=400,
            detail=f"Google authorization failed: {error}. Please try again."
        )

    try:
        result = drive_service.exchange_code_for_tokens(
            code=code,
            user_id=DEFAULT_USER_ID,
            db=db,
        )
        logger.info(f"OAuth2 complete for user: {result['user_id']}")

        # Redirect to the frontend UI — user is now authenticated with Drive
        return RedirectResponse(url="/ui?drive_connected=true")

    except Exception as e:
        logger.error(f"Token exchange failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to exchange authorization code: {str(e)}"
        )


# ---------------------------------------------------------------------------
# GET /api/drive/files
# ---------------------------------------------------------------------------
@drive_router.get(
    "/files",
    summary="List Google Drive files",
    description="Returns the user's Drive files filtered to PDFs and Google Docs.",
)
async def list_drive_files(db: Session = Depends(get_db)):
    """
    List the user's Google Drive files (PDFs and Docs only).

    RESPONSE SHAPE:
    ---------------
    {
        "files": [
            {
                "id": "1BxiMV...",
                "name": "Q3_Report.pdf",
                "mimeType": "application/pdf",
                "modifiedTime": "2024-01-15T10:30:00.000Z",
                "size": "245891",
                "already_ingested": false
            },
            ...
        ],
        "total": 12
    }

    The "already_ingested" flag lets the frontend show which files
    are already in DocuMind vs which ones are new.

    WHAT COULD GO WRONG:
    --------------------
    - User hasn't authenticated yet → 401 with "please visit /auth/google"
    - Token expired and refresh fails (user revoked access) → 401
    """
    try:
        files = drive_service.list_drive_files(
            user_id=DEFAULT_USER_ID,
            db=db,
        )
        return {"files": files, "total": len(files)}

    except ValueError as e:
        # "No token found" — user hasn't authenticated via OAuth yet
        raise HTTPException(
            status_code=401,
            detail=str(e) + " Visit GET /auth/google to connect your Drive."
        )
    except Exception as e:
        logger.error(f"Failed to list Drive files: {e}")
        raise HTTPException(status_code=500, detail=f"Drive API error: {str(e)}")


# ---------------------------------------------------------------------------
# POST /api/drive/ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """
    Request body for ingesting a Drive file.

    Why a Pydantic model?
    ---------------------
    FastAPI uses Pydantic to automatically validate the JSON request body.
    If the client sends anything other than a string for drive_file_id,
    FastAPI returns a 422 Unprocessable Entity automatically.

    We also add webhook_url as an optional field so the client can
    provide their public URL for Drive watch registration.
    If not provided, we'll only ingest (no auto-update on change).
    """
    drive_file_id: str
    register_watch: bool = False     # Whether to register a Drive watch after ingestion
    webhook_url: Optional[str] = None    # Required if register_watch=True


@drive_router.post(
    "/ingest",
    summary="Ingest a Drive file into DocuMind",
    description=(
        "Downloads a file from Google Drive, runs it through the ingestion pipeline "
        "(extract → chunk → embed → FAISS), and optionally registers a Drive watch "
        "to auto-update when the file changes."
    ),
    status_code=202,
)
async def ingest_drive_file(
    body: IngestRequest,
    db: Session = Depends(get_db),
):
    """
    Ingest a specific Google Drive file into DocuMind.

    FLOW:
    -----
    1. Download the file from Drive (streaming, not all-at-once)
    2. Write to temp file on disk
    3. Run document_service.process_document() — SAME pipeline as manual upload
    4. Delete temp file immediately
    5. Optionally register a Drive watch for auto-updates

    The file will appear in GET /api/documents/ just like a manually uploaded file.
    Q&A and search work identically for Drive-ingested documents.

    RESPONSE:
    ---------
    {
        "document_id": "abc-123",
        "filename": "Q3_Report.pdf",
        "status": "completed",
        "chunks_created": 47,
        "watch": {                       ← only present if register_watch=true
            "channel_id": "uuid...",
            "watch_expiry": "2024-01-22T10:30:00Z"
        }
    }
    """
    try:
        result = drive_service.fetch_and_ingest_file(
            drive_file_id=body.drive_file_id,
            user_id=DEFAULT_USER_ID,
            db=db,
        )

        response = {**result}

        # Optionally register a Drive watch for automatic re-ingestion on changes
        if body.register_watch:
            if not body.webhook_url:
                raise HTTPException(
                    status_code=400,
                    detail="webhook_url is required when register_watch=true. "
                           "Use your public URL (e.g. https://your-app.onrender.com) "
                           "or an ngrok tunnel for local development."
                )
            watch_result = drive_service.register_drive_watch(
                drive_file_id=body.drive_file_id,
                user_id=DEFAULT_USER_ID,
                webhook_url=body.webhook_url + "/api/drive/webhook",
                db=db,
            )
            response["watch"] = watch_result

        return response

    except ValueError as e:
        # "Already ingested" or "No token" errors
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Drive ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ---------------------------------------------------------------------------
# POST /api/drive/webhook
# ---------------------------------------------------------------------------
@drive_router.post(
    "/webhook",
    summary="Receive Drive change notifications",
    description=(
        "Called by Google Drive when a watched file changes. "
        "Automatically re-ingests the updated file."
    ),
    include_in_schema=False,  # Hide from Swagger UI — it's for Google, not users
)
async def drive_webhook(
    request: Request,
    db: Session = Depends(get_db),
    # Google sends channel info in HTTP headers, not in the body
    x_goog_channel_id: Optional[str] = Header(None, alias="X-Goog-Channel-ID"),
    x_goog_resource_state: Optional[str] = Header(None, alias="X-Goog-Resource-State"),
):
    """
    Handle Google Drive change push notification.

    HOW GOOGLE CALLS THIS:
    ----------------------
    When a watched file changes, Google sends a POST like:

        POST /api/drive/webhook  HTTP/1.1
        X-Goog-Channel-ID: <our channel_id (a UUID we generated)>
        X-Goog-Resource-State: change
        Content-Length: 0

    The body is empty. All info we need is in the headers.

    We map channel_id → document in our drive_files table, then
    delete old chunks and re-ingest the file.

    WHY 200 EVEN IF WE DON'T KNOW THE CHANNEL?
    --------------------------------------------
    Google retries webhook delivery on any non-2xx response.
    If we return 404 for unknown channels, Google will hammer us
    with retries. Instead we always return 200 and log the issue.
    """
    if not x_goog_channel_id:
        # Shouldn't happen — but guard against malformed requests
        logger.warning("Webhook called without X-Goog-Channel-ID header")
        return JSONResponse(status_code=200, content={"status": "ignored"})

    logger.info(
        f"Drive webhook received: channel={x_goog_channel_id}, "
        f"state={x_goog_resource_state}"
    )

    try:
        result = drive_service.handle_webhook_change(
            channel_id=x_goog_channel_id,
            resource_state=x_goog_resource_state or "change",
            db=db,
        )
        # Always return 200 to Google (see docstring explanation)
        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        logger.error(f"Webhook processing error: {e}", exc_info=True)
        # Still 200 to Google so it doesn't retry
        return JSONResponse(
            status_code=200,
            content={"status": "error", "detail": str(e)}
        )
