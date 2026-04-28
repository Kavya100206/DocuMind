"""
Drive Token Model

WHAT DOES THIS FILE DO?
------------------------
This is the database table that stores Google OAuth2 tokens.

WHY DO WE NEED THIS?
---------------------
When a user logs in with Google, Google gives us two tokens:

  1. access_token  — short-lived (expires in ~1 hour).
                     Used like a password to call Drive API.

  2. refresh_token — long-lived (never expires unless user revokes access).
                     Used to silently get a new access_token when it expires.

The golden rule: NEVER store tokens in Python variables between requests.
If the server restarts, in-memory tokens are gone → user has to re-login.
Storing in PostgreSQL means tokens survive restarts and deployments.

WHAT IS THE FLOW?
-----------------
1. User visits /auth/google → redirected to Google consent screen
2. User approves → Google sends a one-time "code" to our callback URL
3. We exchange that code for access_token + refresh_token
4. We write both into this table immediately
5. Every Drive API call reads the token from this table
6. If access_token is expired, we use refresh_token to get a new one
   and update this table in place

TABLE: drive_tokens
"""

from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.sql import func
from app.database.postgres import Base
import uuid


class DriveToken(Base):
    """
    Stores Google OAuth2 tokens for a user.

    One row per user. If the user re-authenticates, we UPSERT
    (update the existing row), not insert a new one.

    Columns:
    --------
    - id           : UUID primary key
    - user_id      : Which user this token belongs to
    - access_token : Short-lived token (~1 hour). Used for API calls.
    - refresh_token: Long-lived token. Used to renew access_token silently.
    - token_expiry : Exact datetime when access_token expires.
                     We check this before every API call.
    - created_at   : When the user first authenticated
    - updated_at   : Last time we refreshed the token (auto-updated by DB)
    """

    __tablename__ = "drive_tokens"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True
    )

    # Which user this token belongs to.
    # Indexed so we can look up a token by user_id quickly.
    user_id = Column(String, nullable=False, unique=True, index=True)

    # The actual token strings — stored as Text (unlimited length).
    # access_token is typically ~200 characters.
    # refresh_token is typically ~100 characters.
    # Never truncate these — a partial token is useless.
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=True)   # Nullable: first request may not include it

    # When the access_token expires.
    # We compare this to datetime.utcnow() before every API call.
    # If it's within 5 minutes of expiry, we proactively refresh it.
    token_expiry = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),   # PostgreSQL auto-updates this on every UPDATE
        nullable=False
    )

    def __repr__(self):
        return f"<DriveToken(user_id={self.user_id}, expiry={self.token_expiry})>"
