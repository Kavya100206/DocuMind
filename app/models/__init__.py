"""
Data models package
Contains all database models (M in MVC)

WHY DO WE IMPORT HERE?
-----------------------
SQLAlchemy's Base.metadata only knows about a model if that model's
class has been imported at least once before init_db() is called.

init_db() calls Base.metadata.create_all() which creates tables for
every model it knows about. If we don't import a model here, its table
will never be created in PostgreSQL.

We import all models in this __init__.py so that any code that does
    from app.models import document, chunk
will automatically also register the Drive models with SQLAlchemy.
"""

# Existing models
from app.models import document  # noqa: F401
from app.models import chunk     # noqa: F401

# Phase 1 — Google Drive Integration
from app.models import drive_token  # noqa: F401 — creates `drive_tokens` table
from app.models import drive_file   # noqa: F401 — creates `drive_files` table
