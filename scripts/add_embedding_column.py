"""
One-time migration: Add 'embedding' column to the 'chunks' table.

Run this ONCE before deploying the memory-optimised version.
It adds a TEXT column to store embedding vectors as JSON,
so FAISS can be rebuilt from stored vectors without re-running the model.

Usage:
    python scripts/add_embedding_column.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.database.postgres import engine
from sqlalchemy import text


def migrate():
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'embedding'
        """))

        if result.fetchone():
            print("✅ Column 'embedding' already exists in 'chunks' table. Nothing to do.")
            return

        # Add the column
        conn.execute(text("""
            ALTER TABLE chunks ADD COLUMN embedding TEXT
        """))
        conn.commit()
        print("✅ Added 'embedding' column to 'chunks' table.")


if __name__ == "__main__":
    migrate()
