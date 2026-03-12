import os, sys
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

# Show current docs
cur.execute("SELECT id, filename, status FROM documents ORDER BY created_at")
docs = cur.fetchall()
print("Current documents:")
for d in docs:
    print(f"  {d[0][:8]}... | {d[1]} | {d[2]}")

# Delete Batch docs and their chunks
cur.execute("DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE filename ILIKE '%batch%')")
deleted_chunks = cur.rowcount
cur.execute("DELETE FROM documents WHERE filename ILIKE '%batch%'")
deleted_docs = cur.rowcount
conn.commit()
print(f"\nDeleted {deleted_docs} Batch doc(s) and {deleted_chunks} chunk(s)")

# Confirm remaining
cur.execute("SELECT id, filename FROM documents ORDER BY created_at")
remaining = cur.fetchall()
print("Remaining documents:")
for d in remaining:
    print(f"  {d[0][:8]}... | {d[1]}")
cur.close()
conn.close()
