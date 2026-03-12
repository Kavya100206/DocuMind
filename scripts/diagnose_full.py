"""Minimal FAISS diagnostic — no ORM needed."""
import pickle, faiss, os
from dotenv import load_dotenv
load_dotenv()

# --- 1. FAISS state ---
index = faiss.read_index("data/faiss_index.index")
with open("data/faiss_index.meta", "rb") as f:
    meta = pickle.load(f)

print("FAISS vectors:", index.ntotal)
print("Meta entries:", len(meta))

from collections import Counter
doc_counts = Counter(m.get("document_id", "NONE") for m in meta)
print("\nDoc IDs in FAISS meta:")
for did, cnt in doc_counts.items():
    print(f"  {did} -> {cnt} chunks")

# --- 2. DB query via psycopg2 directly ---
import psycopg2
url = os.environ["DATABASE_URL"]
conn = psycopg2.connect(url)
cur = conn.cursor()
cur.execute("SELECT id, filename, status FROM documents ORDER BY created_at DESC")
db_docs = cur.fetchall()
print("\nDB Documents:")
for row in db_docs:
    print(f"  {row[0]} | {row[1][:40]} | {row[2]}")

cur.execute("SELECT COUNT(*) FROM chunks")
print("DB Chunk count:", cur.fetchone()[0])

db_ids = {row[0] for row in db_docs}
faiss_ids = set(doc_counts.keys())
stale = faiss_ids - db_ids
missing = db_ids - faiss_ids
print("\nFAISS ids NOT in DB (stale):", stale if stale else "none")
print("DB ids NOT in FAISS (missing):", missing if missing else "none")

# --- 3. Quick search test ---
from sentence_transformers import SentenceTransformer
import numpy as np

THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.15"))
model = SentenceTransformer("all-MiniLM-L6-v2")

for q in ["who are the team members", "what is the project title", "industries domains"]:
    vec = model.encode(q, normalize_embeddings=True).astype("float32").reshape(1,-1)
    D, I = index.search(vec, 5)
    results_above = [(round(float(D[0][j]),4), meta[I[0][j]].get("document_id","")[:8])
                     for j in range(5) if I[0][j]>=0 and float(D[0][j]) >= THRESHOLD]
    print(f"\n'{q}'")
    print(f"  Threshold={THRESHOLD}  Above threshold: {results_above}")
    print(f"  All top-5 scores: {[round(float(s),4) for s in D[0]]}")

cur.close()
conn.close()
