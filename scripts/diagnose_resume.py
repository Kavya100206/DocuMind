"""Diagnose resume chunks — writes clean UTF-8 output."""
import sys, os, pickle, faiss
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
import psycopg2

OUT = open("diag_resume_out.txt", "w", encoding="utf-8")
def p(*a): print(*a); OUT.write(" ".join(str(x) for x in a) + "\n")

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("SELECT id, filename FROM documents WHERE filename ILIKE '%resume%' OR filename ILIKE '%kavya%' OR filename ILIKE '%cv%' ORDER BY created_at DESC LIMIT 3")
docs = cur.fetchall()
p("Resume docs:", docs)

resume_id = docs[0][0] if docs else None
if not resume_id:
    p("No resume found"); sys.exit(0)

# All resume chunks
cur.execute("SELECT page_number, char_count, LEFT(text,300) FROM chunks WHERE document_id=%s ORDER BY page_number", (resume_id,))
chunks = cur.fetchall()
p(f"\nResume chunk count: {len(chunks)}")
for i, (pg, cc, txt) in enumerate(chunks):
    p(f"\n[Chunk {i+1} | Page {pg} | {cc} chars]")
    p(txt.replace("\n", " "))

# FAISS retrieval test
p("\n\n=== FAISS retrieval for projects query ===")
from app.config.settings import settings
index = faiss.read_index("data/faiss_index.index")
with open("data/faiss_index.meta", "rb") as f:
    meta = pickle.load(f)

from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)

q = "list all projects in the resume"
prefix = "Represent this sentence for searching relevant passages: " if settings.LOCAL_EMBEDDING_MODEL.startswith("BAAI/bge") else ""
vec = model.encode(prefix + q, normalize_embeddings=True).astype("float32").reshape(1,-1)
D, I = index.search(vec, 20)

p(f"Threshold: {settings.SIMILARITY_THRESHOLD}")
p(f"Results (20 candidates):")
for score, idx in zip(D[0], I[0]):
    if idx < 0: continue
    did = meta[idx].get("document_id","")
    flag = "[RESUME]" if did == resume_id else "[other ]"
    txt = str(meta[idx].get("text",""))[:100].replace("\n"," ")
    above = "ABOVE" if float(score) >= settings.SIMILARITY_THRESHOLD else "below"
    p(f"  {above} {float(score):.4f} {flag} {txt}")

# Check score window
p("\n=== Score window analysis ===")
resume_scores = [float(D[0][j]) for j in range(len(D[0])) if I[0][j]>=0 and meta[I[0][j]].get("document_id")==resume_id]
if resume_scores:
    top = max(resume_scores)
    p(f"Top RESUME score: {top:.4f}")
    p(f"Score window (top - 0.05): {top-0.05:.4f}")
    in_window = [s for s in resume_scores if s >= top - 0.05]
    p(f"Resume chunks in window: {len(in_window)} of {len(resume_scores)}")
    p(f"All resume scores: {[round(s,4) for s in resume_scores]}")

cur.close(); conn.close()
OUT.close()
print("Written to diag_resume_out.txt")
