"""Diagnostic script — run from project root with venv active."""
import pickle, faiss, numpy as np

# ── FAISS state ──────────────────────────────────────────────────────
index = faiss.read_index("data/faiss_index.index")
with open("data/faiss_index.meta", "rb") as f:
    meta = pickle.load(f)

print(f"FAISS vectors : {index.ntotal}")
print(f"Meta entries  : {len(meta)}")
if meta:
    print(f"Sample meta[0]: {str(meta[0])[:200]}")
else:
    print("Meta is EMPTY!")

# ── Test search ─────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

queries = [
    "team members names",
    "project title",
    "which industries domains problem",
]

for q in queries:
    vec = model.encode(q, normalize_embeddings=True).astype("float32").reshape(1, -1)
    D, I = index.search(vec, 5)
    print(f"\nQuery: '{q}'")
    print(f"  Top scores: {[round(float(s), 4) for s in D[0]]}")
    for score, idx in zip(D[0], I[0]):
        if idx >= 0:
            text_preview = str(meta[idx].get("text", ""))[:120].replace("\n", " ")
            print(f"  [{float(score):.4f}] {text_preview}")
