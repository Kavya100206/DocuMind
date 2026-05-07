## Assignment 03 — RAG Pipeline Mapping

This project fulfills and significantly exceeds the NotebookLM RAG assignment requirements.

### Marking Scheme Coverage

| Criterion | Marks | How DocuMind covers it |
|---|---|---|
| GitHub Repository | 2 | Public repo at github.com/Kavya100206/DocuMind |
| Live Project | 2 | Deployed at https://web-production-6f28c.up.railway.app/ui |
| RAG Pipeline | 3 | Full 6-stage pipeline (see below) |
| Answer Quality | 2 | Hard refusal contract + qualifier-distance grounding gate |
| Code Quality & Docs | 1 | Modular service architecture + behavioural test suite |

### RAG Pipeline — End to End

**1. Ingestion**
PDF upload via `/api/documents/upload`. File is validated, parsed, and queued for chunking. Raw file is scrubbed from disk post-processing.

**2. Chunking**
Section-aware chunking (`chunking_service.py`) — preserves document structure (section headers, semantic blocks) instead of splitting on raw character count. Config: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`.

**3. Embedding**
`sentence-transformers/all-MiniLM-L6-v2` generates dense vector embeddings per chunk (`embedding_service.py`). Stored in PostgreSQL (NeonDB) and indexed into FAISS.

**4. Storage**
Dual storage — chunk text + metadata in PostgreSQL, dense vectors in FAISS (`faiss_service.py`). FAISS index is rebuilt deterministically from the DB on startup if missing.

**5. Retrieval**
Hybrid retrieval (`retrieval_service.py`) — FAISS semantic search (65%) + BM25 lexical search (35%) merged and reranked by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-2-v2`) on the top 10 candidates. Retrieval latency: ~120ms.

**6. Generation**
LangGraph agentic loop (`agent_service.py`) — an LLM router selects the retrieval tool per query (vector_search / keyword_search / summarize_document), a downstream evaluation node validates grounding, then generates an answer via Groq (LLaMA 3.3 70B). If grounding fails at any gate, the system returns a hard refusal — `has_answer=false, confidence=0.0, citations=[]` — never an answer from the LLM's pretrained knowledge.