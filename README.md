# DocuMind: Intelligent Document Question-Answering System (RAG)

DocuMind is an advanced multi-document question-answering system built using Retrieval-Augmented Generation (RAG). It transforms static PDF document repositories into an interactive knowledge retrieval system, combining semantic search, lexical ranking, and cross-encoder reranking to generate highly grounded responses with precise citations and confidence scores.

## Overview

The pipeline is explicitly designed to minimize Large Language Model (LLM) hallucinations and maximize user trust through a rigorous, multi-stage retrieval and generation process. By integrating conversational memory, context-aware query rewriting, and ambiguity guards, DocuMind provides accurate, evidence-backed answers to complex queries across multiple documents.

## Key Features

*   **Multi-Document Question Answering:** Query semantically across an entire repository of uploaded PDFs simultaneously.
*   **Section-Aware Chunking:** Intelligently detects document structure (headers, bullet lists, semantic blocks) for context-rich text chunking, rather than relying solely on arbitrary character counts.
*   **Hybrid Retrieval Pipeline:** Combines FAISS semantic vector search with BM25 lexical ranking to ensure high recall for both conceptual queries and exact keyword matches.
*   **Lexical Boosting:** Applies specialized re-ranking multipliers for exact token matches, bigram matches, and section header targets.
*   **Two-Stage Reranking:** Deep relevance assessment using a secondary cross-encoder transformer model applied only to the top candidates to balance accuracy and latency.
*   **Conversational Memory & Query Rewriting:** Maintains conversation history and utilizes an LLM to resolve pronouns and implicit references into standalone, search-optimized queries.
*   **Strict Grounded Generation:** System prompts and constraint logic prevent the LLM from relying on its internal training weights, forcing it to generate answers based solely on the retrieved context.
*   **Automated Citations:** Provides explicit references to source documents and specific page numbers for every generated answer.
*   **Confidence Evaluation:** Calculates a quantitative reliability score based on retrieval evidence strength and applies a penalty if the generation model utilizes hedging language.

## Architecture & Pipeline

DocuMind follows a highly optimized, six-stage RAG architecture:

1.  **Query Rewriting:** The user's input question is processed alongside the session history. An LLM resolves ambiguities (e.g., "What were their responsibilities?") and generates a comprehensive search query.
2.  **Dual Retrieval:** Both the user's original query and the rewritten query are executed against the document index to capture both specific intent and broad semantic meaning.
3.  **Hybrid Search (FAISS + BM25):** 
    *   Semantic vectors are compared using an `IndexFlatIP` FAISS index and BGE embeddings.
    *   Lexical keyword frequency is calculated utilizing BM25Okapi.
    *   The scores are normalized and hybridized (65% Semantic / 35% Lexical).
4.  **Thresholding & Lexical Boosting:** Results below a defined similarity threshold are discarded. Remaining candidates receive scoring boosts based on exact token matches and section metadata.
5.  **Cross-Encoder Reranking:** The top 10 merged candidates are evaluated by an `ms-marco-MiniLM` cross-encoder for deep contextual relevance scoring.
6.  **Context Packing & Generation:** The highest-scoring, deduplicated chunks are packed into the prompt context limit. The Groq Llama 3 API generates the final response, which is then parsed for citations and assigned a confidence score.

## Technology Stack

### Backend Infrastructure
*   **Framework:** FastAPI (Python 3.11)
*   **Database:** PostgreSQL (NeonDB) with SQLAlchemy ORM
*   **Data Persistence:** Ephemeral file storage with automatic FAISS index rebuilding from database chunks on startup.

### AI & Retrieval Components
*   **Vector Database:** FAISS (Facebook AI Similarity Search)
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Local, CPU-optimized)
*   **Lexical Search:** `rank_bm25` (BM25Okapi)
*   **Cross-Encoder:** `cross-encoder/ms-marco-MiniLM-L-2-v2` (Local, lazy-loaded)
*   **LLM Provider:** Groq API (`llama-3.3-70b-versatile`)
*   **Document Parsing:** `pdfplumber` with `PyPDF2` fallback

### Frontend Interface
*   **Client:** Vanilla JavaScript and CSS
*   **Features:** Asynchronous document upload polling, dynamic Chat UI, dynamic confidence visualizers, and expandable citation accordions.

## Project Structure

```text
DocuMind/
├── app/
│   ├── config/        # Environment configurations and Pydantic settings
│   ├── controllers/   # FastAPI route handlers (Document, QA, Search, System)
│   ├── database/      # PostgreSQL connection management
│   ├── models/        # SQLAlchemy ORM schemas (Document, Chunk)
│   ├── services/      # Core business logic (Chunking, Embeddings, FAISS, LLM, Retrieval)
│   ├── utils/         # Helper functions (File validation, Logging)
│   ├── views/         # Pydantic response schemas
│   └── main.py        # Application entry point and startup events
├── data/              # FAISS index and metadata binary storage
├── frontend/          # Single-page web application (index.html, static assets)
├── scripts/           # Diagnostic utilities and index rebuilding scripts
├── uploads/           # Temporary PDF storage during processing
├── .env.example       # Template for required environment variables
├── requirements.txt   # Python package dependencies
└── start.sh           # Deployment startup script
```

## Setup and Installation

### Prerequisites
*   Python 3.11+
*   PostgreSQL database instance (e.g., NeonDB)
*   Groq API Key (Available at console.groq.com)

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/documind.git
    cd documind
    ```

2.  **Initialize the Virtual Environment**
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Copy the `.env.example` file to `.env` and populate the required fields.
    ```bash
    cp .env.example .env
    ```
    Required keys:
    *   `DATABASE_URL` (PostgreSQL connection string)
    *   `GROQ_API_KEY`

5.  **Run the Application**
    Start the FastAPI server utilizing uvicorn.
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

6.  **Access the Interface**
    Navigate to `http://localhost:8000/ui` in your web browser.

## Diagnostic Tools

The `scripts/` directory contains utilities for system maintenance:
*   `python scripts/rebuild_faiss.py`: Reconstructs the FAISS vector index from the PostgreSQL chunk database.
*   `python scripts/diagnose_full.py`: Verifies database connectivity and vector index integrity.
*   `python scripts/test_retrieval.py`: CLI execution of the retrieval pipeline for testing specific queries.
