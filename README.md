# DocuMind: Intelligent Document Question-Answering System (RAG)

DocuMind is a multi-document question answering system built using Retrieval-Augmented Generation (RAG).  
It combines semantic search, lexical ranking, and cross-encoder reranking to generate grounded responses with citations and confidence scores.

---

## Overview

DocuMind transforms static document repositories into an interactive knowledge retrieval system. The pipeline is designed to minimize hallucinations and maximize trust through a multi-stage retrieval and generation process.

Typical response latency: 10-20 seconds.

---

## Features

- Multi-Document Question Answering: Query across an entire repository of uploaded PDFs.
- Hybrid Retrieval: Combines FAISS semantic search with BM25 lexical ranking for superior recall.
- Lexical Boosting: Specialized re-ranking for exact term matches and Roman numerals.
- Cross-Encoder Reranking: Deep relevance assessment using a secondary transformer model.
- Conversational Memory: Context-aware query rewriting to resolve pronouns and follow-up questions.
- Grounded Generation: Strict system prompts prevent the LLM from using external training data.
- Automated Citations: Direct links to specific pages and source documents for every answer.
- Confidence Scoring: Quantitative evaluation of answer reliability based on evidence strength.
- Ambiguity Guard: Internal logic to prevent answering queries with unresolved pronouns.

---

## Architecture

The DocuMind pipeline follows a sophisticated multi-stage retrieval architecture:

1. Query Rewriting
   - Input question is processed with conversation history.
   - LLM resolves pronouns and creates a standalone search-optimized query.

2. Hybrid Retrieval
   - Semantic Search: FAISS identifies top candidates using BGE embeddings.
   - Lexical Boost: Candidates receive scores for exact token and bigram matches.
   - BM25 Score: Keyword frequency is calculated across the shortlisted corpus.
   - Hybridization: Scores are merged (65% Semantic / 35% Lexical).

3. Reranking
   - Top candidates are processed by a Cross-Encoder model.
   - Deep text interaction analysis ensures the most relevant chunks are prioritized.

4. Grounded Generation
   - Deduplicated and packed context is sent to Groq (Llama 3).
   - Answer is generated using only the provided snippets.

5. Scoring and Validation
   - Confidence is calculated from retrieval scores.
   - Penalty is applied if the model indicates hedging or uncertainty.

---

## Tech Stack

### Backend
- FastAPI: High-performance Python web framework.
- PostgreSQL (NeonDB): Relational storage for document metadata and chunking structure.

### Search and AI
- FAISS: Efficient vector similarity search.
- BAAI/bge-small-en-v1.5: Embedding model for semantic representation.
- bm25-service: Custom lexical ranking implementation.
- ms-marco-MiniLM-L-2-v2: Cross-encoder model for deep reranking.
- Groq Llama-3: High-speed LLM for grounded answer generation.

### Frontend
- Vanilla JavaScript and HTML: Premium-styled dashboard for document management and chat.

---

## Project Structure

```text
DocuMind/
├── app/
│   ├── services/      # Retrieval, reranking, chunking, LLM pipeline
│   ├── controllers/   # API routes
│   ├── models/        # Database schema
│   ├── config/        # Settings and environment handling
│   └── main.py        # Application entry point
├── frontend/          # Web interface dashboard
├── scripts/           # Debug and operational utilities
├── data/              # FAISS index persistence
├── uploads/           # Document storage
└── render.yaml        # Deployment configuration

```
---

## Installation

### Prerequisites
- Python 3.10+
- PostgreSQL database (NeonDB recommended)
- Groq API Key

### Setup

1. Clone the Repository:
   ```bash
   git clone https://github.com/yourusername/documind
   cd documind
   ```

2. Create Virtual Environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   ```

3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure Environment:
   Create a .env file based on .env.example:
   - DATABASE_URL=your_database_url
   - GROQ_API_KEY=your_groq_key

---

## Running the Application

To start the server locally:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
The interface will be available at: http://localhost:8000

---

## API Documentation

- POST /api/ask: Submit a natural language question.
- POST /api/upload: Ingest new PDF documents.
- GET /health: Monitor system health and component status.

---

## Engineering Highlights

- Recall Optimization: Solved "semantic collapse" where similar document versions (e.g., DSA I vs DSA II) confuse standard embeddings through lexical boosting.
- Hallucination Control: Implementation of an Ambiguity Guard and strict system prompts ensures the model stays within document bounds.
- Latency Management: Orchestrated a multi-stage pipeline where slow cross-encoders only run on a small subset of candidates.
