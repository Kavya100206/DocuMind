# DocuMind — Test Instructions

This document gives the evaluator everything needed to install, run, and verify DocuMind on a fresh machine.

---

## 1. Prerequisites

- **Python 3.11+**
- A **PostgreSQL** database (NeonDB free tier works; any Postgres connection string is fine)
- A **Groq API key** — sign up at https://console.groq.com (free tier is sufficient for the test suite)

---

## 2. Install

```bash
git clone https://github.com/Kavya100206/DocuMind.git
cd DocuMind

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```ini
DATABASE_URL=postgresql://user:password@host/dbname
GROQ_API_KEY=your-groq-api-key
```

The Google Drive variables (`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`) are optional — only needed if you want to test the Drive ingestion path.

---

## 4. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

The server is up when you see `Application startup complete` in the logs. Frontend at **http://localhost:8000**.

---

## 5. Upload the sample PDF

The test suite is anchored to a specific document — the **RBI Annual Report 2024-25**.

1. Open http://localhost:8000 in a browser
2. Click "Upload" and select the PDF (provided alongside the submission, or download from https://www.rbi.org.in/Scripts/AnnualReportPublications.aspx)
3. Wait until the document status shows `completed` (~30–60s for 318 pages)

---

## 6. Run the automated test suite

The suite executes 11 queries, asserts the expected behaviour, and prints PASS / FAIL per assertion. It exits 0 on full pass, 1 on any failure.

```bash
# Find the document_id of your uploaded PDF
curl http://localhost:8000/api/documents/

# If your document_id differs from the default, update DOC_ID at the top of tests/test_suite.py

# Run the full suite
.\venv\Scripts\python.exe tests\test_suite.py
```

Expected output:
```
✓ N/N tests passed (100%)
🎉 All tests passed!
```

For a faster, focused run (V5 / V6 / I3 only, ~1 min):

```bash
.\venv\Scripts\python.exe tests\test_v5_v6.py
```

---

## 7. Manual verification — sample queries

If you prefer to verify capabilities interactively in the UI, the canonical queries are below.

### Grounded answers (must answer correctly with citations)

| Query | Expected behaviour |
|---|---|
| *When did the revised Master Directions on Priority Sector Lending (PSL) come into effect?* | Answer mentions **April 1, 2025**, ≥ 1 citation, confidence ≥ 0.5 |
| *By how much did the Reserve Bank reduce the CRR during 2024-25?* | Answer mentions **50 bps** (two tranches of 25 bps each), ≥ 1 citation |
| *Summarize the main themes of this annual report.* | Multi-section thematic summary, ≥ 1 citation. Routes to `summarize_document` tool — visible in observability panel. |

### Multi-turn (same session — V6)

1. *What was the change in CRR during 2024-25?* — sets context
2. *Why might the Reserve Bank have made this decision based on the broader economic context discussed?* — must resolve "this decision" → CRR cut, retrieve from new pages, synthesize rationale

### Refusals (must show refusal banner — no fabricated answer)

| Query | Why it must refuse |
|---|---|
| *Who is the current Prime Minister of India?* | Outside knowledge — not in an RBI report |
| *What are the lyrics of the Indian national anthem?* | Off-topic — completely unrelated to monetary policy |
| *What is the official rollout date for e₹-Retail in rural India announced in this report?* | **Hallucination trap.** The doc mentions e₹-Retail and rural finance separately but no specific rural-rollout date exists. The qualifier-distance grounding gate must fire. |

For all refusals: `has_answer=false`, `confidence=0`, `citations=[]`.

### Multilingual (Hindi)

| Query | Expected |
|---|---|
| रिज़र्व बैंक के मुख्य वित्तीय अधिकारी कौन हैं? | Answer in **Devanagari script** (regex `[ऀ-ॿ]`). Tests language detection + prompt-directive compliance. |
| भारत के प्रधान मंत्री कौन हैं? | Refusal message in **Hindi**, not English fallback. |

---

## 8. UI features to spot-check

- **Confidence bar** — proportional to confidence score; **0%** on refusals
- **Citation accordion** — expandable, shows page numbers + filenames
- **Refusal banner** — distinct red "Out of Scope" treatment for refusals
- **Retrieval trace toggle** — when ON, the response includes per-chunk score breakdown (FAISS / BM25 / hybrid / reranker)
- **Observability panel** — shows which tool the agent picked, refusal-gate decisions, confidence components
- **Document summary button** — one-click whole-document summarization
- **Hindi rendering** — Devanagari renders correctly in the answer panel

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `psycopg2.OperationalError: could not translate host name` | Neon DB suspended or expired | Wake up at https://console.neon.tech, or update `DATABASE_URL` |
| `429 Too Many Requests` on `/api/ask` | Groq free-tier 6000 TPM exhausted | Wait ~60s; or upgrade to Groq Dev tier ($5/mo) |
| `documents=0` on `/api/documents/` | Sample PDF not uploaded | Upload via UI; wait for status `completed` |
| Test suite fails on H1 only | Groq rate limit during the heavy Hindi prompt | Re-run; or use `tests\test_v5_v6.py` which has a longer inter-call sleep |
| First query slow (~30s) | Cold start — embedding model + FAISS index downloading | One-time only; subsequent queries are fast |
| `Pydantic int_parsing` error in logs | Old build before chunk_trace coercion fix | Pull latest, redeploy |

---

## 10. Test specification

The full specification of every test case (queries, evidence, expected output, pass criteria) lives in **[`tests/test_cases.md`](../tests/test_cases.md)**. That document is the authoritative reference for what the suite proves.
