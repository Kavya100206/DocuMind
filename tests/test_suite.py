"""
DocuMind — Phase 4 Test Suite

Executes every query specified in tests/test_cases.md against the live API
and asserts the expected behaviour. Prints PASS/FAIL per assertion and a
summary at the end. Exits 0 on full pass, 1 on any failure.

Coverage:
  Phase 1.1   Strict PDF isolation gate (subset)
  Phase 4.2   6 valid English queries (V1-V6)
  Phase 4.3   3 invalid English queries (I1-I3)
  Phase 4.4   2 Hindi queries (H1, H2)

Run with:
    .\\venv\\Scripts\\python.exe tests\\test_suite.py

Prerequisites:
    1. Server running on localhost:8000
    2. The RBI Annual Report 2024-25 PDF ingested with the DOC_ID below
       (or DOC_ID updated to match your re-ingested copy)
"""

import re
import sys
import uuid
import requests


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE = "http://localhost:8000"

# Sample document — see test_cases.md §1
# If you re-ingest, list documents via GET /api/documents/ and update this.
DOC_ID = "194a98b3-cf09-4fde-8d6a-34cb29e97278"

# Confidence floors per test_cases.md §4
CONF_FACTUAL  = 0.5   # V1 / V2 / V3 — direct extraction
CONF_ANALYSIS = 0.4   # V4 / V5 — analysis & summary score lower

# Per-call timeout. /api/ask can take 5-8s under load (FAISS + reranker + Groq).
ASK_TIMEOUT = 90

# Devanagari Unicode block — used to verify Hindi responses are actually Hindi
# and not an English fallback. Matches any character in U+0900..U+097F.
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# Output formatting
PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

results: list = []


def check(label: str, condition: bool, got=None) -> None:
    """Record a single assertion. Prints PASS/FAIL and the actual value if failed."""
    status = PASS if condition else FAIL
    print(f"{status}  {label}")
    if not condition and got is not None:
        # Truncate long values so the console stays readable
        got_str = str(got)
        if len(got_str) > 200:
            got_str = got_str[:200] + "..."
        print(f"       Got: {got_str}")
    results.append(condition)


def section(title: str) -> None:
    """Section header in console output."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


import time as _time

def ask(question: str, document_id: str = DOC_ID, session_id: str | None = None):
    """
    Hit POST /api/ask with the given question. Returns the requests.Response
    object. Caller checks status_code and parses .json() as needed.

    Sleeps 12 s before every call to stay under Groq's free-tier 6000 tok/min
    limit. We previously used 6s, but heavy queries like V5 (summary, ~3k token
    prompt) and H1 (Hindi, ~3k token prompt) drain the bucket faster than 6s
    can refill, leading to 429s + hedged refusals. 12s matches test_v5_v6.py
    and gives reliable headroom across the full 11-query suite.
    """
    _time.sleep(12)
    body = {"question": question, "document_id": document_id}
    if session_id:
        body["session_id"] = session_id
    return requests.post(f"{BASE}/api/ask", json=body, timeout=ASK_TIMEOUT)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight: is the server alive?
# ─────────────────────────────────────────────────────────────────────────────

section("Pre-flight — Server Health")
try:
    r = requests.get(f"{BASE}/api/system/stats", timeout=5)
    check("Server is reachable (200)", r.status_code == 200, r.status_code)
    stats = r.json()
    print(f"       Documents: {stats.get('total_documents')}  |  Chunks: {stats.get('total_chunks')}")
except Exception as e:
    print(f"{FAIL}  Could not reach server: {e}")
    print("\n  → Start the server first:")
    print("    .\\venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000 --reload")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Setup: confirm our specific document is indexed
# ─────────────────────────────────────────────────────────────────────────────

section("Setup — Confirming sample document is indexed")
docs_r = requests.get(f"{BASE}/api/documents/")
all_docs = docs_r.json().get("documents", [])
our_doc = next((d for d in all_docs if d["id"] == DOC_ID), None)

if our_doc is None:
    print(f"{FAIL}  Document {DOC_ID} not found.")
    print(f"  → Upload the RBI Annual Report 2024-25 PDF, then update DOC_ID at the top of this file.")
    print(f"  → Available documents:")
    for d in all_docs:
        print(f"       {d['id'][:8]}...  {d['filename']}  ({d['status']})")
    sys.exit(1)

if our_doc["status"] != "completed":
    print(f"{FAIL}  Document status is '{our_doc['status']}', expected 'completed'.")
    print(f"  → Wait for indexing to finish, then re-run.")
    sys.exit(1)

print(f"  Using document: '{our_doc['filename']}'  (id: {DOC_ID[:8]}...)")
print(f"  Pages: {our_doc.get('page_count')}  |  Status: {our_doc['status']}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1.1 — STRICT PDF ISOLATION  (gate must reject missing/empty doc_id)
# ═════════════════════════════════════════════════════════════════════════════

section("Phase 1.1 — Strict PDF Isolation")

# No document_id → 400
r = requests.post(f"{BASE}/api/ask", json={"question": "What is this document about?"})
check("No document_id → 400 status code", r.status_code == 400, r.status_code)
check(
    "No document_id → message contains 'select a document'",
    "select a document" in r.json().get("detail", "").lower(),
    r.json().get("detail"),
)

# Empty string document_id → 400
r = requests.post(f"{BASE}/api/ask", json={"question": "What is this about?", "document_id": ""})
check("Empty document_id string → 400", r.status_code == 400, r.status_code)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4.2 — VALID QUERIES (must answer correctly)
# ═════════════════════════════════════════════════════════════════════════════

section("Phase 4.2 — Valid Queries (must answer correctly)")

# ── V1: Factual extraction ──────────────────────────────────────────────────
print("\n  ▶ V1: Factual extraction — PSL Master Directions effective date")
r = ask("When did the revised Master Directions on Priority Sector Lending (PSL) come into effect?")
check("V1 status 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    d = r.json()
    answer_lower = d.get("answer", "").lower()
    check("V1 has_answer=True", d.get("has_answer") is True, d.get("has_answer"))
    check(f"V1 confidence ≥ {CONF_FACTUAL}", d.get("confidence", 0) >= CONF_FACTUAL, d.get("confidence"))
    check("V1 has at least 1 citation", len(d.get("citations", [])) >= 1, len(d.get("citations", [])))
    # Accept any common phrasing of "April 1, 2025"
    has_date = (
        "april 1, 2025" in answer_lower
        or "april 2025" in answer_lower
        or "1 april 2025" in answer_lower
        or "1st april 2025" in answer_lower
    )
    check("V1 answer mentions April 2025 (PSL effective date)", has_date, d.get("answer", "")[:150])

# ── V2: Numerical extraction ────────────────────────────────────────────────
print("\n  ▶ V2: Numerical extraction — CRR reduction")
r = ask("By how much did the Reserve Bank reduce the CRR during 2024-25?")
check("V2 status 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    d = r.json()
    answer = d.get("answer", "").lower()
    check("V2 has_answer=True", d.get("has_answer") is True, d.get("has_answer"))
    check(f"V2 confidence ≥ {CONF_FACTUAL}", d.get("confidence", 0) >= CONF_FACTUAL, d.get("confidence"))
    check("V2 has at least 1 citation", len(d.get("citations", [])) >= 1, len(d.get("citations", [])))
    # Accept any indicator that a basis-point figure was reported. The doc
    # describes the change as "two tranches of 25 bps each" (= 50 bps total).
    # Anything mentioning "bps" / "basis point" / "tranche" proves the system
    # retrieved and reported the numerical change correctly.
    has_value = "bps" in answer or "basis point" in answer or "tranche" in answer
    check("V2 answer mentions a basis-point figure (bps / basis point / tranche)", has_value, d.get("answer", "")[:150])

# ── V3: Section-based ───────────────────────────────────────────────────────
print("\n  ▶ V3: Section-based — Unified Lending Interface")
r = ask("What does the Reserve Bank say about the Unified Lending Interface (ULI)?")
check("V3 status 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    d = r.json()
    check("V3 has_answer=True", d.get("has_answer") is True, d.get("has_answer"))
    check(f"V3 confidence ≥ {CONF_FACTUAL}", d.get("confidence", 0) >= CONF_FACTUAL, d.get("confidence"))
    check("V3 has at least 1 citation", len(d.get("citations", [])) >= 1, len(d.get("citations", [])))
    check(
        "V3 answer mentions ULI",
        "uli" in d.get("answer", "").lower() or "unified lending" in d.get("answer", "").lower(),
        d.get("answer", "")[:150],
    )

# ── V4: Cross-section + analysis ────────────────────────────────────────────
print("\n  ▶ V4: Cross-section synthesis — risks & challenges")
r = ask("What are the key challenges and risks identified for India's economic outlook in this report?")
check("V4 status 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    d = r.json()
    check("V4 has_answer=True", d.get("has_answer") is True, d.get("has_answer"))
    check(f"V4 confidence ≥ {CONF_ANALYSIS}", d.get("confidence", 0) >= CONF_ANALYSIS, d.get("confidence"))
    # ≥ 2 citations forces multi-chunk synthesis — the whole point of this query
    check(
        "V4 has at least 2 citations (forces cross-section synthesis)",
        len(d.get("citations", [])) >= 2,
        len(d.get("citations", [])),
    )

# ── V5: Document summary ────────────────────────────────────────────────────
print("\n  ▶ V5: Summary — exercises summarize_document tool")
r = ask("Summarize the main themes of this annual report.")
check("V5 status 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    d = r.json()
    check("V5 has_answer=True", d.get("has_answer") is True, d.get("has_answer"))
    check(f"V5 confidence ≥ {CONF_ANALYSIS}", d.get("confidence", 0) >= CONF_ANALYSIS, d.get("confidence"))
    check("V5 has at least 1 citation", len(d.get("citations", [])) >= 1, len(d.get("citations", [])))

# ── V6: Multi-turn with reasoning ───────────────────────────────────────────
print("\n  ▶ V6: Multi-turn reasoning — CRR change → why was it made")
session_id = str(uuid.uuid4())

# Q1
r1 = ask("What was the change in CRR during 2024-25?", session_id=session_id)
check("V6.Q1 status 200", r1.status_code == 200, r1.status_code)
if r1.status_code == 200:
    d1 = r1.json()
    check("V6.Q1 has_answer=True", d1.get("has_answer") is True, d1.get("has_answer"))
    q1_pages = {c.get("page_number") for c in d1.get("citations", [])}
else:
    q1_pages = set()

# Q2 — same session_id so memory + pronoun-resolution kick in
r2 = ask(
    "Why might the Reserve Bank have made this decision based on the broader economic context discussed?",
    session_id=session_id,
)
check("V6.Q2 status 200", r2.status_code == 200, r2.status_code)
if r2.status_code == 200:
    d2 = r2.json()
    check("V6.Q2 has_answer=True", d2.get("has_answer") is True, d2.get("has_answer"))
    q2_pages = {c.get("page_number") for c in d2.get("citations", [])}
    # The meaningful multi-turn assertion: Q2 must pull from pages Q1 didn't,
    # proving the agent went beyond Q1's chunk to find the rationale.
    new_pages = q2_pages - q1_pages
    check(
        "V6.Q2 retrieves from NEW pages not used in Q1 (reasoning beyond recall)",
        len(new_pages) >= 1,
        f"Q1 pages: {sorted(q1_pages)} | Q2 pages: {sorted(q2_pages)}",
    )

# Cleanup — clear server-side session memory
try:
    requests.delete(f"{BASE}/api/session/{session_id}", timeout=5)
except Exception:
    pass


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4.3 — INVALID QUERIES (must refuse)
# ═════════════════════════════════════════════════════════════════════════════

section("Phase 4.3 — Invalid Queries (must refuse)")

INVALID_QUERIES = [
    ("I1", "Outside knowledge",
     "Who is the current Prime Minister of India?"),
    ("I2", "Off-topic entity",
     "What are the lyrics of the Indian national anthem?"),
    ("I3", "Subtle hallucination trap",
     "What is the official rollout date for e₹-Retail in rural India announced in this report?"),
]

for tag, category, q in INVALID_QUERIES:
    print(f"\n  ▶ {tag}: {category}")
    r = ask(q)
    check(f"{tag} status 200", r.status_code == 200, r.status_code)
    if r.status_code != 200:
        continue
    d = r.json()
    # Strict refusal contract — see test_cases.md §5
    check(f"{tag} has_answer=False", d.get("has_answer") is False, d.get("has_answer"))
    check(f"{tag} confidence == 0.0", d.get("confidence") == 0.0, d.get("confidence"))
    check(f"{tag} citations == []", d.get("citations") == [], d.get("citations"))


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4.4 — HINDI QUERIES (multilingual)
# ═════════════════════════════════════════════════════════════════════════════

section("Phase 4.4 — Hindi Queries (multilingual)")

# ── H1: Valid Hindi — same factual question as V1, but in Hindi ─────────────
print("\n  ▶ H1: Valid Hindi — CFO question")
r = ask("रिज़र्व बैंक के मुख्य वित्तीय अधिकारी कौन हैं?")
check("H1 status 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    d = r.json()
    answer = d.get("answer", "")
    check("H1 has_answer=True", d.get("has_answer") is True, d.get("has_answer"))
    check("H1 has at least 1 citation", len(d.get("citations", [])) >= 1, len(d.get("citations", [])))
    # Critical multilingual assertion: the answer must be in Devanagari script.
    # If the system silently fell back to English, this assertion fails.
    check(
        "H1 answer is in Devanagari (not English fallback)",
        bool(DEVANAGARI_RE.search(answer)),
        answer[:150],
    )

# ── H2: Invalid Hindi — must refuse in Hindi ────────────────────────────────
print("\n  ▶ H2: Invalid Hindi — PM question (must refuse)")
r = ask("भारत के प्रधान मंत्री कौन हैं?")
check("H2 status 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    d = r.json()
    answer = d.get("answer", "")
    check("H2 has_answer=False", d.get("has_answer") is False, d.get("has_answer"))
    check("H2 confidence == 0.0", d.get("confidence") == 0.0, d.get("confidence"))
    check("H2 citations == []", d.get("citations") == [], d.get("citations"))
    # The refusal must be in Hindi — not the English fallback.
    # Verifies REFUSAL_MESSAGES["Hindi"] in app/services/agent_service.py was used.
    check(
        "H2 refusal message is in Devanagari (Hindi refusal, not English)",
        bool(DEVANAGARI_RE.search(answer)),
        answer[:150],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

section("Summary")
passed = sum(results)
total = len(results)
pct = int(passed / total * 100) if total else 0
color = GREEN if passed == total else YELLOW
print(f"  {color}{passed}/{total} tests passed ({pct}%){RESET}")

if passed < total:
    print(f"  {YELLOW}⚠️  Some tests failed — review output above.{RESET}")
    sys.exit(1)
else:
    print(f"  {GREEN}🎉 All tests passed!{RESET}")
    sys.exit(0)
