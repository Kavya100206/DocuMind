"""
Phase 1 Test Suite — DocuMind Submission

Tests every Phase 1 requirement:
  1.1  Strict PDF Isolation  (document_id gate)
  1.2  Hard Refusal Logic    (off-topic query)
  1.3  Frontend contract     (has_answer=False format)

Run with:
    .\\venv\\Scripts\\python.exe tests\\phase1_test.py
"""

import requests
import json
import sys

BASE = "http://localhost:8000"
PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

results = []

def check(label, condition, got):
    status = PASS if condition else FAIL
    print(f"{status}  {label}")
    if not condition:
        print(f"       Got: {got}")
    results.append(condition)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Pre-flight: server alive? ──────────────────────────────────
section("Pre-flight — Server Health")
try:
    r = requests.get(f"{BASE}/api/system/stats", timeout=5)
    check("Server is reachable (200)", r.status_code == 200, r.status_code)
    stats = r.json()
    print(f"       Documents: {stats.get('total_documents')}  |  Chunks: {stats.get('total_chunks')}")
except Exception as e:
    print(f"\033[91m  FAIL\033[0m  Could not reach server: {e}")
    print("\n  → Start the server first:")
    print("    .\\venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000 --reload")
    sys.exit(1)


# ── Get a real document ID ─────────────────────────────────────
section("Setup — Finding an indexed document")
docs_r = requests.get(f"{BASE}/api/documents/")
all_docs = docs_r.json().get("documents", [])
completed = [d for d in all_docs if d["status"] == "completed"]

if not completed:
    print("  ⚠️  No completed documents found.")
    print("  → Upload at least ONE PDF via the UI, wait for status=completed, then re-run.")
    sys.exit(1)

DOC = completed[0]
DOC_ID = DOC["id"]
DOC_NAME = DOC["filename"]
print(f"  Using document: '{DOC_NAME}'  (id: {DOC_ID[:8]}...)")

if len(completed) >= 2:
    OTHER_DOC = completed[1]
    OTHER_DOC_ID = OTHER_DOC["id"]
    OTHER_DOC_NAME = OTHER_DOC["filename"]
    print(f"  Second document: '{OTHER_DOC_NAME}'  (id: {OTHER_DOC_ID[:8]}...)")
else:
    OTHER_DOC_ID = None
    print("  (Only one document — cross-doc isolation test will be skipped)")


# ══════════════════════════════════════════════════════════════
#  PHASE 1.1 — STRICT PDF ISOLATION
# ══════════════════════════════════════════════════════════════
section("Phase 1.1 — Strict PDF Isolation")

# Test A: no document_id at all → 400
r = requests.post(f"{BASE}/api/ask", json={"question": "What is this document about?"})
check(
    "No document_id → 400 status code",
    r.status_code == 400,
    r.status_code
)
check(
    "No document_id → message contains 'select a document'",
    "select a document" in r.json().get("detail", "").lower(),
    r.json().get("detail")
)

# Test B: empty string document_id → 400
r = requests.post(f"{BASE}/api/ask", json={"question": "What is this about?", "document_id": ""})
check(
    "Empty document_id string → 400",
    r.status_code == 400,
    r.status_code
)

# Test C: valid doc_id → NOT 400 (should get a real response)
r = requests.post(
    f"{BASE}/api/ask",
    json={"question": "What is this document about?", "document_id": DOC_ID},
    timeout=60
)
check(
    "Valid document_id → NOT 400 (answer attempted)",
    r.status_code == 200,
    r.status_code
)
if r.status_code == 200:
    data = r.json()
    check(
        "Valid document_id → response has 'has_answer' field",
        "has_answer" in data,
        list(data.keys())
    )
    check(
        "Valid document_id → response has 'citations' field",
        "citations" in data,
        list(data.keys())
    )
    check(
        "Valid document_id → response has 'confidence' field",
        "confidence" in data,
        list(data.keys())
    )

# Test D: cross-document isolation (if 2 docs available)
if OTHER_DOC_ID:
    r = requests.post(
        f"{BASE}/api/ask",
        json={
            "question": "What is this document about?",
            "document_id": DOC_ID
        },
        timeout=60
    )
    if r.status_code == 200:
        data = r.json()
        citations = data.get("citations", [])
        leak = [c for c in citations if c.get("document_id") == OTHER_DOC_ID]
        check(
            "Cross-doc isolation: citations only from selected doc",
            len(leak) == 0,
            f"{len(leak)} citation(s) leaked from '{OTHER_DOC_NAME}'"
        )
    else:
        print(f"  SKIP  Cross-doc test — query returned {r.status_code}")
else:
    print("  SKIP  Cross-doc isolation (need 2 documents)")


# ══════════════════════════════════════════════════════════════
#  PHASE 1.2 — HARD REFUSAL LOGIC
# ══════════════════════════════════════════════════════════════
section("Phase 1.2 — Hard Refusal Logic")

REFUSAL_QUERIES = [
    "What is the capital of France?",
    "Who is the Prime Minister of India?",
    "What is 2 + 2?",
]

for q in REFUSAL_QUERIES:
    r = requests.post(
        f"{BASE}/api/ask",
        json={"question": q, "document_id": DOC_ID},
        timeout=60
    )
    check(
        f"Refusal for off-topic: '{q[:40]}' → 200 status",
        r.status_code == 200,
        r.status_code
    )
    if r.status_code == 200:
        data = r.json()
        check(
            f"  └─ has_answer=False",
            data.get("has_answer") == False,
            data.get("has_answer")
        )
        check(
            f"  └─ confidence=0.0",
            data.get("confidence") == 0.0,
            data.get("confidence")
        )
        check(
            f"  └─ citations=[]",
            data.get("citations") == [],
            data.get("citations")
        )


# ══════════════════════════════════════════════════════════════
#  PHASE 1.3 — FRONTEND CONTRACT (API response shape)
# ══════════════════════════════════════════════════════════════
section("Phase 1.3 — API Response Contract (Frontend relies on these fields)")

# On refusal, the response MUST have the exact shape the frontend checks
r = requests.post(
    f"{BASE}/api/ask",
    json={"question": "What is the capital of France?", "document_id": DOC_ID},
    timeout=60
)
if r.status_code == 200:
    data = r.json()
    check("Refusal response: 'answer' is a non-empty string",
          isinstance(data.get("answer"), str) and len(data.get("answer", "")) > 0,
          data.get("answer"))
    check("Refusal response: 'has_answer' is boolean False",
          data.get("has_answer") is False,
          data.get("has_answer"))
    check("Refusal response: 'confidence' is float 0.0",
          data.get("confidence") == 0.0,
          data.get("confidence"))
    check("Refusal response: 'citations' is empty list",
          data.get("citations") == [],
          data.get("citations"))
    check("Refusal response: 'question' echoed back",
          "question" in data,
          list(data.keys()))

# On a real answer, check all fields exist
r2 = requests.post(
    f"{BASE}/api/ask",
    json={"question": "Give me a brief overview of this document.", "document_id": DOC_ID},
    timeout=60
)
if r2.status_code == 200:
    d2 = r2.json()
    check("Valid answer: has all required fields",
          all(k in d2 for k in ["answer", "has_answer", "confidence", "citations", "question"]),
          list(d2.keys()))


# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
section("Summary")
passed = sum(results)
total = len(results)
pct = int(passed / total * 100) if total else 0
color = "\033[92m" if passed == total else "\033[93m"
print(f"  {color}{passed}/{total} tests passed ({pct}%)\033[0m")
if passed < total:
    print("  ⚠️  Some tests failed — review output above.")
else:
    print("  🎉 All Phase 1 tests passed!")
print()
