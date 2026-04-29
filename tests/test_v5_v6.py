"""
DocuMind — Targeted V5 / V6 / I3 test runner

Hits ONLY the queries we're iterating on, so we don't burn the Groq free-tier
6000-tok/min budget re-running the full suite. Four /api/ask calls total:
  V5    — Summarize the main themes of this annual report.
  V6.Q1 — What was the change in CRR during 2024-25?  (sets up session)
  V6.Q2 — Why might the Reserve Bank have made this decision based on the
          broader economic context discussed?         (the actual target)
  I3    — What is the official rollout date for e₹-Retail in rural India
          announced in this report?                   (hallucination trap)

Plus a no-server unit check on _qualifier_distance_check so the heuristic
logic itself can be validated even when the server is down.

Run with:
    .\\venv\\Scripts\\python.exe tests\\test_v5_v6.py

Prerequisites:
  1. Server restarted AFTER the V5/V6/I3 fixes were applied
  2. RBI Annual Report indexed under DOC_ID below
"""

import os
import re
import sys
import uuid
import time
import requests

# Allow `from app.services...` when running this script directly from /tests.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


BASE = "http://localhost:8000"
DOC_ID = "194a98b3-cf09-4fde-8d6a-34cb29e97278"

CONF_ANALYSIS = 0.4
ASK_TIMEOUT = 90

# Bigger gap than the main suite. The main suite uses 6 s, but if the previous
# call generated a long summary answer (~1k tokens out + 2.5k tokens in) that
# can drain the per-minute budget. 12 s leaves plenty of headroom for three
# back-to-back calls without a 429.
INTER_CALL_SLEEP = 12

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

results: list = []


def check(label: str, condition: bool, got=None) -> None:
    status = PASS if condition else FAIL
    print(f"{status}  {label}")
    if not condition and got is not None:
        got_str = str(got)
        if len(got_str) > 220:
            got_str = got_str[:220] + "..."
        print(f"       Got: {got_str}")
    results.append(condition)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def ask(question: str, session_id: str | None = None):
    time.sleep(INTER_CALL_SLEEP)
    body = {"question": question, "document_id": DOC_ID}
    if session_id:
        body["session_id"] = session_id
    return requests.post(f"{BASE}/api/ask", json=body, timeout=ASK_TIMEOUT)


# ── I3 unit check (no server needed) ────────────────────────────────────────
# Validates the qualifier-distance heuristic in isolation. Runs even if the
# server is down, giving a fast first-line signal that the I3 trap logic is
# wired up correctly before we spend tokens on the live /api/ask path.
section("I3 unit — qualifier-distance heuristic (no server)")
try:
    from app.services.agent_service import _qualifier_distance_check

    i3_q = "What is the official rollout date for e₹-Retail in rural India announced in this report?"
    i3_a = "The official rollout date for e₹-Retail in rural India is December 1, 2022, as explicitly stated in the report on page 302 of Source 1."
    i3_chunks = [{
        "text": (
            "November 29, 2022 Operationalisation of CBDC - Pilot for CBDC - "
            "retail (e-R) was launched on December 1, 2022. February 14, 2023 "
            "Second global hackathon - HaRBInger 2023 - was launched. Earlier "
            "the doc discusses Digitalisation of rural finance in India - "
            "pilot for kisan credit card."
        )
    }]
    keep = _qualifier_distance_check(i3_q, i3_a, i3_chunks)
    check("I3 unit — heuristic refuses fabricated 'rural rollout' date", keep is False, keep)

    # V1-shaped sanity: legitimate grounded answer must NOT trip the heuristic.
    v1_q = "When did the revised Master Directions on Priority Sector Lending (PSL) come into effect?"
    v1_a = "The revised Master Directions on PSL came into effect from April 1, 2025, as explicitly stated in the report."
    v1_chunks = [{
        "text": (
            "The Reserve Bank issued the revised Master Directions on PSL, "
            "which has come into effect from April 1, 2025. Priority Sector "
            "Lending norms cover advances to agriculture, MSMEs, and weaker "
            "sections."
        )
    }]
    keep_v1 = _qualifier_distance_check(v1_q, v1_a, v1_chunks)
    check("I3 unit — heuristic does NOT false-refuse V1 (grounded answer)", keep_v1 is True, keep_v1)
except Exception as e:
    check("I3 unit — import/exec _qualifier_distance_check", False, e)


# ── Pre-flight ──────────────────────────────────────────────────────────────
section("Pre-flight")
try:
    r = requests.get(f"{BASE}/api/system/stats", timeout=5)
    if r.status_code != 200:
        print(f"{FAIL}  Server not reachable (status {r.status_code})")
        sys.exit(1)
    print(f"  Server up. Documents: {r.json().get('total_documents')}")
except Exception as e:
    print(f"{FAIL}  Could not reach server: {e}")
    print("  → Start it: .\\venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000 --reload")
    sys.exit(1)


# ── V5: summary intent must route to summarize_document ─────────────────────
section("V5 — Summary (deterministic pre-route to summarize_document)")
print("\n  ▶ Summarize the main themes of this annual report.")
r = ask("Summarize the main themes of this annual report.")
check("V5 status 200", r.status_code == 200, r.status_code)
if r.status_code == 200:
    d = r.json()
    answer = d.get("answer", "")
    check("V5 has_answer=True", d.get("has_answer") is True, d.get("has_answer"))
    check(f"V5 confidence ≥ {CONF_ANALYSIS}", d.get("confidence", 0) >= CONF_ANALYSIS, d.get("confidence"))
    check("V5 has at least 1 citation", len(d.get("citations", [])) >= 1, len(d.get("citations", [])))
    # The fix: router must pick summarize_document. The trace echoes the tool used.
    routing = d.get("routing") or {}
    check(
        "V5 routed to summarize_document (deterministic pre-route)",
        routing.get("tool_chosen") == "summarize_document",
        routing,
    )
    print(f"       Answer opening: {answer[:140]!r}")


# ── V6: multi-turn — Q1 sets up, Q2 is the target ──────────────────────────
section("V6 — Multi-turn reasoning (Q2 must NOT refuse)")
session_id = str(uuid.uuid4())

print("\n  ▶ V6.Q1: What was the change in CRR during 2024-25?")
r1 = ask("What was the change in CRR during 2024-25?", session_id=session_id)
check("V6.Q1 status 200", r1.status_code == 200, r1.status_code)
q1_pages: set = set()
if r1.status_code == 200:
    d1 = r1.json()
    check("V6.Q1 has_answer=True", d1.get("has_answer") is True, d1.get("has_answer"))
    q1_pages = {c.get("page_number") for c in d1.get("citations", [])}

print("\n  ▶ V6.Q2: Why might the Reserve Bank have made this decision based on the broader economic context discussed?")
r2 = ask(
    "Why might the Reserve Bank have made this decision based on the broader economic context discussed?",
    session_id=session_id,
)
check("V6.Q2 status 200", r2.status_code == 200, r2.status_code)
if r2.status_code == 200:
    d2 = r2.json()
    answer = d2.get("answer", "")
    check("V6.Q2 has_answer=True (fix: synthesis directive prevents hedged refusal)",
          d2.get("has_answer") is True, d2.get("has_answer"))
    q2_pages = {c.get("page_number") for c in d2.get("citations", [])}
    new_pages = q2_pages - q1_pages
    check(
        "V6.Q2 retrieves from NEW pages not used in Q1 (reasoning beyond recall)",
        len(new_pages) >= 1,
        f"Q1 pages: {sorted(p for p in q1_pages if p is not None)} | Q2 pages: {sorted(p for p in q2_pages if p is not None)}",
    )
    print(f"       Answer opening: {answer[:200]!r}")

# Cleanup
try:
    requests.delete(f"{BASE}/api/session/{session_id}", timeout=5)
except Exception:
    pass


# ── I3: hallucination trap — must refuse, not fabricate a grounded date ────
section("I3 — Hallucination trap (must refuse, not fabricate)")
print("\n  ▶ I3: What is the official rollout date for e₹-Retail in rural India announced in this report?")
r3 = ask("What is the official rollout date for e₹-Retail in rural India announced in this report?")
check("I3 status 200", r3.status_code == 200, r3.status_code)
if r3.status_code == 200:
    d3 = r3.json()
    answer = d3.get("answer", "")
    answer_lower = answer.lower()
    has_answer = d3.get("has_answer", True)

    # The system must NOT confidently assert "December 1, 2022" as the
    # rural-rollout date. Two acceptable shapes:
    #   (a) hard refusal:  has_answer == False
    #   (b) hedged answer that does not assert the trap date as grounded
    asserts_trap_date = "december 1, 2022" in answer_lower and (
        "explicitly stated" in answer_lower
        or "as stated" in answer_lower
        or "according to" in answer_lower
    )
    check(
        "I3 must not fabricate 'December 1, 2022' as a grounded rural-rollout date",
        not asserts_trap_date,
        answer[:220],
    )
    # Stronger signal: heuristic should flip has_answer to False.
    check(
        "I3 returns has_answer=False (hard refusal preferred)",
        has_answer is False,
        has_answer,
    )


# ── Summary ─────────────────────────────────────────────────────────────────
section("Summary")
passed = sum(results)
total = len(results)
pct = int(passed / total * 100) if total else 0
color = GREEN if passed == total else YELLOW
print(f"  {color}{passed}/{total} assertions passed ({pct}%){RESET}")
sys.exit(0 if passed == total else 1)
