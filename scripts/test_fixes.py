"""Test the 2 remaining failing cases after fixes."""
import sys, os, json
sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
import urllib.request, urllib.error, psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("SELECT id FROM documents WHERE filename ILIKE '%orbit%' ORDER BY created_at DESC LIMIT 1")
row = cur.fetchone()
cur.close(); conn.close()

orbit_id = row[0] if row else None
print(f"OrbitMind doc ID: {orbit_id}\n")

tests = [
    # (question, use_doc_filter, expect_helpful_message)
    ("title?",                                  False, True),   # short query guard
    ("What machine learning technique is used", True,  False),  # single-doc improved retrieval
    ("What is the project title",               True,  False),  # full question, single doc
    ("who are the team members names",          True,  False),  # bigram boost test
]

for q, use_filter, expect_guardrail in tests:
    payload = {"question": q, "k": 10}
    if use_filter and orbit_id:
        payload["document_id"] = orbit_id

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://localhost:8000/api/ask", data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        r = urllib.request.urlopen(req, timeout=45)
        resp = json.loads(r.read())
        ans = resp.get("answer", "")[:200].replace("\n", " ")
        has_ans = resp.get("has_answer")
        conf = resp.get("confidence")
        filter_note = "(doc filter)" if use_filter else "(global)"
        print(f"Q [{filter_note}]: {q}")
        print(f"   has_answer={has_ans} | conf={conf}")
        print(f"   answer: {ans}\n")
    except urllib.error.HTTPError as e:
        print(f"Q: {q}")
        print(f"   HTTP {e.code}: {e.read().decode()[:200]}\n")
    except Exception as e:
        print(f"Q: {q}  ERROR: {e}\n")
