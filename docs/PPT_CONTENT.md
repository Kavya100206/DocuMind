# DocuMind — PPT Content (slide-by-slide)

Ready-to-paste content for the submission deck. Each slide block has: a title, what to put on the slide, and a one-line speaker note for what to say while it's up.

Keep slides **light** — bullets, not paragraphs. Evaluators skim. The TECHNICAL.md doc has the depth; the PPT just needs to land the story.

---

## Slide 1 — Title

**Title:** DocuMind
**Subtitle:** PDF-Constrained Conversational Agent
**On the slide:**
- Live URL: https://documind-production-58e5.up.railway.app/ui
- GitHub: https://github.com/Kavya100206/DocuMind
- Your name · submission date

**Speaker note:** "DocuMind is a Q&A agent that answers from a single PDF — and refuses cleanly when the answer isn't there."

---

## Slide 2 — The Problem

**Title:** Why standard RAG fails

**On the slide:**
- Vector-only similarity misses exact keyword matches (*"DSA I" vs "DSA II"*)
- Naive chunking destroys document structure
- **No hard contract** preventing the LLM from drawing on pretrained knowledge when chunks are weak
- → "Grounded" RAG can still confidently fabricate answers with no source in the user's PDF

**Speaker note:** "The hidden failure mode isn't off-topic queries — those are easy to refuse. It's queries whose vocabulary appears in the chunks but whose specific fact doesn't."

---

## Slide 3 — The Solution

**Title:** A PDF-constrained contract

**On the slide:**
Every response is **one of two shapes** — there is no middle ground:

| Grounded | Refusal |
|---|---|
| ✓ Answer from chunks | ✗ has_answer=false |
| ✓ Confidence score | ✗ confidence=0.0 |
| ✓ Page citations | ✗ citations=[] |
| | ✗ Localized refusal message |

**Speaker note:** "The `has_answer / confidence / citations` triple is a contract, not a UI hint. That's what makes the answers trustable."

---

## Slide 4 — Architecture

**Title:** How it works

**On the slide:** (one image — the diagram from TECHNICAL.md §1)

```
User Query
    │
    ▼
[ Language Detect ] ── EN / HI
    │
    ▼
[ Query Rewriter ] ── pronoun resolution from session memory
    │
    ▼
[ LangGraph Agent Loop ]
    │
    ├─▶ router_node      (LLM picks tool: vector / lexical / summarize)
    │
    ├─▶ tool_node        (hybrid retrieval + cross-encoder rerank)
    │
    └─▶ evaluate_node    (3-layer refusal gate)
            │
            ├── pass → answer + citations
            │
            └── fail → structural refusal (localized)
```

**Speaker note:** "Hybrid retrieval combines FAISS and BM25, reranked by a cross-encoder. The agent loop lets an LLM pick the retrieval tool per query type."

---

## Slide 5 — What makes this different

**Title:** Key design decisions

**On the slide:**
- **Hard refusal contract** — refusal is a structured response shape, not a fallback string
- **Qualifier-distance grounding gate** — catches the *realistic* hallucination trap (chunks share vocabulary with the question, but the specific fact isn't there)
- **LangGraph agent loop** — LLM chooses the retrieval tool per query; no hardcoded routing
- **Hybrid retrieval** — FAISS (65%) + BM25 (35%) + cross-encoder rerank
- **Multilingual** — EN + HI with Devanagari verification (catches silent English fallbacks)

**Speaker note:** "The qualifier-distance gate is the part I'm most happy with — it's what handles the I3 trap that defeats most RAG implementations."

---

## Slide 6 — The I3 hallucination trap

**Title:** Catching what other RAG systems miss

**On the slide:**

> *"What is the official rollout date for **e₹-Retail** in **rural India** announced in this report?"*

- Document mentions e₹-Retail (CBDC pilot, Dec 1 2022) ✓
- Document mentions rural India (kisan credit cards, financial inclusion) ✓
- Document does **not** mention a *rural-specific rollout date* for e₹-Retail ✗

**Most RAG systems:** retrieve the e₹-Retail chunk → fabricate a date → confident answer
**DocuMind:** qualifier-distance gate fires → structural refusal

**Speaker note:** "Vocabulary overlap with the chunks isn't enough. The gate verifies that the question's distinctive qualifiers actually cluster around the cited fact — not just appear somewhere in the same document."

---

## Slide 7 — Test categories

**Title:** What's verified

**On the slide:**

| Category | Count | What it proves |
|---|---|---|
| Valid English (V1–V6) | 6 | Factual / numerical / section / synthesis / summary / multi-turn |
| Invalid English (I1–I3) | 3 | Refuses outside-knowledge, off-topic, **and the I3 hallucination trap** |
| Hindi (H1–H2) | 2 | Answer + refusal in Devanagari |
| Strict isolation | — | Missing/empty `document_id` rejected at controller |

**Reproducibility:** `tests/test_suite.py` · full spec in `tests/test_cases.md`

**Speaker note:** "Every category exercises a different contract. The I3 case is the strongest signal — it's the failure mode you can't detect with a confidence threshold alone."

---

## Slide 8 — Live demo

**Title:** Live demo

**On the slide:** (just the demo flow — keep this minimal so attention is on the screen, not the slide)

1. **V1** — *"When did the revised Master Directions on PSL come into effect?"* → factual + citation
2. **V5** — *"Summarize the main themes of this annual report."* → exercises summarize tool
3. **I3** — *"What is the official rollout date for e₹-Retail in rural India announced in this report?"* → refusal banner
4. **H1** — Hindi CFO question → Devanagari response

**Speaker note:** *(Switch to live deploy or play recorded video. Pause on I3's refusal banner — that's your strongest demo moment. Pause for 25–30s between heavy queries on free Groq tier to avoid 429s.)*

---

## Slide 9 — Tech Stack

**Title:** Tech stack

**On the slide:**

| Layer | Technology |
|---|---|
| LLM | Groq · LLaMA 3.3 70B Versatile |
| Agent framework | LangGraph |
| Vector store | FAISS |
| Lexical search | BM25 (`rank_bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-2-v2` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Backend | FastAPI · SQLAlchemy · PostgreSQL (NeonDB) |
| Frontend | Vanilla JS SPA (no build step) |
| Multilingual | `langdetect` + prompt-directive answer generation |
| External | Google Drive API (OAuth2) |
| Deploy | Railway |

**Speaker note:** "Chose Groq over OpenAI for ~10× cost at similar quality on this workload. FAISS over Pinecone because it's dependency-free and rebuildable from Postgres on cold start."

---

## Slide 10 — Trade-offs and what's next

**Title:** Trade-offs · limitations · roadmap

**On the slide:**

**Trade-offs I'd flag:**
- Hybrid retrieval costs ~30ms more than vector-only; gain is exact-match queries
- Cross-encoder reranker is lazy-loaded — first query after cold start is slow
- Layered refusal gates over a single confidence threshold — handles different failure modes separately

**Known limitations:**
- Embedding model is English-tuned; Hindi works but is borderline
- Groq free-tier rate limits (6000 TPM) cap throughput at ~1.7 heavy queries/min
- Drive integration is single-user (`user_id` hardcoded)

**Roadmap:**
- Switch to `paraphrase-multilingual-MiniLM-L12-v2` for native Hindi semantic search
- Distinguish "service unavailable, retry" from "answer not in document" in the refusal layer
- Multi-tenant Drive with session/JWT layer

**Speaker note:** "Honest list — these aren't oversights, they're scoped trade-offs given the assignment timeline."

---

## Slide 11 — Closing

**Title:** Thank you

**On the slide:**
- 🌐 Live: https://documind-production-58e5.up.railway.app/ui
- 🐙 Repo: https://github.com/Kavya100206/DocuMind
- 📄 Technical Note: `docs/TECHNICAL.md`
- 🧪 Test Instructions: `docs/TESTING.md`
- 🎥 Demo Video: *(insert link)*

**Speaker note:** "Happy to walk through the code, the test suite, or the design decisions in any depth."

---

## Design tips for tomorrow

- **Use one font, one color palette.** Pick a dark accent (deep blue / charcoal) + one highlight (the same red/orange you'd use for a refusal).
- **Diagram on Slide 4 should be the hero visual.** Make it large, recreate in `Mermaid` or draw.io rather than pasting ASCII art.
- **Slide 6 (I3 trap) deserves visual emphasis.** Three columns: question / chunks / why other RAG fails. This is your strongest design-decision moment.
- **Slide 8 (demo)** should be the shortest slide — the live screen does the work.
- **Avoid wall-of-text.** Each bullet should be ≤ 8 words on screen; expand verbally.
- **Export as PDF** as a backup before submitting.
- **Total slides: 11.** Aim for ~10-12 minutes including the demo.
