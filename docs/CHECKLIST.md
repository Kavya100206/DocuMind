# DocuMind — Submission Checklist

**Status legend:** ✅ Done · ⏳ In progress · ⬜ Not started

---

## Mandatory Deliverables

| # | Item | Status | Location / Notes |
|---|---|---|---|
| 1 | 📄 **Technical Note** | ✅ | [TECHNICAL.md](TECHNICAL.md) — architecture, design decisions, trade-offs |
| 2 | 🧪 **Test Instructions** | ✅ | [TESTING.md](TESTING.md) — install, run, verify, sample queries |
| 3 | 🎥 **Demo Video** | ⬜ | Record tomorrow — see "Demo Video Plan" below |
| 4 | 📊 **PPT Deck** | ⬜ | Build tomorrow — see "PPT Outline" below |
| 5 | 🐙 **GitHub Repo** | ✅ | https://github.com/Kavya100206/DocuMind |
| 6 | 🌐 **Live URL** | ✅ | https://documind-production-58e5.up.railway.app/ui |
| 7 | 📕 **Sample PDF** | ✅ | RBI Annual Report 2024-25 — already uploaded on live deploy; bundle PDF with submission |
| 8 | 🧪 **Test Cases** | ✅ | [../tests/test_cases.md](../tests/test_cases.md) — 11 query specs (V1–V6, I1–I3, H1–H2) + pass criteria |
| 9 | 📋 **Submission Checklist** | ✅ | This file |

---

## Demo Video Plan

### 7-query flow (covers every capability without bloating the runtime)

1. **V1** — *"When did the revised Master Directions on Priority Sector Lending (PSL) come into effect?"* → factual extraction, citation
2. **V2** — *"By how much did the Reserve Bank reduce the CRR during 2024-25?"* → numerical extraction
3. **V5** — *"Summarize the main themes of this annual report."* → exercises `summarize_document` tool
4. **V6 multi-turn** (same session):
   - Q1: *"What was the change in CRR during 2024-25?"*
   - Q2: *"Why might the Reserve Bank have made this decision based on the broader economic context discussed?"*
5. **I3 — hallucination trap** (the headliner): *"What is the official rollout date for e₹-Retail in rural India announced in this report?"* → must show refusal banner. **Spend extra time on this — it's your strongest signal.**
6. **H1 — Hindi answer**: *"रिज़र्व बैंक के मुख्य वित्तीय अधिकारी कौन हैं?"* → Devanagari response
7. **H2 — Hindi refusal**: *"भारत के प्रधान मंत्री कौन हैं?"* → Hindi refusal message

### Voiceover beats (one sentence per beat)

- Open with the problem: "Standard RAG hallucinates when the answer isn't in the document."
- Show V1: "Direct factual extraction with a citation."
- Show V5: "The agent picks the summarize tool — visible in the observability panel."
- Show V6 multi-turn: "Pronoun resolution + retrieval beyond the first chunk."
- **Pause on I3**: "This is the hard case — the document mentions e₹-Retail and rural India separately, so a weak system would invent a date. DocuMind refuses cleanly."
- Show H1, H2: "Multilingual works both ways — answer in Hindi, refusal in Hindi, no English fallback."
- Close with the hard refusal contract: "Every response is either grounded in the user's PDF or a structural refusal. No middle ground."

### What to show on screen

- Citation accordion (expand on V1, V2)
- Confidence bar (full on V1/V5, **0% on I3/H2**)
- Refusal banner (I3, H2)
- Retrieval trace toggle (open during V5 to show chunk breakdown)
- Observability panel (show tool selection on V5)

### Rate-limit gotcha for the recording

Groq free-tier is 6000 TPM. Heavy queries chain ~3000 tokens each. **Pause 25–30s between heavy queries** during recording (edit out the wait), or upgrade to Groq Dev Tier for the recording session ($5).

---

## PPT Outline

1. **Title slide** — DocuMind · live URL · GitHub link · your name
2. **Problem** — Standard RAG hallucinates; "grounded" RAG without a hard refusal contract still fabricates
3. **Solution overview** — 6-stage hybrid retrieval + LangGraph agent loop + 3-layer refusal gate
4. **Architecture diagram** — copy from [TECHNICAL.md §1](TECHNICAL.md#1-architecture)
5. **Key design decisions** — hard refusal contract · qualifier-distance grounding · hybrid retrieval · multilingual
6. **Test categories** — table from [test_cases.md §3](../tests/test_cases.md#3-test-categories). Mention reproducibility (`tests/test_suite.py`) without quoting a "X/X passing" headline — the grader will run it themselves.
7. **Live demo** — embed video OR live walkthrough
8. **Tech stack** — copy table from [README.md "Tech Stack"](../README.md#tech-stack)
9. **Trade-offs + what's next** — copy from [TECHNICAL.md §3](TECHNICAL.md#3-trade-offs)

---

## Pre-submission verification

- [ ] All 4 verification queries on live deploy answer correctly (V1, V5, I3, H1)
- [ ] `tests\test_suite.py` passes locally end-to-end
- [ ] README links resolve (no 404s on relative paths)
- [ ] TECHNICAL.md and TESTING.md links resolve
- [ ] Demo video uploaded to a stable URL (YouTube unlisted / Drive shared link)
- [ ] PPT exported as PDF (backup in case the .pptx breaks on the grader's machine)
- [ ] Repo is public (or evaluator added as collaborator)
- [ ] `.env` is in `.gitignore` (no leaked secrets)
- [ ] Sample PDF bundled with submission OR linked
