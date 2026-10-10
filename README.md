# Revenue Leakage Investigation

A human-in-the-loop AI agent that finds money a SaaS company earned but never billed, proves it with evidence, and fixes it only after a named approver says yes.

**Stack:** LangGraph · LangChain · Groq · FastAPI · Streamlit · SQLite

| | |
|---|---|
| UI | Streamlit, http://localhost:8501 |
| API | FastAPI, http://localhost:8000 (docs at `/docs`) |
| Tests | 137 passed, 0 failed ([QA report](tests/QA_REPORT.md)) |

## Features
- Stateful conversations that remember previous investigations (LangGraph SQLite checkpointer).
- Tool calling to query and reconcile billing data; money math is deterministic Python (`Decimal`), not LLM output.
- Evidence-backed anomaly detection: on the seed data it finds 17 anomalies, $2,557.60 lost revenue, 9 proposed actions.
- Human approval (`interrupt()`) before any credit memo or invoice change.
- Audit log, retry handling, and role-based protection against unauthorized actions.
- Bring-your-own Groq key: entered in the UI, kept in session memory only, never stored.

## Run it
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn api.main:app --port 8000            # terminal 1
.venv\Scripts\python -m streamlit run ui/app.py --server.port 8501  # terminal 2
```
Open http://localhost:8501, paste your Groq key (free at https://console.groq.com/keys, starts with `gsk_`), pick the `analyst` or `approver` role, and ask it to investigate revenue leakage. Switch to `approver` to approve or reject proposed fixes.

**Demo mode (no key):** set `$env:FAKE_LLM = "1"` before starting uvicorn; any `gsk_...` value connects. For tests and demos only.

## Tests
```powershell
.venv\Scripts\python -m pytest tests -v
```
All tests run offline with a fake model. See [tests/QA_REPORT.md](tests/QA_REPORT.md).

## Docs
- [summary.md](summary.md): what was built, architecture, key decisions, limitations.

## Layout
```
agent/   LangGraph workflow, tools, SQLite seed data, service facade
api/     FastAPI routes
ui/      Streamlit app and API client
tests/   unit, integration, UI and live smoke tests
```

## Limitations
Role headers are demo-grade gating, not real authentication. Data is synthetic. The live Groq path is untested in CI (tests use a fake model).
