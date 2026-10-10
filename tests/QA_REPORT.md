# QA Report - Revenue Leakage Investigation Agent

Command: `.venv\Scripts\python -m pytest tests -v` (Python 3.14.6, pytest 9.1.1; `FAKE_LLM=1`, temp DBs, no network, no real Groq key).
Raw output of the final run: `tests/last_run.txt`.

## Result: 137 passed, 0 failed, 0 errors (39 s)

| Suite | File(s) | Tests | Pass | Fail |
|---|---|---|---|---|
| Unit | `tests/unit/test_seed_billing.py` | 14 | 14 | 0 |
| Unit | `tests/unit/test_reconcile.py` | 22 | 22 | 0 |
| Unit | `tests/unit/test_workflow.py` | 21 | 21 | 0 |
| Unit | `tests/unit/test_keys_errors_retry.py` | 35 | 35 | 0 |
| Integration (FastAPI TestClient) | `tests/integration/test_api.py` | 26 | 26 | 0 |
| UI (Streamlit AppTest, API routed in-process) | `tests/ui/test_streamlit_app.py` | 14 | 14 | 0 |
| Live smoke (real uvicorn + streamlit processes, spare ports) | `tests/e2e/test_live_smoke.py` | 5 | 5 | 0 |
| **Total** | | **137** | **137** | **0** |

## What is covered
- **Billing accuracy** - invoices recomputed independently from raw plans/subscriptions/usage (does not reuse `expected_lines`); per-customer leakage hand-computed: C3 150.00, C5 90.00, C7 79.60, C9 120.00, C11 999.00, C12 10.00, C13 110.00, C14 999.00, C15 over-billed 20.00; total lost 2,557.60; customers 1,2,4,6,8,10 clean. `reconcile_customer` matches for every case (type, period, amount).
- **Anomaly evidence** - every anomaly has root cause, expected/billed, subscription, plan, invoice lines/expected lines; expected - billed == lost - overbilled.
- **Human-in-the-loop** - graph suspended at `human_approval` (interrupt) with no write to invoices / invoice_lines / credit_memos; write tools reject unapproved approvals, approvals decided by a non-approver, wrong action type, and double execution; repeat message while pending does not re-run; second thread does not duplicate proposals.
- **Rejection** - all 9 denied: data byte-identical, no `write_executed`.
- **Approval** - invoice corrected / missing invoice generated / credit memo of $20 created; `reconcile_customer` returns no anomalies afterwards; approving all 9 leaves zero leakage.
- **RBAC** - analyst, unknown, upper-case, empty and None roles get `PermissionError`; the denied attempt is audit-logged (actor, role, approval id); approval stays pending.
- **Audit log** - required actions present, actor/role/ts on every row, newest first, limit honoured.
- **Memory** - state kept across turns on one thread_id, survives a simulated process restart (new checkpointer), threads isolated.
- **Retries** - RateLimit/5xx/connection/timeout/sqlite-locked retried then succeed; gives up after max attempts; auth/bad-request/ValueError not retried; full workflow recovers from a transient 429.
- **Friendly errors** - mapping for auth, permission, rate limit, not found, bad request, 5xx, connection, timeout, unexpected; no raw text or key in messages.
- **API key never persisted** - not in billing DB, checkpoint DB (incl. serialized state), any file in the run dir, audit log, history, responses, response headers, logs (caplog), os.environ, model cache, source files (no hard-coded `gsk_...` key).
- **API** - all routes, 401 (missing headers / key), 403 (bad role, analyst decision), 404, 409, 422, 429 (rate limit), 401 (provider auth), 502 (AgentError), 500 with no leak; OpenAPI/docs; CORS only for the Streamlit origin.
- **UI** - key gate blocks chat even with `GROQ_API_KEY` set; key input is a password field; empty / whitespace / non-`gsk_` keys rejected; connect, forget-key, new-investigation; approve/reject disabled for analyst and enabled for approver; metric "Total revenue lost" = $2,557.60; key never rendered; API-offline friendly error.
- **Live** - `/health`, `/docs`, Streamlit `/_stcore/health`, Streamlit root page, one HTTP investigation round-trip, analyst 403 on decision, key absent from server logs.

## Bugs found
None blocking. No test required a fix from the owners.

## Known gaps / notes
- The real Groq path (`ChatGroq`, model discovery) is only tested with mocks; never exercised against the live API (no key by design).
- Streamlit AppTest does not render a real browser (no CSS / JS / `st.rerun` visual behaviour); the live smoke test only checks that Streamlit serves and is healthy, not interactive browser flows.
- `tenacity` is imported by `agent/graph.py` but is not listed in `requirements.txt` (it arrives transitively via langchain-core). Recommend adding it explicitly.
- No concurrency / load tests (SQLite is single-writer; two approvers deciding simultaneously is untested).
- Authentication is header-based (`X-User`, `X-User-Role`) and trusts the client: it is role gating for the demo, not real authentication.
- Retry timings are set to zero in tests; actual back-off durations are not measured.
