# Revenue Leakage Investigation Agent

A human-in-the-loop AI agent that finds money a SaaS company earned but never billed, proves it with evidence, and fixes it only after a named approver says yes.

| | |
|---|---|
| UI | Streamlit, http://localhost:8501 |
| API | FastAPI, http://localhost:8000 (docs at http://localhost:8000/docs) |
| Agent | LangGraph + Groq (bring-your-own key), SQLite |
| Tests | 137 passed, 0 failed (see [tests/QA_REPORT.md](tests/QA_REPORT.md)) |

## 1. What was built

- **Seeded billing database** (`agent/db.py`): 15 customers, 3 plans, periods 2025-04 to 2025-06, with nine planted leakage cases (wrong plan price, missing overage, expired discount still applied, unbilled seats, missing invoices, one over-billing).
- **Investigation agent** (`agent/graph.py`, `agent/tools.py`): a LangGraph workflow that plans, calls read-only billing tools, quantifies leakage, proposes actions, pauses for approval, executes, and summarises.
- **Service facade** (`agent/service.py`): `make_llm`, `validate_api_key`, `send_message`, `list_pending_approvals`, `decide_approval`, `get_audit_log`, `get_history`.
- **REST API** (`api/main.py`): identity headers, error mapping, CORS.
- **Chat UI** (`ui/app.py`, `ui/api_client.py`): key gate, chat, findings with evidence, approval cards, audit tab.
- **137 offline tests** (`tests/`).

On the seed data a full investigation finds **17 anomalies, $2,557.60 of lost revenue and 9 proposed actions** (8 invoice adjustments and 1 credit memo for a $20 over-billing).

## 2. Architecture

```
Browser -> Streamlit UI :8501   (key only in st.session_state)
   -- HTTP + X-User, X-User-Role, X-Groq-Api-Key -->
FastAPI :8000   (role check, error mapping, key never stored or logged)
   -> agent/service.py   (facade)
      -> LangGraph workflow (agent/graph.py)  <-checkpoints->  data/checkpoints.db
      -> agent/tools.py   (read tools for the LLM; write tools only after approval)
         -> data/billing.db  (customers, plans, subscriptions, invoices, usage,
                              approvals, credit_memos, audit_log)

START -> plan -> investigate (LLM tool loop, read-only) -> analyze (Decimal math)
      -> propose_actions -> no proposals? -> summarize -> END
                         -> proposals?   -> human_approval (interrupt, graph suspended)
                                         -> resume with approver decision -> execute (only write path)
                                         -> more pending? -> human_approval, else summarize -> END
```

## 3. Key decisions

- **SQLite for everything.** Zero setup, transactional, easy to reset and seed. The LangGraph SQLite checkpointer keeps conversation memory and pending approvals across restarts. Trade-off: a single writer.
- **`interrupt()` for human approval.** Proposals are stored as `pending` and the graph is genuinely suspended. Write tools are never given to the LLM, are called only from the `execute` node, and each re-checks that the approval was made by an `approver` and not already executed.
- **Role headers (`X-User`, `X-User-Role`).** `analyst` can investigate and propose; `approver` can also decide. Enforced in the API (403) and again in the service and tools. A denied attempt is audit-logged as `unauthorized_decision_attempt`. This is demo-grade gating, not real authentication.
- **Bring-your-own Groq key.** Typed into a password field, held only in Streamlit session memory, sent per request in `X-Groq-Api-Key`, used to build the client, then discarded. Never written to files, databases, checkpoints, logs, responses or caches. An environment key does not bypass the UI gate. Tests assert this.
- **Deterministic money math.** The LLM chooses tools and writes the narrative; all amounts are computed in Python with `Decimal`. If the model skips reconciliation, a fallback reconciles all customers. Tests recompute expected values independently from the raw tables.
- **Separate FastAPI backend.** Gives the UI a stable HTTP contract, keeps key and identity handling in one place, and maps provider errors to 401/403/404/409/422/429/502 without leaking raw messages. The brief named FastAPI, and approvals, audit and role checks justified it.
- **Audit everything.** Starts, tool calls, proposals, decisions, writes, failures and denials, each with actor, role and timestamp.
- **Retries with exponential back-off (tenacity)** on rate limits, 5xx, connection errors, timeouts and locked SQLite; never on auth or bad-request errors.

## 4. How to run

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn api.main:app --port 8000            # terminal 1
.venv\Scripts\python -m streamlit run ui/app.py --server.port 8501  # terminal 2
```

Open http://localhost:8501 (app) and http://localhost:8000/docs (API docs). Health checks: http://localhost:8000/health and http://localhost:8501/_stcore/health. The database is created and seeded on first use; override paths with `BILLING_DB_PATH` and `CHECKPOINT_DB_PATH`.

**Using it with a real key:** get a free key at https://console.groq.com/keys (starts with `gsk_`), paste it into the sidebar and click Connect. Pick the `analyst` or `approver` role, ask "Investigate revenue leakage in last quarter's billing data.", review the findings and evidence, switch to `approver` to approve or reject each proposed action, and check the Audit log tab.

**Demo mode without a key:** set `$env:FAKE_LLM = "1"` before starting uvicorn. This uses a deterministic, network-free model and only checks the `gsk_` prefix, so any `gsk_...` value connects. For tests and demos only.

## 5. Test results

**137 passed, 0 failed** (unit 92, integration 26, Streamlit AppTest 14, live smoke 5). Run `.venv\Scripts\python -m pytest tests -v`. Details and known gaps are in [tests/QA_REPORT.md](tests/QA_REPORT.md).

## 6. Limitations

- Identity is header-based and trusted from the client; there is no login or token verification.
- The data is synthetic; there are no connectors to a real billing system.
- The real Groq path (model discovery, `ChatGroq`) has not been exercised live; all tests are offline.
- SQLite is single-writer; concurrent approvers are untested.
- The UI was checked with Streamlit AppTest and a live health smoke test, not browser automation.
- Actions are limited to invoice corrections, missing-invoice generation and credit memos; nothing is emailed or posted to an accounting system.
