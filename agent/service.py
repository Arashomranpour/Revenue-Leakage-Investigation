"""Service facade for the Revenue Leakage Investigation Agent (the ONLY module backend should call).

All functions are synchronous and return JSON-serializable dicts/lists.

    make_llm(api_key) -> chat model            # FAKE_LLM=1 -> deterministic offline model
    validate_api_key(key) -> None | str        # None = OK, else a friendly error message
    send_message(thread_id, text, llm, role, actor) -> dict
        {thread_id, reply, anomalies[], total_lost, pending_approvals[], status: 'done'|'awaiting_approval'}
    list_pending_approvals(thread_id=None) -> list[dict]
    decide_approval(approval_id, approved, role, actor, llm) -> dict   (same shape as send_message + decision)
    get_audit_log(limit=100) -> list[dict]     # newest first
    get_history(thread_id) -> list[{role: 'user'|'assistant', content}]

Roles: 'analyst' may investigate/propose; only 'approver' may decide (approve/deny) -> PermissionError
(and an `unauthorized_decision_attempt` audit entry) otherwise. Writes run only after an approved decision.

Env (read at call time): BILLING_DB_PATH (default data/billing.db), CHECKPOINT_DB_PATH
(default data/checkpoints.db), FAKE_LLM=1.

The API key is used only to construct the ChatGroq client; it is never stored, cached or logged.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from . import tools as T
from .db import ROOT, default_db_path, init_db, log_audit, session
from .graph import build_graph

PREFERRED_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3-32b",
                    "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.1-8b-instant"]
_SKIP_MODEL_WORDS = ("whisper", "guard", "tts", "orpheus", "playai", "distil", "embed", "safeguard")
_MODEL_CACHE: dict = {}  # holds only a public model id + timestamp, never key material
_SAVERS: dict[str, SqliteSaver] = {}
ROLES = {"analyst", "approver"}


class AgentError(RuntimeError):
    """Friendly, user-displayable error (never contains keys or org ids)."""


# ------------------------------------------------------------------ errors / key handling
def friendly_error(exc: BaseException) -> str:
    try:
        import groq
    except Exception:  # pragma: no cover
        groq = None
    if groq is not None:
        if isinstance(exc, groq.AuthenticationError):
            return "The Groq API key was rejected. Please check the key and try again."
        if isinstance(exc, groq.PermissionDeniedError):
            return "This Groq API key does not have permission for that request."
        if isinstance(exc, groq.RateLimitError):
            return "Groq rate limit reached. Please wait a moment and try again."
        if isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
            return "Could not reach Groq (network problem or timeout). Please try again."
        if isinstance(exc, groq.NotFoundError):
            return "The selected Groq model is not available for this key. Please try again."
        if isinstance(exc, groq.BadRequestError):
            return "Groq rejected the request (bad request). Try rephrasing or retry."
        if isinstance(exc, groq.APIStatusError):
            return f"Groq returned an error (HTTP {getattr(exc, 'status_code', '?')}). Please try again later."
    return f"Unexpected error: {type(exc).__name__}"


def _pick_model(client) -> str:
    ids = []
    for m in client.models.list().data:
        mid = getattr(m, "id", "")
        if getattr(m, "active", True) is False or any(w in mid.lower() for w in _SKIP_MODEL_WORDS):
            continue
        ids.append(mid)
    for pref in PREFERRED_MODELS:
        if pref in ids:
            return pref
    if ids:
        return sorted(ids)[0]
    raise AgentError("No usable chat model is available for this Groq key.")


def _fake() -> bool:
    return os.environ.get("FAKE_LLM") == "1"


def _prefix_error(key) -> str | None:
    if not isinstance(key, str) or not key.strip():
        return "Please enter your Groq API key."
    if not key.strip().startswith("gsk_"):
        return "That does not look like a Groq API key (it should start with 'gsk_')."
    return None


def validate_api_key(key: str) -> str | None:
    """Return None if the key is usable, else a friendly error message."""
    err = _prefix_error(key)
    if err or _fake():
        return err
    try:
        from groq import Groq
        _pick_model(Groq(api_key=key.strip(), max_retries=1, timeout=15))
        return None
    except AgentError as e:
        return str(e)
    except Exception as e:  # noqa: BLE001
        return friendly_error(e)


def make_llm(api_key: str):
    """Build the chat model. The key is passed straight to the client and not retained by us."""
    if _fake():
        from .fake_llm import FakeBillingLLM
        return FakeBillingLLM()
    err = _prefix_error(api_key)
    if err:
        raise AgentError(err)
    try:
        from groq import Groq
        from langchain_groq import ChatGroq
        now = time.time()
        model = _MODEL_CACHE.get("model") if now - _MODEL_CACHE.get("t", 0) < 600 else None
        if not model:
            model = _pick_model(Groq(api_key=api_key.strip(), max_retries=1, timeout=15))
            _MODEL_CACHE.update(model=model, t=now)
        return ChatGroq(model=model, api_key=api_key.strip(), temperature=0, max_retries=0, timeout=60)
    except AgentError:
        raise
    except Exception as e:  # noqa: BLE001
        raise AgentError(friendly_error(e)) from None


# ------------------------------------------------------------------ plumbing
def _db() -> str:
    path = default_db_path()
    if not Path(path).exists():
        init_db(path)
    return path


def _saver() -> SqliteSaver:
    path = os.environ.get("CHECKPOINT_DB_PATH") or str(ROOT / "data" / "checkpoints.db")
    if path not in _SAVERS:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        _SAVERS[path] = SqliteSaver(conn)
    return _SAVERS[path]


def _graph(llm):
    return build_graph(llm, _db(), _saver())


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}


def _result(graph, thread_id: str, extra: str = "") -> dict:
    snap = graph.get_state(_cfg(thread_id))
    v = snap.values or {}
    pending = list_pending_approvals(thread_id)
    status = "awaiting_approval" if (snap.next or pending) else "done"
    if status == "done":
        reply = v.get("reply", "")
    else:
        reply = (extra + "\n\n" if extra else "") + (
            f"Found {len(v.get('anomalies', []))} anomalies, lost revenue ${v.get('total_lost', 0):,.2f}. "
            f"{len(pending)} proposed action(s) await approver sign-off: " +
            "; ".join(f"#{p['id']} {p['action']} for {p['customer_name']} (${p['amount']:,.2f})" for p in pending))
    return {"thread_id": thread_id, "reply": reply, "anomalies": v.get("anomalies", []),
            "total_lost": v.get("total_lost", 0.0), "pending_approvals": pending, "status": status}


# ------------------------------------------------------------------ public API
def send_message(thread_id: str, text: str, llm, role: str, actor: str) -> dict:
    db = _db()
    if role not in ROLES:
        log_audit(db, actor, str(role), "permission_denied", {"reason": "unknown role"})
        raise PermissionError(f"Unknown role '{role}'. Use 'analyst' or 'approver'.")
    if not text or not text.strip():
        raise ValueError("Message is empty.")
    if llm is None:
        raise AgentError("No LLM configured. Provide your Groq API key first.")
    graph = _graph(llm)
    if list_pending_approvals(thread_id):
        res = _result(graph, thread_id)
        res["reply"] = ("This conversation has approvals awaiting decision; an approver must approve or "
                        "deny them before a new investigation starts.\n\n" + res["reply"])
        return res
    inp = {"messages": [{"role": "user", "content": text.strip()}], "thread_id": thread_id, "role": role,
           "actor": actor, "plan": "", "anomalies": [], "reconciled": [], "total_lost": 0.0,
           "proposals": [], "notes": [], "reply": ""}
    try:
        graph.invoke(inp, _cfg(thread_id))
    except PermissionError:
        raise
    except Exception as e:  # noqa: BLE001
        log_audit(db, actor, role, "error", {"where": "send_message", "type": type(e).__name__})
        raise AgentError(friendly_error(e)) from None
    return _result(graph, thread_id)


def list_pending_approvals(thread_id: str | None = None) -> list[dict]:
    q, args = "SELECT id FROM approvals WHERE status='pending'", []
    if thread_id:
        q += " AND thread_id=?"
        args.append(thread_id)
    with session(_db()) as c:
        ids = [r[0] for r in c.execute(q + " ORDER BY id", args)]
    return [T.get_approval(default_db_path(), i) for i in ids]


def decide_approval(approval_id: int, approved: bool, role: str, actor: str, llm=None) -> dict:
    db = _db()
    if role != T.APPROVER_ROLE:  # checked first: needs no llm, always audited
        log_audit(db, actor, str(role), "unauthorized_decision_attempt",
                  {"approval_id": approval_id, "approved": bool(approved)})
        raise PermissionError("Only users with the 'approver' role can approve or deny actions.")
    appr = T.get_approval(db, int(approval_id))
    if appr is None:
        raise ValueError(f"Approval {approval_id} not found.")
    if appr["status"] != "pending":
        raise ValueError(f"Approval {approval_id} is already {appr['status']}.")
    graph = _graph(llm)
    tid = appr["thread_id"]
    if not graph.get_state(_cfg(tid)).next:
        raise ValueError("No suspended investigation found for this approval (checkpoint missing).")
    try:
        graph.invoke(Command(resume={"approval_id": int(approval_id), "approved": bool(approved),
                                     "role": role, "actor": actor}), _cfg(tid))
    except PermissionError:
        raise
    except Exception as e:  # noqa: BLE001
        log_audit(db, actor, role, "error", {"where": "decide_approval", "type": type(e).__name__})
        raise AgentError(friendly_error(e)) from None
    now = T.get_approval(db, int(approval_id))
    verb = {"executed": "approved and executed", "approved": "approved", "denied": "denied"}.get(
        now["status"], now["status"])
    res = _result(graph, tid, extra=f"Approval #{approval_id} {verb} by {actor}.")
    res.update(decision=now["status"], approval_id=int(approval_id))
    return res


def get_audit_log(limit: int = 100) -> list[dict]:
    with session(_db()) as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (int(limit),))]
    for r in rows:
        try:
            r["detail"] = json.loads(r["detail"])
        except (TypeError, ValueError):
            pass
    return rows


def get_history(thread_id: str) -> list[dict]:
    graph = _graph(None)
    msgs = (graph.get_state(_cfg(thread_id)).values or {}).get("messages", [])
    out = []
    for m in msgs:
        role = "user" if m.type == "human" else "assistant"
        out.append({"role": role, "content": m.content if isinstance(m.content, str) else str(m.content)})
    return out
