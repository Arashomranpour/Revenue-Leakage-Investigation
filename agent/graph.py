"""LangGraph workflow: plan -> investigate (tool loop) -> analyze -> propose_actions ->
human_approval (interrupt) -> execute -> (loop while approvals pending) -> summarize.

    graph = build_graph(llm, db_path, checkpointer)
    graph.invoke({... see service.send_message ...}, {"configurable": {"thread_id": t}})
    graph.invoke(Command(resume={"approval_id": 3, "approved": True, "role": "approver", "actor": "bob"}), cfg)

Safety model:
  * the LLM only gets READ tools; amounts come from tools.reconcile_customer (Decimal math).
  * proposals are rows in `approvals` (status pending). Writes (adjust_invoice/apply_credit_memo) are
    called only in the `execute` node, and only for approvals whose status is 'approved' and whose
    decision was made by role 'approver' (re-verified inside the write tools).
  * every step is written to audit_log.
"""
from __future__ import annotations

import json
import operator
import os
import sqlite3
from decimal import Decimal
from typing import Annotated, Any

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from typing_extensions import TypedDict

from . import tools as T
from .db import log_audit, session, utcnow

MAX_TOOL_STEPS = 4
RETRY_ATTEMPTS = int(os.environ.get("AGENT_RETRY_ATTEMPTS", "4"))
RETRY_WAIT = float(os.environ.get("AGENT_RETRY_WAIT", "1.0"))


class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    memory: Annotated[list[str], operator.add]  # one line per past investigation (thread memory)
    thread_id: str
    role: str
    actor: str
    plan: str
    anomalies: list[dict]
    reconciled: list[int]
    total_lost: float
    proposals: list[dict]
    notes: list[str]
    reply: str


# ------------------------------------------------------------------ retry helpers
def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (sqlite3.OperationalError, ConnectionError, TimeoutError)):
        return True
    try:
        import groq
        return isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError, groq.RateLimitError,
                                groq.InternalServerError))
    except Exception:  # pragma: no cover
        return False


def _retrying():
    return retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(RETRY_ATTEMPTS),
                 wait=wait_exponential(multiplier=RETRY_WAIT, max=10), reraise=True)


def call_llm(llm, messages):
    """LLM call with retry on transient errors (rate limit / connection / 5xx)."""
    return _retrying()(lambda: llm.invoke(messages))()


def call_tool(tool, args):
    """Tool call with retry on transient errors (e.g. sqlite locked)."""
    return _retrying()(lambda: tool.invoke(args))()


# ------------------------------------------------------------------ helpers
def _text(m: BaseMessage) -> str:
    c = m.content
    return c if isinstance(c, str) else json.dumps(c)


def _history(messages: list[BaseMessage], n: int = 8) -> list[BaseMessage]:
    """Last n plain Human/AI turns (tool-free) for LLM context."""
    out = [HumanMessage(content=_text(m)) if isinstance(m, HumanMessage) else AIMessage(content=_text(m))
           for m in messages if isinstance(m, (HumanMessage, AIMessage)) and _text(m)]
    return out[-n:]


def _money(x) -> str:
    return f"${x:,.2f}"


def _approvals(db_path, ids):
    if not ids:
        return []
    return [a for a in (T.get_approval(db_path, i) for i in ids) if a]


def build_graph(llm, db_path: str, checkpointer=None):
    read_tools = T.make_read_tools(db_path)
    registry = {t.name: t for t in read_tools}

    def audit(s, action, detail=None):
        log_audit(db_path, s.get("actor", "unknown"), s.get("role", "unknown"), action, detail)

    # ---------------------------------------------------------------- nodes
    def plan(s: State):
        role = s.get("role")
        if role not in T.PROPOSER_ROLES:
            audit(s, "permission_denied", {"reason": f"role '{role}' may not investigate"})
            raise PermissionError(f"Role '{role}' is not allowed to investigate (use 'analyst' or 'approver').")
        request = _text(s["messages"][-1])
        audit(s, "investigation_started", {"thread_id": s.get("thread_id"), "request": request[:500]})
        mem = "\n".join(f"- {m}" for m in s.get("memory", [])[-5:]) or "none"
        text = ""
        if llm is not None:
            sys = SystemMessage(content=(
                "[PLAN] You are a revenue-leakage investigator. Write a 2-3 line investigation plan. "
                f"Previous investigations in this conversation:\n{mem}"))
            text = _text(call_llm(llm, [sys, *_history(s["messages"])]))
        return {"plan": text}

    def investigate(s: State):
        found: dict[str, dict] = {}
        reconciled: list[int] = []
        if llm is None:
            return {"anomalies": [], "reconciled": []}
        bound = llm.bind_tools(read_tools)
        sys = SystemMessage(content=(
            "[INVESTIGATE] You investigate billing discrepancies between customer plans and invoices. "
            "Use the tools: list_customers to find ids, then reconcile_customer for each customer in scope "
            "(all customers if the user does not specify). NEVER compute money yourself; amounts come "
            "from tools. When all needed reconciliations are done, reply briefly.\n"
            f"Plan: {s.get('plan', '')}\nPast investigations: {s.get('memory', [])[-5:]}"))
        convo: list[BaseMessage] = [sys, *_history(s["messages"])]
        for _ in range(MAX_TOOL_STEPS):
            ai = call_llm(bound, convo)
            convo.append(ai)
            if not getattr(ai, "tool_calls", None):
                break
            for tc in ai.tool_calls:
                name, args = tc["name"], tc.get("args") or {}
                audit(s, f"tool:{name}", args)
                tool = registry.get(name)
                if tool is None:
                    content = json.dumps({"error": f"unknown tool {name}"})
                else:
                    try:
                        result = call_tool(tool, args)
                        content = json.dumps(result, default=str)
                        if name == "reconcile_customer":
                            cid = int(args.get("customer_id"))
                            reconciled.append(cid)
                            for a in result:
                                found[a["anomaly_id"]] = a
                    except Exception as e:  # noqa: BLE001 - surfaced to the LLM, not fatal
                        content = json.dumps({"error": f"{type(e).__name__}: {e}"})
                convo.append(ToolMessage(content=content, tool_call_id=tc["id"], name=name))
        return {"anomalies": list(found.values()), "reconciled": sorted(set(reconciled))}

    def analyze(s: State):
        anomalies = {a["anomaly_id"]: a for a in s.get("anomalies", [])}
        reconciled = set(s.get("reconciled", []))
        notes = list(s.get("notes", []))
        if not reconciled:  # safety net: the model skipped reconcile; do it deterministically for everyone
            audit(s, "analysis_fallback_reconcile_all", {})
            for c in T.list_customers(db_path):
                reconciled.add(c["id"])
                for a in T.reconcile_customer(db_path, c["id"]):
                    anomalies[a["anomaly_id"]] = a
            notes.append("The model did not run a reconciliation; all customers were reconciled.")
        ordered = sorted(anomalies.values(), key=lambda a: (-a["amount_lost"], a["customer_id"], a["period"]))
        total = sum((Decimal(str(a["amount_lost"])) for a in ordered), Decimal(0))
        audit(s, "analysis_done", {"anomalies": len(ordered), "total_lost": float(total)})
        return {"anomalies": ordered, "reconciled": sorted(reconciled), "total_lost": float(total), "notes": notes}

    def propose_actions(s: State):
        groups: dict[tuple, list[dict]] = {}
        for a in s.get("anomalies", []):
            if a["amount_lost"] > 0:
                it = dict(anomaly_id=a["anomaly_id"], invoice_id=a["invoice_id"], period=a["period"],
                          component=a["component"], amount=a["amount_lost"],
                          description=f"{a['type']} {a['period']}: {a['root_cause']}")
                if a["invoice_id"] is None:
                    it["new_invoice_lines"] = a["evidence"]["expected_lines"]
                groups.setdefault((a["customer_id"], "adjust_invoice"), []).append(it)
            if a["amount_overbilled"] > 0:
                it = dict(anomaly_id=a["anomaly_id"], invoice_id=a["invoice_id"], period=a["period"],
                          component=a["component"], amount=a["amount_overbilled"],
                          description=f"{a['type']} {a['period']}: {a['root_cause']}")
                groups.setdefault((a["customer_id"], "credit_memo"), []).append(it)
        proposals, notes = [], list(s.get("notes", []))
        for (cid, action), items in sorted(groups.items()):
            fn = T.propose_credit_memo if action == "credit_memo" else T.propose_invoice_adjustment
            reason = "; ".join(i["description"] for i in items)[:1000]
            res = fn(db_path, s.get("thread_id", ""), cid, items, reason, s.get("actor"), s.get("role"))
            if res["created"]:
                proposals.append({"approval_id": res["approval_id"], "action": action, "customer_id": cid})
                audit(s, "proposal_created", {"approval_id": res["approval_id"], "action": action,
                                              "customer_id": cid, "items": len(items)})
            else:
                notes.append(f"Action '{action}' for customer {cid} is already pending (approval #{res['approval_id']}).")
        return {"proposals": proposals, "notes": notes}

    def route_after_propose(s: State):
        return "human_approval" if s.get("proposals") else "summarize"

    def pending_ids(s: State) -> list[int]:
        return [a["id"] for a in _approvals(db_path, [p["approval_id"] for p in s.get("proposals", [])])
                if a["status"] == "pending"]

    def human_approval(s: State):
        pend = pending_ids(s)
        if not pend:
            return {}
        decision = interrupt({"type": "approval_required", "thread_id": s.get("thread_id"),
                              "pending": [T.get_approval(db_path, i) for i in pend]})
        if not isinstance(decision, dict):
            return {}
        role, actor = decision.get("role"), decision.get("actor", "unknown")
        try:
            aid = int(decision.get("approval_id"))
        except (TypeError, ValueError):
            return {}
        if role != T.APPROVER_ROLE:
            log_audit(db_path, actor, str(role), "unauthorized_decision_attempt", {"approval_id": aid})
            return {}
        if aid not in pend:
            log_audit(db_path, actor, role, "decision_ignored", {"approval_id": aid, "reason": "not pending here"})
            return {}
        status = "approved" if decision.get("approved") else "denied"
        with session(db_path) as c:
            c.execute("UPDATE approvals SET status=?, decided_by=?, decided_role=?, decided_at=? "
                      "WHERE id=? AND status='pending'", (status, actor, role, utcnow(), aid))
        log_audit(db_path, actor, role, f"approval_{status}", {"approval_id": aid})
        return {}

    def execute(s: State):
        """The ONLY place write tools are called, and only for approver-approved approvals."""
        for a in _approvals(db_path, [p["approval_id"] for p in s.get("proposals", [])]):
            if a["status"] != "approved":
                continue
            actor = a["decided_by"] or "unknown"
            fn = T.apply_credit_memo if a["action"] == "credit_memo" else T.adjust_invoice
            try:
                res = fn(db_path, a["id"], actor)
                log_audit(db_path, actor, T.APPROVER_ROLE, "write_executed",
                          {"approval_id": a["id"], "action": a["action"], **res})
            except Exception as e:  # noqa: BLE001
                log_audit(db_path, actor, T.APPROVER_ROLE, "write_failed",
                          {"approval_id": a["id"], "error": f"{type(e).__name__}: {e}"})
        return {}

    def route_after_execute(s: State):
        return "human_approval" if pending_ids(s) else "summarize"

    def summarize(s: State):
        anomalies = s.get("anomalies", [])
        outcomes = _approvals(db_path, [p["approval_id"] for p in s.get("proposals", [])])
        facts = [f"**Findings** ({len(anomalies)} anomalies, lost revenue {_money(s.get('total_lost', 0))})"]
        for a in anomalies:
            amt = (f"lost {_money(a['amount_lost'])}" if a["amount_lost"] else
                   f"over-billed {_money(a['amount_overbilled'])}")
            facts.append(f"- {a['customer_name']} {a['period']} [{a['type']}]: {amt}. {a['root_cause']}")
        if outcomes:
            facts.append("**Actions**")
            for o in outcomes:
                facts.append(f"- #{o['id']} {o['action']} for {o['customer_name']} "
                             f"{_money(o['amount'])}: {o['status']}")
        facts += [f"_Note: {n}_" for n in s.get("notes", [])]
        text = ""
        if llm is not None:
            try:
                sys = SystemMessage(content=(
                    "[SUMMARIZE] Write a short executive summary (max 4 sentences) of the investigation. "
                    "Do not invent or recompute numbers; use only these facts:\n" + "\n".join(facts)))
                text = _text(call_llm(llm, [sys, *_history(s["messages"], 4)]))
            except Exception:  # noqa: BLE001 - writes may already be done; never lose the result
                text = ""
        reply = (text + "\n\n" if text else "") + "\n".join(facts)
        audit(s, "investigation_summarized", {"total_lost": s.get("total_lost", 0),
                                              "anomalies": len(anomalies)})
        mem = (f"{_text(s['messages'][-1])[:80]} -> {len(anomalies)} anomalies, lost "
               f"{_money(s.get('total_lost', 0))}; actions: " +
               (", ".join(f"#{o['id']} {o['status']}" for o in outcomes) or "none"))
        return {"reply": reply, "messages": [AIMessage(content=reply)], "memory": [mem]}

    g = StateGraph(State)
    for name, fn in [("plan", plan), ("investigate", investigate), ("analyze", analyze),
                     ("propose_actions", propose_actions), ("human_approval", human_approval),
                     ("execute", execute), ("summarize", summarize)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "investigate")
    g.add_edge("investigate", "analyze")
    g.add_edge("analyze", "propose_actions")
    g.add_conditional_edges("propose_actions", route_after_propose, ["human_approval", "summarize"])
    g.add_edge("human_approval", "execute")
    g.add_conditional_edges("execute", route_after_execute, ["human_approval", "summarize"])
    g.add_edge("summarize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
