"""End-to-end agent workflow (FAKE_LLM): interrupt, approval gating, RBAC, audit, memory."""
import sqlite3

import pytest

from agent import tools as T

WRITE_TABLES_SQL = {
    "credit_memos": "select * from credit_memos order by id",
    "invoices": "select * from invoices order by id",
    "invoice_lines": "select * from invoice_lines order by id",
}


def snapshot(db):
    c = sqlite3.connect(db)
    try:
        return {k: c.execute(q).fetchall() for k, q in WRITE_TABLES_SQL.items()}
    finally:
        c.close()


def audit_actions(svc):
    return [r["action"] for r in svc.get_audit_log(1000)]


@pytest.fixture
def full(svc, llm, db_path):
    before = snapshot(db_path)
    res = svc.send_message("t1", "Investigate all customers for revenue leakage", llm, "analyst", "alice")
    return svc, llm, db_path, before, res


def test_full_investigation_numbers(full):
    svc, llm, db, before, res = full
    assert res["status"] == "awaiting_approval"
    assert len(res["anomalies"]) == 17
    assert res["total_lost"] == pytest.approx(2557.60)
    assert len(res["pending_approvals"]) == 9
    assert res["reply"]


def test_interrupt_before_any_write(full):
    svc, llm, db, before, res = full
    assert snapshot(db) == before
    assert all(a["status"] == "pending" for a in res["pending_approvals"])
    acts = audit_actions(svc)
    assert "proposal_created" in acts and "write_executed" not in acts


def test_graph_is_suspended_at_human_approval(full):
    svc, llm, *_ = full
    snap = svc._graph(llm).get_state(svc._cfg("t1"))
    assert snap.next == ("human_approval",)


def _writer(db, aid):
    return T.adjust_invoice if T.get_approval(db, aid)["action"] == "adjust_invoice" else T.apply_credit_memo


def test_no_write_without_approval_via_tools(full):
    svc, llm, db, before, res = full
    aid = res["pending_approvals"][0]["id"]
    with pytest.raises(PermissionError):
        _writer(db, aid)(db, aid, "mallory")
    assert snapshot(db) == before


def test_write_tool_rejects_approval_decided_by_non_approver(full):
    svc, llm, db, before, res = full
    aid = res["pending_approvals"][0]["id"]
    c = sqlite3.connect(db)
    c.execute("update approvals set status='approved', decided_role='analyst', decided_by='eve' where id=?", (aid,))
    c.commit()
    c.close()
    with pytest.raises(PermissionError):
        _writer(db, aid)(db, aid, "eve")
    assert snapshot(db) == before


def test_repeat_message_while_pending_does_not_rerun(full):
    svc, llm, db, before, res = full
    n = len(svc.list_pending_approvals())
    r2 = svc.send_message("t1", "run again", llm, "analyst", "alice")
    assert r2["status"] == "awaiting_approval"
    assert len(svc.list_pending_approvals()) == n
    assert snapshot(db) == before


def test_second_thread_does_not_duplicate_proposals(full):
    svc, llm, db, before, res = full
    svc.send_message("t2", "Investigate all customers", llm, "analyst", "alice")
    assert len(svc.list_pending_approvals()) == 9


def test_analyst_cannot_approve_and_is_audited(full):
    svc, llm, db, before, res = full
    aid = res["pending_approvals"][0]["id"]
    with pytest.raises(PermissionError):
        svc.decide_approval(aid, True, "analyst", "alice", llm)
    with pytest.raises(PermissionError):
        svc.decide_approval(aid, True, "analyst", "alice", None)
    assert snapshot(db) == before
    assert T.get_approval(db, aid)["status"] == "pending"
    denied = [r for r in svc.get_audit_log(1000) if r["action"] == "unauthorized_decision_attempt"]
    assert len(denied) == 2
    assert denied[0]["actor"] == "alice" and denied[0]["role"] == "analyst"
    assert denied[0]["detail"]["approval_id"] == aid


@pytest.mark.parametrize("role", ["", "admin", "ANALYST", None])
def test_unknown_roles_cannot_decide(full, role):
    svc, llm, db, before, res = full
    with pytest.raises(PermissionError):
        svc.decide_approval(res["pending_approvals"][0]["id"], True, role, "x", llm)
    assert snapshot(db) == before


def test_rejection_leaves_data_unchanged(full):
    svc, llm, db, before, res = full
    for a in res["pending_approvals"]:
        out = svc.decide_approval(a["id"], False, "approver", "bob", llm)
    assert out["decision"] == "denied"
    assert snapshot(db) == before
    assert out["status"] == "done" and svc.list_pending_approvals() == []
    acts = audit_actions(svc)
    assert acts.count("approval_denied") == 9 and "write_executed" not in acts


def test_approval_executes_write_and_fixes_leakage(full):
    svc, llm, db, before, res = full
    appr = {a["customer_id"]: a for a in res["pending_approvals"]}
    out = svc.decide_approval(appr[3]["id"], True, "approver", "bob", llm)
    assert out["decision"] == "executed"
    assert T.reconcile_customer(db, 3) == []
    assert snapshot(db) != before
    svc.decide_approval(appr[11]["id"], True, "approver", "bob", llm)
    assert T.reconcile_customer(db, 11) == []
    inv = T.get_invoices(db, 11, "2025-06")
    assert len(inv) == 1 and inv[0]["amount"] == 999.0
    out = svc.decide_approval(appr[15]["id"], True, "approver", "bob", llm)
    c = sqlite3.connect(db)
    memos = c.execute("select customer_id, amount, status, approval_id, created_by from credit_memos").fetchall()
    assert memos == [(15, 20.0, "applied", appr[15]["id"], "bob")]
    assert T.reconcile_customer(db, 15) == []
    assert len(out["pending_approvals"]) == 6


def test_cannot_decide_twice(full):
    svc, llm, db, before, res = full
    aid = res["pending_approvals"][0]["id"]
    svc.decide_approval(aid, True, "approver", "bob", llm)
    with pytest.raises(ValueError):
        svc.decide_approval(aid, True, "approver", "bob", llm)
    fn = T.adjust_invoice if T.get_approval(db, aid)["action"] == "adjust_invoice" else T.apply_credit_memo
    with pytest.raises(PermissionError):
        fn(db, aid, "bob")  # already executed -> no double write


def test_unknown_approval(svc, llm):
    with pytest.raises(ValueError):
        svc.decide_approval(9999, True, "approver", "bob", llm)


def test_all_approved_total_clean(full):
    svc, llm, db, before, res = full
    for a in res["pending_approvals"]:
        out = svc.decide_approval(a["id"], True, "approver", "bob", llm)
    assert out["status"] == "done"
    for c in T.list_customers(db):
        assert T.reconcile_customer(db, c["id"]) == [], c


def test_audit_log_completeness(full):
    svc, llm, db, before, res = full
    aid = res["pending_approvals"][0]["id"]
    svc.decide_approval(aid, True, "approver", "bob", llm)
    log = svc.get_audit_log(1000)
    acts = {r["action"] for r in log}
    for need in ("investigation_started", "tool:list_customers", "tool:reconcile_customer", "analysis_done",
                 "proposal_created", "approval_approved", "write_executed"):
        assert need in acts, need
    assert all(r["actor"] and r["role"] and r["ts"] for r in log)
    ids = [r["id"] for r in log]
    assert ids == sorted(ids, reverse=True)
    started = [r for r in log if r["action"] == "investigation_started"][0]
    assert started["actor"] == "alice" and started["role"] == "analyst"
    ex = [r for r in log if r["action"] == "write_executed"][0]
    assert ex["actor"] == "bob" and ex["role"] == "approver"
    assert len(svc.get_audit_log(3)) == 3


def test_scoped_investigation_and_memory_across_turns(svc, llm, db_path):
    r1 = svc.send_message("m", "Check customer 1 please", llm, "analyst", "alice")
    assert r1["status"] == "done" and r1["anomalies"] == []
    r2 = svc.send_message("m", "Now check customer 15", llm, "analyst", "alice")
    assert r2["status"] == "awaiting_approval" and len(r2["anomalies"]) == 1
    h = svc.get_history("m")
    assert h[0] == {"role": "user", "content": "Check customer 1 please"}
    assert h[1]["role"] == "assistant"
    assert any(m["content"] == "Now check customer 15" for m in h)
    mem = svc._graph(llm).get_state(svc._cfg("m")).values["memory"]
    assert len(mem) >= 1 and "customer 1" in mem[0].lower()


def test_memory_survives_restart_and_threads_isolated(svc, llm):
    svc.send_message("a", "Check customer 1", llm, "analyst", "alice")
    svc.send_message("b", "Check customer 2", llm, "analyst", "alice")
    for s in svc._SAVERS.values():
        s.conn.close()
    svc._SAVERS.clear()  # simulate process restart
    ha, hb = svc.get_history("a"), svc.get_history("b")
    assert "customer 1" in ha[0]["content"] and "customer 2" in hb[0]["content"]
    assert not any("customer 2" in m["content"].lower() for m in ha if m["role"] == "user")
    assert svc.get_history("never-used") == []


def test_input_validation(svc, llm):
    with pytest.raises(ValueError):
        svc.send_message("v", "   ", llm, "analyst", "alice")
    with pytest.raises(svc.AgentError):
        svc.send_message("v", "hi", None, "analyst", "alice")
    with pytest.raises(PermissionError):
        svc.send_message("v", "hi", llm, "hacker", "eve")
    assert "permission_denied" in audit_actions(svc)
