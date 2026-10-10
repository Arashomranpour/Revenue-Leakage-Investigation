"""Smoke script (no pytest): `FAKE_LLM=1 python -m agent.smoke` from repo root. Uses temp DBs."""
import os
import tempfile

os.environ["FAKE_LLM"] = "1"
d = tempfile.mkdtemp()
os.environ["BILLING_DB_PATH"] = os.path.join(d, "b.db")
os.environ["CHECKPOINT_DB_PATH"] = os.path.join(d, "c.db")

from agent import service as S  # noqa: E402

assert S.validate_api_key("nope") and S.validate_api_key("gsk_x") is None
llm = S.make_llm("gsk_fake")
r = S.send_message("t1", "Investigate all customers for revenue leakage", llm, "analyst", "alice")
print(r["status"], len(r["anomalies"]), r["total_lost"], len(r["pending_approvals"]))
assert r["status"] == "awaiting_approval" and r["pending_approvals"]
import sqlite3  # noqa: E402
n_memos = lambda: sqlite3.connect(os.environ["BILLING_DB_PATH"]).execute(  # noqa: E731
    "select count(*) from credit_memos").fetchone()[0]
total_before = r["total_lost"]
assert abs(total_before - 3 * 50 - 40 - 50 - 2 * 39.8 - 2 * 60 - 3 * 30 - 20 - 999 - 999 - 10) < 0.01, total_before

try:
    S.decide_approval(r["pending_approvals"][0]["id"], True, "analyst", "alice", None)
    raise SystemExit("analyst was allowed!")
except PermissionError as e:
    print("denied analyst:", e)
r2 = S.send_message("t1", "again", llm, "analyst", "alice")
assert "awaiting" in r2["reply"]

ids = [p["id"] for p in r["pending_approvals"]]
d1 = S.decide_approval(ids[0], False, "approver", "bob", llm)
print("deny ->", d1["decision"], d1["status"])
d2 = S.decide_approval(ids[1], True, "approver", "bob", llm)
print("approve ->", d2["decision"], d2["status"])
for i in ids[2:]:
    last = S.decide_approval(i, True, "approver", "bob", llm)
assert last["status"] == "done", last["status"]
print(last["reply"][:300])
r3 = S.send_message("t1", "Re-check customer 9 and customer 15", llm, "analyst", "alice")
print("recheck:", r3["status"], [a["anomaly_id"] for a in r3["anomalies"]])
assert r3["status"] == "done" and not r3["anomalies"], r3["anomalies"]
log = S.get_audit_log(500)
acts = {x["action"] for x in log}
print(sorted(acts))
assert {"unauthorized_decision_attempt", "approval_denied", "approval_approved", "write_executed"} <= acts
assert len(S.get_history("t1")) == 4
print("OK")
