"""Smoke test: FAKE_LLM=1 python -m api.smoke_test"""
import os

os.environ["FAKE_LLM"] = "1"
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

c = TestClient(app, raise_server_exceptions=False)
A = {"X-User": "alice", "X-User-Role": "analyst"}
P = {"X-User": "bob", "X-User-Role": "approver"}
assert c.get("/health").json()["status"] == "ok"
assert c.get("/api/audit").status_code == 401
assert c.get("/api/audit", headers={"X-User": "x", "X-User-Role": "boss"}).status_code == 403
tid = c.post("/api/threads", headers=A).json()["thread_id"]
r = c.post(f"/api/threads/{tid}/messages", headers=A, json={"text": "Investigate revenue leakage in invoices"})
print("message", r.status_code, r.json())
assert r.status_code == 200
print("history", c.get(f"/api/threads/{tid}/history", headers=A).json())
ap = c.get("/api/approvals", params={"thread_id": tid}, headers=A).json()["approvals"]
print("approvals", ap)
if ap:
    aid = ap[0].get("approval_id") or ap[0].get("id")
    assert c.post(f"/api/approvals/{aid}/decision", headers=A, json={"approved": True}).status_code == 403
    r = c.post(f"/api/approvals/{aid}/decision", headers=P, json={"approved": True})
    print("decision", r.status_code, r.json())
print("audit", c.get("/api/audit", params={"limit": 10}, headers=P).json())
print("SMOKE OK")
