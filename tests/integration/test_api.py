"""FastAPI integration tests via TestClient (FAKE_LLM=1, temp DBs, no network)."""
import logging

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

SECRET = "gsk_IntegrationSecretKey9876543210zzzz"
ANALYST = {"X-User": "alice", "X-User-Role": "analyst"}
APPROVER = {"X-User": "bob", "X-User-Role": "approver"}
KEY = {"X-Groq-Api-Key": SECRET}


@pytest.fixture
def client(svc):
    from api.main import app
    return TestClient(app, raise_server_exceptions=False)


def hdr(*ds):
    out = {}
    for d in ds:
        out.update(d)
    return out


def new_thread(client):
    r = client.post("/api/threads", headers=ANALYST)
    assert r.status_code == 201
    return r.json()["thread_id"]


def investigate(client, tid, text="Investigate all customers"):
    return client.post(f"/api/threads/{tid}/messages", json={"text": text}, headers=hdr(ANALYST, KEY))


# ---------------------------------------------------------------- basic routes
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["fake_llm"] is True


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/health", "/api/key/validate", "/api/threads", "/api/threads/{thread_id}/messages",
              "/api/approvals", "/api/approvals/{approval_id}/decision", "/api/audit"):
        assert p in paths


def test_key_validate(client):
    ok = client.post("/api/key/validate", json={"api_key": SECRET})
    assert ok.status_code == 200 and ok.json()["valid"] is True
    bad = client.post("/api/key/validate", json={"api_key": "sk-nope"})
    assert bad.status_code == 200 and bad.json()["valid"] is False and bad.json()["message"]
    assert client.post("/api/key/validate", json={"api_key": ""}).status_code == 422
    assert client.post("/api/key/validate", json={}).status_code == 422


def test_key_validate_does_not_echo_key(client):
    for k in (SECRET, "bad" + SECRET):
        r = client.post("/api/key/validate", json={"api_key": k})
        assert k not in r.text and k not in str(r.headers)


# ---------------------------------------------------------------- identity headers
@pytest.mark.parametrize("method,url", [("post", "/api/threads"), ("get", "/api/approvals"),
                                        ("get", "/api/audit"), ("get", "/api/threads/x/history")])
def test_missing_identity_is_401(client, method, url):
    r = getattr(client, method)(url)
    assert r.status_code == 401 and set(r.json()) == {"error", "detail"}


def test_missing_user_header_401_and_bad_role_403(client):
    assert client.get("/api/audit", headers={"X-User-Role": "analyst"}).status_code == 401
    assert client.get("/api/audit", headers={"X-User": "a", "X-User-Role": "admin"}).status_code == 403


def test_role_header_case_insensitive(client):
    assert client.get("/api/audit", headers={"X-User": "a", "X-User-Role": "ANALYST"}).status_code == 200


def test_message_requires_groq_header_when_not_fake(client, monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "0")
    r = client.post("/api/threads/t/messages", json={"text": "hi"}, headers=ANALYST)
    assert r.status_code == 401 and "Groq" in r.json()["detail"]


# ---------------------------------------------------------------- investigation + approval flow
def test_full_flow_over_http(client, db_path):
    tid = new_thread(client)
    r = investigate(client, tid)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "awaiting_approval" and len(body["anomalies"]) == 17
    assert body["total_lost"] == pytest.approx(2557.60) and len(body["pending_approvals"]) == 9

    pend = client.get("/api/approvals", params={"thread_id": tid}, headers=ANALYST).json()["approvals"]
    assert len(pend) == 9
    hist = client.get(f"/api/threads/{tid}/history", headers=ANALYST).json()["messages"]
    assert hist[0]["role"] == "user"

    aid = pend[0]["id"]
    # analyst denied: 403 + audit entry, approval still pending
    d = client.post(f"/api/approvals/{aid}/decision", json={"approved": True}, headers=hdr(ANALYST, KEY))
    assert d.status_code == 403
    assert d.json()["error"] and d.json()["detail"]
    audit = client.get("/api/audit", params={"limit": 500}, headers=ANALYST).json()["entries"]
    att = [e for e in audit if e["action"] == "unauthorized_decision_attempt"]
    assert att and att[0]["actor"] == "alice" and att[0]["role"] == "analyst"
    assert len(client.get("/api/approvals", headers=ANALYST).json()["approvals"]) == 9

    # approver approves
    ok = client.post(f"/api/approvals/{aid}/decision", json={"approved": True}, headers=hdr(APPROVER, KEY))
    assert ok.status_code == 200 and ok.json()["decision"] == "executed"
    assert len(client.get("/api/approvals", headers=ANALYST).json()["approvals"]) == 8
    # deciding again -> 409
    again = client.post(f"/api/approvals/{aid}/decision", json={"approved": True}, headers=hdr(APPROVER, KEY))
    assert again.status_code == 409
    # unknown -> 404 ; analyst on unknown still 403
    assert client.post("/api/approvals/99999/decision", json={"approved": True},
                       headers=hdr(APPROVER, KEY)).status_code == 404
    # reject another
    aid2 = pend[1]["id"]
    rj = client.post(f"/api/approvals/{aid2}/decision", json={"approved": False}, headers=hdr(APPROVER, KEY))
    assert rj.status_code == 200 and rj.json()["decision"] == "denied"
    actions = [e["action"] for e in client.get("/api/audit", params={"limit": 500}, headers=ANALYST).json()["entries"]]
    assert "write_executed" in actions and "approval_denied" in actions


def test_non_numeric_approval_id_does_not_500(client):
    r = client.post("/api/approvals/abc/decision", json={"approved": True}, headers=hdr(APPROVER, KEY))
    assert r.status_code in (404, 409, 422)


def test_approver_decision_needs_key_when_not_fake(client, monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "0")
    r = client.post("/api/approvals/1/decision", json={"approved": True}, headers=APPROVER)
    assert r.status_code == 401


def test_analyst_decision_is_403_without_key(client):
    r = client.post("/api/approvals/1/decision", json={"approved": True}, headers=ANALYST)
    assert r.status_code == 403


def test_audit_limit_validation_and_order(client):
    new_thread(client)
    investigate(client, "tt")
    assert client.get("/api/audit", params={"limit": 0}, headers=ANALYST).status_code == 422
    assert client.get("/api/audit", params={"limit": 501}, headers=ANALYST).status_code == 422
    e = client.get("/api/audit", params={"limit": 5}, headers=ANALYST).json()["entries"]
    assert len(e) == 5 and [x["id"] for x in e] == sorted((x["id"] for x in e), reverse=True)


def test_message_validation(client):
    assert client.post("/api/threads/t/messages", json={"text": ""}, headers=hdr(ANALYST, KEY)).status_code == 422
    assert client.post("/api/threads/t/messages", json={}, headers=hdr(ANALYST, KEY)).status_code == 422
    big = client.post("/api/threads/t/messages", json={"text": "x" * 8001}, headers=hdr(ANALYST, KEY))
    assert big.status_code == 422
    r = client.post("/api/threads/t/messages", json={"text": "   "}, headers=hdr(ANALYST, KEY))
    assert r.status_code in (400, 409, 422)


def test_thread_ids_unique(client):
    assert len({new_thread(client) for _ in range(5)}) == 5


def test_memory_across_http_turns(client):
    tid = new_thread(client)
    investigate(client, tid, "Check customer 1")
    investigate(client, tid, "Check customer 2")
    h = client.get(f"/api/threads/{tid}/history", headers=ANALYST).json()["messages"]
    assert [m["role"] for m in h].count("user") == 2 and len(h) == 4


# ---------------------------------------------------------------- error mapping
def _raiser(exc):
    def f(*a, **k):
        raise exc
    return f


def _resp(status):
    return httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/x"))


def test_rate_limit_maps_to_429(client, svc, monkeypatch):
    monkeypatch.setattr(svc, "send_message", _raiser(groq.RateLimitError(f"limit {SECRET}", response=_resp(429), body=None)))
    r = client.post("/api/threads/t/messages", json={"text": "hi"}, headers=hdr(ANALYST, KEY))
    assert r.status_code == 429 and r.json()["error"] == "rate_limited"
    assert SECRET not in r.text


def test_auth_error_maps_to_401(client, svc, monkeypatch):
    monkeypatch.setattr(svc, "send_message", _raiser(groq.AuthenticationError(f"bad {SECRET}", response=_resp(401), body=None)))
    r = client.post("/api/threads/t/messages", json={"text": "hi"}, headers=hdr(ANALYST, KEY))
    assert r.status_code == 401 and r.json()["error"] == "invalid_api_key" and SECRET not in r.text


def test_agent_error_maps_to_502(client, svc, monkeypatch):
    monkeypatch.setattr(svc, "send_message", _raiser(svc.AgentError("Could not reach Groq.")))
    r = client.post("/api/threads/t/messages", json={"text": "hi"}, headers=hdr(ANALYST, KEY))
    assert r.status_code == 502 and r.json()["error"] == "agent_error"


def test_permission_error_maps_to_403(client, svc, monkeypatch):
    monkeypatch.setattr(svc, "send_message", _raiser(PermissionError("nope")))
    assert client.post("/api/threads/t/messages", json={"text": "hi"}, headers=hdr(ANALYST, KEY)).status_code == 403


def test_unhandled_error_is_500_and_leaks_nothing(client, svc, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(svc, "send_message", _raiser(RuntimeError(f"db exploded {SECRET}")))
    r = client.post("/api/threads/t/messages", json={"text": "hi"}, headers=hdr(ANALYST, KEY))
    assert r.status_code == 500 and r.json()["error"] == "internal_error"
    assert SECRET not in r.text and SECRET not in caplog.text and "exploded" not in r.text


# ---------------------------------------------------------------- key never echoed / stored
def test_key_not_echoed_or_persisted(client, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    tid = new_thread(client)
    r = investigate(client, tid, "Investigate customer 15")
    aid = r.json()["pending_approvals"][0]["id"]
    d = client.post(f"/api/approvals/{aid}/decision", json={"approved": True}, headers=hdr(APPROVER, KEY))
    outputs = [r, d, client.get("/api/audit", params={"limit": 500}, headers=ANALYST),
               client.get(f"/api/threads/{tid}/history", headers=ANALYST),
               client.get("/api/approvals", headers=ANALYST)]
    for o in outputs:
        assert SECRET not in o.text
        assert SECRET not in str(dict(o.headers))
    assert SECRET not in caplog.text
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert SECRET.encode() not in f.read_bytes(), f


def test_cors_only_allows_streamlit_origin(client):
    ok = client.options("/health", headers={"Origin": "http://localhost:8501",
                                            "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8501"
    bad = client.options("/health", headers={"Origin": "http://evil.example",
                                             "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers
