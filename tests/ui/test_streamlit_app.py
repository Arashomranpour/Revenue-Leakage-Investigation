"""Streamlit AppTest tests. HTTP calls from ui/api_client are routed in-process to the FastAPI app
via TestClient (FAKE_LLM=1, temp DBs), so no servers and no network are needed."""
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "ui" / "app.py")
KEY = "gsk_UiTestKey1234567890abcdefghijkl"


@pytest.fixture
def api_routed(svc, monkeypatch):
    from api.main import app
    tc = TestClient(app, raise_server_exceptions=False)

    def fake_request(method, url, headers=None, json=None, params=None, timeout=None, **kw):
        path = url.split("://", 1)[1].split("/", 1)[1]
        return tc.request(method, "/" + path, headers=headers, json=json, params=params)

    monkeypatch.setattr(httpx, "request", fake_request)
    return tc


def fresh():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def connect(at, key=KEY):
    at.text_input(key="key_input").set_value(key)
    next(b for b in at.button if b.label == "Connect").click()
    at.run()
    return at


def test_gate_blocks_chat_even_with_env_key(api_routed, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    at = fresh()
    assert not at.exception
    assert len(at.chat_input) == 0
    assert not at.session_state["connected"]
    assert any("Connect your Groq API key" in i.value for i in at.info)
    assert len(at.tabs) == 0


def test_key_field_is_password(api_routed):
    at = fresh()
    ti = at.text_input(key="key_input")
    assert ti.proto.type == 1  # PASSWORD


@pytest.mark.parametrize("bad", ["", "   ", "sk-123", "hello world"])
def test_malformed_key_rejected(api_routed, bad):
    at = fresh()
    connect(at, bad)
    assert not at.session_state["connected"]
    assert any("gsk_" in e.value for e in at.error)
    assert len(at.chat_input) == 0
    assert at.session_state["api_key"] == ""


def test_valid_key_connects_and_unlocks_chat(api_routed):
    at = connect(fresh())
    assert at.session_state["connected"] and at.session_state["api_key"] == KEY
    assert len(at.chat_input) == 1
    assert [t.label for t in at.tabs] == ["Investigation", "Audit log"]
    # key field is gone after connecting (not displayed anywhere)
    assert all(KEY not in str(getattr(e, "value", "")) for e in at.markdown)
    assert all(getattr(t, "value", "") != KEY for t in at.text_input)


def test_forget_key_relocks(api_routed):
    at = connect(fresh())
    at.button(key="forget_key").click().run()
    assert not at.session_state["connected"] and at.session_state["api_key"] == ""
    assert len(at.chat_input) == 0


def test_investigation_flow_and_role_gating(api_routed):
    at = connect(fresh())
    at.chat_input(key="chat_input").set_value("Investigate all customers").run()
    assert not at.exception
    assert at.session_state["status"] == "awaiting_approval"
    assert len(at.session_state["anomalies"]) == 17
    assert at.session_state["total_lost"] == pytest.approx(2557.60)
    assert any(m.label == "Total revenue lost" and m.value == "$2,557.60" for m in at.metric)

    approve = [b for b in at.button if (b.key or "").startswith("approve_")]
    reject = [b for b in at.button if (b.key or "").startswith("reject_")]
    assert len(approve) == 9 and len(reject) == 9
    assert all(b.disabled for b in approve + reject)  # analyst role
    assert any("approver role" in w.value for w in at.warning)

    # switch to approver: buttons enable and approval works
    at.radio(key="role").set_value("approver").run()
    approve = [b for b in at.button if (b.key or "").startswith("approve_")]
    assert all(not b.disabled for b in approve)
    approve[0].click().run()
    assert not at.exception
    assert len([b for b in at.button if (b.key or "").startswith("approve_")]) == 8
    api_audit = api_routed.get("/api/audit", params={"limit": 500},
                               headers={"X-User": "x", "X-User-Role": "analyst"}).json()["entries"]
    assert any(e["action"] == "write_executed" for e in api_audit)


def test_reject_leaves_one_less_pending(api_routed):
    at = connect(fresh())
    at.chat_input(key="chat_input").set_value("Check customer 15").run()
    at.radio(key="role").set_value("approver").run()
    [b for b in at.button if (b.key or "").startswith("reject_")][0].click().run()
    assert not [b for b in at.button if (b.key or "").startswith("reject_")]


def test_new_investigation_resets(api_routed):
    at = connect(fresh())
    at.chat_input(key="chat_input").set_value("Check customer 1").run()
    assert at.session_state["thread_id"]
    at.button(key="new_investigation").click().run()
    assert at.session_state["thread_id"] is None and at.session_state["messages"] == []
    assert at.session_state["connected"]  # key kept


def test_audit_tab_lists_entries(api_routed):
    at = connect(fresh())
    at.chat_input(key="chat_input").set_value("Check customer 15").run()
    at.button(key="audit_refresh").click().run()
    assert not at.exception
    assert len(at.dataframe) >= 1


def test_key_not_rendered_anywhere(api_routed):
    at = connect(fresh())
    at.chat_input(key="chat_input").set_value("Check customer 15").run()
    rendered = " ".join(str(getattr(e, "value", "")) for e in
                        list(at.markdown) + list(at.text) + list(at.caption) + list(at.error) + list(at.info))
    assert KEY not in rendered


def test_api_offline_shows_friendly_error(svc, monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "request", boom)
    at = fresh()
    assert not at.exception
    connect(at)
    assert not at.session_state["connected"]
    assert any("Cannot reach the API" in e.value for e in at.error)
