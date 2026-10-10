"""Live smoke test: boots uvicorn + streamlit as real processes (FAKE_LLM=1, temp DBs, spare ports)
and checks their health endpoints plus one HTTP investigation round-trip. Offline."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.e2e


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code == 200:
                return r
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise AssertionError(f"{url} did not become healthy in {timeout}s")


@pytest.fixture(scope="module")
def servers(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("live")
    api_port, ui_port = free_port(), free_port()
    env = {**os.environ, "FAKE_LLM": "1", "AGENT_RETRY_WAIT": "0",
           "BILLING_DB_PATH": str(tmp / "billing.db"), "CHECKPOINT_DB_PATH": str(tmp / "ck.db"),
           "API_URL": f"http://127.0.0.1:{api_port}", "PYTHONUNBUFFERED": "1"}
    env.pop("GROQ_API_KEY", None)
    logs = [open(tmp / "api.log", "wb"), open(tmp / "ui.log", "wb")]
    procs = [
        subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(api_port),
                          "--host", "127.0.0.1"], cwd=ROOT, env=env, stdout=logs[0], stderr=subprocess.STDOUT),
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.port", str(ui_port),
                          "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                         cwd=ROOT, env=env, stdout=logs[1], stderr=subprocess.STDOUT),
    ]
    try:
        yield {"api": f"http://127.0.0.1:{api_port}", "ui": f"http://127.0.0.1:{ui_port}", "tmp": tmp}
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in logs:
            f.close()


def test_api_health(servers):
    r = wait_for(servers["api"] + "/health")
    assert r.json()["status"] == "ok" and r.json()["fake_llm"] is True


def test_streamlit_health(servers):
    r = wait_for(servers["ui"] + "/_stcore/health")
    assert r.text.strip() == "ok"


def test_docs_page(servers):
    wait_for(servers["api"] + "/health")
    assert httpx.get(servers["api"] + "/docs").status_code == 200


def test_http_roundtrip_against_live_api(servers):
    wait_for(servers["api"] + "/health")
    h = {"X-User": "alice", "X-User-Role": "analyst", "X-Groq-Api-Key": "gsk_live_smoke_key_123456"}
    tid = httpx.post(servers["api"] + "/api/threads", headers=h).json()["thread_id"]
    r = httpx.post(servers["api"] + f"/api/threads/{tid}/messages", json={"text": "Check customer 15"},
                   headers=h, timeout=120)
    assert r.status_code == 200 and r.json()["status"] == "awaiting_approval"
    assert "gsk_live_smoke_key_123456" not in r.text
    aid = r.json()["pending_approvals"][0]["id"]
    d = httpx.post(servers["api"] + f"/api/approvals/{aid}/decision", json={"approved": True}, headers=h)
    assert d.status_code == 403


def test_streamlit_serves_page_and_logs_have_no_key(servers):
    wait_for(servers["ui"] + "/_stcore/health")
    assert httpx.get(servers["ui"] + "/").status_code == 200
    time.sleep(1)
    for name in ("api.log", "ui.log"):
        assert b"gsk_live_smoke_key" not in (servers["tmp"] / name).read_bytes()
