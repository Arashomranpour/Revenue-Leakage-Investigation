"""API-key hygiene, friendly errors, retry handling. Fully mocked: no network."""
import logging
import re
import os
import sqlite3
from pathlib import Path

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage

from agent import graph as G

SECRET = "gsk_SuperSecretTestKey1234567890abcdef"


def _resp(status):
    return httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/x"))


def groq_err(cls, status):
    return cls("boom", response=_resp(status), body=None)


# ------------------------------------------------------------------ friendly errors
@pytest.mark.parametrize("exc,needle", [
    (groq_err(groq.AuthenticationError, 401), "rejected"),
    (groq_err(groq.PermissionDeniedError, 403), "permission"),
    (groq_err(groq.RateLimitError, 429), "rate limit"),
    (groq_err(groq.NotFoundError, 404), "not available"),
    (groq_err(groq.BadRequestError, 400), "bad request"),
    (groq_err(groq.InternalServerError, 500), "HTTP 500"),
    (groq.APIConnectionError(request=httpx.Request("POST", "https://x")), "reach Groq"),
    (groq.APITimeoutError(request=httpx.Request("POST", "https://x")), "reach Groq"),
])
def test_friendly_error_mapping(svc, exc, needle):
    msg = svc.friendly_error(exc)
    assert needle.lower() in msg.lower()
    assert "Traceback" not in msg and "boom" not in msg


def test_friendly_error_unexpected_hides_details(svc):
    msg = svc.friendly_error(RuntimeError(f"secret {SECRET}"))
    assert SECRET not in msg


def test_send_message_wraps_llm_failure_in_agent_error_and_audits(svc, monkeypatch):
    class Boom:
        def invoke(self, *_a, **_k):
            raise groq_err(groq.AuthenticationError, 401)

        def bind_tools(self, *_a, **_k):
            return self

    with pytest.raises(svc.AgentError) as ei:
        svc.send_message("e1", "check customer 1", Boom(), "analyst", "alice")
    assert "rejected" in str(ei.value)
    assert any(r["action"] == "error" for r in svc.get_audit_log(50))


# ------------------------------------------------------------------ validate_api_key
@pytest.mark.parametrize("bad", ["", "   ", None, "sk-abc", "hello", 123])
def test_malformed_key_rejected(svc, bad):
    err = svc.validate_api_key(bad)
    assert isinstance(err, str) and err


def test_wellformed_key_ok_in_fake_mode(svc):
    assert svc.validate_api_key(SECRET) is None


def test_validate_real_path_auth_error(svc, monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "0")

    class FakeGroq:
        def __init__(self, **kw):
            self.models = self

        def list(self):
            raise groq_err(groq.AuthenticationError, 401)

    monkeypatch.setattr(groq, "Groq", FakeGroq)
    err = svc.validate_api_key(SECRET)
    assert "rejected" in err and SECRET not in err


def test_validate_real_path_success_picks_preferred_model(svc, monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "0")

    class M:
        def __init__(self, i):
            self.id = i

    class FakeGroq:
        def __init__(self, **kw):
            self.models = self

        def list(self):
            return type("R", (), {"data": [M("whisper-large"), M("llama-3.1-8b-instant"), M("openai/gpt-oss-20b")]})()

    monkeypatch.setattr(groq, "Groq", FakeGroq)
    assert svc.validate_api_key(SECRET) is None
    assert svc._pick_model(FakeGroq()) == "openai/gpt-oss-20b"


def test_make_llm_rejects_malformed_key_without_fake(svc, monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "0")
    with pytest.raises(svc.AgentError):
        svc.make_llm("not-a-key")


# ------------------------------------------------------------------ key never persisted
def test_key_not_persisted_anywhere(svc, tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    llm = svc.make_llm(SECRET)
    assert svc.validate_api_key(SECRET) is None
    res = svc.send_message("k1", "Investigate customer 15", llm, "analyst", "alice")
    svc.decide_approval(res["pending_approvals"][0]["id"], True, "approver", "bob", llm)
    # 1. databases + any file the run created
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert SECRET.encode() not in f.read_bytes(), f
    # 2. audit log / history / results
    assert SECRET not in str(svc.get_audit_log(1000))
    assert SECRET not in str(svc.get_history("k1")) and SECRET not in str(res)
    # 3. logs, env, model cache
    assert SECRET not in caplog.text
    assert not any(SECRET in v for v in os.environ.values())
    assert SECRET not in repr(svc._MODEL_CACHE)
    # checkpoint blob (serialized graph state) contains no key
    c = sqlite3.connect(tmp_path / "checkpoints.db")
    for (blob,) in c.execute("select checkpoint from checkpoints"):
        assert SECRET.encode() not in blob


def test_real_path_key_only_goes_to_client_and_not_cached(svc, monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "0")
    seen = {}

    class M:
        id = "llama-3.1-8b-instant"

    class FakeGroq:
        def __init__(self, **kw):
            self.models = self

        def list(self):
            return type("R", (), {"data": [M()]})()

    class FakeChat:
        def __init__(self, **kw):
            seen.update(kw)

    import langchain_groq
    monkeypatch.setattr(groq, "Groq", FakeGroq)
    monkeypatch.setattr(langchain_groq, "ChatGroq", FakeChat)
    svc.make_llm(SECRET)
    assert seen["api_key"] == SECRET
    assert SECRET not in repr(svc._MODEL_CACHE)
    assert not any(SECRET in str(v) for v in svc._SAVERS)


def test_key_not_in_source_or_data_files():
    root = Path(__file__).resolve().parents[2]
    for p in list((root / "agent").glob("*.py")) + list((root / "api").glob("*.py")) + list((root / "ui").glob("*.py")):
        assert not re.search(r"gsk_[A-Za-z0-9]{20,}", p.read_text(encoding="utf-8")), f"hard-coded key in {p}"
    gi = (root / ".gitignore").read_text()
    assert ".env" in gi and ".claude/" in gi


# ------------------------------------------------------------------ retries
class Flaky:
    def __init__(self, fails, exc):
        self.calls, self.fails, self.exc = 0, fails, exc

    def invoke(self, *_a, **_k):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc
        return AIMessage(content="ok")


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    monkeypatch.setattr(G, "RETRY_WAIT", 0.0)
    monkeypatch.setattr(G, "RETRY_ATTEMPTS", 4)


@pytest.mark.parametrize("exc", [
    groq_err(groq.RateLimitError, 429), groq_err(groq.InternalServerError, 500),
    groq.APIConnectionError(request=httpx.Request("POST", "https://x")),
    TimeoutError("t"), ConnectionError("c"), sqlite3.OperationalError("database is locked"),
])
def test_transient_errors_are_retried(exc):
    f = Flaky(2, exc)
    assert G.call_llm(f, []).content == "ok"
    assert f.calls == 3


def test_retry_gives_up_after_max_attempts():
    f = Flaky(99, TimeoutError("t"))
    with pytest.raises(TimeoutError):
        G.call_llm(f, [])
    assert f.calls == 4


@pytest.mark.parametrize("exc", [groq_err(groq.AuthenticationError, 401), ValueError("bad"),
                                 groq_err(groq.BadRequestError, 400)])
def test_non_transient_errors_not_retried(exc):
    f = Flaky(99, exc)
    with pytest.raises(type(exc)):
        G.call_llm(f, [])
    assert f.calls == 1


def test_tool_call_retries_on_locked_db():
    class Tool:
        n = 0

        def invoke(self, args):
            Tool.n += 1
            if Tool.n < 3:
                raise sqlite3.OperationalError("database is locked")
            return [1]

    assert G.call_tool(Tool(), {}) == [1] and Tool.n == 3


def test_workflow_recovers_from_transient_llm_failure(svc, llm):
    class Wrap:
        def __init__(self, inner):
            self.inner, self.n = inner, 0

        def bind_tools(self, t):
            return Wrap(self.inner.bind_tools(t)) if False else self

        def invoke(self, msgs, *a, **k):
            self.n += 1
            if self.n == 1:
                raise groq_err(groq.RateLimitError, 429)
            return self.inner.invoke(msgs, *a, **k)

    w = Wrap(llm)
    res = svc.send_message("r1", "Check customer 1", w, "analyst", "alice")
    assert res["status"] == "done" and w.n > 1
