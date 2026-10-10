"""Shared fixtures. All tests are offline: FAKE_LLM=1 and no real Groq key."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Never let a real key leak into tests; force deterministic LLM.
os.environ["FAKE_LLM"] = "1"
os.environ.pop("GROQ_API_KEY", None)

VALID_KEY = "gsk_" + "A1b2C3d4E5f6G7h8I9j0" * 2  # fake, well-formed


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    yield


@pytest.fixture
def valid_key():
    return VALID_KEY


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """agent.service bound to throwaway billing + checkpoint DBs (offline, FAKE_LLM=1)."""
    monkeypatch.setenv("BILLING_DB_PATH", str(tmp_path / "billing.db"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.db"))
    from agent import graph, service

    service._SAVERS.clear()
    service._MODEL_CACHE.clear()
    monkeypatch.setattr(graph, "RETRY_WAIT", 0.0)
    service._db()  # create + seed
    yield service
    for s in list(service._SAVERS.values()):
        try:
            s.conn.close()
        except Exception:
            pass
    service._SAVERS.clear()


@pytest.fixture
def llm(svc):
    return svc.make_llm("gsk_" + "x" * 30)


@pytest.fixture
def db_path(svc, tmp_path):
    return str(tmp_path / "billing.db")
