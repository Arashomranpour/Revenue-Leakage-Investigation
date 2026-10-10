"""FastAPI wrapper around agent/service.py.

Run: uvicorn api.main:app --port 8000

Security: the Groq key is read from the X-Groq-Api-Key header per request, used to
build an LLM for that request only, and never stored, cached, logged or echoed.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("api")

VALID_ROLES = ("analyst", "approver")

app = FastAPI(title="Revenue Leakage Investigation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _svc():
    """Import the service lazily so the API boots even before the agent exists."""
    from agent import service  # noqa: WPS433

    return service


# ---------- models ----------
class KeyValidateRequest(BaseModel):
    api_key: str = Field(min_length=1)


class KeyValidateResponse(BaseModel):
    valid: bool
    message: str


class ThreadCreated(BaseModel):
    thread_id: str


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class DecisionRequest(BaseModel):
    approved: bool


class Caller(BaseModel):
    actor: str
    role: str


# ---------- identity / llm dependencies ----------
def get_caller(
    x_user: Optional[str] = Header(default=None),
    x_user_role: Optional[str] = Header(default=None),
) -> Caller:
    role = (x_user_role or "").strip().lower()
    if not role:
        raise HTTPException(401, "Missing X-User-Role header (analyst|approver).")
    if role not in VALID_ROLES:
        raise HTTPException(403, "Invalid role. Use analyst or approver.")
    actor = (x_user or "").strip()
    if not actor:
        raise HTTPException(401, "Missing X-User header.")
    return Caller(actor=actor[:100], role=role)


def _build_llm(key: Optional[str]):
    """Build the LLM for this request only. The key is never kept."""
    key = (key or "").strip()
    if not key and os.environ.get("FAKE_LLM") != "1":
        raise HTTPException(401, "Missing X-Groq-Api-Key header.")
    return _svc().make_llm(key or None)


def get_llm(x_groq_api_key: Optional[str] = Header(default=None)):
    return _build_llm(x_groq_api_key)


# ---------- error handling ----------
def _json(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


def _provider_status(exc: BaseException) -> Optional[int]:
    """Classify provider (groq/openai-style) errors by class name; raw text is never returned."""
    for cls in type(exc).__mro__:
        name = cls.__name__
        if name == "RateLimitError":
            return 429
        if name in ("AuthenticationError", "PermissionDeniedError"):
            return 401
    return None


@app.exception_handler(HTTPException)
async def http_exc(_: Request, exc: HTTPException):
    return _json(exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exc(_: Request, exc: RequestValidationError):
    fields = ", ".join(".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]) for e in exc.errors())
    return _json(422, "validation_error", f"Invalid request: {fields}")


@app.exception_handler(PermissionError)
async def perm_exc(_: Request, exc: PermissionError):
    return _json(403, "forbidden", "You are not permitted to perform this action.")


@app.exception_handler(ValueError)
async def value_exc(_: Request, exc: ValueError):
    msg = str(exc).lower()
    if "not found" in msg or "unknown" in msg:
        return _json(404, "not_found", "Approval not found.")
    return _json(409, "conflict", "This approval has already been decided or the request conflicts with its state.")


@app.exception_handler(RuntimeError)
async def runtime_exc(request: Request, exc: RuntimeError):
    if type(exc).__name__ == "AgentError":  # message is user-safe
        return _json(502, "agent_error", str(exc))
    return await unhandled(request, exc)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    status = _provider_status(exc)
    if status == 429:
        return _json(429, "rate_limited", "The LLM provider rate limit was reached. Please wait a moment and retry.")
    if status == 401:
        return _json(401, "invalid_api_key", "The Groq API key was rejected. Check the key and try again.")
    # log only the exception type, never its message (may embed secrets)
    logger.error("unhandled error: %s", type(exc).__name__)
    return _json(500, "internal_error", "Something went wrong while processing the request.")


# ---------- routes ----------
@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "fake_llm": os.environ.get("FAKE_LLM") == "1"}


@app.post("/api/key/validate", response_model=KeyValidateResponse)
def validate_key(body: KeyValidateRequest):
    err = _svc().validate_api_key(body.api_key)
    if err:
        return KeyValidateResponse(valid=False, message=err)
    return KeyValidateResponse(valid=True, message="API key is valid.")


@app.post("/api/threads", response_model=ThreadCreated, status_code=201)
def create_thread(_: Caller = Depends(get_caller)):
    return ThreadCreated(thread_id=uuid.uuid4().hex)


@app.post("/api/threads/{thread_id}/messages")
def post_message(
    thread_id: str,
    body: MessageRequest,
    caller: Caller = Depends(get_caller),
    llm=Depends(get_llm),
):
    return _svc().send_message(thread_id, body.text, llm, caller.role, caller.actor)


@app.get("/api/threads/{thread_id}/history")
def history(thread_id: str, _: Caller = Depends(get_caller)):
    return {"thread_id": thread_id, "messages": _svc().get_history(thread_id)}


@app.get("/api/approvals")
def approvals(thread_id: Optional[str] = None, _: Caller = Depends(get_caller)):
    return {"approvals": _svc().list_pending_approvals(thread_id)}


@app.post("/api/approvals/{approval_id}/decision")
def decide(
    approval_id: str,
    body: DecisionRequest,
    caller: Caller = Depends(get_caller),
    x_groq_api_key: Optional[str] = Header(default=None),
):
    svc = _svc()
    if caller.role != "approver":
        try:  # service audits unauthorized_decision_attempt, then raises PermissionError
            svc.decide_approval(approval_id, body.approved, caller.role, caller.actor, None)
        except PermissionError:
            pass
        except Exception as e:
            logger.error("denial path: %s", type(e).__name__)
        raise HTTPException(403, "Only an approver may decide on approvals.")
    llm = _build_llm(x_groq_api_key)
    return svc.decide_approval(approval_id, body.approved, caller.role, caller.actor, llm)


@app.get("/api/audit")
def audit(limit: int = Query(50, ge=1, le=500), _: Caller = Depends(get_caller)):
    return {"entries": _svc().get_audit_log(limit)}
