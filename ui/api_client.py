"""Thin HTTP client for the Revenue Leakage API. All fetch calls live here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.environ.get("API_TIMEOUT", "120"))


class ApiError(Exception):
    """Error with a user-safe message (never a raw provider error)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.status = status


def friendly_message(status: int, detail: str = "") -> str:
    if status == 401:
        return "Your Groq API key was rejected. Check it, or forget it and connect again."
    if status == 403:
        return "You do not have permission for this action. Switch to the 'approver' role in the sidebar."
    if status == 404:
        return "That item was not found. It may belong to an older investigation."
    if status == 429:
        return "Rate limit reached. Please wait a moment and try again."
    if status == 409:
        return "This request was already decided or conflicts with the current state. Refresh to see the latest."
    if status == 422:
        return "The request was not valid. Please check your input."
    if status >= 500:
        return "The service hit a problem processing this request. Please try again."
    return "Something went wrong. Please try again."


@dataclass
class Session:
    api_key: str = ""
    user: str = "analyst1"
    role: str = "analyst"

    def headers(self) -> dict[str, str]:
        h = {"X-User": self.user or "unknown", "X-User-Role": self.role}
        if self.api_key:
            h["X-Groq-Api-Key"] = self.api_key
        return h


def _request(method: str, path: str, sess: Session | None = None,
             json: dict | None = None, params: dict | None = None) -> Any:
    headers = sess.headers() if sess else {}
    try:
        r = httpx.request(method, f"{API_URL}{path}", headers=headers, json=json,
                          params=params, timeout=TIMEOUT)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise ApiError(f"Cannot reach the API at {API_URL}. Is the backend running?") from None
    except httpx.TimeoutException:
        raise ApiError("The request timed out. Please try again.") from None
    except httpx.HTTPError:
        raise ApiError("Network error while contacting the API.") from None
    if r.status_code >= 400:
        detail = ""
        try:
            body = r.json()
            detail = body.get("detail", "") if isinstance(body, dict) else ""
        except Exception:
            pass
        raise ApiError(friendly_message(r.status_code, str(detail)), r.status_code)
    try:
        return r.json()
    except Exception:
        raise ApiError("The API returned an unexpected response.", r.status_code) from None


def health() -> dict:
    return _request("GET", "/health")


def validate_key(sess: Session, api_key: str) -> dict:
    return _request("POST", "/api/key/validate", sess, json={"api_key": api_key})


def create_thread(sess: Session) -> dict:
    return _request("POST", "/api/threads", sess)


def send_message(sess: Session, thread_id: str, text: str) -> dict:
    return _request("POST", f"/api/threads/{thread_id}/messages", sess, json={"text": text})


def get_history(sess: Session, thread_id: str) -> dict:
    return _request("GET", f"/api/threads/{thread_id}/history", sess)


def list_approvals(sess: Session, thread_id: str | None = None) -> Any:
    params = {"thread_id": thread_id} if thread_id else None
    return _request("GET", "/api/approvals", sess, params=params)


def decide_approval(sess: Session, approval_id: str, approved: bool) -> dict:
    return _request("POST", f"/api/approvals/{approval_id}/decision", sess,
                    json={"approved": approved})


def get_audit(sess: Session, limit: int = 100) -> Any:
    return _request("GET", "/api/audit", sess, params={"limit": limit})
