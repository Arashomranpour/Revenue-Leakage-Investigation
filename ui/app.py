"""Revenue Leakage Investigation Agent - Streamlit UI.

Run: streamlit run ui/app.py --server.port 8501
The Groq key lives ONLY in st.session_state (bring-your-own-key); an env key never bypasses the gate.
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api_client as api  # noqa: E402

st.set_page_config(page_title="Revenue Leakage Investigator", page_icon=":mag:", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); border-radius: 10px; padding: 12px 16px;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600;}
.pill-ok {background:#d1fae5; color:#065f46;} .pill-warn {background:#fef3c7; color:#92400e;}
.pill-bad {background:#fee2e2; color:#991b1b;}
</style>
""",
    unsafe_allow_html=True,
)

ss = st.session_state
ss.setdefault("api_key", "")
ss.setdefault("connected", False)
ss.setdefault("thread_id", None)
ss.setdefault("messages", [])      # [{role, content}]
ss.setdefault("anomalies", [])
ss.setdefault("total_lost", 0.0)
ss.setdefault("status", "")
ss.setdefault("user", "analyst1")
ss.setdefault("role", "analyst")


def sess() -> api.Session:
    return api.Session(api_key=ss.api_key, user=ss.user, role=ss.role)


def money(v) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return str(v)


def forget_key():
    ss.api_key = ""
    ss.connected = False
    ss.thread_id = None
    ss.messages = []
    ss.anomalies = []
    ss.total_lost = 0.0
    ss.status = ""


def new_investigation():
    ss.thread_id = None
    ss.messages = []
    ss.anomalies = []
    ss.total_lost = 0.0
    ss.status = ""


def ensure_thread():
    if not ss.thread_id:
        data = api.create_thread(sess())
        ss.thread_id = data.get("thread_id") or data.get("id")
    return ss.thread_id


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("Revenue Leakage")
    st.caption("Investigation agent")

    st.subheader("Groq API key")
    if not ss.connected:
        with st.form("key_form", clear_on_submit=True):
            key_in = st.text_input("Groq API key", type="password", placeholder="gsk_...",
                                   key="key_input")
            submitted = st.form_submit_button("Connect", width="stretch")
        st.markdown("Get a free key at [console.groq.com/keys](https://console.groq.com/keys).")
        st.caption("Your key is kept only in this browser session's memory. It is never stored "
                   "or logged, and it is gone when you close the tab or click Forget key.")
        if submitted:
            key = (key_in or "").strip()
            if not key.startswith("gsk_"):
                st.error("That does not look like a Groq key. It should start with gsk_.")
            else:
                try:
                    api.validate_key(api.Session(api_key=key, user=ss.user, role=ss.role), key)
                    ss.api_key = key
                    ss.connected = True
                    st.rerun()
                except api.ApiError as e:
                    st.error(e.message)
    else:
        st.markdown('<span class="pill pill-ok">Connected</span>', unsafe_allow_html=True)
        st.caption("Key held in session memory only.")
        st.button("Forget key", on_click=forget_key, width="stretch", key="forget_key")

    st.divider()
    st.subheader("Identity")
    st.text_input("Your name", key="user")
    st.radio("Role", ["analyst", "approver"], key="role", horizontal=True,
             help="Only approvers can approve or reject proposed actions.")

    st.divider()
    st.button("New investigation", on_click=new_investigation, width="stretch",
              key="new_investigation", disabled=not ss.connected)
    if ss.thread_id:
        st.caption(f"Thread: {ss.thread_id}")
    try:
        api.health()
        st.markdown('<span class="pill pill-ok">API online</span>', unsafe_allow_html=True)
    except api.ApiError:
        st.markdown('<span class="pill pill-bad">API offline</span>', unsafe_allow_html=True)

# ----------------------------------------------------------------- gate
st.title("Revenue Leakage Investigation")
if not ss.connected:
    st.info("Connect your Groq API key in the sidebar to start. "
            "The key is required for every session, even if one is set in the environment.")
    st.stop()

tab_chat, tab_audit = st.tabs(["Investigation", "Audit log"])


# ----------------------------------------------------------------- helpers
def handle_error(e: api.ApiError):
    if e.status == 401:
        ss.connected = False
        ss.api_key = ""
        st.error(e.message + " The key has been cleared.")
    else:
        st.error(e.message)


def render_anomalies():
    st.metric("Total revenue lost", money(ss.total_lost))
    if not ss.anomalies:
        st.caption("No anomalies found yet. Ask the agent to investigate.")
        return
    cols = ["customer_name", "period", "type", "component", "invoice_id",
            "expected", "billed", "amount_lost", "amount_overbilled", "root_cause"]
    rows = [{c: a.get(c) for c in cols if isinstance(a, dict) and a.get(c) is not None}
            for a in ss.anomalies if isinstance(a, dict)]
    st.dataframe(rows, width="stretch", hide_index=True)
    with st.expander("Evidence"):
        for i, a in enumerate(ss.anomalies, 1):
            if not isinstance(a, dict):
                continue
            st.markdown(f"**{i}. {a.get('customer_name', '')} - {a.get('type', '')}** "
                        f"({a.get('anomaly_id', '')})")
            st.json(a.get("evidence") or {})


def render_approvals():
    try:
        data = api.list_approvals(sess(), ss.thread_id)
    except api.ApiError as e:
        handle_error(e)
        return
    items = data.get("approvals", []) if isinstance(data, dict) else data
    pending = [x for x in (items or []) if str(x.get("status", "pending")).lower() == "pending"]
    st.subheader(f"Pending approvals ({len(pending)})")
    if not pending:
        st.caption("Nothing is waiting for approval.")
        return
    is_approver = ss.role == "approver"
    if not is_approver:
        st.warning("Only users with the approver role can approve or reject. "
                   "Switch role in the sidebar.")
    for it in pending:
        aid = it.get("id") or it.get("approval_id")
        with st.container(border=True):
            act = str(it.get("action", "action")).replace("_", " ")
            st.markdown(f"**{act.title()} - {it.get('customer_name', '')}**")
            st.caption(f"Amount: {money(it.get('amount'))} | requested by {it.get('requested_by', '?')}")
            if it.get("reason"):
                st.write(it["reason"])
            if it.get("items"):
                with st.expander("Items"):
                    st.json(it["items"])
            c1, c2 = st.columns(2)
            if c1.button("Approve", key=f"approve_{aid}", disabled=not is_approver,
                         type="primary", width="stretch"):
                _decide(aid, True)
            if c2.button("Reject", key=f"reject_{aid}", disabled=not is_approver,
                         width="stretch"):
                _decide(aid, False)


def _decide(aid, approved: bool):
    try:
        res = api.decide_approval(sess(), aid, approved)
        if isinstance(res, dict):
            if res.get("reply"):
                ss.messages.append({"role": "assistant", "content": res["reply"]})
            if "anomalies" in res:
                ss.anomalies = res["anomalies"] or ss.anomalies
            if "total_lost" in res:
                ss.total_lost = res["total_lost"]
            ss.status = res.get("status", ss.status)
        st.toast("Approved" if approved else "Rejected")
        st.rerun()
    except api.ApiError as e:
        handle_error(e)


# ----------------------------------------------------------------- chat
with tab_chat:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Chat")
        box = st.container(height=480, border=True)
        with box:
            if not ss.messages:
                st.caption("Try: \"Investigate revenue leakage in last quarter's billing data.\"")
            for m in ss.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
        prompt = st.chat_input("Ask the agent to investigate...", key="chat_input")
        if prompt:
            ss.messages.append({"role": "user", "content": prompt})
            try:
                tid = ensure_thread()
                with st.spinner("Investigating..."):
                    res = api.send_message(sess(), tid, prompt)
                ss.messages.append({"role": "assistant", "content": res.get("reply", "")})
                if "anomalies" in res:
                    ss.anomalies = res.get("anomalies") or []
                if "total_lost" in res:
                    ss.total_lost = res.get("total_lost") or 0.0
                ss.status = res.get("status", "")
            except api.ApiError as e:
                ss.messages.append({"role": "assistant", "content": f"Error: {e.message}"})
                handle_error(e)
            st.rerun()
    with right:
        st.subheader("Findings")
        if ss.status:
            st.caption(f"Status: {ss.status}")
        render_anomalies()
        st.divider()
        render_approvals()

# ----------------------------------------------------------------- audit
with tab_audit:
    st.subheader("Audit log")
    limit = st.number_input("Entries", 10, 500, 100, step=10, key="audit_limit")
    st.button("Refresh", key="audit_refresh")
    try:
        data = api.get_audit(sess(), int(limit))
        entries = data.get("entries", data.get("audit", [])) if isinstance(data, dict) else data
        if entries:
            st.dataframe(entries, width="stretch", hide_index=True)
        else:
            st.caption("No audit entries yet.")
    except api.ApiError as e:
        handle_error(e)
