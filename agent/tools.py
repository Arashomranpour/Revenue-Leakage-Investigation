"""Billing tools.

READ tools (safe, SQL SELECT only; exposed to the LLM through `make_read_tools`):
    list_customers, get_subscription, get_invoices, get_usage, reconcile_customer

WRITE tools (never exposed to the LLM, only called by the graph's execute node after an
approver approved the decision):
    propose_credit_memo / propose_invoice_adjustment  -> create a *pending* approval row only
    apply_credit_memo / adjust_invoice                -> mutate billing data; they re-verify the
                                                         approval row (status=approved by an 'approver')

All money math is done here in Python with Decimal; the LLM never computes amounts.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP

from langchain_core.tools import StructuredTool

from .db import (PERIODS, connect, default_db_path, discount_active, expected_lines, seats_for,
                 session, utcnow)

PROPOSER_ROLES = {"analyst", "approver"}
APPROVER_ROLE = "approver"


def _d(x) -> Decimal:
    return Decimal(str(x))


def _f(x: Decimal) -> float:
    return float(x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


# --------------------------------------------------------------------------- read tools
def list_customers(db_path: str | None = None) -> list[dict]:
    with session(db_path or default_db_path()) as c:
        return _rows(c.execute(
            "SELECT c.id, c.name, c.segment, p.name AS plan, s.seats FROM customers c "
            "JOIN subscriptions s ON s.customer_id=c.id JOIN plans p ON p.id=s.plan_id ORDER BY c.id"))


def get_subscription(db_path: str | None, customer_id: int) -> dict:
    with session(db_path or default_db_path()) as c:
        row = c.execute(
            "SELECT s.*, c.name AS customer_name FROM subscriptions s JOIN customers c ON c.id=s.customer_id "
            "WHERE s.customer_id=?", (customer_id,)).fetchone()
        if not row:
            return {"error": f"customer {customer_id} not found"}
        plan = dict(c.execute("SELECT * FROM plans WHERE id=?", (row["plan_id"],)).fetchone())
        return {**dict(row), "plan": plan}


def get_invoices(db_path: str | None, customer_id: int, period: str | None = None) -> list[dict]:
    with session(db_path or default_db_path()) as c:
        q = "SELECT * FROM invoices WHERE customer_id=?"
        args: list = [customer_id]
        if period:
            q += " AND period=?"
            args.append(period)
        invs = _rows(c.execute(q + " ORDER BY period, id", args))
        for inv in invs:
            inv["lines"] = _rows(c.execute(
                "SELECT id, kind, description, quantity, unit_price, amount FROM invoice_lines "
                "WHERE invoice_id=? ORDER BY id", (inv["id"],)))
        return invs


def get_usage(db_path: str | None, customer_id: int) -> list[dict]:
    with session(db_path or default_db_path()) as c:
        return _rows(c.execute(
            "SELECT id, period, units FROM usage WHERE customer_id=? ORDER BY period", (customer_id,)))


_TYPE_BY_KIND = {"base": "wrong_plan_price", "discount": "expired_discount",
                 "seats": "unbilled_seats", "overage": "missing_overage"}


def reconcile_customer(db_path: str | None, customer_id: int) -> list[dict]:
    """Compare contracted charges with invoiced charges per period.

    Returns anomalies: {anomaly_id, customer_id, customer_name, period, type, component, invoice_id,
    expected, billed, amount_lost, amount_overbilled, root_cause, evidence{...rows...}}.
    amount_lost > 0 means under-billing (lost revenue); amount_overbilled > 0 means the customer was
    overcharged (credit memo case). Amounts are computed with Decimal, not by the LLM.
    """
    db_path = db_path or default_db_path()
    sub = get_subscription(db_path, customer_id)
    if "error" in sub:
        return []
    plan = sub["plan"]
    usage = {u["period"]: u for u in get_usage(db_path, customer_id)}
    invoices = get_invoices(db_path, customer_id)
    with session(db_path) as c:
        name = c.execute("SELECT name FROM customers WHERE id=?", (customer_id,)).fetchone()[0]
        credits = {r["invoice_id"]: _d(r["t"]) for r in c.execute(
            "SELECT invoice_id, SUM(amount) t FROM credit_memos WHERE customer_id=? AND status='applied' "
            "GROUP BY invoice_id", (customer_id,))}
    sub_row = {k: v for k, v in sub.items() if k != "plan"}
    out: list[dict] = []
    for period in PERIODS:
        if period < sub["start_date"][:7] or (sub["end_date"] and period > sub["end_date"][:7]):
            continue
        units = usage.get(period, {}).get("units", 0)
        exp_lines = expected_lines(plan, sub, units, period)
        invs = [i for i in invoices if i["period"] == period]
        base_ev = {"subscription": sub_row, "plan": plan, "usage": usage.get(period),
                   "seats_in_period": seats_for(sub, period), "discount_active": discount_active(sub, period)}
        if not invs:
            total = sum((_d(l["amount"]) for l in exp_lines), Decimal(0))
            out.append(dict(
                anomaly_id=f"{customer_id}:{period}:missing_invoice", customer_id=customer_id,
                customer_name=name, period=period, type="missing_invoice", component="invoice",
                invoice_id=None, expected=_f(total), billed=0.0, amount_lost=_f(total),
                amount_overbilled=0.0,
                root_cause=f"No invoice was generated for {period} although the subscription was active.",
                evidence={**base_ev, "invoices": [], "expected_lines": exp_lines}))
            continue
        lines = [l for i in invs for l in i["lines"]]
        inv_id = invs[0]["id"]
        credit = sum((credits.get(i["id"], Decimal(0)) for i in invs), Decimal(0))
        for kind in ("base", "discount", "seats", "overage"):
            exp = sum((_d(l["amount"]) for l in exp_lines if l["kind"] == kind), Decimal(0))
            bil = sum((_d(l["amount"]) for l in lines if l["kind"] == kind), Decimal(0))
            diff = exp - bil  # >0 under-billed
            if diff > 0:
                lost, over = diff, Decimal(0)
            else:
                lost, over = Decimal(0), -diff
                if kind == "base":
                    over = max(over - credit, Decimal(0))  # already refunded via applied credit memos
            if lost == 0 and over == 0:
                continue
            atype = _TYPE_BY_KIND[kind]
            if kind == "discount" and exp == 0 and bil < 0:
                cause = (f"A discount of {abs(_f(bil))} is still applied in {period} but the contractual "
                         f"discount ended {sub['discount_end']}.")
            elif kind == "discount":
                atype, cause = "discount_mismatch", f"Discount billed {_f(bil)} vs contractual {_f(exp)}."
            elif kind == "base":
                cause = (f"Plan fee billed {_f(bil)} but {plan['name']} list price is {_f(exp)} "
                         f"({'under' if lost else 'over'}-billed).")
            elif kind == "seats":
                cause = (f"{seats_for(sub, period)} seats contracted ({plan['included_seats']} included) but "
                         f"seat charges billed {_f(bil)} vs expected {_f(exp)}.")
            else:
                cause = (f"Usage {units} exceeds {plan['included_usage']} included units; overage billed "
                         f"{_f(bil)} vs expected {_f(exp)}.")
            out.append(dict(
                anomaly_id=f"{customer_id}:{period}:{atype}", customer_id=customer_id, customer_name=name,
                period=period, type=atype, component=kind, invoice_id=inv_id, expected=_f(exp),
                billed=_f(bil), amount_lost=_f(lost), amount_overbilled=_f(over), root_cause=cause,
                evidence={**base_ev, "invoices": [{k: v for k, v in i.items() if k != "lines"} for i in invs],
                          "invoice_lines": [l for l in lines if l["kind"] == kind],
                          "expected_lines": [l for l in exp_lines if l["kind"] == kind]}))
    return out


def make_read_tools(db_path: str) -> list[StructuredTool]:
    """LangChain tool wrappers bound to db_path. Read-only."""

    def _list_customers() -> list:
        """List all customers with plan and seat count."""
        return list_customers(db_path)

    def _get_subscription(customer_id: int) -> dict:
        """Get the subscription (plan, seats, discount) for a customer id."""
        return get_subscription(db_path, customer_id)

    def _get_invoices(customer_id: int, period: str = "") -> list:
        """Get invoices with line items for a customer; optional period like 2025-05."""
        return get_invoices(db_path, customer_id, period or None)

    def _get_usage(customer_id: int) -> list:
        """Get metered usage per period for a customer."""
        return get_usage(db_path, customer_id)

    def _reconcile_customer(customer_id: int) -> list:
        """Reconcile billing plan vs invoices for a customer; returns anomalies with evidence and
        deterministic amount_lost."""
        return reconcile_customer(db_path, customer_id)

    fns = [(_list_customers, "list_customers"), (_get_subscription, "get_subscription"),
           (_get_invoices, "get_invoices"), (_get_usage, "get_usage"),
           (_reconcile_customer, "reconcile_customer")]
    return [StructuredTool.from_function(f, name=n, description=f.__doc__) for f, n in fns]


# --------------------------------------------------------------------------- write tools
def _create_approval(db_path, thread_id, action, customer_id, items, reason, requested_by, requested_role):
    if requested_role not in PROPOSER_ROLES:
        raise PermissionError(f"Role '{requested_role}' may not propose actions.")
    amount = _f(sum((_d(i["amount"]) for i in items), Decimal(0)))
    key = hashlib.sha256(json.dumps(
        [action, customer_id, sorted(i["anomaly_id"] for i in items)]).encode()).hexdigest()[:24]
    payload = json.dumps({"customer_id": customer_id, "items": items, "reason": reason}, default=str)
    with session(db_path) as c:
        dup = c.execute("SELECT id FROM approvals WHERE dedupe_key=? AND status IN ('pending','approved')",
                        (key,)).fetchone()
        if dup:
            return {"approval_id": dup["id"], "created": False}
        cur = c.execute(
            "INSERT INTO approvals(thread_id,action,customer_id,amount,payload,dedupe_key,status,"
            "requested_by,requested_role,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (thread_id, action, customer_id, amount, payload, key, "pending", requested_by,
             requested_role, utcnow()))
        return {"approval_id": cur.lastrowid, "created": True}


def propose_credit_memo(db_path, thread_id, customer_id, items, reason, requested_by, requested_role):
    """Create a PENDING credit-memo approval (no billing data is changed). items: [{anomaly_id,
    invoice_id, period, amount, description}]."""
    return _create_approval(db_path, thread_id, "credit_memo", customer_id, items, reason,
                            requested_by, requested_role)


def propose_invoice_adjustment(db_path, thread_id, customer_id, items, reason, requested_by, requested_role):
    """Create a PENDING invoice-adjustment approval (no billing data is changed)."""
    return _create_approval(db_path, thread_id, "adjust_invoice", customer_id, items, reason,
                            requested_by, requested_role)


def get_approval(db_path: str | None, approval_id: int) -> dict | None:
    with session(db_path or default_db_path()) as c:
        r = c.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    p = json.loads(d.pop("payload"))
    d.update(items=p["items"], reason=p["reason"])
    d["customer_name"] = _customer_name(db_path, d["customer_id"])
    return d


def _customer_name(db_path, cid):
    with session(db_path or default_db_path()) as c:
        r = c.execute("SELECT name FROM customers WHERE id=?", (cid,)).fetchone()
    return r[0] if r else None


def _claim_for_execution(conn, approval_id: int, action: str) -> dict:
    """Guard: only an approved (by an 'approver') and not-yet-executed approval may be executed."""
    row = conn.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
    if row is None:
        raise ValueError(f"approval {approval_id} not found")
    if row["action"] != action:
        raise ValueError(f"approval {approval_id} is a {row['action']}, not {action}")
    if row["status"] != "approved" or row["decided_role"] != APPROVER_ROLE:
        raise PermissionError(f"approval {approval_id} is not approved by an approver (status={row['status']})")
    cur = conn.execute("UPDATE approvals SET status='executed', executed_at=? WHERE id=? AND status='approved'",
                       (utcnow(), approval_id))
    if cur.rowcount != 1:
        raise PermissionError(f"approval {approval_id} already executed")
    return {**dict(row), **json.loads(row["payload"])}


def apply_credit_memo(db_path: str | None, approval_id: int, executed_by: str) -> dict:
    """WRITE: create credit memos for an approved credit_memo approval."""
    db_path = db_path or default_db_path()
    with session(db_path) as c:
        a = _claim_for_execution(c, approval_id, "credit_memo")
        ids = []
        for it in a["items"]:
            cur = c.execute(
                "INSERT INTO credit_memos(customer_id,invoice_id,amount,reason,status,approval_id,created_by,"
                "created_at) VALUES (?,?,?,?,?,?,?,?)",
                (a["customer_id"], it.get("invoice_id"), it["amount"], a["reason"], "applied",
                 approval_id, executed_by, utcnow()))
            ids.append(cur.lastrowid)
    return {"approval_id": approval_id, "credit_memo_ids": ids, "amount": a["amount"]}


def adjust_invoice(db_path: str | None, approval_id: int, executed_by: str) -> dict:
    """WRITE: add correction lines to invoices (or generate a missing invoice) for an approved
    adjust_invoice approval."""
    db_path = db_path or default_db_path()
    touched = []
    with session(db_path) as c:
        a = _claim_for_execution(c, approval_id, "adjust_invoice")
        for it in a["items"]:
            inv_id = it.get("invoice_id")
            if inv_id is None:  # generate the never-issued invoice
                inv_id = c.execute("SELECT COALESCE(MAX(id),0)+1 FROM invoices").fetchone()[0]
                total = sum((_d(l["amount"]) for l in it["new_invoice_lines"]), Decimal(0))
                c.execute("INSERT INTO invoices VALUES (?,?,?,?,?,?)",
                          (inv_id, a["customer_id"], it["period"], utcnow()[:10], _f(total), "issued"))
                lines = it["new_invoice_lines"]
            else:
                lines = [dict(kind=it["component"], description=f"Correction: {it['description']}",
                              quantity=1, unit_price=it["amount"], amount=it["amount"])]
            for l in lines:
                c.execute("INSERT INTO invoice_lines(invoice_id,kind,description,quantity,unit_price,amount) "
                          "VALUES (?,?,?,?,?,?)",
                          (inv_id, l["kind"], l["description"], l["quantity"], l["unit_price"], l["amount"]))
            if it.get("invoice_id") is not None:
                c.execute("UPDATE invoices SET amount=(SELECT ROUND(SUM(amount),2) FROM invoice_lines "
                          "WHERE invoice_id=?) WHERE id=?", (inv_id, inv_id))
            touched.append(inv_id)
    return {"approval_id": approval_id, "invoice_ids": touched, "amount": a["amount"]}
