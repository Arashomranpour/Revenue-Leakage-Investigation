"""SQLite layer for the Revenue Leakage Investigation Agent.

Usage:
    from agent.db import init_db, log_audit, connect
    path = init_db()                 # creates + seeds data/billing.db if missing (idempotent)
    path = init_db(p, reset=True)    # wipe and reseed deterministically
    log_audit(path, actor="alice", role="analyst", action="note", detail={"k": "v"})

The seed is fully deterministic (no randomness, fixed timestamps) so tests can rely on it.
Periods seeded: 2025-04, 2025-05, 2025-06.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(ROOT / "data" / "billing.db")
PERIODS = ["2025-04", "2025-05", "2025-06"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, segment TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS plans(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, monthly_price REAL NOT NULL,
  included_seats INTEGER NOT NULL, per_seat_price REAL NOT NULL,
  included_usage INTEGER NOT NULL, overage_rate REAL NOT NULL);
CREATE TABLE IF NOT EXISTS subscriptions(
  id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
  plan_id INTEGER NOT NULL REFERENCES plans(id),
  seats INTEGER NOT NULL, prior_seats INTEGER NOT NULL, seats_from TEXT NOT NULL,
  start_date TEXT NOT NULL, end_date TEXT,
  discount_pct REAL NOT NULL DEFAULT 0, discount_end TEXT);
CREATE TABLE IF NOT EXISTS invoices(
  id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
  period TEXT NOT NULL, issued_date TEXT NOT NULL, amount REAL NOT NULL, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS invoice_lines(
  id INTEGER PRIMARY KEY AUTOINCREMENT, invoice_id INTEGER NOT NULL REFERENCES invoices(id),
  kind TEXT NOT NULL, description TEXT NOT NULL, quantity REAL NOT NULL,
  unit_price REAL NOT NULL, amount REAL NOT NULL);
CREATE TABLE IF NOT EXISTS usage(
  id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id INTEGER NOT NULL REFERENCES customers(id),
  period TEXT NOT NULL, units INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS credit_memos(
  id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id INTEGER NOT NULL, invoice_id INTEGER,
  amount REAL NOT NULL, reason TEXT NOT NULL, status TEXT NOT NULL,
  approval_id INTEGER, created_by TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS approvals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL, action TEXT NOT NULL,
  customer_id INTEGER NOT NULL, amount REAL NOT NULL, payload TEXT NOT NULL,
  dedupe_key TEXT NOT NULL, status TEXT NOT NULL,
  requested_by TEXT, requested_role TEXT, created_at TEXT NOT NULL,
  decided_by TEXT, decided_role TEXT, decided_at TEXT, executed_at TEXT);
CREATE TABLE IF NOT EXISTS audit_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, actor TEXT NOT NULL,
  role TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL);
"""

# (id, name, monthly_price, included_seats, per_seat_price, included_usage, overage_rate)
PLANS = [
    (1, "Starter", 49.0, 3, 10.0, 1000, 0.02),
    (2, "Growth", 199.0, 10, 15.0, 10000, 0.01),
    (3, "Enterprise", 999.0, 50, 12.0, 100000, 0.005),
]

# id, name, segment, plan, seats, prior_seats, seats_from, discount_pct, discount_end, usage(3 periods)
CUSTOMERS = [
    (1, "Acme Corp", "SMB", 2, 8, 8, "2025-01", 0, None, (6000, 6500, 7000)),
    (2, "Birch Labs", "SMB", 1, 3, 3, "2025-01", 0, None, (800, 900, 700)),
    (3, "Cobalt Systems", "SMB", 2, 9, 9, "2025-01", 0, None, (5000, 5200, 5100)),
    (4, "Dune Analytics", "Enterprise", 3, 40, 40, "2025-01", 0, None, (90000, 91000, 92000)),
    (5, "Ember Retail", "Mid", 2, 10, 10, "2025-01", 0, None, (9000, 14000, 15000)),
    (6, "Fjord Media", "SMB", 1, 3, 3, "2025-01", 0, None, (1500, 1200, 900)),
    (7, "Granite Health", "Mid", 2, 10, 10, "2025-01", 20, "2025-04-30", (7000, 7000, 7500)),
    (8, "Helix Bio", "Enterprise", 3, 60, 60, "2025-01", 0, None, (80000, 85000, 88000)),
    (9, "Ion Logistics", "Mid", 2, 14, 10, "2025-05", 0, None, (6000, 6200, 6400)),
    (10, "Juniper Foods", "SMB", 1, 3, 3, "2025-01", 10, "2025-12-31", (500, 600, 700)),
    (11, "Kestrel Energy", "Enterprise", 3, 50, 50, "2025-01", 0, None, (70000, 72000, 74000)),
    (12, "Lumen Studio", "SMB", 1, 3, 3, "2025-01", 0, None, (400, 450, 500)),
    (13, "Meridian Travel", "Mid", 2, 12, 12, "2025-01", 0, None, (9000, 9500, 12000)),
    (14, "Nimbus Cloud", "Enterprise", 3, 50, 50, "2025-01", 0, None, (60000, 65000, 66000)),
    (15, "Orchid Pharma", "Mid", 2, 10, 10, "2025-01", 0, None, (4000, 4100, 4200)),
]

# Planted leakage (customer_id -> description) kept for documentation / tests:
PLANTED = {
    3: "wrong_plan_price: Growth billed 149 instead of 199 in all periods",
    5: "missing_overage: usage above 10000 in 2025-05/06 not billed",
    7: "expired_discount: 20% discount ended 2025-04-30 but still applied in 2025-05/06",
    9: "unbilled_seats: 4 extra seats since 2025-05 never billed",
    11: "missing_invoice: no invoice generated for 2025-06",
    12: "wrong_plan_price: Starter billed 39 instead of 49 in 2025-06",
    13: "unbilled_seats (2 extra, all periods) + missing_overage in 2025-06",
    14: "missing_invoice: no invoice generated for 2025-05",
    15: "over-billing: Growth billed 219 instead of 199 in 2025-06 (credit memo case)",
}

FIXED_TS = "2025-07-05T00:00:00+00:00"


def _q(x) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def session(db_path: str | None = None):
    """Transactional connection: commits on success, rolls back on error, always closes."""
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def default_db_path() -> str:
    return os.environ.get("BILLING_DB_PATH") or DEFAULT_DB_PATH


def seats_for(sub: dict, period: str) -> int:
    return sub["seats"] if period >= sub["seats_from"] else sub["prior_seats"]


def discount_active(sub: dict, period: str) -> bool:
    if not sub["discount_pct"]:
        return False
    end = sub["discount_end"]
    return end is None or period <= end[:7]


def expected_lines(plan: dict, sub: dict, units: int, period: str) -> list[dict]:
    """Single source of truth for what an invoice SHOULD contain (Decimal math, 2dp)."""
    lines = []
    price = Decimal(str(plan["monthly_price"]))
    lines.append(dict(kind="base", description=f"{plan['name']} plan monthly fee", quantity=1,
                      unit_price=_q(price), amount=_q(price)))
    if discount_active(sub, period):
        d = (price * Decimal(str(sub["discount_pct"])) / 100)
        lines.append(dict(kind="discount", description=f"{sub['discount_pct']:g}% discount", quantity=1,
                          unit_price=-_q(d), amount=-_q(d)))
    extra = max(seats_for(sub, period) - plan["included_seats"], 0)
    if extra:
        amt = Decimal(extra) * Decimal(str(plan["per_seat_price"]))
        lines.append(dict(kind="seats", description=f"Additional seats ({extra})", quantity=extra,
                          unit_price=plan["per_seat_price"], amount=_q(amt)))
    over = max(units - plan["included_usage"], 0)
    if over:
        amt = Decimal(over) * Decimal(str(plan["overage_rate"]))
        lines.append(dict(kind="overage", description=f"Usage overage ({over} units)", quantity=over,
                          unit_price=plan["overage_rate"], amount=_q(amt)))
    return lines


def _flawed_lines(cid: int, period: str, lines: list[dict]) -> list[dict] | None:
    """Apply planted billing errors. Return None to skip invoice generation."""
    L = [dict(l) for l in lines]

    def by(kind):
        return next((l for l in L if l["kind"] == kind), None)

    if cid == 3:
        b = by("base"); b["unit_price"] = b["amount"] = 149.0
    elif cid == 5 and period >= "2025-05":
        L = [l for l in L if l["kind"] != "overage"]
    elif cid == 7 and period >= "2025-05":
        price = by("base")["amount"]
        d = _q(Decimal(str(price)) * Decimal("0.20"))
        L.append(dict(kind="discount", description="20% discount", quantity=1, unit_price=-d, amount=-d))
    elif cid == 9 and period >= "2025-05":
        L = [l for l in L if l["kind"] != "seats"]
    elif cid == 11 and period == "2025-06":
        return None
    elif cid == 12 and period == "2025-06":
        b = by("base"); b["unit_price"] = b["amount"] = 39.0
    elif cid == 13:
        L = [l for l in L if l["kind"] != "seats"]
        if period == "2025-06":
            L = [l for l in L if l["kind"] != "overage"]
    elif cid == 14 and period == "2025-05":
        return None
    elif cid == 15 and period == "2025-06":
        b = by("base"); b["unit_price"] = b["amount"] = 219.0
    return L


def init_db(path: str | None = None, reset: bool = False) -> str:
    """Create and deterministically seed the billing DB. Idempotent unless reset=True."""
    path = path or default_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if reset and Path(path).exists():
        Path(path).unlink()
    with closing(connect(path)) as conn:
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]:
            return path
        conn.executemany("INSERT INTO plans VALUES (?,?,?,?,?,?,?)", PLANS)
        plans = {p[0]: dict(zip(["id", "name", "monthly_price", "included_seats", "per_seat_price",
                                  "included_usage", "overage_rate"], p)) for p in PLANS}
        inv_id = 0
        for (cid, name, seg, pid, seats, prior, sfrom, dpct, dend, usage) in CUSTOMERS:
            conn.execute("INSERT INTO customers VALUES (?,?,?)", (cid, name, seg))
            conn.execute(
                "INSERT INTO subscriptions(id,customer_id,plan_id,seats,prior_seats,seats_from,start_date,"
                "end_date,discount_pct,discount_end) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (cid, cid, pid, seats, prior, sfrom, "2025-01-01", None, dpct, dend))
            sub = dict(seats=seats, prior_seats=prior, seats_from=sfrom, discount_pct=dpct, discount_end=dend)
            for period, units in zip(PERIODS, usage):
                conn.execute("INSERT INTO usage(customer_id,period,units) VALUES (?,?,?)", (cid, period, units))
                lines = _flawed_lines(cid, period, expected_lines(plans[pid], sub, units, period))
                if lines is None:
                    continue
                inv_id += 1
                total = _q(sum(Decimal(str(l["amount"])) for l in lines))
                year, month = map(int, period.split("-"))
                issued = f"{year + (month == 12)}-{(month % 12) + 1:02d}-01"
                conn.execute("INSERT INTO invoices VALUES (?,?,?,?,?,?)",
                             (inv_id, cid, period, issued, total, "issued"))
                conn.executemany(
                    "INSERT INTO invoice_lines(invoice_id,kind,description,quantity,unit_price,amount) "
                    "VALUES (?,?,?,?,?,?)",
                    [(inv_id, l["kind"], l["description"], l["quantity"], l["unit_price"], l["amount"])
                     for l in lines])
        conn.commit()
    return path


def log_audit(db_path: str | None, actor: str, role: str, action: str, detail=None) -> int:
    """Append an audit record. `detail` may be a str or any JSON-serializable object."""
    if not isinstance(detail, str):
        detail = json.dumps(detail if detail is not None else {}, default=str, sort_keys=True)
    with session(db_path or default_db_path()) as conn:
        cur = conn.execute(
            "INSERT INTO audit_log(ts,actor,role,action,detail) VALUES (?,?,?,?,?)",
            (utcnow(), actor or "unknown", role or "unknown", action, detail))
        return cur.lastrowid


AUDIT = log_audit  # alias: `AUDIT` helper per the team contract


if __name__ == "__main__":
    print(init_db(reset=True))
