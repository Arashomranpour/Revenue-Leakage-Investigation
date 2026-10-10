"""Independent billing oracle: recompute expected invoices from raw seed tables
(plans / subscriptions / usage) with our own Decimal logic, and compare with the
planted leakage the seed claims. Does not call agent.db.expected_lines."""
import sqlite3
from decimal import Decimal as D

import pytest

from agent.db import init_db

PERIODS = ["2025-04", "2025-05", "2025-06"]

# Hand-computed leakage (expected - billed) per customer, total across periods.
EXPECTED_LEAKAGE = {
    3: D("150.00"),    # 50 x 3
    5: D("90.00"),     # 4000*.01 + 5000*.01
    7: D("79.60"),     # 39.80 x 2
    9: D("120.00"),    # 4 seats * 15 * 2
    11: D("999.00"),   # missing Jun invoice
    12: D("10.00"),
    13: D("110.00"),   # 2*15*3 + 2000*.01
    14: D("999.00"),   # missing May invoice
    15: D("-20.00"),   # overbilled
}


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    path = init_db(str(tmp_path_factory.mktemp("db") / "b.db"), reset=True)
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def expected_total(conn, cid, period):
    s = conn.execute("select * from subscriptions where customer_id=?", (cid,)).fetchone()
    p = conn.execute("select * from plans where id=?", (s["plan_id"],)).fetchone()
    u = conn.execute("select units from usage where customer_id=? and period=?", (cid, period)).fetchone()[0]
    total = D(str(p["monthly_price"]))
    if s["discount_pct"] and (s["discount_end"] is None or period <= s["discount_end"][:7]):
        total -= total * D(str(s["discount_pct"])) / 100
    seats = s["seats"] if period >= s["seats_from"] else s["prior_seats"]
    total += max(seats - p["included_seats"], 0) * D(str(p["per_seat_price"]))
    total += max(u - p["included_usage"], 0) * D(str(p["overage_rate"]))
    return total.quantize(D("0.01"))


def billed_total(conn, cid, period):
    r = conn.execute("select amount from invoices where customer_id=? and period=?", (cid, period)).fetchone()
    return D(str(r[0])).quantize(D("0.01")) if r else D("0.00")


def test_seed_shape(conn):
    assert conn.execute("select count(*) from customers").fetchone()[0] == 15
    assert conn.execute("select count(*) from plans").fetchone()[0] == 3
    # 15 customers * 3 periods - 2 missing invoices
    assert conn.execute("select count(*) from invoices").fetchone()[0] == 43


def test_invoice_header_matches_lines(conn):
    for inv in conn.execute("select id, amount from invoices"):
        s = conn.execute("select sum(amount) from invoice_lines where invoice_id=?", (inv["id"],)).fetchone()[0]
        assert D(str(s)).quantize(D("0.01")) == D(str(inv["amount"])).quantize(D("0.01"))


@pytest.mark.parametrize("cid,leak", sorted(EXPECTED_LEAKAGE.items()))
def test_planted_leakage_amount(conn, cid, leak):
    total = sum(expected_total(conn, cid, p) - billed_total(conn, cid, p) for p in PERIODS)
    assert total == leak


def test_clean_customers_have_no_leakage(conn):
    for cid in range(1, 16):
        if cid in EXPECTED_LEAKAGE:
            continue
        for p in PERIODS:
            assert expected_total(conn, cid, p) == billed_total(conn, cid, p), (cid, p)


def test_seed_is_deterministic(tmp_path):
    a = init_db(str(tmp_path / "a.db"), reset=True)
    b = init_db(str(tmp_path / "b.db"), reset=True)
    dump = lambda p: list(sqlite3.connect(p).execute("select customer_id,period,amount from invoices order by id"))
    assert dump(a) == dump(b)


def test_init_db_idempotent(tmp_path):
    p = str(tmp_path / "x.db")
    init_db(p)
    init_db(p)
    assert sqlite3.connect(p).execute("select count(*) from customers").fetchone()[0] == 15
