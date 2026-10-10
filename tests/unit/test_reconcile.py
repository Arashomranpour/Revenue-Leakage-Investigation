"""reconcile_customer accuracy + evidence, checked against independently computed numbers."""
from decimal import Decimal as D

import pytest

from agent import tools as T

# customer -> list of (period, type, amount_lost, amount_overbilled)
EXPECTED = {
    3: [("2025-04", "wrong_plan_price", 50.0, 0), ("2025-05", "wrong_plan_price", 50.0, 0),
        ("2025-06", "wrong_plan_price", 50.0, 0)],
    5: [("2025-05", "missing_overage", 40.0, 0), ("2025-06", "missing_overage", 50.0, 0)],
    7: [("2025-05", "expired_discount", 39.8, 0), ("2025-06", "expired_discount", 39.8, 0)],
    9: [("2025-05", "unbilled_seats", 60.0, 0), ("2025-06", "unbilled_seats", 60.0, 0)],
    11: [("2025-06", "missing_invoice", 999.0, 0)],
    12: [("2025-06", "wrong_plan_price", 10.0, 0)],
    13: [("2025-04", "unbilled_seats", 30.0, 0), ("2025-05", "unbilled_seats", 30.0, 0),
         ("2025-06", "unbilled_seats", 30.0, 0), ("2025-06", "missing_overage", 20.0, 0)],
    14: [("2025-05", "missing_invoice", 999.0, 0)],
    15: [("2025-06", "wrong_plan_price", 0, 20.0)],
}


@pytest.mark.parametrize("cid", sorted(EXPECTED))
def test_planted_case(db_path, cid):
    got = sorted((a["period"], a["type"], a["amount_lost"], a["amount_overbilled"])
                 for a in T.reconcile_customer(db_path, cid))
    assert got == sorted(EXPECTED[cid])


@pytest.mark.parametrize("cid", [1, 2, 4, 6, 8, 10])
def test_clean_customers_have_no_anomalies(db_path, cid):
    assert T.reconcile_customer(db_path, cid) == []


def test_total_leakage_matches_independent_sum(db_path):
    total = D(0)
    for c in T.list_customers(db_path):
        for a in T.reconcile_customer(db_path, c["id"]):
            total += D(str(a["amount_lost"]))
    assert total == D("2557.60")


def test_unknown_customer_returns_empty(db_path):
    assert T.reconcile_customer(db_path, 999) == []


def test_evidence_present_and_consistent(db_path):
    for cid in EXPECTED:
        for a in T.reconcile_customer(db_path, cid):
            for k in ("anomaly_id", "customer_name", "period", "type", "expected", "billed", "root_cause",
                      "evidence"):
                assert a[k] not in (None, ""), (cid, k)
            ev = a["evidence"]
            assert ev["subscription"] and ev["plan"] and "expected_lines" in ev
            # amounts are consistent: expected - billed == lost - overbilled
            assert round(a["expected"] - a["billed"], 2) == round(a["amount_lost"] - a["amount_overbilled"], 2)
            if a["type"] != "missing_invoice":
                assert ev["invoice_lines"] is not None and a["invoice_id"]
            else:
                assert a["invoice_id"] is None and ev["invoices"] == []


def test_wrong_price_evidence_shows_billed_vs_list(db_path):
    a = T.reconcile_customer(db_path, 3)[0]
    assert a["billed"] == 149.0 and a["expected"] == 199.0
    assert a["evidence"]["invoice_lines"][0]["amount"] == 149.0
    assert "199" in a["root_cause"]


def test_expired_discount_evidence(db_path):
    a = T.reconcile_customer(db_path, 7)[0]
    assert a["evidence"]["discount_active"] is False
    assert a["evidence"]["subscription"]["discount_end"] == "2025-04-30"


def test_overbilling_is_not_counted_as_loss(db_path):
    a = T.reconcile_customer(db_path, 15)[0]
    assert a["amount_lost"] == 0 and a["amount_overbilled"] == 20.0


def test_read_tools_are_select_only(db_path):
    names = {t.name for t in T.make_read_tools(db_path)}
    assert names == {"list_customers", "get_subscription", "get_invoices", "get_usage", "reconcile_customer"}
    assert not any("apply" in n or "adjust" in n or "propose" in n for n in names)
