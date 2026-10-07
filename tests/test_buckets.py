"""Tests for savings buckets: balances, goals, the leftover figure and reconciliation.

A bucket's figures cover the months loaded, so an off-by-one in counting them,
or a sign slip on spending, is multiplied across every bucket. Each rule gets
its own test.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from budget.ingest import ingest
from budget.reports import (
    BUCKETS_PATH,
    bucket_balance,
    latest_period,
    left_after_savings,
    load_buckets,
    load_savings_pool,
    months_loaded,
    planned_allocation,
    save_buckets,
    savings_reconciliation,
    savings_summary,
)
from budget.schema import empty_transactions

HEADER = "Date,Description,Institution,Account,Category,Is Hidden,Is Pending,Amount"


def frame_of(*rows: str) -> pd.DataFrame:
    frame, _ = ingest(io.BytesIO("\n".join([HEADER, *rows, ""]).encode("utf-8")))
    return frame


def bucket(**overrides) -> dict:
    spec = {"contribution": 100, "adjustment": 0, "goal": 0}
    spec.update(overrides)
    return spec


def balance_of(frame: pd.DataFrame, **overrides) -> dict:
    return bucket_balance(frame, "Vacation", bucket(**overrides))


# Three months loaded: June, July and August.
THREE_MONTHS = (
    "10/06/2026,A,INST,C,Groceries_,No,No,($10)",
    "10/07/2026,B,INST,C,Groceries_,No,No,($10)",
    "10/08/2026,C,INST,C,Groceries_,No,No,($10)",
)


# --- months loaded -------------------------------------------------------


def test_months_loaded_counts_first_through_newest_inclusive() -> None:
    assert months_loaded(frame_of(*THREE_MONTHS)) == 3


def test_a_full_year_to_september_is_nine_months() -> None:
    frame = frame_of(
        "02/01/2026,A,INST,C,Groceries_,No,No,($10)",
        "20/09/2026,B,INST,C,Groceries_,No,No,($10)",
    )
    assert months_loaded(frame) == 9


def test_a_quiet_month_in_between_still_counts() -> None:
    frame = frame_of(
        "10/06/2026,A,INST,C,Groceries_,No,No,($10)",
        "10/08/2026,C,INST,C,Groceries_,No,No,($10)",
    )
    assert months_loaded(frame) == 3


def test_months_loaded_carries_on_across_a_new_year() -> None:
    frame = frame_of(
        "10/11/2026,A,INST,C,Groceries_,No,No,($10)",
        "10/02/2027,B,INST,C,Groceries_,No,No,($10)",
    )
    assert months_loaded(frame) == 4


def test_one_month_loaded_is_one() -> None:
    assert months_loaded(frame_of("10/06/2026,A,INST,C,Groceries_,No,No,($10)")) == 1


def test_no_data_is_no_months() -> None:
    assert months_loaded(empty_transactions()) == 0
    assert latest_period(empty_transactions()) is None


# --- a bucket's figures --------------------------------------------------


def test_contributed_is_the_contribution_times_months_loaded() -> None:
    """The month in progress counts too: three months, three contributions."""
    assert balance_of(frame_of(*THREE_MONTHS))["contributed"] == 30_000


def test_changing_the_contribution_reprices_every_month_loaded() -> None:
    frame = frame_of(*THREE_MONTHS)
    assert balance_of(frame, contribution=50)["contributed"] == 15_000
    assert balance_of(frame, contribution=0)["contributed"] == 0


def test_spent_covers_every_month_loaded() -> None:
    frame = frame_of(
        *THREE_MONTHS,
        "15/06/2026,FLIGHT,INST,C,Vacation_,No,No,($400)",
        "15/08/2026,HOTEL,INST,C,Vacation_,No,No,($150)",
    )
    assert balance_of(frame)["spent"] == 55_000


def test_a_refund_in_the_category_goes_back_into_the_bucket() -> None:
    frame = frame_of(
        *THREE_MONTHS,
        "15/07/2026,HOTEL,INST,C,Vacation_,No,No,($150)",
        "20/07/2026,HOTEL REFUND,INST,C,Vacation_,No,No,$50",
    )
    assert balance_of(frame)["spent"] == 10_000


def test_spending_in_other_categories_does_not_touch_the_bucket() -> None:
    frame = frame_of(*THREE_MONTHS, "15/07/2026,SHOES,INST,C,Clothing_,No,No,($90)")
    assert balance_of(frame)["spent"] == 0


def test_balance_is_contributed_plus_adjustment_less_spent() -> None:
    frame = frame_of(*THREE_MONTHS, "15/07/2026,HOTEL,INST,C,Vacation_,No,No,($150)")
    assert balance_of(frame, adjustment=1_000) == {
        "contributed": 30_000,
        "adjustment": 100_000,
        "spent": 15_000,
        "balance": 30_000 + 100_000 - 15_000,
    }


def test_a_negative_adjustment_takes_money_out() -> None:
    """Money moved by hand that the data does not show."""
    assert balance_of(frame_of(*THREE_MONTHS), adjustment=-50)["balance"] == 30_000 - 5_000


def test_overspending_leaves_a_negative_balance() -> None:
    frame = frame_of(*THREE_MONTHS, "15/06/2026,FLIGHT,INST,C,Vacation_,No,No,($900)")
    assert balance_of(frame)["balance"] == 30_000 - 90_000


def test_no_data_leaves_just_the_adjustment() -> None:
    assert balance_of(empty_transactions(), adjustment=75) == {
        "contributed": 0,
        "adjustment": 7_500,
        "spent": 0,
        "balance": 7_500,
    }


# --- goals ---------------------------------------------------------------


def summary_row(frame: pd.DataFrame, **overrides) -> pd.Series:
    return savings_summary(frame, {"Vacation": bucket(**overrides)}).iloc[0]


def test_progress_is_the_balance_as_a_share_of_the_goal() -> None:
    row = summary_row(frame_of(*THREE_MONTHS), contribution=100, goal=1_000)
    assert row["goal_cents"] == 100_000
    assert row["pct_saved"] == pytest.approx(0.3)
    assert row["remaining_cents"] == 70_000


def test_a_goal_met_leaves_nothing_to_save() -> None:
    row = summary_row(frame_of(*THREE_MONTHS), adjustment=900, goal=1_000)
    assert row["pct_saved"] == pytest.approx(1.2)
    assert row["remaining_cents"] == 0


def test_an_overdrawn_bucket_has_more_than_the_goal_left() -> None:
    row = summary_row(frame_of(*THREE_MONTHS), adjustment=-400, goal=1_000)
    assert row["pct_saved"] < 0
    assert row["remaining_cents"] == 110_000


def test_no_goal_means_no_progress_figures() -> None:
    row = summary_row(frame_of(*THREE_MONTHS))
    assert row["goal_cents"] == 0
    assert row["pct_saved"] is None or pd.isna(row["pct_saved"])
    assert row["remaining_cents"] is None or pd.isna(row["remaining_cents"])


def test_summary_columns_agree_with_the_balance() -> None:
    frame = frame_of(*THREE_MONTHS, "15/07/2026,HOTEL,INST,C,Vacation_,No,No,($150)")
    row = summary_row(frame, adjustment=50)
    assert row["category"] == "Vacation"
    assert row["monthly_cents"] == 10_000
    assert row["contributed_cents"] == 30_000
    assert row["adjustment_cents"] == 5_000
    assert row["spent_cents"] == 15_000
    assert row["balance_cents"] == 30_000 + 5_000 - 15_000


def test_summary_rows_are_in_category_order() -> None:
    summary = savings_summary(frame_of(*THREE_MONTHS), {"Vacation": bucket(), "Car Payment": bucket()})
    assert list(summary["category"]) == ["Car Payment", "Vacation"]


# --- income left after spending and saving -------------------------------


def test_planned_allocation_is_the_contributions_for_the_months_in_scope() -> None:
    buckets = {"Vacation": bucket(contribution=150), "Car Payment": bucket(contribution=100)}
    assert planned_allocation(buckets, ["2026-08"]) == 25_000
    assert planned_allocation(buckets, ["2026-07", "2026-08"]) == 50_000


def test_left_after_savings_subtracts_spend_and_savings_from_income() -> None:
    frame = frame_of(
        "05/08/2026,PAY,INST,C,Paycheck/Salary,No,No,$1000",
        "10/08/2026,FOOD,INST,C,Groceries_,No,No,($300)",
    )
    result = left_after_savings(frame, {"Vacation": bucket(contribution=150)}, ["2026-08"])
    assert result == {"income": 100_000, "spend": -30_000, "allocated": 15_000, "left": 55_000}


# --- reconciliation against cash in the bank -----------------------------


def funded() -> tuple[pd.DataFrame, dict]:
    return frame_of(*THREE_MONTHS), {"Vacation": bucket(), "Car Payment": bucket()}  # $300 each


def test_every_dollar_assigned_leaves_nothing_unallocated() -> None:
    frame, buckets = funded()
    assert savings_reconciliation(frame, buckets, 60_000) == {
        "pool": 60_000,
        "allocated": 60_000,
        "unallocated": 0,
    }


def test_cash_not_in_any_bucket_is_unallocated() -> None:
    frame, buckets = funded()
    assert savings_reconciliation(frame, buckets, 70_000)["unallocated"] == 10_000


def test_buckets_claiming_more_than_the_cash_are_over_allocated() -> None:
    frame, buckets = funded()
    assert savings_reconciliation(frame, buckets, 50_000)["unallocated"] == -10_000


def test_an_adjustment_allocates_more_cash() -> None:
    frame, buckets = funded()
    buckets["Vacation"]["adjustment"] = 100
    assert savings_reconciliation(frame, buckets, 70_000)["unallocated"] == 0


def test_an_overdrawn_bucket_counts_against_the_allocated_total() -> None:
    frame = frame_of(*THREE_MONTHS, "15/06/2026,FLIGHT,INST,C,Vacation_,No,No,($500)")
    result = savings_reconciliation(frame, {"Vacation": bucket()}, 0)
    assert result["allocated"] == -20_000


def test_no_buckets_leaves_the_whole_pool_unallocated() -> None:
    assert savings_reconciliation(frame_of(*THREE_MONTHS), {}, 12_300)["unallocated"] == 12_300


# --- the buckets file ----------------------------------------------------


def test_buckets_and_pool_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "buckets.yaml"
    buckets = {
        "Vacation": bucket(contribution=150, adjustment=-20, goal=3_000),
        "Gifts": bucket(contribution=25),
    }
    save_buckets(buckets, savings_pool=4_000, path=path)
    assert load_buckets(path) == buckets
    assert load_savings_pool(path) == 4_000


@pytest.mark.parametrize("old_name", ["current_balance", "opening_balance"])
def test_an_older_balance_field_reads_as_the_adjustment(tmp_path: Path, old_name: str) -> None:
    """Files from earlier versions keep their figure, and a leftover as_of is ignored."""
    path = tmp_path / "buckets.yaml"
    path.write_text(
        f"buckets:\n  Vacation:\n    contribution: 150\n    {old_name}: 40\n    as_of: '2026-09'\n",
        encoding="utf-8",
    )
    assert load_buckets(path)["Vacation"] == {"contribution": 150, "adjustment": 40, "goal": 0}


def test_the_pool_keeps_its_cents(tmp_path: Path) -> None:
    """The pool is a bank balance, so cents matter to the reconciliation."""
    path = tmp_path / "buckets.yaml"
    save_buckets({}, savings_pool=1234.56, path=path)
    assert load_savings_pool(path) == pytest.approx(1234.56)


def test_saving_buckets_alone_keeps_the_pool(tmp_path: Path) -> None:
    path = tmp_path / "buckets.yaml"
    save_buckets({}, savings_pool=900, path=path)
    save_buckets({"Vacation": bucket()}, path=path)
    assert load_savings_pool(path) == 900


def test_missing_file_means_no_buckets_and_no_pool(tmp_path: Path) -> None:
    assert load_buckets(tmp_path / "absent.yaml") == {}
    assert load_savings_pool(tmp_path / "absent.yaml") == 0


def test_malformed_buckets_are_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "buckets.yaml"
    path.write_text(
        "savings_pool: lots\n"
        "buckets:\n"
        "  Vacation: {contribution: 50}\n"
        "  Auto: {contribution: plenty, adjustment: -20, goal: -5}\n"
        "  NotADict: 7\n",
        encoding="utf-8",
    )
    buckets = load_buckets(path)
    assert list(buckets) == ["Vacation", "Auto"]
    assert buckets["Vacation"] == bucket(contribution=50)
    assert buckets["Auto"] == {"contribution": 0, "adjustment": -20, "goal": 0}
    assert load_savings_pool(path) == 0


def test_shipped_buckets_file_is_valid() -> None:
    buckets = load_buckets(BUCKETS_PATH)
    assert buckets, "the shipped buckets file should load at least one bucket"
    for spec in buckets.values():
        assert spec["contribution"] >= 0
        assert spec["goal"] >= 0
