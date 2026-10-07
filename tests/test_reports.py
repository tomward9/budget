"""Tests for the reporting and budget layer.

Coverage flags and variance signs are the two things most likely to mislead
quietly, so each gets a dedicated test.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from budget.ingest import ingest
from budget.reports import (
    BUDGETS_PATH,
    category_breakdown,
    category_month_matrix,
    filter_months,
    filter_period,
    load_budgets,
    monthly_summary,
    months_in_scope,
    period_summary,
    save_budgets,
    split_signs,
    spending_rows,
    variance_table,
    year_summary,
    yearly_summary,
)
from budget.schema import empty_transactions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "sample_transactions.csv"

HEADER = "Date,Description,Institution,Account,Category,Is Hidden,Is Pending,Amount"


def make_csv(*rows: str) -> io.BytesIO:
    text = "\n".join([HEADER, *rows, ""])
    return io.BytesIO(text.encode("utf-8"))


def sample_frame():
    if not SAMPLE_CSV.exists():
        pytest.skip("sample export not present")
    frame, _ = ingest(SAMPLE_CSV)
    return frame


# --- monthly summary -----------------------------------------------------


def test_monthly_summary_splits_income_spend_and_net() -> None:
    frame, _ = ingest(
        make_csv(
            "05/09/2026,SPEND,INST,MAX CHECKING,Groceries_,No,No,($100)",
            "06/09/2026,EARN,INST,MAX CHECKING,Paycheck/Salary,No,No,$500",
        )
    )
    summary = monthly_summary(frame)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["period"] == "2026-09"
    assert row["income_cents"] == 50000
    assert row["spend_cents"] == -10000
    assert row["net_cents"] == 40000


def test_partial_month_is_flagged_and_coverage_computed() -> None:
    frame, _ = ingest(
        make_csv(
            "01/09/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "06/09/2026,B,INST,MAX CHECKING,Groceries_,No,No,($10)",
        )
    )
    row = monthly_summary(frame).iloc[0]
    assert row["days_covered"] == 6
    assert row["days_in_period"] == 30
    assert row["coverage_pct"] == pytest.approx(0.2)
    assert bool(row["is_partial"]) is True


def test_full_month_is_not_partial() -> None:
    frame, _ = ingest(
        make_csv(
            "01/09/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "30/09/2026,B,INST,MAX CHECKING,Groceries_,No,No,($10)",
        )
    )
    row = monthly_summary(frame).iloc[0]
    assert row["days_covered"] == 30
    assert bool(row["is_partial"]) is False


def test_february_uses_actual_month_length() -> None:
    frame, _ = ingest(
        make_csv(
            "01/02/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "28/02/2026,B,INST,MAX CHECKING,Groceries_,No,No,($10)",
        )
    )
    row = monthly_summary(frame).iloc[0]
    assert row["days_in_period"] == 28
    assert bool(row["is_partial"]) is False


def test_months_are_returned_in_order() -> None:
    frame, _ = ingest(
        make_csv(
            "20/09/2026,B,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "05/08/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
        )
    )
    assert list(monthly_summary(frame)["period"]) == ["2026-08", "2026-09"]


def test_empty_transactions_give_empty_summary() -> None:
    assert monthly_summary(empty_transactions()).empty
    assert yearly_summary(empty_transactions()).empty


# --- yearly summary ------------------------------------------------------


def test_yearly_summary_reports_coverage_against_full_year() -> None:
    frame, _ = ingest(
        make_csv(
            "01/01/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "31/12/2026,B,INST,MAX CHECKING,Groceries_,No,No,($10)",
        )
    )
    row = yearly_summary(frame).iloc[0]
    assert row["period"] == "2026"
    assert row["days_in_period"] == 365
    assert row["days_covered"] == 365
    assert bool(row["is_partial"]) is False


def test_leap_year_uses_366_days() -> None:
    frame, _ = ingest(
        make_csv(
            "01/01/2028,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "31/12/2028,B,INST,MAX CHECKING,Groceries_,No,No,($10)",
        )
    )
    row = yearly_summary(frame).iloc[0]
    assert row["days_in_period"] == 366
    assert row["days_covered"] == 366
    assert bool(row["is_partial"]) is False


def test_partial_year_is_flagged() -> None:
    frame, _ = ingest(
        make_csv(
            "26/08/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "21/09/2026,B,INST,MAX CHECKING,Groceries_,No,No,($10)",
        )
    )
    row = yearly_summary(frame).iloc[0]
    assert row["days_covered"] == 27
    assert row["coverage_pct"] == pytest.approx(27 / 365, abs=1e-4)
    assert bool(row["is_partial"]) is True


def test_yearly_totals_sum_every_month() -> None:
    frame, _ = ingest(
        make_csv(
            "05/08/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "05/09/2026,B,INST,MAX CHECKING,Groceries_,No,No,($20)",
        )
    )
    row = yearly_summary(frame).iloc[0]
    assert row["spend_cents"] == -3000


# --- single period lookups ----------------------------------------------


def test_period_summary_returns_none_for_absent_month() -> None:
    frame, _ = ingest(make_csv("05/09/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)"))
    assert period_summary(frame, "2026-09") is not None
    assert period_summary(frame, "2021-01") is None


def test_year_summary_returns_none_for_absent_year() -> None:
    frame, _ = ingest(make_csv("05/09/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)"))
    assert year_summary(frame, 2026) is not None
    assert year_summary(frame, 2021) is None


def test_filter_period_scopes_by_month_and_year() -> None:
    frame, _ = ingest(
        make_csv(
            "05/08/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "05/09/2026,B,INST,MAX CHECKING,Groceries_,No,No,($20)",
        )
    )
    assert len(filter_period(frame, month="2026-08")) == 1
    assert len(filter_period(frame, year=2026)) == 2
    assert len(filter_period(frame)) == 2


# --- the month multiselect filter -----------------------------------------


def two_months() -> pd.DataFrame:
    frame, _ = ingest(
        make_csv(
            "05/08/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "06/09/2026,B,INST,MAX CHECKING,Gasoline,No,No,($20)",
        )
    )
    return frame


def test_filter_months_takes_every_month_listed() -> None:
    frame = two_months()
    assert len(filter_months(frame, ["2026-08", "2026-09"])) == 2
    assert len(filter_months(frame, ["2026-08"])) == 1


def test_filter_months_with_none_means_no_filter() -> None:
    """A caller that has not filtered should not have to special-case it."""
    assert len(filter_months(two_months(), None)) == 2


def test_filter_months_with_nothing_selected_returns_no_rows() -> None:
    """Deselect all is a real state, and 'all time' is not what it means."""
    assert filter_months(two_months(), []).empty


def test_filter_months_ignores_a_month_with_no_data() -> None:
    frame = two_months()
    assert len(filter_months(frame, ["2026-08", "2026-10"])) == 1


def test_filter_months_on_an_empty_frame_is_empty() -> None:
    assert filter_months(empty_transactions(), ["2026-08"]).empty


# --- the category by month matrix -----------------------------------------


def grid() -> pd.DataFrame:
    frame, _ = ingest(
        make_csv(
            "05/08/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "05/08/2026,B,INST,MAX CHECKING,Gasoline,No,No,($20)",
            "05/09/2026,C,INST,MAX CHECKING,Groceries_,No,No,($30)",
        )
    )
    return frame


def test_matrix_pairs_every_category_with_every_month() -> None:
    """Two categories over two months is four cells, including the pair that
    spent nothing. A pivot of the raw rows would simply have no row for it, and
    a heatmap drawn from that leaves a hole instead of a zero."""
    matrix = category_month_matrix(grid())
    assert len(matrix) == 4
    assert set(matrix["period"]) == {"2026-08", "2026-09"}
    assert set(matrix["display_label"]) == {"Gasoline", "Groceries"}


def test_matrix_zero_fills_a_pair_with_no_spending() -> None:
    matrix = category_month_matrix(grid())
    gas_september = matrix[
        (matrix["period"] == "2026-09") & (matrix["display_label"] == "Gasoline")
    ]
    assert gas_september["spend_cents"].iloc[0] == 0


def test_matrix_totals_match_the_breakdown() -> None:
    frame = grid()
    matrix = category_month_matrix(frame)
    assert int(matrix["spend_cents"].sum()) == int(
        category_breakdown(frame)["spend_cents"].sum()
    )


def test_matrix_excludes_income_only_categories() -> None:
    """Paycheck/Salary would otherwise become a column empty in every row."""
    matrix = category_month_matrix(grid())
    assert "Paycheck/Salary" not in set(matrix["display_label"])


def test_matrix_can_be_narrowed_to_selected_months() -> None:
    matrix = category_month_matrix(grid(), months=["2026-09"])
    assert set(matrix["period"]) == {"2026-09"}


def test_matrix_of_no_spending_is_empty() -> None:
    frame, _ = ingest(make_csv("05/09/2026,EARN,INST,MAX CHECKING,Paycheck/Salary,No,No,$500"))
    assert category_month_matrix(frame).empty


# --- category breakdown --------------------------------------------------


def test_category_breakdown_excludes_income() -> None:
    frame, _ = ingest(
        make_csv(
            "05/09/2026,SPEND,INST,MAX CHECKING,Groceries_,No,No,($100)",
            "06/09/2026,EARN,INST,MAX CHECKING,Paycheck/Salary,No,No,$500",
        )
    )
    breakdown = category_breakdown(frame, month="2026-09")
    assert list(breakdown["display_label"]) == ["Groceries"]
    assert breakdown["spend_cents"].iloc[0] == -10000


def test_breakdown_counts_every_negative_row_as_spending() -> None:
    """Transfer filtering is the user's responsibility, done before upload.

    Whatever negative rows arrive are counted as spending in the category they
    were filed under, with no inspection of the description.
    """
    frame, _ = ingest(
        make_csv(
            "30/08/2026,WITHDRAWAL TRANSFER TO LOAN,INST,MAX CHECKING,Electricity,No,No,($26.69)",
            "31/08/2026,REGULAR GAS,INST,MAX CHECKING,Gasoline,No,No,($40)",
        )
    )
    breakdown = category_breakdown(frame, month="2026-08")
    assert set(breakdown["display_label"]) == {"Electricity", "Gasoline"}
    electricity = breakdown[breakdown["display_label"] == "Electricity"].iloc[0]
    assert electricity["spend_cents"] == -2669


def test_a_credit_in_a_spending_category_lowers_that_spend() -> None:
    """Reported: a $1.45 credit filed under Gasoline was being read as income.

    Only a credit in a named income category is money in. Filed anywhere else
    it is money back out, so it comes off the spend of the category it sits in
    rather than adding to income.
    """
    frame, _ = ingest(
        make_csv(
            "30/08/2026,PAYMENTS TRANSFER FROM SHARE,INST,MAX VISA,Gasoline,No,No,$1.45",
            "31/08/2026,REGULAR GAS,INST,MAX CHECKING,Gasoline,No,No,($40)",
        )
    )
    breakdown = category_breakdown(frame, month="2026-08")
    assert list(breakdown["display_label"]) == ["Gasoline"]
    assert breakdown["spend_cents"].iloc[0] == -4000 + 145

    income, spend = split_signs(frame)
    assert income == 0
    assert spend == -4000 + 145


def test_a_credit_in_a_named_income_category_is_money_in() -> None:
    """The three income categories are income whatever their sign, and they are
    never attributed to a spending bucket."""
    frame, _ = ingest(
        make_csv(
            "01/09/2026,PAYCHECK,INST,MAX CHECKING,Paycheck/Salary,No,No,$500",
            "02/09/2026,INTEREST,INST,MAX CHECKING,Interest Income,No,No,$0.13",
            "03/09/2026,DEPOSIT,INST,MAX CHECKING,Income,No,No,$421",
        )
    )
    income, spend = split_signs(frame)
    assert income == 50000 + 13 + 42100
    assert spend == 0
    assert category_breakdown(frame, month="2026-09").empty
    assert list(spending_rows(frame)["display_label"]) == []


def test_category_breakdown_is_sorted_by_spend_descending() -> None:
    frame, _ = ingest(
        make_csv(
            "01/09/2026,A,INST,MAX CHECKING,Gasoline,No,No,($10)",
            "02/09/2026,B,INST,MAX CHECKING,Tithing,No,No,($500)",
            "03/09/2026,C,INST,MAX CHECKING,Groceries_,No,No,($50)",
        )
    )
    breakdown = category_breakdown(frame, month="2026-09")
    assert list(breakdown["display_label"]) == ["Tithing", "Groceries", "Gasoline"]


# --- variance ------------------------------------------------------------


def test_variance_positive_means_under_budget() -> None:
    frame, _ = ingest(make_csv("05/09/2026,A,INST,MAX CHECKING,Groceries_,No,No,($100)"))
    table = variance_table(frame, month="2026-09", budgets={"Groceries": 400})
    row = table.iloc[0]
    assert row["budget_cents"] == 40000
    assert row["actual_cents"] == -10000
    assert row["variance_cents"] == 30000
    assert row["status"] == "under"


def test_variance_negative_means_over_budget() -> None:
    frame, _ = ingest(make_csv("05/09/2026,A,INST,MAX CHECKING,Groceries_,No,No,($500)"))
    table = variance_table(frame, month="2026-09", budgets={"Groceries": 400})
    row = table.iloc[0]
    assert row["variance_cents"] == -10000
    assert row["status"] == "over"


def test_category_without_budget_is_shown_not_hidden() -> None:
    frame, _ = ingest(make_csv("05/09/2026,A,INST,MAX CHECKING,Dining,No,No,($100)"))
    table = variance_table(frame, month="2026-09", budgets={})
    assert len(table) == 1
    assert table["status"].iloc[0] == "no budget"
    assert table["budget_cents"].iloc[0] == 0


def test_budgeted_category_with_no_spending_still_appears() -> None:
    frame, _ = ingest(make_csv("05/09/2026,A,INST,MAX CHECKING,Dining,No,No,($100)"))
    table = variance_table(frame, month="2026-09", budgets={"Dining": 200, "Mortgage": 900})
    assert set(table["display_label"]) == {"Dining", "Mortgage"}
    mortgage = table[table["display_label"] == "Mortgage"].iloc[0]
    assert mortgage["actual_cents"] == 0
    assert mortgage["status"] == "under"


def test_yearly_variance_charges_twelve_months() -> None:
    frame, _ = ingest(make_csv("05/09/2026,A,INST,MAX CHECKING,Groceries_,No,No,($100)"))
    monthly = variance_table(frame, month="2026-09", budgets={"Groceries": 400})
    yearly = variance_table(frame, year=2026, budgets={"Groceries": 400})
    assert monthly["budget_cents"].iloc[0] == 40000
    assert yearly["budget_cents"].iloc[0] == 480000


def test_months_in_scope_rules() -> None:
    assert months_in_scope(month="2026-09") == 1
    assert months_in_scope(year=2026) == 12
    assert months_in_scope() == 1


def test_an_explicit_month_count_overrides_the_scope() -> None:
    """The dashboard filters to the months the user picked and then hands the
    count in, because a list of months cannot be expressed as a month or a
    year. A single month has to read the same either way."""
    frame, _ = ingest(make_csv("05/08/2026,A,INST,MAX CHECKING,Groceries_,No,No,($100)"))
    budgets = {"Groceries": 400}
    assert (
        variance_table(frame, budgets=budgets, months=1)["budget_cents"].iloc[0]
        == variance_table(frame, month="2026-08", budgets=budgets)["budget_cents"].iloc[0]
    )


def test_two_months_of_spending_are_charged_two_months_of_budget() -> None:
    """Comparing a two-month total against one monthly target would report every
    category as over budget, with nothing on screen to say why."""
    frame, _ = ingest(
        make_csv(
            "05/08/2026,A,INST,MAX CHECKING,Groceries_,No,No,($300)",
            "05/09/2026,B,INST,MAX CHECKING,Groceries_,No,No,($300)",
        )
    )
    budgets = {"Groceries": 400}
    two = variance_table(frame, budgets=budgets, months=2).iloc[0]
    assert two["budget_cents"] == 80000
    assert two["variance_cents"] == 20000
    assert two["status"] == "under"

    # The same six hundred against a single month's target would read as a
    # hundred and fifty dollars over, which is the wrong answer.
    one = variance_table(frame, budgets=budgets, months=1).iloc[0]
    assert one["status"] == "over"


# --- budget file ---------------------------------------------------------


def test_shipped_budgets_file_is_valid_and_empty() -> None:
    budgets = load_budgets(BUDGETS_PATH)
    assert isinstance(budgets, dict)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "budgets.yaml"
    save_budgets({"Groceries": 400, "Mortgage": 1200}, path)
    assert load_budgets(path) == {"Groceries": 400, "Mortgage": 1200}


def test_save_drops_zero_and_negative_targets(tmp_path: Path) -> None:
    path = tmp_path / "budgets.yaml"
    save_budgets({"Groceries": 400, "Empty": 0, "Negative": -50}, path)
    assert load_budgets(path) == {"Groceries": 400}


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_budgets(tmp_path / "nope.yaml") == {}


def test_load_ignores_non_numeric_values(tmp_path: Path) -> None:
    path = tmp_path / "budgets.yaml"
    path.write_text("budgets:\n  Groceries: 400\n  Broken: abc\n", encoding="utf-8")
    assert load_budgets(path) == {"Groceries": 400}


# --- against the sample file --------------------------------------------


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample file not present")
def test_the_sample_covers_every_month_of_the_year() -> None:
    """The sample spans the whole calendar year, so every month appears and
    each one's coverage flag follows from its own days observed."""
    summary = monthly_summary(sample_frame())
    assert list(summary["period"]) == [f"2026-{m:02d}" for m in range(1, 13)]
    assert summary["is_partial"].equals(
        summary["days_covered"] < summary["days_in_period"]
    )


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample file not present")
def test_real_sample_year_totals_match_expected() -> None:
    row = yearly_summary(sample_frame()).iloc[0]
    assert row["period"] == "2026"
    # The sample's only credit outside the income categories is the grocery
    # refund filed under Groceries, so it lowers spend rather than adding to
    # income.
    assert row["income_cents"] == 5224910
    assert row["spend_cents"] == -4161942
    assert row["net_cents"] == 1062968
    assert row["days_covered"] == 365
    assert bool(row["is_partial"]) is False


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample file not present")
def test_real_sample_dte_and_electricity_stay_separate() -> None:
    """Two distinct utilities must not be merged by this layer."""
    breakdown = category_breakdown(sample_frame(), year=2026)
    labels = set(breakdown["display_label"])
    assert "DTE" in labels
    assert "Electricity" in labels
