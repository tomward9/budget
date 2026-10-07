"""Tests for the ingest layer.

Each test pins a quirk observed in the real Quicken export so the parsing
cannot silently regress.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from budget.ingest import (
    combine,
    describe_ingest,
    income_transactions,
    ingest,
    spending_transactions,
    total_cents,
)
from budget.schema import AmountParseError, empty_transactions, format_cents, parse_amount_to_cents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "sample_transactions.csv"

HEADER = "Date,Description,Institution,Account,Category,Is Hidden,Is Pending,Amount"


def make_csv(*rows: str) -> io.BytesIO:
    text = "\n".join([HEADER, *rows, ""])
    return io.BytesIO(text.encode("utf-8"))


# --- amount parsing ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected_cents"),
    [
        ("($164)", -16400),
        ("($54.29)", -5429),
        ("($0.01)", -1),
        ("($18)", -1800),
        ("$190 ", 19000),
        ("$0.90 ", 90),
        ("($383.98)", -38398),
        ("(1,234.56)", -123456),
        ("42.74", 4274),
    ],
)
def test_parse_amount_handles_accounting_notation(raw: str, expected_cents: int) -> None:
    assert parse_amount_to_cents(raw) == expected_cents


@pytest.mark.parametrize("raw", [None, "", "   ", "abc", "(", "($)"])
def test_parse_amount_rejects_malformed(raw: object) -> None:
    with pytest.raises(AmountParseError):
        parse_amount_to_cents(raw)


def test_format_cents_round_trips() -> None:
    assert format_cents(-16400) == "-$164.00"
    assert format_cents(19000) == "$190.00"
    assert format_cents(-1) == "-$0.01"


# --- date handling -------------------------------------------------------


def test_dates_are_day_first_not_month_first() -> None:
    """21/09/2026 must be 21 September, not 9 December."""
    stream = make_csv("21/09/2026,TEST MERCHANT,INST,MAX CHECKING,Groceries_,No,No,($10)")
    frame, _ = ingest(stream)
    assert frame["date"].iloc[0] == pd.Timestamp("2026-09-21")
    assert frame["date"].iloc[0].month == 9
    assert frame["date"].iloc[0].day == 21


def test_single_digit_days_and_months_parse() -> None:
    """The export omits leading zeros: '11/9/2026' and '3/9/2026'."""
    stream = make_csv(
        "11/9/2026,MERCHANT A,INST,MAX CHECKING,Groceries_,No,No,($10)",
        "3/9/2026,MERCHANT B,INST,MAX CHECKING,Groceries_,No,No,($10)",
    )
    frame, report = ingest(stream)
    assert report.ok
    assert len(frame) == 2
    assert sorted(frame["date"].dt.day) == [3, 11]


def test_unparseable_date_is_reported_not_fatal() -> None:
    stream = make_csv(
        "21/09/2026,GOOD MERCHANT,INST,MAX CHECKING,Groceries_,No,No,($10)",
        "not-a-date,BAD MERCHANT,INST,MAX CHECKING,Groceries_,No,No,($10)",
    )
    frame, report = ingest(stream)
    assert len(frame) == 1
    assert report.unparsed_dates == ["not-a-date"]
    assert not report.ok


# --- structural behaviour ------------------------------------------------


def test_missing_columns_report_ok() -> None:
    frame, report = ingest(io.BytesIO(b"Date,Amount\n21/09/2026,10\n"))
    assert frame.empty
    assert "Description" in report.missing_columns
    assert not report.ok


def test_empty_frame_has_canonical_dtypes() -> None:
    frame = empty_transactions()
    assert frame.empty
    assert str(frame["date"].dtype) == "datetime64[ns]"
    assert str(frame["amount_cents"].dtype) == "int64"


def test_duplicate_transactions_are_dropped() -> None:
    stream = make_csv(
        "21/09/2026,REPEATED MERCHANT,INST,MAX CHECKING,Groceries_,No,No,($10)",
        "21/09/2026,REPEATED MERCHANT,INST,MAX CHECKING,Groceries_,No,No,($10)",
        "20/09/2026,OTHER MERCHANT,INST,MAX CHECKING,Groceries_,No,No,($5)",
    )
    frame, report = ingest(stream)
    assert len(frame) == 2
    assert report.duplicate_count == 1


def test_all_rows_are_retained_regardless_of_description() -> None:
    """Nothing is dropped for looking like a transfer. The user filters
    those out of the file before uploading it."""
    stream = make_csv(
        "30/08/2026,PAYMENTS TRANSFER FROM SHARE,INST,MAX REWARDS VISA,Gasoline,No,No,$1.45",
        "31/08/2026,REGULAR SPEND,INST,MAX CHECKING,Groceries_,No,No,($10)",
    )
    frame, _ = ingest(stream)
    assert len(frame) == 2
    assert "PAYMENTS TRANSFER FROM SHARE" in set(frame["description"])


def test_trailing_underscore_stripped_for_display_only() -> None:
    stream = make_csv("21/09/2026,MERCHANT,INST,MAX CHECKING,Groceries_,No,No,($10)")
    frame, _ = ingest(stream)
    assert frame["source_category"].iloc[0] == "Groceries_"
    assert frame["display_label"].iloc[0] == "Groceries"


def test_amounts_are_stored_as_signed_cents() -> None:
    stream = make_csv(
        "21/09/2026,SPEND,INST,MAX CHECKING,Groceries_,No,No,($54.29)",
        "20/09/2026,EARN,INST,MAX CHECKING,Paycheck/Salary,No,No,$784.99",
    )
    frame, _ = ingest(stream)
    assert frame["amount_cents"].dtype == "int64"
    assert sorted(frame["amount_cents"]) == [-5429, 78499]


def test_boolean_flags_parse() -> None:
    stream = make_csv("21/09/2026,MERCHANT,INST,MAX CHECKING,Groceries_,Yes,Yes,($10)")
    frame, _ = ingest(stream)
    assert bool(frame["is_hidden"].iloc[0]) is True
    assert bool(frame["is_pending"].iloc[0]) is True


# --- the sample file ----------------------------------------------------


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample file not present")
def test_real_sample_file_parses_completely() -> None:
    frame, report = ingest(SAMPLE_CSV)
    assert report.ok, "\n".join(report.summary_lines())
    assert report.rows_parsed == report.row_count
    assert report.date_parse_rate == 1.0
    assert frame["amount_cents"].dtype == "int64"
    assert report.date_min == pd.Timestamp("2026-01-01")
    assert report.date_max == pd.Timestamp("2026-12-31")
    assert "CLUB CHECKING" in report.accounts
    assert frame["date"].is_monotonic_increasing


@pytest.mark.skipif(not SAMPLE_CSV.exists(), reason="sample file not present")
def test_real_sample_paycheque_is_split_across_two_rows() -> None:
    """The paycheque arrives as two same-day rows, which is why the
    allocation rule groups by date rather than counting transactions."""
    frame, _ = ingest(SAMPLE_CSV)
    paycheques = frame[frame["source_category"] == "Paycheck/Salary"]
    assert paycheques["date"].nunique() == 24
    assert len(paycheques) == 48
    per_day = paycheques.groupby("date").size()
    assert set(per_day) == {2}


# --- helpers -------------------------------------------------------------


def test_combine_drops_cross_file_duplicates() -> None:
    first, _ = ingest(
        make_csv("21/09/2026,MERCHANT,INST,MAX CHECKING,Groceries_,No,No,($10)")
    )
    second, _ = ingest(
        make_csv(
            "21/09/2026,MERCHANT,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "20/09/2026,ANOTHER,INST,MAX CHECKING,Dining,No,No,($5)",
        )
    )
    combined, removed = combine([first, second])
    assert len(combined) == 2
    assert removed == 1


def test_combine_handles_all_empty() -> None:
    combined, removed = combine([empty_transactions(), empty_transactions()])
    assert combined.empty
    assert removed == 0


def test_spending_and_income_split() -> None:
    frame, _ = ingest(
        make_csv(
            "21/09/2026,SPEND,INST,MAX CHECKING,Groceries_,No,No,($10)",
            "20/09/2026,EARN,INST,MAX CHECKING,Paycheck/Salary,No,No,$100",
        )
    )
    assert total_cents(spending_transactions(frame)) == -1000
    assert total_cents(income_transactions(frame)) == 10000
    assert total_cents(frame) == 9000


def test_describe_ingest_mentions_net_movement() -> None:
    frame, report = ingest(
        make_csv("21/09/2026,SPEND,INST,MAX CHECKING,Groceries_,No,No,($10)")
    )
    text = describe_ingest(frame, report)
    assert "money out" in text
    assert "-$10.00" in text
