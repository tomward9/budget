"""Canonical schema for budget transactions.

Amounts are held as signed integer cents throughout. Float arithmetic on
currency accumulates error ($3.99 becomes 3.9899999999999995) and a balance
that is off by a cent per transaction is off by a meaningful amount over a
year of data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import os

import pandas as pd

# When this file was loaded, so the app can tell a stale copy from a fresh one.
LOADED_MTIME = os.path.getmtime(__file__)

# Money is stored in cents to keep arithmetic exact.
CENTS_PER_UNIT = 100

# The Quicken export uses DD/MM/YYYY and omits leading zeros on single-digit
# days and months. 36 of the 39 rows in the sample file are ambiguous under
# US month-first parsing, so the format is always explicit.
DATE_FORMAT = "%d/%m/%Y"

# Columns we expect to find in a Quicken export, mapped to canonical names.
QUICKEN_COLUMN_MAP = {
    "Date": "date",
    "Description": "description",
    "Institution": "institution",
    "Account": "account",
    "Category": "source_category",
    "Is Hidden": "is_hidden",
    "Is Pending": "is_pending",
    "Amount": "amount",
}

CANONICAL_COLUMNS = [
    "txn_id",
    "date",
    "description",
    "institution",
    "account",
    "source_category",
    "display_label",
    "amount_cents",
    "is_hidden",
    "is_pending",
]

CANONICAL_DTYPES = {
    "txn_id": "string",
    "date": "datetime64[ns]",
    "description": "string",
    "institution": "string",
    "account": "string",
    "source_category": "string",
    "display_label": "string",
    "amount_cents": "int64",
    "is_hidden": "bool",
    "is_pending": "bool",
}

# Categories whose credits are genuine money in. A credit filed anywhere else
# is a refund or a transfer sitting in a spending category, so it comes off that
# category's spend instead of adding to income. 'Paycheck/Salary' stays the
# allocation trigger on its own.
INCOME_CATEGORIES = frozenset({"Paycheck/Salary", "Interest Income", "Income"})

SALARY_CATEGORY = "Paycheck/Salary"

# Amounts below this are typically partial authorisations, where a merchant
# places a small hold that is replaced by the full charge.
PARTIAL_AUTHORISATION_CENTS = 100


class AmountParseError(ValueError):
    """Raised when an amount cannot be interpreted as currency."""


def parse_amount_to_cents(raw: object) -> int:
    """Parse a Quicken amount string into signed integer cents.

    Quicken writes negative amounts in accounting notation with parentheses,
    e.g. '($164)' and '($54.29)', and positive amounts as '$190' with a
    trailing space. Plain float() fails on every one of those, so the sign is
    detected before the currency symbol and separators are stripped.

    Returns positive cents for money in and negative cents for money out.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        raise AmountParseError("amount is empty")

    text = str(raw).strip()
    if not text:
        raise AmountParseError("amount is empty")

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    text = text.replace("$", "").replace(",", "").replace(" ", "").strip()

    if text.startswith("-"):
        negative = True
        text = text[1:]
    elif text.startswith("+"):
        text = text[1:]

    if not text:
        raise AmountParseError(f"amount has no digits: {raw!r}")

    try:
        value = float(text)
    except ValueError as exc:
        raise AmountParseError(f"amount is not numeric: {raw!r}") from exc

    cents = int(round(value * CENTS_PER_UNIT))
    return -cents if negative else cents


def cents_to_float(amount_cents: int) -> float:
    """Convert cents back to a float for charting only.

    Never feed the result back into arithmetic; it reintroduces the float
    error that cents storage exists to avoid.
    """
    return amount_cents / CENTS_PER_UNIT


def format_cents(amount_cents: int) -> str:
    """Render cents as a currency string, e.g. -16400 -> '-$164.00'."""
    sign = "-" if amount_cents < 0 else ""
    whole, remainder = divmod(abs(int(amount_cents)), CENTS_PER_UNIT)
    return f"{sign}${whole:,}.{remainder:02d}"


def strip_trailing_underscore(source_category: object) -> str:
    """Return a display label for a source category.

    Quicken appends an underscore to several categories in the export
    ('Groceries_', 'Income_'), but not to all of them ('Eating Out',
    'Gasoline'). The underscore is stripped for display only; the original
    value stays in source_category.
    """
    if source_category is None or pd.isna(source_category):
        return "Uncategorized"
    return str(source_category).strip().rstrip("_") or "Uncategorized"


def make_txn_id(
    date: pd.Timestamp,
    description: str,
    amount_cents: int,
    account: str,
) -> str:
    """Build a stable identifier for a transaction.

    Derived from the fields that make a transaction distinct rather than from
    the row position, so re-uploading an overlapping file range does not
    create phantom duplicates.
    """
    return f"{pd.Timestamp(date).date()}|{account}|{description}|{amount_cents}"


def empty_transactions() -> pd.DataFrame:
    """An empty frame with canonical columns and dtypes.

    Lets downstream code rely on column presence and types even before any
    file has been loaded.
    """
    return pd.DataFrame({name: pd.Series(dtype="string") for name in CANONICAL_COLUMNS}).astype(
        CANONICAL_DTYPES
    )


@dataclass
class ImportReport:
    """Outcome of parsing one uploaded CSV.

    Collected rather than raised so the dashboard can show a full picture of
    what happened instead of stopping at the first bad row.
    """

    source: str = ""
    row_count: int = 0
    rows_parsed: int = 0
    unparsed_dates: list[str] = field(default_factory=list)
    unparsed_amounts: list[str] = field(default_factory=list)
    duplicate_count: int = 0
    duplicate_examples: list[str] = field(default_factory=list)
    accounts: dict[str, int] = field(default_factory=dict)
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    untrimmed_amounts: int = 0
    missing_columns: list[str] = field(default_factory=list)
    failed_rows: list[tuple[int, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the file parsed without structural problems."""
        return not (
            self.missing_columns
            or self.unparsed_dates
            or self.unparsed_amounts
            or self.failed_rows
        )

    @property
    def date_parse_rate(self) -> float:
        if self.row_count == 0:
            return 0.0
        return 1.0 - (len(self.unparsed_dates) / self.row_count)

    def summary_lines(self) -> list[str]:
        """Human-readable findings for the dashboard."""
        lines = [
            f"Rows: {self.rows_parsed} of {self.row_count} parsed",
            f"Dates parsed: {self.date_parse_rate:.1%}",
        ]
        if self.date_min is not None and self.date_max is not None:
            span = self.date_max.date() - self.date_min.date()
            lines.append(
                f"Date range: {self.date_min.date()} to {self.date_max.date()} ({span.days} days)"
            )
        if self.duplicate_count:
            lines.append(f"Duplicates dropped: {self.duplicate_count}")
        if self.accounts:
            breakdown = ", ".join(f"{name} ({count})" for name, count in self.accounts.items())
            lines.append(f"Accounts: {breakdown}")
        if self.untrimmed_amounts:
            lines.append(f"Untrimmed amount values: {self.untrimmed_amounts}")
        if self.missing_columns:
            lines.append(f"Missing columns: {', '.join(self.missing_columns)}")
        if self.unparsed_dates:
            lines.append(f"Unparsed dates ({len(self.unparsed_dates)}): "
                         f"{', '.join(self.unparsed_dates[:5])}")
        if self.unparsed_amounts:
            lines.append(f"Unparsed amounts ({len(self.unparsed_amounts)}): "
                         f"{', '.join(self.unparsed_amounts[:5])}")
        if self.failed_rows:
            lines.append(f"Rows dropped: {len(self.failed_rows)}")
            for index, reason in self.failed_rows[:5]:
                lines.append(f"  row {index}: {reason}")
        return lines
