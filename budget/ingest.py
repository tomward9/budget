"""Read Quicken transaction exports into the canonical schema.

The export is a fixed eight-column format with two quirks that defeat naive
loading: dates are DD/MM/YYYY without leading zeros, and negative amounts use
accounting parentheses. Both are handled here so the rest of the application
only ever sees typed, cent-based data.

Rows that cannot be parsed are skipped and recorded in an ImportReport rather
than aborting the load, so a partially bad file still yields usable data
alongside a list of what went wrong.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .schema import (
    CANONICAL_DTYPES,
    DATE_FORMAT,
    QUICKEN_COLUMN_MAP,
    AmountParseError,
    ImportReport,
    empty_transactions,
    format_cents,
    make_txn_id,
    parse_amount_to_cents,
    strip_trailing_underscore,
)

# When this file was loaded, so the app can tell a stale copy from a fresh one.
LOADED_MTIME = Path(__file__).stat().st_mtime

# Values accepted for the boolean flag columns, which arrive as 'Yes'/'No'.
_TRUE_VALUES = frozenset({"yes", "y", "true", "t", "1"})
_FALSE_VALUES = frozenset({"no", "n", "false", "f", "0", ""})


def _to_bool(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return False


def _load_raw(source: str | Path | io.BytesIO | pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Read a source into a raw string-typed frame, without type guessing."""
    if isinstance(source, pd.DataFrame):
        return source.copy(), "DataFrame"
    if isinstance(source, (str, Path)):
        path = Path(source)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
        return frame, path.name
    frame = pd.read_csv(source, dtype=str, keep_default_na=False, na_values=[])
    return frame, getattr(source, "name", "uploaded.csv")


def _validate_columns(raw: pd.DataFrame, report: ImportReport) -> None:
    missing = [name for name in QUICKEN_COLUMN_MAP if name not in raw.columns]
    report.missing_columns = missing


def ingest(source: str | Path | io.BytesIO | pd.DataFrame) -> tuple[pd.DataFrame, ImportReport]:
    """Parse one Quicken export into canonical transactions.

    Returns the transactions and a report describing what was read. The
    returned frame has one row per successfully parsed transaction, with
    duplicate identifiers removed.
    """
    raw, report_source = _load_raw(source)
    report = ImportReport(source=report_source, row_count=len(raw))
    _validate_columns(raw, report)

    if report.missing_columns:
        return empty_transactions(), report

    frame = raw.rename(columns=QUICKEN_COLUMN_MAP)

    # Dates are parsed with an explicit format. Inference would read
    # 21/09/2026 as December 9th under US conventions, and the sample file
    # contains 36 such rows out of 39.
    parsed_dates = pd.to_datetime(frame["date"], format=DATE_FORMAT, errors="coerce")
    bad_dates = parsed_dates.isna()
    for value in frame.loc[bad_dates, "date"].head(10):
        report.unparsed_dates.append(str(value))

    report.untrimmed_amounts = int(
        sum(1 for value in frame["amount"] if str(value) != str(value).strip())
    )

    records: list[dict[str, object]] = []
    seen: set[str] = set()

    for position, row in enumerate(frame.itertuples(index=False), start=2):
        if bad_dates.iloc[position - 2]:
            report.failed_rows.append((position, f"unparseable date {row.date!r}"))
            continue

        try:
            amount_cents = parse_amount_to_cents(row.amount)
        except AmountParseError as exc:
            report.unparsed_amounts.append(str(row.amount))
            report.failed_rows.append((position, str(exc)))
            continue

        date = parsed_dates.iloc[position - 2]
        description = str(row.description).strip()
        account = str(row.account).strip()

        txn_id = make_txn_id(date, description, amount_cents, account)
        if txn_id in seen:
            report.duplicate_count += 1
            if len(report.duplicate_examples) < 5:
                report.duplicate_examples.append(f"{date.date()}  {description}")
            continue
        seen.add(txn_id)

        source_category = str(row.source_category).strip()
        records.append(
            {
                "txn_id": txn_id,
                "date": date,
                "description": description,
                "institution": str(row.institution).strip(),
                "account": account,
                "source_category": source_category,
                "display_label": strip_trailing_underscore(source_category),
                "amount_cents": amount_cents,
                "is_hidden": _to_bool(row.is_hidden),
                "is_pending": _to_bool(row.is_pending),
            }
        )

    if not records:
        return empty_transactions(), report

    transactions = pd.DataFrame.from_records(records)
    for column, dtype in CANONICAL_DTYPES.items():
        transactions[column] = transactions[column].astype(dtype)

    transactions = transactions.sort_values(["date", "description"], kind="stable").reset_index(
        drop=True
    )

    report.rows_parsed = len(transactions)
    report.date_min = transactions["date"].min()
    report.date_max = transactions["date"].max()
    report.accounts = dict(
        transactions.groupby("account", observed=True).size().sort_values(ascending=False).items()
    )

    return transactions, report


def combine(
    frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, int]:
    """Concatenate per-file results, dropping duplicate transactions.

    Re-uploading a file whose date range overlaps one already loaded should
    not double-count spending, so identifier collisions are resolved here
    rather than in the dashboard.
    """
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return empty_transactions(), 0

    combined = pd.concat(usable, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["txn_id"], keep="first")
    combined = combined.sort_values(["date", "description"], kind="stable").reset_index(drop=True)
    return combined, before - len(combined)


def spending_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Rows where money left an account."""
    if transactions.empty:
        return transactions
    return transactions[transactions["amount_cents"] < 0]


def income_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Rows where money entered an account."""
    if transactions.empty:
        return transactions
    return transactions[transactions["amount_cents"] > 0]


def total_cents(transactions: pd.DataFrame) -> int:
    """Net movement across a set of transactions, in cents."""
    if transactions.empty:
        return 0
    return int(transactions["amount_cents"].sum())


def describe_ingest(transactions: pd.DataFrame, report: ImportReport) -> str:
    """A short text block summarising an import, for the dashboard."""
    lines = report.summary_lines()
    if not transactions.empty:
        spend = total_cents(spending_transactions(transactions))
        income = total_cents(income_transactions(transactions))
        lines.append("")
        lines.append(f"Net movement: {format_cents(income + spend)}")
        lines.append(f"  money out: {format_cents(spend)}")
        lines.append(f"  money in:  {format_cents(income)}")
        categories = transactions["display_label"].nunique()
        lines.append(f"Categories present: {categories}")
    return "\n".join(lines)
