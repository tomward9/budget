"""Aggregate transactions into monthly and yearly summaries.

Every rollup carries coverage information. A month or year built from a
partial slice of the account history produces a total that is arithmetically
correct but easy to misread as a full-period figure, so `days_covered`,
`coverage_pct` and `is_partial` travel with the numbers rather than being
recomputed by the caller.

All money stays in integer cents.
"""

from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd
import yaml

from budget.schema import INCOME_CATEGORIES

# When this file was loaded, so the app can tell a stale copy from a fresh one.
LOADED_MTIME = Path(__file__).stat().st_mtime

BUDGETS_PATH = Path(__file__).resolve().parent / "rules" / "budgets.yaml"
BUCKETS_PATH = Path(__file__).resolve().parent / "rules" / "buckets.yaml"

SUMMARY_COLUMNS = [
    "period",
    "income_cents",
    "spend_cents",
    "net_cents",
    "txn_count",
    "first_date",
    "last_date",
    "days_covered",
    "days_in_period",
    "coverage_pct",
    "is_partial",
]

# A February that is fully observed has 28 days, so the completeness test is
# expressed against the actual length of the period rather than a fixed
# number of days.


def _days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def _money_in(frame: pd.DataFrame) -> pd.Series:
    """Which rows are money in: a credit filed under a named income category.

    A credit anywhere else is a refund or a transfer sitting in a spending
    category, so it does not count as income.
    """
    return (frame["amount_cents"] > 0) & frame["display_label"].isin(
        INCOME_CATEGORIES
    )


def spending_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows that count as money out.

    Every debit counts, wherever it was filed. A credit in a named income
    category is money in and is left out; a credit anywhere else comes off the
    spend of the category it sits in, so the two must be kept apart by name
    rather than by sign alone.
    """
    if frame.empty:
        return frame
    money_in = _money_in(frame)
    return frame[(frame["amount_cents"] != 0) & ~money_in]


def split_signs(frame: pd.DataFrame) -> tuple[int, int]:
    """Return (income_cents, spend_cents) for a set of rows.

    Income is a credit in a named income category; spend is everything else
    that moved money, including a credit that lowers what a spending category
    spent. Spend stays negative, so the two figures can be summed directly
    into a net without any further sign juggling.
    """
    if frame.empty:
        return 0, 0
    amounts = frame["amount_cents"]
    money_in = _money_in(frame)
    income = int(amounts[money_in].sum())
    spend = int(amounts[(amounts != 0) & ~money_in].sum())
    return income, spend


def _coverage(
    first: pd.Timestamp, last: pd.Timestamp, days_in_period: int
) -> tuple[int, float, bool]:
    """Days observed, share of the period covered, and whether it is partial.

    Counting from the first to the last transaction is a lower bound on
    coverage: a period with no activity in the middle still counts those
    days, which is intentional since the goal is to reflect the span of
    data present rather than to infer gaps.
    """
    if first is None or pd.isna(first) or last is None or pd.isna(last):
        return 0, 0.0, True
    days_covered = int((last - first).days + 1)
    coverage = days_covered / days_in_period if days_in_period else 0.0
    return days_covered, round(coverage, 4), days_covered < days_in_period


def _rows_to_summary(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        frame = pd.DataFrame(columns=SUMMARY_COLUMNS)
        return frame
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def monthly_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Income, spend and net per calendar month, with coverage flags."""
    if transactions.empty:
        return _rows_to_summary([])

    frame = transactions.copy()
    frame["_month"] = frame["date"].dt.to_period("M")

    rows: list[dict] = []
    for period, group in frame.groupby("_month", sort=True, observed=True):
        income, spend = split_signs(group)
        days_in = calendar.monthrange(period.year, period.month)[1]
        covered, coverage, partial = _coverage(
            group["date"].min(), group["date"].max(), days_in
        )
        rows.append(
            {
                "period": str(period),
                "income_cents": income,
                "spend_cents": spend,
                "net_cents": income + spend,
                "txn_count": len(group),
                "first_date": group["date"].min(),
                "last_date": group["date"].max(),
                "days_covered": covered,
                "days_in_period": days_in,
                "coverage_pct": coverage,
                "is_partial": partial,
            }
        )
    return _rows_to_summary(rows)


def yearly_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Income, spend and net per calendar year, with coverage flags."""
    if transactions.empty:
        return _rows_to_summary([])

    frame = transactions.copy()
    frame["_year"] = frame["date"].dt.year

    rows: list[dict] = []
    for year, group in frame.groupby("_year", sort=True, observed=True):
        income, spend = split_signs(group)
        days_in = _days_in_year(int(year))
        covered, coverage, partial = _coverage(
            group["date"].min(), group["date"].max(), days_in
        )
        rows.append(
            {
                "period": str(int(year)),
                "income_cents": income,
                "spend_cents": spend,
                "net_cents": income + spend,
                "txn_count": len(group),
                "first_date": group["date"].min(),
                "last_date": group["date"].max(),
                "days_covered": covered,
                "days_in_period": days_in,
                "coverage_pct": coverage,
                "is_partial": partial,
            }
        )
    return _rows_to_summary(rows)


def period_summary(transactions: pd.DataFrame, month: str) -> dict | None:
    """The single monthly row matching `month`, as a dict.

    `month` is a 'YYYY-MM' string. Returns None when the month is absent so
    callers can distinguish "no data" from "zero activity".
    """
    summary = monthly_summary(transactions)
    if summary.empty:
        return None
    matches = summary[summary["period"] == month]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def year_summary(transactions: pd.DataFrame, year: int) -> dict | None:
    """The single yearly row matching `year`, as a dict."""
    summary = yearly_summary(transactions)
    if summary.empty:
        return None
    matches = summary[summary["period"] == str(int(year))]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def filter_period(
    transactions: pd.DataFrame, month: str | None = None, year: int | None = None
) -> pd.DataFrame:
    """Rows within a month or a year. Omit both to return everything."""
    if transactions.empty:
        return transactions
    frame = transactions
    if month is not None:
        frame = frame[frame["date"].dt.to_period("M").astype(str) == month]
    if year is not None:
        frame = frame[frame["date"].dt.year == int(year)]
    return frame


def filter_months(transactions: pd.DataFrame, months: list[str] | None) -> pd.DataFrame:
    """Rows in any of the given 'YYYY-MM' periods.

    This is the filter the dashboard's month multiselect drives, so a selection
    of several months has to narrow the data once, up front, rather than every
    report having to understand a list of periods.

    An empty list is honoured as an empty selection and returns no rows, which
    is what "deselect all" means. None means "no month filter at all" and
    returns everything, so a caller that has not filtered does not have to
    special-case it.
    """
    if transactions.empty or months is None:
        return transactions
    wanted = {str(period) for period in months}
    if not wanted:
        return transactions.iloc[0:0]
    return transactions[
        transactions["date"].dt.to_period("M").astype(str).isin(wanted)
    ]


def category_breakdown(
    transactions: pd.DataFrame,
    month: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """Spend per category for the given scope, ordered by amount.

    Only money out is counted, and the sign is decided by name rather than by
    the amount alone. A credit in a spending category is money back out, so it
    reduces that category's total; a credit in a named income category is left
    out altogether, since income is not a spending bucket.
    """
    frame = filter_period(transactions, month=month, year=year)
    if frame.empty:
        return pd.DataFrame(
            columns=["display_label", "source_category", "spend_cents", "txn_count"]
        )

    spend = spending_rows(frame)

    if spend.empty:
        return pd.DataFrame(
            columns=["display_label", "source_category", "spend_cents", "txn_count"]
        )

    grouped = (
        spend.groupby(["display_label", "source_category"], observed=True)
        .agg(spend_cents=("amount_cents", "sum"), txn_count=("amount_cents", "size"))
        .reset_index()
    )
    # Spend is negative, so sorting on the raw value descending would put the
    # smallest outlay first. Rank on magnitude to get largest spend on top.
    grouped = grouped.sort_values(
        "spend_cents", key=lambda series: series.abs(), ascending=False
    ).reset_index(drop=True)
    return grouped


def category_month_matrix(
    transactions: pd.DataFrame, months: list[str] | None = None
) -> pd.DataFrame:
    """Spend for every category in every month, as long rows.

    Returns one row per (period, display_label) with `spend_cents`, zero-filled
    across the full cross product of the months and categories present.

    The zero fill is the point. A pivot of the raw rows would simply have no
    cell for 'Groceries in August', and a heatmap drawn from that leaves a hole
    in the grid rather than showing that nothing was spent. Filling the gaps
    keeps every category comparable across every month, which is the only reason
    to read a matrix in the first place.

    Categories are taken from spending only, so an income-only category such as
    Paycheck/Salary does not become a column that is empty in every row.
    """
    columns = ["period", "display_label", "spend_cents"]
    frame = filter_months(transactions, months)
    if frame.empty:
        return pd.DataFrame(columns=columns)

    spend = spending_rows(frame).copy()
    if spend.empty:
        return pd.DataFrame(columns=columns)

    periods = sorted(frame["date"].dt.to_period("M").astype(str).unique())
    labels = sorted(spend["display_label"].unique())

    spend["_period"] = spend["date"].dt.to_period("M").astype(str)
    observed = (
        spend.groupby(["_period", "display_label"], observed=True)["amount_cents"]
        .sum()
        .rename("spend_cents")
        .reset_index()
        .rename(columns={"_period": "period"})
    )

    grid = pd.MultiIndex.from_product(
        [periods, labels], names=["period", "display_label"]
    ).to_frame(index=False)
    return grid.merge(observed, on=["period", "display_label"], how="left").assign(
        spend_cents=lambda merged: merged["spend_cents"].fillna(0).astype("int64")
    )[columns]


# --- budget targets ------------------------------------------------------


def load_budgets(path: Path | str = BUDGETS_PATH) -> dict[str, int]:
    """Read monthly targets as whole dollars, keyed by display label.

    A missing or empty file yields an empty dict rather than an error, so
    the dashboard opens cleanly before any target has been set.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    budgets = data.get("budgets", {}) or {}
    cleaned: dict[str, int] = {}
    for key, value in budgets.items():
        try:
            cleaned[str(key)] = int(round(float(value)))
        except (TypeError, ValueError):
            continue
    return cleaned


def save_budgets(targets: dict[str, int], path: Path | str = BUDGETS_PATH) -> Path:
    """Write targets back as whole dollars, dropping zero and negative values.

    Removing unset entries keeps the file honest about what has actually
    been configured rather than accumulating empty placeholders.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "budgets": {
            str(key): int(value)
            for key, value in sorted(targets.items())
            if value and int(value) > 0
        }
    }
    header = (
        "# Monthly budget targets in whole dollars, keyed by dashboard category label.\n"
        "# Edited from the Budgets page in the app; edit by hand if preferred.\n"
        "# 0 or absent means no target is set for that category.\n"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump(payload, handle, sort_keys=True, default_flow_style=False)
    return path


def months_in_scope(month: str | None = None, year: int | None = None) -> int:
    """How many months of budget a comparison covers.

    A yearly comparison is charged twelve times a monthly target, matching
    the decision to compare partial periods against full budgets rather than
    prorating them.
    """
    if month is not None:
        return 1
    if year is not None:
        return 12
    return 1


def variance_table(
    transactions: pd.DataFrame,
    month: str | None = None,
    year: int | None = None,
    budgets: dict[str, int] | None = None,
    months: int | None = None,
) -> pd.DataFrame:
    """Budget against actual per category for the given scope.

    Every category present in the data appears, even without a target, so an
    unbudgeted category is visible rather than silently omitted. Positive
    `variance_cents` means under budget; negative means over.

    Categories holding a target but absent from the data are included with
    zero actual, so a budgeted category that received no spending is not
    lost from the table.

    `pct_used` is the share of the target consumed, as a fraction. It is None
    when no target exists, since there is nothing to be a percentage of, and
    the table leaves the bar empty for those rows.

    `months` is how many months of budget the comparison covers, used when the
    caller has already narrowed the rows itself (the dashboard filters to the
    months the user picked) and so cannot express the window as a month or a
    year. It overrides the count derived from `month`/`year`, which otherwise
    stay the way to ask for a single period or a whole year.
    """
    budgets = budgets or {}
    breakdown = category_breakdown(transactions, month=month, year=year)

    observed: dict[str, int] = {}
    counts: dict[str, int] = {}
    if not breakdown.empty:
        for row in breakdown.itertuples(index=False):
            observed[row.display_label] = int(row.spend_cents)
            counts[row.display_label] = int(row.txn_count)

    labels = sorted(set(observed) | set(budgets))
    if months is None:
        months = months_in_scope(month=month, year=year)

    rows: list[dict] = []
    for label in labels:
        budget_cents = int(budgets.get(label, 0)) * 100 * months
        actual_cents = observed.get(label, 0)
        # Actual spend is negative, so adding it to the budget subtracts the
        # amount spent. Positive variance is money left unspent.
        variance_cents = budget_cents + actual_cents
        # Share of the target consumed. Actual is negative, so negating it
        # gives the amount spent as a positive figure to divide by the target.
        # Spend exactly equal to the target is 1.0, which counts as over: the
        # category has used everything allowed and any further spend is a
        # breach rather than a rounding detail.
        pct_used = -actual_cents / budget_cents if budget_cents > 0 else None
        if budget_cents == 0:
            status = "no budget"
        elif variance_cents > 0:
            status = "under"
        else:
            status = "over"
        rows.append(
            {
                "display_label": label,
                "budget_cents": budget_cents,
                "actual_cents": actual_cents,
                "variance_cents": variance_cents,
                "pct_used": pct_used,
                "txn_count": counts.get(label, 0),
                "status": status,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "display_label",
            "budget_cents",
            "actual_cents",
            "variance_cents",
            "pct_used",
            "txn_count",
            "status",
        ],
    )
    if frame.empty:
        return frame
    return frame.sort_values(
        "actual_cents", key=lambda series: series.abs(), ascending=False
    ).reset_index(drop=True)


# --- savings buckets -----------------------------------------------------
#
# A bucket is money set aside each month for one spend category, and drawn down
# by what is actually spent in that category. A bucket is named by its category,
# so there is one bucket per category at most, and the two models are connected
# through that name: a vacation charge still shows as ordinary spending in the
# budget views and also comes out of the Vacation bucket.
#
# Everything covers the months loaded, from the first through the newest:
#
#     contributed = monthly contribution x months loaded
#     spent       = net spending in the category over those months
#     balance     = contributed + adjustment - spent
#
# 'Months loaded' counts every calendar month from the first to the newest,
# including a quiet month with no transactions and the month in progress. The
# contribution is applied to all of them as it stands now, so changing it
# re-prices the whole period. The adjustment is a manual offset for anything
# the data does not show, such as savings already held or money moved by hand.


def _whole_dollars(value: object) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def load_buckets(path: Path | str = BUCKETS_PATH) -> dict[str, dict]:
    """Read savings buckets, keyed by the spend category each one belongs to.

    Each bucket is a dict of `contribution` (whole dollars a month), `adjustment`
    (whole dollars, a manual offset to the balance, may be negative) and `goal`
    (the target balance in whole dollars, 0 for none).

    Files from earlier versions named the adjustment `current_balance` or
    `opening_balance`; either is read as the adjustment, and a leftover `as_of`
    is ignored. A missing or unreadable figure reads as 0, and a negative
    contribution or goal as 0, so one bad hand edit does not take the page down.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    raw = data.get("buckets", {}) or {}
    if not isinstance(raw, dict):
        return {}

    buckets: dict[str, dict] = {}
    for category, spec in raw.items():
        category = str(category).strip()
        if not category or not isinstance(spec, dict):
            continue
        adjustment = spec.get(
            "adjustment", spec.get("current_balance", spec.get("opening_balance", 0))
        )
        buckets[category] = {
            "contribution": max(_whole_dollars(spec.get("contribution", 0)) or 0, 0),
            "adjustment": _whole_dollars(adjustment) or 0,
            "goal": max(_whole_dollars(spec.get("goal", 0)) or 0, 0),
        }
    return buckets


def load_savings_pool(path: Path | str = BUCKETS_PATH) -> float:
    """The total cash available to allocate, in dollars and cents (0 when unset).

    Unlike targets, this is a bank balance, so it keeps its cents.
    """
    path = Path(path)
    if not path.exists():
        return 0.0
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    try:
        return round(float(data.get("savings_pool", 0) or 0), 2)
    except (TypeError, ValueError):
        return 0.0


def save_buckets(
    buckets: dict[str, dict],
    savings_pool: float | None = None,
    path: Path | str = BUCKETS_PATH,
) -> Path:
    """Write buckets as whole dollars and the savings pool in dollars and cents.

    `savings_pool=None` keeps whatever pool is already on file, so saving the
    buckets alone cannot wipe the cash figure, and vice versa.
    """
    path = Path(path)
    if savings_pool is None:
        savings_pool = load_savings_pool(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pool = round(float(savings_pool), 2)
    payload = {
        "savings_pool": int(pool) if pool == int(pool) else pool,
        "buckets": {
            str(category): {
                "contribution": int(spec.get("contribution", 0)),
                "adjustment": int(spec.get("adjustment", 0)),
                "goal": int(spec.get("goal", 0)),
            }
            for category, spec in sorted(buckets.items())
        },
    }
    header = (
        "# Savings buckets, in whole dollars. Edited from the Savings page in the app.\n"
        "# savings_pool: total cash in the bank, to check every dollar is assigned.\n"
        "# Each bucket is named by the spend category that draws it down:\n"
        "# contribution (per month loaded), adjustment (manual offset to the\n"
        "# balance, may be negative), goal (target balance, 0 for none).\n"
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    return path


def latest_period(transactions: pd.DataFrame) -> str | None:
    """The newest 'YYYY-MM' month in the data, or None when there is none."""
    if transactions.empty:
        return None
    return str(transactions["date"].max().to_period("M"))


def months_loaded(transactions: pd.DataFrame) -> int:
    """Calendar months from the first loaded through the newest, inclusive.

    A month with no transactions in between still counts, and so does the
    month in progress. August to September is 2; a full January to September
    is 9.
    """
    if transactions.empty:
        return 0
    first = transactions["date"].min().to_period("M")
    last = transactions["date"].max().to_period("M")
    return (last - first).n + 1


def bucket_balance(transactions: pd.DataFrame, category: str, bucket: dict) -> dict[str, int]:
    """A bucket's figures over the months loaded, in cents.

    `contributed` is the monthly contribution times the months loaded. `spent` is
    the net outflow in the category, so a refund is netted off. `balance` is
    `contributed + adjustment - spent`.
    """
    contributed = int(bucket.get("contribution", 0)) * 100 * months_loaded(transactions)
    adjustment = int(bucket.get("adjustment", 0)) * 100
    spent = 0
    if not transactions.empty:
        in_category = transactions["display_label"] == category
        spent = -int(transactions.loc[in_category, "amount_cents"].sum())
    return {
        "contributed": contributed,
        "adjustment": adjustment,
        "spent": spent,
        "balance": contributed + adjustment - spent,
    }


SAVINGS_COLUMNS = [
    "category",
    "monthly_cents",
    "contributed_cents",
    "adjustment_cents",
    "spent_cents",
    "balance_cents",
    "goal_cents",
    "pct_saved",
    "remaining_cents",
]


def savings_summary(transactions: pd.DataFrame, buckets: dict[str, dict]) -> pd.DataFrame:
    """One row per bucket, in category order, over the months loaded.

    `pct_saved` is the balance as a share of the goal (a fraction, which can pass
    1.0 or fall below 0), and `remaining_cents` is what is still to save, never
    below 0. Both are None when the bucket has no goal.
    """
    rows = []
    for category, bucket in sorted(buckets.items()):
        figures = bucket_balance(transactions, category, bucket)
        goal = int(bucket.get("goal", 0)) * 100
        rows.append(
            {
                "category": category,
                "monthly_cents": int(bucket.get("contribution", 0)) * 100,
                "contributed_cents": figures["contributed"],
                "adjustment_cents": figures["adjustment"],
                "spent_cents": figures["spent"],
                "balance_cents": figures["balance"],
                "goal_cents": goal,
                "pct_saved": figures["balance"] / goal if goal > 0 else None,
                "remaining_cents": max(goal - figures["balance"], 0) if goal > 0 else None,
            }
        )
    return pd.DataFrame(rows, columns=SAVINGS_COLUMNS)


def planned_allocation(buckets: dict[str, dict], months: list[str]) -> int:
    """Cents set aside for savings over `months`, as planned.

    Every bucket sets aside its contribution for each month, including the
    month still in progress. This is the plan, not what has been credited: a
    month's savings are spoken for from the start of the month, even though the
    bucket is only credited once the month is complete.
    """
    monthly = sum(int(bucket.get("contribution", 0)) for bucket in buckets.values())
    return monthly * 100 * len(months)


def left_after_savings(
    scoped: pd.DataFrame, buckets: dict[str, dict], months: list[str]
) -> dict[str, int]:
    """Income, spend, savings allocated and what is left, in cents, for `months`.

    Spend is negative, as everywhere else; `left` is income plus spend minus the
    planned savings, so a negative figure means the months spent and saved more
    than came in.
    """
    income, spend = split_signs(scoped)
    allocated = planned_allocation(buckets, months)
    return {
        "income": income,
        "spend": spend,
        "allocated": allocated,
        "left": income + spend - allocated,
    }


def savings_reconciliation(
    transactions: pd.DataFrame, buckets: dict[str, dict], pool_cents: int
) -> dict[str, int]:
    """Cash on hand against what the buckets claim.

    `unallocated` is cash not yet assigned to any bucket; zero means every dollar
    has a job, and a negative figure means the buckets claim more than is held.
    An overdrawn bucket's negative balance counts against the allocated total.
    """
    summary = savings_summary(transactions, buckets)
    allocated = int(summary["balance_cents"].sum()) if not summary.empty else 0
    return {
        "pool": int(pool_cents),
        "allocated": allocated,
        "unallocated": int(pool_cents) - allocated,
    }

