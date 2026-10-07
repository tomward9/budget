"""Pure display helpers shared by the dashboard and its tests.

These live apart from app.py on purpose. Streamlit executes app.py top to
bottom as a script, so a test that imported it would run the whole page as a
side effect. Keeping the side-effect-free helpers here makes them directly
testable.

Nothing in this module touches Streamlit state or reads files.
"""

from __future__ import annotations

import os

import pandas as pd

# When this file was loaded, so the app can tell a stale copy from a fresh one.
LOADED_MTIME = os.path.getmtime(__file__)

# Budget-status colours. The red is deliberately a light fill with dark text
# rather than a saturated background, so the figure stays readable and the
# table does not shimmer.
OVER_FILL = "#ffd9d9"
OVER_TEXT = "#8b1a1a"
UNDER_FILL = "#dcf5e3"
UNDER_TEXT = "#14532d"

# Status values produced by reports.variance_table.
STATUS_OVER = "over"
STATUS_UNDER = "under"
STATUS_NO_BUDGET = "no budget"

# Bar label placement. A label drawn inside a bar is white on the bar's fill; one
# drawn outside sits on the plot background, so it has to be dark or it vanishes
# against the light theme.
LABEL_ON_BAR = "#ffffff"
LABEL_OFF_BAR = "#31333f"

# The estimate of how wide a rendered bar chart is, in pixels. Streamlit does not
# report the width of the container a chart lands in, so deciding whether a
# label fits has to assume one. This is the plot area of a full-width chart on an
# ordinary desktop window; the cost of being wrong is that one or two labels sit
# outside that would have fitted, or the reverse, which is a small cosmetic
# difference rather than a wrong number.
PLOT_WIDTH_PX = 880.0
# Average glyph width as a fraction of font size. Digits, the currency symbol and
# the middot in a label like '$523  ·  36%' are all wide, so this errs towards
# reserving more room than the label needs, which is the safe direction: a label
# that is moved outside unnecessarily still reads.
CHAR_WIDTH_RATIO = 0.62
# Room left inside a bar between its edge and a label drawn within it.
LABEL_PADDING_PX = 10.0
# Cap on how much of the axis given over to outside labels. Past this the bars
# get too short to compare, which defeats the chart.
MAX_HEADROOM = 0.6
# The budget comparison draws two bars per category, side by side, so each bar's
# longest possible length is a share of the plot width rather than the whole of
# it. The bar label plan's width estimate has to know that share or it puts a
# label inside a bar that is actually too short for it: plotly shrinks that
# label's font to fit, and a shrunken amount stops being the readable number the
# label is there to be.
GROUPED_BAR_SLOT_SHARE = 0.45


def period_label(period: object) -> str:
    """'2026-09' -> 'Sep 2026', for display only.

    A value that is not a monthly period is returned as given rather than
    raising, so a malformed row cannot blank the whole page.
    """
    try:
        stamp = pd.Period(str(period), freq="M")
    except (ValueError, TypeError):
        return str(period)
    return f"{stamp.strftime('%b')} {stamp.year}"


def resolve_period(periods: list[str], label: object) -> str | None:
    """Map a clicked or selected label back to its 'YYYY-MM' period.

    Charts display the friendly label, so a click arrives as 'Sep 2026' while
    the rest of the app keys off the raw period. Returns None when the label
    matches nothing, so an unexpected click is ignored rather than blanking
    the month selector.
    """
    for period in periods:
        if period_label(period) == str(label):
            return str(period)
    return None


def color_status(value: object) -> str:
    """CSS for one budget-status cell: red once the target is reached.

    Returns a single CSS declaration string, not a list of declarations.
    Streamlit's Styler marshaller unpacks each style entry as a
    (property, value) pair, so returning a list raises
    'too many values to unpack' at render time.

    A row with no target returns an empty string, leaving the cell unstyled
    rather than implying a judgement that was never made.
    """
    if value == STATUS_OVER:
        return f"background-color: {OVER_FILL}; color: {OVER_TEXT}"
    if value == STATUS_UNDER:
        return f"background-color: {UNDER_FILL}; color: {UNDER_TEXT}"
    return ""


# The one row picked in the budget table. A tint rather than a verdict, so it
# reads as a selection and not as a second status.
SELECT_FILL = "#dbe9ff"
SELECT_TEXT = "#0b3d91"


def budget_style(
    table: pd.DataFrame,
    status: pd.Series | list[str],
    selected: str | None = None,
) -> pd.io.formats.style.Styler:
    """Colour the Difference cell of a budget table from the report's status.

    The colour rides on Difference rather than on a separate Status column, so
    the judgement is visible on the number the user actually acts on. A separate
    status column had to be read across the row to find the verdict, and it
    repeated what the sign of Difference already said.

    It is still driven by the status values rather than by the amount, so the
    threshold stays the single decision reports.variance_table makes. That is
    what keeps spending exactly equal to the target red: the sign alone would
    call a zero difference neutral, and a zero difference is a target used up
    in full, not money left over.

    `selected` tints the one row the reader picked. Difference keeps its own
    colour there, since the tint would otherwise cover up whether that row is
    under or over budget, and bolding it is enough to find it again.
    """
    styler = table.style.apply(
        lambda _column: [color_status(value) for value in status],
        subset=["Difference"],
    )
    if selected is None:
        return styler

    # The row to tint, found in the Category column and then applied across
    # every column: a cell in Budget or Actual holds an amount, not the
    # category, so it cannot be matched on its own value.
    marked = {
        position
        for position, label in enumerate(table["Category"].astype(str))
        if label == str(selected)
    }

    def tint(column: pd.Series) -> list[str]:
        # Applied column by column, so the column's own name decides whether
        # it can take the tint.
        css = (
            "font-weight: bold"
            if column.name == "Difference"
            else f"font-weight: bold; background-color: {SELECT_FILL}; "
            f"color: {SELECT_TEXT}"
        )
        return [
            css if position in marked else "" for position in range(len(column))
        ]

    return styler.apply(tint)


def months_label(months: list[str], available: list[str] | None = None) -> str:
    """Name a month selection for a heading: 'All months', 'Sep 2026', and so on.

    The filter is a multiselect, so there is no single period to name. One and
    two months are spelled out because those are the cases where naming them
    costs nothing and tells the reader exactly what is being reported on.
    Longer selections are counted instead: a list of nine month names is a
    worse heading than '4 months', and it is no more informative.
    """
    chosen = sorted(str(period) for period in months)
    if not chosen:
        return "No months selected"
    if len(chosen) == 1:
        return period_label(chosen[0])
    if available is not None and len(chosen) == len(set(available)):
        return "All months"
    if len(chosen) == 2:
        return f"{period_label(chosen[0])} to {period_label(chosen[1])}"
    return f"{len(chosen)} months"


def spend_bar_labels(spend_cents: pd.Series, total_cents: int) -> pd.Series:
    """'amount and share of total' labels, drawn inside the category bars.

    The share is of the whole period, taken before any category filter, because
    that is the only denominator that means anything. A filtered chart shows one
    bar, and dividing it by its own amount would print 100% for a category that
    is really 12% of the spending.

    Both arguments are signed spend, so the sign is dropped here and a caller
    can pass the sum of a spend column straight in. Dividing by the raw sum
    instead would hit a negative denominator and print 0% on every bar, which
    looks like a plausible chart rather than an obvious failure.

    Whole dollars, matching the axis and the hover. The table beside the chart
    carries the cents.
    """
    magnitudes = spend_cents.abs() / 100.0
    total_dollars = abs(total_cents) / 100.0
    if total_dollars <= 0:
        shares = pd.Series(0.0, index=spend_cents.index)
    else:
        shares = magnitudes / total_dollars * 100.0
    return pd.Series(
        [
            f"${amount:,.0f}  ·  {share:.0f}%"
            for amount, share in zip(magnitudes, shares)
        ],
        index=spend_cents.index,
    )


def bar_label_plan(
    values: pd.Series,
    labels: pd.Series,
    plot_width_px: float = PLOT_WIDTH_PX,
    font_size: float = 10.0,
) -> tuple[list[str], list[str], float]:
    """Decide for each bar whether its label goes inside or outside, and the axis
    range that leaves room for the ones that go outside.

    A label that will not fit inside its bar is squashed: plotly shrinks the text
    to the bar's width or clips it, and either way the amount stops being
    readable, which is the one thing the label is there for. So a label without
    room is drawn just past the bar's tip instead, in a dark colour, because it
    is now on the background rather than on the bar's fill.

    The axis range comes out of this same calculation rather than being passed
    in, because the two have to agree. A label placed outside a bar needs empty
    axis beyond that bar's tip; if the axis stopped at the longest bar, which is
    what plotly picks by default, those labels would be cut off at the edge of
    the plot. Widening the axis to fit the widest outside label is what makes the
    placement safe rather than merely tidy.

    Both the fit and the headroom are worked out in pixels against an assumed
    plot width, since a label's rendered size is not knowable before the browser
    draws it. It is an estimate, but the error is a label sitting on the other
    side of the line, not a wrong number.

    Returns (textposition, font colour, axis maximum), each list aligned with the
    order of `values`.
    """
    amounts = pd.Series(values).abs().astype(float)
    texts = pd.Series(labels).astype(str)
    if amounts.empty or float(amounts.max()) <= 0:
        # Nothing to scale against, so no bar can hold a label.
        return (
            ["outside"] * len(amounts),
            [LABEL_OFF_BAR] * len(amounts),
            0.0,
        )

    peak = float(amounts.max())
    widths = texts.str.len().to_numpy() * (font_size * CHAR_WIDTH_RATIO)
    # Each bar's share of the longest one, which is also the share of the axis
    # it covers, however much headroom the axis ends up with.
    shares = (amounts / peak).to_numpy()

    # Headroom needed so that every bar plus its label fits: a bar covering
    # `share` of the peak ends at share * width / (1 + headroom) from the axis
    # origin, and an outside label then needs its own width beyond that. Solving
    # for the headroom that keeps the worst case inside the plot gives this.
    headroom = min(
        max(
            (
                share * plot_width_px / max(plot_width_px - width, 1.0) - 1.0
                for share, width in zip(shares, widths)
            ),
            default=0.0,
        ),
        MAX_HEADROOM,
    )
    axis_max = peak * (1.0 + headroom)

    bar_px = shares * plot_width_px / (1.0 + headroom)
    positions = [
        "inside" if room >= wanted + LABEL_PADDING_PX else "outside"
        for room, wanted in zip(bar_px, widths)
    ]
    colors = [
        LABEL_ON_BAR if position == "inside" else LABEL_OFF_BAR
        for position in positions
    ]
    return positions, colors, axis_max


def dollars(cents: pd.Series) -> pd.Series:
    """Convert a cents column to dollars for display.

    Tables hand their columns to Streamlit's NumberColumn with a '$%.2f'
    format string, which only adds a symbol. It does not rescale, so a column
    left in cents renders 50300 cents as '$50300.00' rather than '$503.00'.
    Every money column therefore passes through here first.

    Dividing by a float 100 rather than an int keeps the result a float dtype,
    which is what the '$%.2f' format expects.
    """
    return cents / 100.0


def percent_bar(pct_used: pd.Series) -> tuple[pd.Series, float]:
    """Values and upper bound for a 'percent used' progress bar column.

    Two separate scaling decisions, and the second one is the important one.

    The value: Streamlit applies a ProgressColumn's format string with printf
    semantics, so '%.0f%%' against the raw fraction 0.75 prints '1%' rather
    than '75%'. Converting to percentage points first makes the printed label
    correct.

    The bound: a ProgressColumn draws its bar as value / max_value, so the
    bound has to be 100 for the bar's *length* to be the percentage. Stretching
    it to the worst overspend keeps an over-budget bar off the right edge, but
    it rescales every other row to match: with a 174% worst case, a 48% category
    draws at 27% of the track. The label then reads '48%' beside a bar that
    looks like a little over a quarter full, which is simply a wrong chart.

    So the bound is fixed at 100. Overspend reads correctly because the bar
    fills completely, the Difference column and its colour both say 'over',
    and the percentage label says by how much.
    """
    return pct_used * 100.0, 100.0


def category_filter(frame: pd.DataFrame, selected) -> pd.DataFrame:
    """Restrict a frame to the chosen display labels, or leave it untouched.

    `selected` is the list of categories picked in the multiselect. Empty (or
    None) means no category filter, so every category is kept, which is what the
    multiselect's 'All categories' placeholder stands for. A chart or table
    click seeds this with a single category, and the same value narrows the
    transactions view, so drilling either way lands on the same rows.
    """
    if not selected or frame.empty:
        return frame
    if isinstance(selected, str):
        selected = [selected]
    return frame[frame["display_label"].isin(selected)]


# --- the filter selectors --------------------------------------------------
#
# The month filter is a single choice: 'All months' or one month. It carries its
# own 'everything' entry rather than leaning on a separate button, so the scope
# of the selection is named inside the control that applies it. The sentinel
# lives here rather than in app.py because app.py is a script that runs top to
# bottom on import, and the value has to be assertable from a test. It is not a
# valid month period, so nothing real can collide with it.
#
# The category filter needs no such sentinel: it is a multiselect where an empty
# selection already means 'all', shown as a placeholder.

ALL_MONTHS = "\x00all-months"


def month_filter_label(value: object) -> str:
    """What a month filter option reads as: 'All months' or 'Sep 2026'."""
    if value == ALL_MONTHS:
        return "All months"
    return period_label(value)


def budget_months(choice: object, periods: list[str]) -> int:
    """How many months of budget to compare the chosen spending against.

    A single month is one month of budget. 'All months' is read as the year: a
    full twelve months of budget for each calendar year the data touches, not
    the number of months loaded so far. Charging only the loaded months would
    make a year that is half over look exactly on track when it is really half
    the annual allowance spent.
    """
    if choice != ALL_MONTHS:
        return 1
    years = {str(period)[:4] for period in periods}
    return 12 * max(len(years), 1)


def budget_totals(table: pd.DataFrame) -> tuple[int, int, int]:
    """Total budgeted, total spent (as a positive amount) and the difference, in
    cents, over the rows of a variance table.

    The difference is the sum of the table's own Difference column, so the
    cards always agree with the rows beneath them. Spending in a category with
    no target counts against the difference, just as it does in its own row.
    """
    budgeted = int(table["budget_cents"].sum())
    spent = -int(table["actual_cents"].sum())
    return budgeted, spent, int(table["variance_cents"].sum())


def net_style(values) -> list[str]:
    """CSS per cell for a Net column: green when positive, red when negative.

    Zero is left uncoloured, since breaking even is neither. The colours are the
    same pair the budget table uses, so green and red mean the same thing
    everywhere on the page.
    """
    styles = []
    for value in values:
        if pd.isna(value) or value == 0:
            styles.append("")
        elif value > 0:
            styles.append(f"background-color: {UNDER_FILL}; color: {UNDER_TEXT}")
        else:
            styles.append(f"background-color: {OVER_FILL}; color: {OVER_TEXT}")
    return styles


def unallocated_status(unallocated_cents: int) -> str:
    """What the gap between cash on hand and the buckets means, in words."""
    if unallocated_cents == 0:
        return "Every dollar assigned"
    if unallocated_cents > 0:
        return "Not yet assigned to a bucket"
    return "Over-allocated: buckets claim more than you hold"


def parse_bucket_rows(rows: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """Turn the bucket editor's rows into buckets keyed by category, with a list
    of problems.

    Each row needs a category. Blank rows (a half-added row in the editor) are
    skipped. A category listed twice is refused, since its spending would come
    out of both and be counted twice. Nothing is saved while any problem remains.
    """

    def whole(value: object) -> int:
        try:
            return 0 if pd.isna(value) else int(round(float(value)))
        except (TypeError, ValueError):
            return 0

    buckets: dict[str, dict] = {}
    problems: list[str] = []
    for number, row in enumerate(rows, start=1):
        category = str(row.get("Category") or "").strip()
        contribution_raw = row.get("Monthly contribution")
        adjustment_raw = row.get("Adjustment")
        goal_raw = row.get("Goal")
        if not category:
            if any(
                value is not None and not pd.isna(value) and value != 0
                for value in (contribution_raw, adjustment_raw, goal_raw)
            ):
                problems.append(f"Row {number} needs a category.")
            continue
        if category in buckets:
            problems.append(f"{category} is listed twice; a category has one bucket.")
            continue
        contribution = whole(contribution_raw)
        if contribution < 0:
            problems.append(f"{category} has a negative monthly contribution.")
            continue
        goal = whole(goal_raw)
        if goal < 0:
            problems.append(f"{category} has a negative goal.")
            continue
        buckets[category] = {
            "contribution": contribution,
            "adjustment": whole(adjustment_raw),
            "goal": goal,
        }
    return buckets, problems


