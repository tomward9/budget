"""Local budgeting dashboard.

Run with:
    conda activate budget
    streamlit run app.py

Data comes from CSV exports uploaded in the sidebar, or dropped into
data/raw/ and they will be picked up on the next run. No data leaves the
machine.
"""

from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Streamlit re-runs this script on every refresh, but keeps the budget/ modules
# it imported when the server started. After an edit to one of them, a server
# left running would pair this new script with the old module, and fail on
# whatever the old one lacks. Each module records the file time it was loaded
# from; when any file is newer, or a module predates that record, the whole
# package is reloaded, in dependency order, before anything is imported from it.
BUDGET_MODULES = ["budget.schema", "budget.ingest", "budget.reports", "budget.presentation"]


def reload_changed_budget_modules() -> bool:
    loaded = [sys.modules.get(name) for name in BUDGET_MODULES]
    stale = any(
        module is not None
        and (
            getattr(module, "LOADED_MTIME", None) is None
            or Path(module.__file__).stat().st_mtime > module.LOADED_MTIME
        )
        for module in loaded
    )
    if stale:
        for module in loaded:
            if module is not None:
                importlib.reload(module)
    return stale


if reload_changed_budget_modules():
    # Cached results were computed by the old code.
    st.cache_data.clear()

from budget.ingest import ingest
from budget.presentation import (
    ALL_MONTHS,
    GROUPED_BAR_SLOT_SHARE,
    PLOT_WIDTH_PX,
    bar_label_plan,
    budget_months,
    budget_style,
    budget_totals,
    category_filter,
    dollars,
    month_filter_label,
    months_label,
    net_style,
    parse_bucket_rows,
    percent_bar,
    period_label,
    resolve_period,
    spend_bar_labels,
    unallocated_status,
)
from budget.reports import (
    BUCKETS_PATH,
    BUDGETS_PATH,
    category_breakdown,
    category_month_matrix,
    filter_months,
    left_after_savings,
    load_budgets,
    load_buckets,
    load_savings_pool,
    monthly_summary,
    months_loaded,
    save_budgets,
    save_buckets,
    savings_reconciliation,
    savings_summary,
    split_signs,
    variance_table,
)
from budget.schema import INCOME_CATEGORIES, format_cents

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

st.set_page_config(page_title="Budget", page_icon=None, layout="wide")

# The fixed-width report panes leave less room than the classic page padding
# gives them, so the main column is moved a step toward the sidebar: the row
# then needs fewer sideways scrolls to show the whole chart.
st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CURRENCY = format_cents


@st.cache_data(show_spinner=False)
def parse_transactions(payload: bytes, filename: str) -> pd.DataFrame:
    """Parse one uploaded file, cached on its exact bytes."""
    frame, _report = ingest(io.BytesIO(payload))
    return frame


@st.cache_data(show_spinner=False)
def read_budgets(path: str, mtime: float) -> dict[str, int]:
    """Budget targets, cached on the file's modification time.

    Keying on mtime means an edit made in a text editor invalidates the
    cache without needing a manual refresh.
    """
    return load_budgets(Path(path))


@st.cache_data(show_spinner=False)
def read_buckets(path: str, mtime: float) -> tuple[dict[str, dict], float]:
    """Savings buckets and the cash pool, cached on the file's modification time."""
    return load_buckets(Path(path)), load_savings_pool(Path(path))


def file_mtime(path: Path) -> float:
    """A file's modification time, or 0 when it does not exist yet."""
    return path.stat().st_mtime if path.exists() else 0.0


def load_all_sources(uploaded: list) -> tuple[pd.DataFrame, list[str]]:
    """Combine uploaded files, then data/raw/, then the project sample.

    Returns the combined frame and a list of the files that were actually read.
    """
    frames: list[pd.DataFrame] = []
    used: list[str] = []

    for upload in uploaded:
        frames.append(parse_transactions(upload.getvalue(), upload.name))
        used.append(upload.name)

    if not uploaded and RAW_DIR.exists():
        for path in sorted(RAW_DIR.glob("*.csv")):
            frames.append(parse_transactions(path.read_bytes(), path.name))
            used.append(path.name)

    if not used:
        sample = PROJECT_ROOT / "sample_transactions.csv"
        if sample.exists():
            frames.append(parse_transactions(sample.read_bytes(), sample.name))
            used.append(f"{sample.name} (project sample)")

    if not frames:
        return pd.DataFrame(), []

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["txn_id"], keep="first")
    combined = combined.sort_values(["date", "description"], kind="stable").reset_index(
        drop=True
    )

    return combined, used


def kpi_row(income_cents: int, spend_cents: int) -> None:
    """Income, spend and net, side by side."""
    net = income_cents + spend_cents
    columns = st.columns(3)
    columns[0].metric("Income", CURRENCY(income_cents), border=True)
    columns[1].metric("Spend", CURRENCY(spend_cents), delta_color="inverse", border=True)
    columns[2].metric(
        "Income minus spend", CURRENCY(net), delta_color="off", border=True
    )


def budget_kpis(table: pd.DataFrame) -> None:
    """Budgeted, spent and difference for whatever the filters leave in scope.

    A row picked in the budget table works like a month does: the caller
    narrows the table handed in to that category's own row, so the three cards
    stand alone for the pick rather than totalling the whole in-scope report.
    """
    budgeted, spent, difference = budget_totals(table)
    columns = st.columns(3)
    columns[0].metric("Total budgeted", CURRENCY(budgeted), border=True)
    columns[1].metric("Total spend", CURRENCY(spent), border=True)
    columns[2].metric(
        "Difference",
        CURRENCY(difference),
        # Green while money remains, red once it is gone, as in the table.
        delta="Under budget" if difference > 0 else "Over budget",
        delta_color="normal" if difference > 0 else "inverse",
        delta_arrow="off",
        border=True,
        help="Total budgeted minus total spend. Negative means over budget.",
    )


def variance_table_panel(table: pd.DataFrame, height: int) -> None:
    """The budget table, with one row pickable to mark a category.

    The pick does not narrow this table: the comparison is the point of it, and
    hiding the other categories would hide what is being compared. It marks the
    row here, marks its bar in the chart, narrows the trend and the
    transactions, and narrows the budget cards to the picked category's own
    figures instead. The report arrives already built, because the row
    picked in it has to resolve before any view is drawn. Its title, with the
    note on reading the figures behind a hovered question mark, and the chart's
    style buttons all sit above the row.

    The dataframe is given an explicit `height` - enough for the first 17 rows
    plus its header - because Streamlit otherwise draws it at its own default
    height, which shows only about ten rows before scrolling, and the chart
    being as long as the table would be a lie.
    """
    pct_values, bar_max = percent_bar(table["pct_used"])

    shown = pd.DataFrame(
        {
            "Category": table["display_label"],
            "Budget": dollars(table["budget_cents"]),
            "Actual": dollars(table["actual_cents"]),
            "Difference": dollars(table["variance_cents"]),
            "Used": pct_values,
        }
    )
    # The mark is written into the table's own state before it is created,
    # which is the only point where writing a widget's key is allowed. That is
    # what keeps a bar click and a row click showing the same mark here.
    matches = [
        position
        for position, label in enumerate(shown["Category"].astype(str))
        if label == str(table_selection)
    ]
    rows_shown = [matches[0]] if matches else []
    st.session_state["variance_table"] = {"selection": {"rows": rows_shown}}
    # ...and the same rows are remembered, so the next run can tell a fresh
    # click from the state this one just wrote.
    st.session_state["table_selection_rows_seen"] = rows_shown
    st.dataframe(
        budget_style(shown, table["status"], selected=table_selection),
        hide_index=True,
        width="stretch",
        height=height,
        on_select="rerun",
        selection_mode="single-row",
        key="variance_table",
        column_config={
            "Budget": st.column_config.NumberColumn("Budget", format="dollar"),
            "Actual": st.column_config.NumberColumn("Actual", format="dollar"),
            "Difference": st.column_config.NumberColumn(
                "Difference",
                format="dollar",
                help=(
                    "Budget left unspent. Green while money remains, red once the "
                    "target is met or passed. Negative means over budget."
                ),
            ),
            "Used": st.column_config.ProgressColumn(
                "Used",
                min_value=0.0,
                max_value=bar_max,
                format="%.0f%%",
                help=(
                    "Share of the target spent, as a length out of 100%. "
                    "A full bar means the target is met or passed."
                ),
            ),
        },
    )


def budget_editor(transactions: pd.DataFrame, budgets: dict[str, int]) -> None:
    """Editable table of monthly targets, saved back to budgets.yaml."""
    st.subheader("Budgets", anchor=False)
    if transactions.empty:
        st.info("Upload a transaction file to set budgets.")
        return

    breakdown = category_breakdown(transactions)
    labels = sorted(set(breakdown["display_label"]) | set(budgets))
    if not labels:
        st.info("No categories found.")
        return

    frame = pd.DataFrame(
        {
            "Category": labels,
            "Monthly target": [float(budgets.get(label, 0)) for label in labels],
        }
    )
    edited = st.data_editor(
        frame,
        hide_index=True,
        width="stretch",
        disabled=["Category"],
        column_config={
            "Monthly target": st.column_config.NumberColumn(
                "Monthly target",
                min_value=0.0,
                step=10.0,
                format="dollar",
                help="Leave at 0 for no budget on this category.",
            )
        },
        key="budget_editor",
    )

    if st.button("Save budgets", type="primary"):
        labels_column = edited["Category"].astype(str)
        amounts_column = pd.to_numeric(edited["Monthly target"], errors="coerce").fillna(0.0)
        targets = {
            label: int(round(float(amount)))
            for label, amount in zip(labels_column, amounts_column)
            if float(amount) > 0
        }
        save_budgets(targets, BUDGETS_PATH)
        st.cache_data.clear()
        st.success(f"Saved {len(targets)} target(s) to {BUDGETS_PATH.name}.")


def transactions_view(
    transactions: pd.DataFrame, label: str, months: list[str]
) -> None:
    """The raw transaction table for whatever the global filters select.

    The month and category filters are not repeated here: they are global, so
    showing a second copy would let the two disagree, and a copy that resets
    when you switch views is what 'maintained across views' rules out. Only the
    filters specific to this view are local, and the table takes the rest of
    the screen, since the rows themselves are the point of this page.
    """
    st.subheader("Transactions", anchor=False)
    if not months:
        st.info("Select at least one month in **Filters** to see transactions.")
        return
    if transactions.empty:
        st.info("No transactions to show.")
        return

    control_left, control_right = st.columns([2, 3])
    control_left.caption(f"Months: **{label}**")
    kind = control_right.segmented_control(
        "Show",
        ["Spend", "Income", "Everything"],
        key="txn_kind",
    )

    search_left, search_right = st.columns(2)
    search = search_left.text_input("Search description", key="txn_search")
    search_right.caption(
        f"Categories: **{', '.join(narrowed_categories) or 'All categories'}**"
    )

    frame = transactions
    if kind == "Spend":
        frame = frame[frame["amount_cents"] < 0]
    elif kind == "Income":
        frame = frame[frame["amount_cents"] > 0]
    if selected_categories:
        frame = frame[frame["display_label"].isin(selected_categories)]
    # The row marked in the budget table narrows this too: it is the same pick
    # the trend answers to, so the two never disagree about what was chosen.
    if table_selection:
        frame = frame[frame["display_label"] == table_selection]
    if search:
        needle = search.strip().lower()
        if needle:
            frame = frame[frame["description"].str.lower().str.contains(needle, na=False)]

    frame = frame.sort_values(["date", "description"], ascending=[False, True], kind="stable")
    income, spend = split_signs(frame)

    st.caption(
        f"{len(frame):,} transactions | income {CURRENCY(income)} | "
        f"spend {CURRENCY(spend)} | net {CURRENCY(income + spend)}"
    )

    if frame.empty:
        st.info("No transactions match these filters.")
        return

    shown = pd.DataFrame(
        {
            "Date": frame["date"].dt.date,
            "Description": frame["description"],
            "Category": frame["display_label"],
            "Account": frame["account"],
            "Amount": dollars(frame["amount_cents"]),
        }
    )
    # A pane that reaches down to where the month section of the Summary page
    # starts, with the table filling it: a month of transactions runs past
    # that either way, and this way they scroll inside the table instead of
    # dragging the page down behind them.
    with st.container(height=TRANSACTIONS_PANE_HEIGHT_PX, key="txn_pane", border=False):
        st.dataframe(
            shown,
            hide_index=True,
            width="stretch",
            height=TRANSACTIONS_PANE_HEIGHT_PX,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "Amount": st.column_config.NumberColumn(
                    "Amount", format="dollar", help="Negative means money out."
                ),
            },
        )


def month_overview_table(months: pd.DataFrame, height: int) -> None:
    """A clickable month-by-month summary beside the trend.

    Clicking a row sets the global time frame, which every view then honours.
    The chart does the same thing, but a two-point line is a small target and
    this table also answers 'what happened in every month' at a glance.

    The dataframe gets an explicit `height` (enough for a full year of rows
    plus its header) so every month reads without the table scrolling, the same
    reason the budget table is told its height rather than left to Streamlit's
    ten-row default.
    """
    st.subheader("Months", anchor=False)
    shown = pd.DataFrame(
        {
            "Month": months["period"].map(period_label),
            "Income": dollars(months["income_cents"]),
            "Spend": dollars(months["spend_cents"]),
            "Net": dollars(months["net_cents"]),
        }
    )
    event = st.dataframe(
        # Net is green when the month took in more than it spent, red when it
        # spent more, the same colours the budget table uses.
        shown.style.apply(net_style, subset=["Net"]),
        hide_index=True,
        width="stretch",
        height=height,
        on_select="rerun",
        selection_mode="single-row",
        key="month_overview",
        column_config={
            "Income": st.column_config.NumberColumn("Income", format="dollar"),
            "Spend": st.column_config.NumberColumn("Spend", format="dollar"),
            "Net": st.column_config.NumberColumn("Net", format="dollar"),
        },
    )

    rows = getattr(event.selection, "rows", []) or []
    if not rows:
        return
    chosen = str(shown.iloc[rows[0]]["Month"])
    # As with the charts, the row selection persists across reruns, so a marker
    # records the last row acted on to avoid re-filtering on every later rerun.
    if st.session_state.get("handled_month_overview") == chosen:
        return
    st.session_state["handled_month_overview"] = chosen

    target = resolve_period(list(months["period"]), chosen)
    if target:
        st.session_state["pending_month"] = target
        # The filter widgets were created before this table, so the change is
        # applied at the top of the next run; this rerun is what starts it.
        st.rerun()


def polish(fig) -> None:
    """Apply currency formatting to a figure with a numeric dollar axis.

    Charts plot numeric dollars; the formatted string is only ever used for
    hover text. Passing a pre-formatted string to plotly turns the axis
    categorical, which destroys the scale.
    """
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", separatethousands=True)
    fig.update_layout(legend_title_text="", margin=dict(t=50, l=10, r=10, b=10))
    return fig


def category_spend_chart(
    transactions: pd.DataFrame, key: str, pane_height: int
) -> None:
    """Horizontal bars of spend per category, each labelled with its amount and
    share of the period's total.

    The share comes from the whole period rather than from the bars on screen, so
    a chart already filtered to one category still reports that category's true
    size instead of 100%.

    A label goes inside its bar when there is room and just past the tip when
    there is not, because a label squeezed into a short bar stops being
    readable, and a category's amount is the whole point of the chart. Deciding
    that needs the axis to reach past the longest bar, so the axis is widened to
    fit the labels rather than left to plotly's own default of stopping at the
    longest bar.

    Horizontal bars keep long category names readable. The chart's canvas is a
    row of fixed thickness per bar - twice a table row, so the bars draw twice
    as thick - rather than whatever the pane's height leaves per category. Past
    eight or so chunky bars the canvas runs longer than the pane, and the pane
    scrolls to each bar rather than stretching the page; with fewer bars the
    blank room stays in the pane, which keeps the charted box the very same
    length as the budget table it reads against. `pane_height` is that box's
    height.
    """
    breakdown = category_breakdown(transactions)
    if breakdown.empty:
        st.info("No spending recorded in this period.")
        return

    total_cents = int(breakdown["spend_cents"].sum())
    frame = category_filter(breakdown, selected_categories).copy()
    if frame.empty:
        st.info("No spending in this category for this period.")
        return
    frame["dollars"] = frame["spend_cents"].abs() / 100
    frame["label"] = spend_bar_labels(frame["spend_cents"], total_cents)
    # Ascending, because a horizontal bar chart draws the first row at the
    # bottom; this puts the largest category at the top. Done before the label
    # plan so the per-bar positions line up with the bars in the order px lays
    # them out.
    frame = frame.sort_values("dollars", ascending=True)
    positions, colors, axis_max = bar_label_plan(
        frame["dollars"],
        frame["label"],
        font_size=SUMMARY_CHART_LABEL_FONT_SIZE,
    )

    fig = px.bar(
        frame,
        x="dollars",
        y="display_label",
        orientation="h",
        # Resolved by px against the column, not handed to the trace as the
        # literal string "label". By the time update_traces runs, px has already
        # turned the frame into arrays, so a column name there is a string that
        # every bar would print.
        text="label",
        title="Spend by category",
        labels={"display_label": "Category", "dollars": "Amount"},
        # One flat colour. The bar's length carries the amount, and the label
        # carries the number; a second colour would only encode what is already
        # on the axis.
        color_discrete_sequence=[SPEND_BAR],
        custom_data=["display_label"],
    )
    fig.update_layout(
        showlegend=False,
        # The marked category's colour and the category names are the only
        # legends this chart needs, so both axis titles are dropped and the
        # margins trimmed to what the tick labels actually occupy. The bars
        # then start closer to the names and stretch further for the same
        # dollar figure.
        margin=dict(t=50, l=4, r=10, b=4),
        # The canvas grows a row of fixed thickness per bar, header and floor
        # around it, so a bar's band is twice a table row and the bars draw
        # twice as thick. Whatever the row count, the pane scrolls to the rows
        # that outrun it rather than squeezing the bars.
        height=(
            SUMMARY_TABLE_HEADER_PX
            + SUMMARY_CHART_ROW_PX * len(frame)
            + SUMMARY_TABLE_FLOOR_PX
        ),
        # The same spacing as the budget comparison, so reading one chart and
        # the other feels like reading the same ruler: 12% of a row's band
        # between adjacent categories, and bars that fill the rest of it. A
        # spend bar is then exactly as wide as the budget and actual pair in
        # the comparison beside it.
        bargap=0.12,
    )
    # Headroom for the labels drawn past the end of their bar, so those labels
    # are not clipped by the right edge of the plot.
    fig.update_xaxes(
        range=[0, axis_max],
        tickprefix="$",
        tickformat=",.0f",
        separatethousands=True,
        title=None,
    )
    fig.update_yaxes(title=None)
    fig.update_traces(
        texttemplate="%{text}",
        # One entry per bar: a long bar holds its label, a short one has it drawn
        # outside in a dark colour against the background. The font colour is
        # per bar for the same reason, since a white label on the background
        # would be invisible.
        textposition=positions,
        textfont={"size": SUMMARY_CHART_LABEL_FONT_SIZE, "color": colors},
        # Only meaningful for the labels drawn inside; an outside label is
        # anchored past the tip either way.
        insidetextanchor="end",
        # Without this, an outside label near the right edge is cut off.
        cliponaxis=False,
        hovertemplate="%{y}<br>$%{x:,.2f}<extra></extra>",
    )
    # The marked category keeps the strong blue and every other bar pales, so
    # the pick is visible without any bar disappearing. The colours follow the
    # sorted frame, which is the order px draws the bars in.
    if table_selection:
        fig.update_traces(
            marker={
                "color": [
                    SPEND_BAR
                    if str(label) == str(table_selection)
                    else SPEND_BAR_PALE
                    for label in frame["display_label"]
                ]
            }
        )
    # The chart is drawn inside a pane exactly as long as the budget table
    # beside it. The canvas inside is taller whenever the categories outrun the
    # pane's eight or so chunky rows, so the pane scrolls to each bar rather
    # than the page stretching to fit them; the pane keeps the table's length
    # either way.
    with st.container(height=pane_height, border=False):
        st.plotly_chart(fig, width="stretch", key=key, on_select="rerun")


def budget_vs_actual_chart(
    transactions: pd.DataFrame,
    months: int,
    budgets: dict[str, int],
    pane_height: int,
    ascending: bool = False,
) -> None:
    """The budget target and the actual spending, bar by bar, for each category.

    Horizontal bars, like the spend chart, with one bar of fixed thickness per
    category - the same twice-a-table-row band - so the two read side by side.
    Every bar carries its amount in a label inside or just past its tip, the
    same way the spend chart does. The categories are ordered by actual spend,
    largest first, so the row the eye lands on first is the row that matters;
    `ascending` flips that to smallest first. `pane_height` is the pane's
    height - the budget table's length - and the pane scrolls to the bars that
    outrun it.

    The bars are drawn as positive magnitudes so the two lengths can be
    compared by eye. The sign lives in the variance table above, where the
    Difference column carries it and is coloured.
    """
    table = variance_table(transactions, budgets=budgets, months=months)
    budgeted = table[table["budget_cents"] > 0]
    if budgeted.empty:
        st.info(
            "No budget targets set yet, so there is nothing to compare against. "
            "Use the budget editor below to set one."
        )
        return
    budgeted = category_filter(budgeted, selected_categories)
    if budgeted.empty:
        st.info("No budgeted spending in this category for this period.")
        return

    # Horizontal bars draw the first row at the bottom, so the biggest spender
    # goes last to sit on top by default.
    budgeted = budgeted.sort_values(
        "actual_cents", ascending=ascending, kind="stable"
    )

    melted = budgeted.melt(
        id_vars=["display_label"],
        value_vars=["budget_cents", "actual_cents"],
        var_name="kind",
        value_name="cents",
    )
    melted["kind"] = melted["kind"].map(
        {"budget_cents": "Budget", "actual_cents": "Actual"}
    )
    melted["dollars"] = melted["cents"].abs() / 100
    # Whole dollars, matching the axis and the hover; the table beside the
    # chart carries the cents.
    melted["label"] = melted["dollars"].map(lambda amount: f"${amount:,.0f}")

    # Each trace opts into its own label placement, because the two bars of a
    # category have their own lengths to fit a label into. The bars of a pair
    # sit side by side, so neither can ever be as wide as the whole plot: the
    # width assumed per bar is the plot width times each bar's share of its
    # grouping, or a short bar would be judged wide enough to hold its label,
    # plotly would shrink the label's font to squeeze it, and a shrunken amount
    # is not the readable number the label is there to be. The axis headroom
    # has to cover the longest outside label of either trace.
    plans: dict[str, tuple[list[str], list[str]]] = {}
    axis_max = 0.0
    for kind in ("Budget", "Actual"):
        rows = melted[melted["kind"] == kind]
        positions, colors, rows_max = bar_label_plan(
            rows["dollars"].reset_index(drop=True),
            rows["label"].reset_index(drop=True),
            font_size=SUMMARY_CHART_LABEL_FONT_SIZE,
            plot_width_px=PLOT_WIDTH_PX * GROUPED_BAR_SLOT_SHARE,
        )
        plans[kind] = (positions, colors)
        axis_max = max(axis_max, rows_max)

    fig = px.bar(
        melted,
        x="dollars",
        y="display_label",
        color="kind",
        orientation="h",
        barmode="group",
        # Resolved against the column so the template below can print it.
        text="label",
        # The canvas grows a row of fixed thickness per category, twice a table
        # row, so both bars of the pair draw twice as thick. Whatever the row
        # count, the pane scrolls to the rows that outrun it rather than
        # squeezing the bars.
        height=(
            SUMMARY_TABLE_HEADER_PX
            + SUMMARY_CHART_ROW_PX * len(budgeted)
            + SUMMARY_TABLE_FLOOR_PX
        ),
        title="Budget vs actual by category",
        labels={"display_label": "Category", "dollars": "Amount"},
        color_discrete_map={"Budget": BUDGET_BAR, "Actual": ACTUAL_BAR},
    )
    # Horizontal bars, so a category name reads along the same line as its two
    # bars and the bars themselves can stretch right. No $ on the tick numbers
    # here: the bars carry their amounts in their labels, and a $0..$500 scale
    # running up the front of every category row reads as a $ in front of the
    # names.
    fig.update_layout(
        showlegend=True,
        margin=dict(t=50, l=4, r=10, b=4),
        legend_title_text="",
        bargap=0.12,
    )
    fig.update_yaxes(title=None)
    fig.update_xaxes(
        title=None,
        # Headroom for the labels drawn past the end of their bar.
        range=[0, axis_max],
        nticks=6,
    )
    for kind, (positions, colors) in plans.items():
        fig.update_traces(
            selector={"name": kind},
            texttemplate="%{text}",
            textposition=positions,
            textfont={"size": SUMMARY_CHART_LABEL_FONT_SIZE, "color": colors},
            insidetextanchor="end",
            cliponaxis=False,
            hovertemplate="%{y}<br>$%{x:,.2f}<extra>%{fullData.name}</extra>",
        )
    # Drawn inside a pane exactly as long as the budget table beside it, like
    # the spend chart. The canvas is taller whenever the categories outrun the
    # pane's eight or so chunky rows, so the pane scrolls to each bar rather
    # than the page stretching to fit them.
    with st.container(height=pane_height, border=False):
        st.plotly_chart(fig, width="stretch", key="budget_vs_actual")


def trend_chart(
    summary: pd.DataFrame,
    monthly_budget_cents: int = 0,
    show_income: bool = True,
    height: int = 400,
) -> None:
    """Income, spend and the monthly budget per month.

    A month-based line chart rather than daily: a two-point sample is the
    only case where this looks sparse, and it fills in as history accumulates.
    The x values stay as the 'YYYY-MM' period strings so a click maps cleanly
    back to a month in the selector.

    `monthly_budget_cents` is the sum of the monthly targets in scope, drawn as
    a flat dashed line so each month's spend can be read against it; zero draws
    no line. `show_income` is off when the chosen categories take in no income,
    so a spending-only selection does not carry an Income line flat at $0.
    `height` is the months table's pane's length, so the chart reads as long as
    the table it stands beside.
    """
    if summary.empty:
        return

    series = {"spend_cents": "Spend"}
    if show_income:
        series = {"income_cents": "Income", **series}
    melted = summary.melt(
        id_vars=["period"],
        value_vars=list(series),
        var_name="kind",
        value_name="cents",
    )
    melted["kind"] = melted["kind"].map(series)
    if monthly_budget_cents > 0:
        budget_rows = pd.DataFrame(
            {"period": summary["period"], "kind": "Budget", "cents": monthly_budget_cents}
        )
        melted = pd.concat([melted, budget_rows], ignore_index=True)
    # Both series are plotted as positive magnitudes so their heights can be
    # compared directly. Signed figures live in the KPI row, where the sign
    # is meaningful for the net total.
    melted["dollars"] = melted["cents"].abs() / 100
    melted["label"] = melted["period"].map(period_label)

    fig = px.line(
        melted,
        x="label",
        y="dollars",
        color="kind",
        markers=True,
        title="Income and spend by month" if show_income else "Spend by month",
        labels={"label": "Month", "dollars": "Amount", "kind": ""},
        category_orders={
            "label": list(summary["period"].map(period_label)),
            "kind": ["Income", "Spend", "Budget"],
        },
        color_discrete_map={"Income": "#2ca02c", "Spend": "#1f77b4", "Budget": "#7f7f7f"},
        height=height,
    )
    fig.update_traces(
        hovertemplate="%{x}<br>$%{y:,.2f}<extra>%{fullData.name}</extra>"
    )
    fig.update_traces(line={"dash": "dash"}, marker={"size": 4}, selector={"name": "Budget"})
    polish(fig)
    # The pane is wider than it is tall, so the legend runs along the top of the
    # chart in one horizontal line instead of claiming a column of plot width.
    fig.update_layout(
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom"),
        margin=dict(t=70, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, width="stretch", key="trend", on_select="rerun")


def category_month_grid(matrix: pd.DataFrame) -> None:
    """Spend as a grid of amounts: categories down the side, months across the
    top, the dollar figure printed in every cell.

    Laid out like a table rather than a chart on purpose. The two charts above
    answer 'which category is biggest' and 'is this month over budget'; this
    answers 'what did this category cost in that month', which is a number to be
    read, not a length to be compared. So there is no colour and no bar: every
    cell simply shows its amount, and a pair with no spending shows $0 rather
    than a gap, so the grid stays square and the eye can run along a row or down
    a column without losing its place.

    Months run along the top because that is where a table's column headings go,
    and because on a grid tall enough to need scrolling the headings would
    otherwise sit at the bottom out of sight.
    """
    st.subheader("Spend by category and month", anchor=False)
    if matrix.empty:
        st.info("No spending recorded yet.")
        return

    wide = matrix.pivot(index="display_label", columns="period", values="spend_cents")
    # Months left to right in time order; categories largest at the bottom of
    # the frame so that, with plotly drawing the first row at the foot of the
    # grid, the biggest spender ends up at the top.
    wide = wide.reindex(sorted(wide.columns), axis=1)
    order = wide.fillna(0).abs().sum(axis=1).sort_values(ascending=True).index
    wide = wide.loc[order]

    # Spend is stored negative; the amount is the magnitude. A zero-filled cell
    # is a real 'nothing spent', not missing data, so it reads as $0.
    amounts = wide.fillna(0).abs() / 100.0
    # Two summary columns after the months: the average per month across every
    # month in the grid (a month with nothing spent counts as $0, so a category
    # bought once a quarter is not averaged as if it were monthly), and the total.
    amounts[AVERAGE_COLUMN] = amounts[list(wide.columns)].mean(axis=1)
    amounts[TOTAL_COLUMN] = amounts[list(wide.columns)].sum(axis=1)
    months = [period_label(period) for period in wide.columns]
    columns = [*months, AVERAGE_COLUMN, TOTAL_COLUMN]
    categories = list(wide.index)
    text = [[f"${value:,.2f}" for value in row] for row in amounts.to_numpy()]

    fig = go.Figure(
        go.Heatmap(
            x=columns,
            y=categories,
            # z is handed over as a plain list of lists because plotly encodes a
            # numpy array as a binary blob the browser reads but nothing else
            # can. It is here only for the hover and the grid shape.
            z=amounts.to_numpy().tolist(),
            text=text,
            texttemplate="%{text}",
            textfont={"size": LABEL_FONT_SIZE},
            # A flat, single-colour scale: the cell fill carries no meaning, so
            # every cell is the same pale wash whatever its amount. This is the
            # 'not heat coloured' part; the number, not the shade, is the value.
            colorscale=[[0, "#eef2f6"], [1, "#eef2f6"]],
            showscale=False,
            # Thin gaps draw the grid lines that make it read as a table.
            xgap=2,
            ygap=2,
            hovertemplate="%{x}<br>%{y}<br>$%{z:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Spend by category and month",
        margin=dict(t=70, l=10, r=10, b=10),
        # One row per category, so the height follows the category count rather
        # than stretching a few rows into tall bands.
        height=max(240, 34 * len(categories) + 130),
    )
    # Column headings across the top, the way a table reads.
    fig.update_xaxes(side="top", tickfont={"size": 12}, fixedrange=True)
    fig.update_yaxes(tickfont={"size": 12}, fixedrange=True)
    st.plotly_chart(fig, width="stretch", key="category_months", on_select="rerun")


def read_click(key: str, *fields: str) -> str | None:
    """Return the label of a *new* point click in a plotly chart.

    Streamlit keeps a chart's selection in session state under the chart's own
    key, and that selection persists across reruns, so a plain rerun would
    otherwise re-apply the same click forever.

    The selection is never written to: Streamlit forbids assigning to a
    widget's key, and doing so raises StreamlitValueAssignmentNotAllowedError
    from the chart call itself. Instead the label last acted on is recorded in
    a separate, non-widget key, and a click is treated as new only when it
    differs from that marker.

    `fields` names the point keys to try, in order. There is no single answer:
    plotly's line trace reports the clicked category under 'x', a horizontal
    bar puts it under 'y', and a heatmap cell reports its row under 'y' while
    also carrying the column under 'x'. The caller lists the ones its chart
    means, so a click is never read from a field the trace does not emit and
    silently dropped. 'customdata' is tried last and unwrapped from the list
    plotly puts it in, since it is the field the category chart populates.
    """
    state = st.session_state.get(key)
    if not state:
        return None
    points = state.get("selection", {}).get("points", [])
    if not points:
        return None
    point = points[0]

    label = None
    for field in (*fields, "customdata"):
        value = point.get(field)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is not None:
            label = value
            break
    if label is None:
        return None

    handled_key = f"handled_{key}"
    if st.session_state.get(handled_key) == str(label):
        return None
    return str(label)


# --- page ----------------------------------------------------------------

VIEWS = ["Summary", "Transactions", "Savings"]

# Data-label font on the charts. A touch larger than plotly's default so the
# amounts on the bars and in the grid are read at a glance rather than squinted
# at.
LABEL_FONT_SIZE = 13
# The two bar charts' data labels in the summary, once the bars doubled. A
# label on a thicker bar has the room to be read without squinting, so it is
# bumped up with the bar; the grid's cells keep the base size.
SUMMARY_CHART_LABEL_FONT_SIZE = 16

# The spend bars' two states. With nothing marked every bar is the strong blue;
# once a category is picked, that bar keeps it and the others pale, so the pick
# is visible without any bar disappearing.
SPEND_BAR = "#1f77b4"
SPEND_BAR_PALE = "#bcd9ef"
# The comparison chart draws its two bars in the same language as the trend: the
# target in grey, the spending it is measured against in blue.
BUDGET_BAR = "#7f7f7f"
ACTUAL_BAR = "#1f77b4"

# Heights of the two columns in each summary row: the table on the left, the
# chart on the right. The table's dataframe gets an explicit height - enough
# for its first 17 rows plus its header - because Streamlit draws a dataframe
# at its own default height (~400px is only ten rows) unless told how tall to
# be. Fewer categories than the 17-row cap leave blank space in the table
# rather than fake empty rows, and rows past the cap make it scroll inside
# itself. The pane the chart sits in is given that very same height, so the two
# read as the same length whatever the row count, and the row itself is a pair
# of columns that share the page rather than a fixed-width pair that would have
# to scroll sideways (the table's column is the wider of the two, since the
# table carries a row of named figures where the chart carries bars). The few
# pixels the title line and the chart's buttons take above the two bodies are
# small and alike, so the bodies stay the matched length.
SUMMARY_TABLE_HEADER_PX = 60
SUMMARY_TABLE_ROW_PX = 40
SUMMARY_TABLE_MAX_ROWS = 17
SUMMARY_TABLE_FLOOR_PX = 8
# A bar chart row is twice a table row: the bars draw twice as thick, so a
# category's bar carries its number with room to spare in the taller band
# rather than being squeezed into a 40px one. The chart's canvas grows a row
# per bar up to whatever categories are in scope, so past eight or so chunky
# bars it outruns the pane beside the table, and the pane scrolls inside itself
# to each row rather than stretching the page.
SUMMARY_CHART_ROW_PX = 80
# The months table holds a full year before it needs to scroll, and grows to
# whatever months are loaded past that; the trend chart beside it is given the
# dataframe's very own height, so the two match however many months there are.
TREND_TABLE_HEADER_PX = 60
TREND_TABLE_ROW_PX = 40
TREND_TABLE_MAX_ROWS = 12
TREND_TABLE_FLOOR_PX = 8
# How far the transactions table reaches: the rows themselves are the point of
# that page, so the table runs down to where the Summary page's month section
# sits, and scrolls inside itself rather than dragging the page behind it.
TRANSACTIONS_PANE_HEIGHT_PX = 1100
# The gap between a section's table and its chart, in the summary and the
# trend: fixed to the whitespace between the sidebar and the report content,
# so a table and its chart sit together as closely as the section sits to the
# sidebar, and reading one pair against the other feels like the same spacing.
# (Streamlit's named gaps are terse; a brand-new "medium" is wider than this,
# and wide enough to read as a separate column on its own.)
SECTION_GAP_PX = 24

# The summary columns at the right of the category-by-month grid.
AVERAGE_COLUMN = "Monthly avg"
TOTAL_COLUMN = "Total"

# What the two charts on the summary can be switched between.
SUMMARY_CHARTS = ["Spend by category", "Budget vs actual"]


def clear_filters() -> None:
    """Reset every filter, the single way out of a filtered dashboard.

    Runs as a button callback, which Streamlit invokes before the script body
    and therefore before any filter widget is recreated, so writing their keys
    here is legal.
    """
    st.session_state["month_filter"] = ALL_MONTHS
    st.session_state["category_filter"] = []
    st.session_state["table_selection"] = None
    st.session_state["txn_kind"] = "Everything"
    st.session_state["txn_search"] = ""


st.title("Budget", anchor=False)

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    help="Leave empty to use data/raw/ or the project sample.",
)

transactions, used = load_all_sources([uploaded] if uploaded else [])

st.sidebar.caption(f"Loaded: {', '.join(used) if used else 'nothing yet'}")

if transactions.empty:
    st.info(
        "No transactions loaded. Upload a CSV export, or place one in "
        "`data/raw/`, then reload."
    )
    st.stop()

months = monthly_summary(transactions)
month_options = list(months["period"])

budgets = read_budgets(str(BUDGETS_PATH), BUDGETS_PATH.stat().st_mtime)
buckets, savings_pool = read_buckets(str(BUCKETS_PATH), file_mtime(BUCKETS_PATH))

# --- global filter state -------------------------------------------------
#
# A set of months and a category, shared by every view. Clicking a chart or a
# table row sets one of these, and it stays set until Clear filters, which is
# what 'maintained across views' requires: a filter that reset on a view change
# would not be maintained.
#
# Streamlit forbids writing a widget's session state after that widget has been
# created in the same script run, so a click cannot apply its change here and
# now. It records the intended value under a pending_* key, which is applied to
# the widgets below before they are instantiated.

# The trend reports the clicked month on 'x', where a line trace puts the point
# it was clicked at.
clicked = read_click("trend", "x")
if clicked is not None:
    # Recorded as acted on, since the chart's own selection cannot be cleared.
    st.session_state["handled_trend"] = clicked
    target = resolve_period(month_options, clicked)
    if target:
        st.session_state["pending_month"] = target

# The one chart that selects a category: a horizontal bar reports it on 'y'. A
# click picks that category, the same as clicking its row in the budget table:
# the row and the bar are marked, the trend and the transactions narrow to it,
# and the budget cards report just that category once it is marked, like a
# chosen month reports just that month. The table and the chart keep every
# category in view, since the point of a selection is to mark one rather than
# to hide the rest.
clicked = read_click("category_spend", "y")
if clicked is not None:
    st.session_state["handled_category_spend"] = clicked
    st.session_state["table_selection"] = clicked
    # Flagged so the budget table, which reads its own selection before it
    # draws, does not read a stale row back over this choice.
    st.session_state["table_selection_from_chart"] = clicked

# The category-by-month grid puts the months across the top, so a clicked cell
# reports its month on 'x' as the formatted label, which resolves back to the
# period the selector keys off. The category on 'y' is deliberately ignored: a
# cell click narrows the months, not the category.
clicked = read_click("category_months", "x")
if clicked is not None:
    st.session_state["handled_category_months"] = clicked
    target = resolve_period(month_options, clicked)
    if target:
        st.session_state["pending_month"] = target

pending_month = st.session_state.pop("pending_month", None)
if pending_month is not None:
    st.session_state["month_filter"] = pending_month

# The filter controls are widgets, so their keys must exist before they are
# created. The month starts on 'All months' and the category multiselect starts
# empty, which both mean 'everything', so the dashboard reconciles against the
# whole file until one is narrowed.
st.session_state.setdefault("view", "Summary")
st.session_state.setdefault("month_filter", ALL_MONTHS)
st.session_state.setdefault("category_filter", [])
st.session_state.setdefault("table_selection", None)
st.session_state.setdefault("txn_kind", "Everything")
st.session_state.setdefault("txn_account", [])
st.session_state.setdefault("txn_search", "")

# A value can outlive the file it came from, from an earlier upload, and a widget
# whose stored value is not among its options raises. The month falls back to
# 'All months'; the categories drop any that are no longer present.
category_options = sorted(set(transactions["display_label"]))
if st.session_state["month_filter"] not in [ALL_MONTHS, *month_options]:
    st.session_state["month_filter"] = ALL_MONTHS
st.session_state["category_filter"] = [
    category
    for category in st.session_state["category_filter"]
    if category in category_options
]

view = st.segmented_control(
    "View",
    VIEWS,
    key="view",
    label_visibility="collapsed",
)

st.sidebar.header("Filters")
st.sidebar.button(
    "Clear filters",
    on_click=clear_filters,
    width="stretch",
    key="clear_filters",
)
# 'Month' is a checkbox dropdown that holds one answer: tick a month and the one
# before it falls away, tick nothing and it reads 'All months'. The multiselect
# is the widget; month_filter stays the plain value every report reads, so a
# chart click or Clear filters can set it without touching the widget's state.
month_picks = st.session_state.get("month_picks", [])
seen_picks = st.session_state.get("month_picks_seen")
if month_picks != seen_picks:
    # The reader changed the widget: keep whichever month was added last, or
    # fall back to 'All months' when the list was emptied.
    added = [pick for pick in month_picks if pick not in (seen_picks or [])]
    month_choice = (
        added[-1]
        if added
        else (month_picks[0] if month_picks else ALL_MONTHS)
    )
else:
    month_choice = st.session_state["month_filter"]
st.session_state["month_filter"] = month_choice
st.session_state["month_picks"] = [month_choice]
st.session_state["month_picks_seen"] = [month_choice]

st.sidebar.multiselect(
    "Month",
    [ALL_MONTHS, *month_options],
    format_func=month_filter_label,
    select_all=False,
    key="month_picks",
)
# Categories are a multiselect: pick as many as you want to compare, or leave it
# empty for all of them. The placeholder names that empty state, and the checkbox
# list inside ticks every category in one go.
st.sidebar.multiselect(
    "Categories",
    category_options,
    select_all=True,
    key="category_filter",
    placeholder="All categories",
)

# 'All months' means every month in the report; a single choice means just that
# one. No categories, or every category, both mean no category filter; the
# second is normalised to the first so captions and labels do not list all of
# them as though it were a narrowing.
month_choice = st.session_state["month_filter"]
selected_months = (
    list(month_options) if month_choice == ALL_MONTHS else [month_choice]
)
selected_categories = list(st.session_state["category_filter"])
if set(selected_categories) >= set(category_options):
    selected_categories = []
# Months of budget to compare against: one for a single month, a full twelve per
# calendar year for 'All months'.
budget_month_count = budget_months(month_choice, month_options)

# One narrowing, done once. Every report below reads the scoped frame, so no
# view can quietly disagree with the filter about which rows are in scope.
scoped = filter_months(transactions, selected_months)
window = months_label(selected_months, month_options)

# The budget report, built once for the run. The table draws it as it stands,
# and the row picked in it becomes a category here, before any view is drawn:
# the trend, the transactions and the sidebar button all read that choice, and
# resolving it from the element inside the table would leave them a run behind.
budget_report = category_filter(
    variance_table(scoped, budgets=budgets, months=budget_month_count),
    selected_categories,
)
budget_labels = [str(label) for label in budget_report["display_label"]]
# A bar click sets the mark itself; a click in the table only carries a row
# number, which means nothing until it is matched against the report above.
from_chart = st.session_state.pop("table_selection_from_chart", None)
if from_chart is not None:
    st.session_state["table_selection"] = from_chart
else:
    # Rows that have not moved are not a fresh pick. The table keeps its own
    # state between runs, but a view that does not draw it has it dropped, and
    # an absent widget state must not read as 'the user cleared the mark'.
    stored = st.session_state.get("variance_table")
    if stored is not None:
        rows = (stored.get("selection") or {}).get("rows") or []
        if rows != list(st.session_state.get("table_selection_rows_seen") or []):
            st.session_state["table_selection"] = (
                budget_labels[rows[0]] if rows and rows[0] < len(budget_labels) else None
            )
# The mark must still be a category in scope: it is dropped when the file no
# longer carries it or the slicer excludes it, rather than being left to narrow
# the trend to nothing.
if st.session_state.get("table_selection") not in category_options:
    st.session_state["table_selection"] = None
elif (
    selected_categories
    and st.session_state["table_selection"] not in selected_categories
):
    st.session_state["table_selection"] = None
table_selection = st.session_state["table_selection"]
# What the trend and the transactions narrow to: the marked row when there is
# one, otherwise the categories the slicer chose.
narrowed_categories = [table_selection] if table_selection else selected_categories

bits = []
if selected_categories:
    bits.append(", ".join(selected_categories))
if selected_months and len(selected_months) != len(month_options):
    bits.append(window)
if bits:
    st.sidebar.caption("Active: " + ", ".join(bits))

# From a number on the summary to the transactions behind it, without hunting
# for the view. It sits with the filters because it carries their scope across,
# and only offers itself where there is a number to follow up.
if view == "Summary":
    if len(narrowed_categories) == 1:
        lead = f"the {narrowed_categories[0]} "
    elif narrowed_categories:
        lead = "these "
    else:
        lead = ""
    st.sidebar.button(
        f"See {lead}transactions behind this",
        on_click=lambda: st.session_state.update(view="Transactions"),
        key="to_txn_period",
        width="stretch",
    )


def summary_view() -> None:
    """Everything about the selected months, on one page."""
    st.header(window, anchor=False)
    if scoped.empty:
        st.info("No transactions in the selected months.")
        return

    income_cents, spend_cents = split_signs(scoped)
    kpi_row(income_cents, spend_cents)
    st.caption(
        f"{len(scoped)} transactions between "
        f"{scoped['date'].min().date()} and {scoped['date'].max().date()}."
    )

    st.subheader("Spend compared to budget", anchor=False)
    if budget_report.empty:
        st.info(
            "No spending in this category for this period."
            if selected_categories
            else "Nothing to compare yet."
        )
    else:
        # The budget cards get the same full row as the page's other cards:
        # spending is compared to a target on the open page, not within the
        # cramped width of a table column. A row picked in the table works like
        # a chosen month: the cards narrow to that category's own figures, so
        # the pick's target, spend and difference read on their own rather than
        # mixed into the page's totals. The table beneath keeps every row,
        # since the comparison is the point of it.
        kpi_table = (
            budget_report[budget_report["display_label"] == table_selection]
            if table_selection
            else budget_report
        )
        budget_kpis(kpi_table)
        # The table body and the chart are the same length, and the story they
        # tell is read together, so they share a row of two columns rather than
        # stacking. Two columns always fit the page, where a pair of fixed-width
        # panes forced the row to scroll sideways on a narrower window.
        #
        # The body of both is given the very same height: room for the table's
        # header, a row for each of the first 17 categories, and a little floor
        # under it. The dataframe is handed that height explicitly, because
        # Streamlit otherwise draws it at its own default height of about ten
        # rows, which is the whole difference between seeing 17 categories and
        # seeing a tenth of them. The chart sits in a pane of that very same
        # height, so the box the chart reads in is the same length as the table.
        # Inside the pane the chart's canvas grows a row of fixed thickness per
        # category - twice a table row, so the bars draw twice as thick - and
        # the pane scrolls to the categories that outrun it rather than
        # squeezing the bars or stretching the page. With fewer categories the
        # blank room stays in both the table and the pane, and the two always
        # line up exactly.
        body_height = (
            SUMMARY_TABLE_HEADER_PX
            + SUMMARY_TABLE_ROW_PX * SUMMARY_TABLE_MAX_ROWS
            + SUMMARY_TABLE_FLOOR_PX
        )
        # The table's column is slightly wider than the chart's, since the
        # table carries five named figures where the chart carries bars, and
        # the width it takes is the chart's loss.
        table_col, chart_col = st.columns([1.2, 1], gap=SECTION_GAP_PX)
        with table_col:
            # The table's title, with the note on reading the figures behind a
            # question mark that is hovered, like the hint on a metric, rather
            # than clicked.
            notes = []
            if budget_month_count != 1:
                years = budget_month_count // 12
                notes.append(
                    "Targets are monthly, so each budget here is the monthly "
                    f"target times {budget_month_count} "
                    f"({'a full year' if years == 1 else f'{years} full years'}). "
                    "Spending covers only the months loaded so far, so a year still "
                    "in progress reads under budget until it is complete."
                )
            unbudgeted = budget_report[budget_report["status"] == "no budget"]
            if not unbudgeted.empty:
                notes.append(
                    f"{len(unbudgeted)} categor{'y has' if len(unbudgeted) == 1 else 'ies have'} "
                    "no target set. Use the budget editor below to add one."
                )
            if notes:
                head, hint = st.columns([4, 1], vertical_alignment="center")
                with head:
                    st.subheader("Budget by Category", anchor=False)
                with hint:
                    st.markdown(
                        '<div style="text-align:right">'
                        f'<span title="{"&#10;".join(notes)}" '
                        'style="cursor:help; display:inline-block; font-size:0.78em; '
                        'line-height:1.2em; width:1.2em; height:1.2em; text-align:center; '
                        'color:var(--primary-color); border:1px solid currentColor; '
                        'border-radius:50%;">?</span>'
                        "</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.subheader("Budget by Category", anchor=False)
            variance_table_panel(budget_report, body_height)
        with chart_col:
            # Only one chart is drawn at a time, behind a switch: side by side,
            # the two were each half the width, and the category names, the
            # labels and the scale all had to be squeezed to fit half a chart's
            # worth of room. One chart at a time has the room, and the switch
            # costs one click.
            chart_view = st.segmented_control(
                "Show",
                SUMMARY_CHARTS,
                default=SUMMARY_CHARTS[0],
                label_visibility="collapsed",
                key="summary_chart",
                help="Both charts cover the same categories; this only changes which is drawn.",
            )
            if chart_view == "Budget vs actual":
                # Ordered by actual spend, largest first, which is the row the
                # eye lands on first at the top of the chart. The toggle flips
                # the order; it only exists for this chart, whose ordering is
                # what a reader would want to change.
                ascending = st.toggle(
                    "Smallest actual first",
                    key="comparison_asc",
                    help="Budget vs actual lists the categories by actual "
                    "spend, largest first. Tick this to put the smallest "
                    "first instead.",
                )
                budget_vs_actual_chart(
                    scoped, budget_month_count, budgets, body_height, ascending
                )
            else:
                category_spend_chart(
                    scoped, key="category_spend", pane_height=body_height
                )


def savings_left_over() -> None:
    """Income, spend, savings set aside and what is left, for the Month filter."""
    st.subheader("Left after spending and saving", anchor=False)
    figures = left_after_savings(scoped, buckets, selected_months)
    columns = st.columns(4)
    columns[0].metric("Income", CURRENCY(figures["income"]), border=True)
    columns[1].metric("Spend", CURRENCY(-figures["spend"]), border=True)
    columns[2].metric(
        "Set aside for savings",
        CURRENCY(figures["allocated"]),
        border=True,
        help="Every started bucket's monthly contribution, for each month in scope, "
        "including the month still in progress.",
    )
    left = figures["left"]
    columns[3].metric(
        "Left over",
        CURRENCY(left),
        delta="Spare" if left >= 0 else "Short",
        delta_color="normal" if left >= 0 else "inverse",
        delta_arrow="off",
        border=True,
        help="Income minus spend minus savings. Negative means the months spent "
        "and saved more than came in.",
    )
    st.caption(
        f"For {window}. Follows the **Month** filter; the **Categories** filter "
        "does not apply here, since income is not tied to a category."
    )


def parse_pool_text(raw: str) -> tuple[int | None, str | None]:
    """A cash total as entered, into cents, or a message the page can print.

    Accepts '137000', '137000.00' and '$137,000.00', so the field can be typed
    however a keyboard hands it, and commas or a dollar sign cost nothing. The
    page renders the field's value as text, which is also what lets it show
    '.00' dot decimals: a numeric input would hand the browser the number, and
    the browser would re-render it in the machine's own locale, comma and all.
    """
    text = raw.strip().replace("$", "").replace(",", "").strip()
    if text == "":
        return None, "Enter a cash total to save it."
    try:
        cents = int(round(float(text) * 100))
    except ValueError:
        return None, "The cash total must be a number, like 137000.00."
    if cents < 0:
        return None, "The cash total cannot be negative."
    return cents, None


def savings_cash_check() -> None:
    """Cash in the bank against what the buckets hold, and its input."""
    st.subheader("Cash in the bank", anchor=False)
    input_column, _ = st.columns([1, 2])
    pool_cents = int(round(float(savings_pool) * 100))
    with input_column:
        st.session_state.setdefault("savings_pool_input", f"${pool_cents / 100:,.2f}")
        pool_text = st.text_input(
            "Total cash available for savings",
            key="savings_pool_input",
            help="The total in your bank that the buckets should account for. "
            "Enter it by hand; it is saved to buckets.yaml.",
        )
        pool_cents, pool_problem = parse_pool_text(pool_text)
        if st.button("Save cash total", key="save_savings_pool"):
            if pool_problem:
                st.error(pool_problem)
            else:
                save_buckets(buckets, savings_pool=pool_cents / 100, path=BUCKETS_PATH)
                st.cache_data.clear()
                st.session_state["savings_saved"] = (
                    f"Saved cash total of ${pool_cents / 100:,.2f}."
                )
                st.rerun()

    # Reconciled against the figure as entered, so the check updates as you type
    # rather than only after saving. When the entry does not parse, the check
    # keeps the last saved total so the page does not blank out mid-edit.
    if pool_cents is None:
        pool_cents = int(round(float(savings_pool) * 100))
    check = savings_reconciliation(transactions, buckets, pool_cents)
    unallocated = check["unallocated"]
    columns = st.columns(3)
    columns[0].metric("Total cash", CURRENCY(check["pool"]), border=True)
    columns[1].metric("In buckets", CURRENCY(check["allocated"]), border=True)
    columns[2].metric(
        "Unallocated",
        CURRENCY(unallocated),
        delta=unallocated_status(unallocated),
        # Red only when the buckets claim more than is held; cash not yet given
        # a bucket is shown in grey, as something to assign rather than a fault.
        delta_color="inverse" if unallocated < 0 else ("normal" if unallocated == 0 else "off"),
        delta_arrow="off",
        border=True,
        help="Total cash minus the sum of bucket balances. $0 means every dollar "
        "is assigned to a bucket.",
    )


def savings_buckets_report() -> None:
    """Contributions, spending, balance and goal progress per bucket."""
    st.subheader("Buckets", anchor=False)
    if not buckets:
        st.info("No savings buckets yet. Add one in the editor below.")
        return

    summary = savings_summary(transactions, buckets)
    has_goal = summary["goal_cents"] > 0
    shown = pd.DataFrame(
        {
            "Category": summary["category"],
            "Monthly contribution": dollars(summary["monthly_cents"]),
            "Contributed": dollars(summary["contributed_cents"]),
            "Adjustment": summary["adjustment_cents"] / 100,
            "Spent": dollars(summary["spent_cents"]),
            "Balance": summary["balance_cents"] / 100,
            # Blank rather than $0 for a bucket with no goal, so 'no goal' does
            # not read as 'a goal of nothing'.
            "Goal": (summary["goal_cents"] / 100).where(has_goal),
            # Clamped to the bar's 0-100% range; a bucket past its goal shows a
            # full bar, and an overdrawn one an empty bar.
            "Progress": (summary["pct_saved"].astype("float") * 100).clip(0, 100),
            "Left to save": (summary["remaining_cents"].astype("float") / 100),
        }
    )
    months = months_loaded(transactions)
    st.dataframe(
        # Balance is green while there is money in the bucket and red once it is
        # overdrawn, the same colours as everywhere else.
        shown.style.apply(net_style, subset=["Balance"]),
        hide_index=True,
        width="stretch",
        key="savings_table",
        column_config={
            "Monthly contribution": st.column_config.NumberColumn(
                "Monthly contribution", format="dollar"
            ),
            "Contributed": st.column_config.NumberColumn(
                "Contributed",
                format="dollar",
                help=f"Monthly contribution times the {months} month(s) loaded.",
            ),
            "Adjustment": st.column_config.NumberColumn(
                "Adjustment",
                format="dollar",
                help="Manual offset, set in the editor below.",
            ),
            "Spent": st.column_config.NumberColumn(
                "Spent",
                format="dollar",
                help="Net spending in the category over the months loaded; a refund "
                "goes back in.",
            ),
            "Balance": st.column_config.NumberColumn(
                "Balance", format="dollar", help="Contributed plus Adjustment minus Spent."
            ),
            "Goal": st.column_config.NumberColumn(
                "Goal", format="dollar", help="Target balance. Blank means no goal."
            ),
            "Progress": st.column_config.ProgressColumn(
                "Progress",
                min_value=0.0,
                max_value=100.0,
                format="%.0f%%",
                help="Balance as a share of the goal.",
            ),
            "Left to save": st.column_config.NumberColumn(
                "Left to save", format="dollar", help="Goal minus Balance; $0 once met."
            ),
        },
    )
    first, last = transactions["date"].min(), transactions["date"].max()
    st.caption(
        f"Covers the {months} month(s) loaded, {period_label(first.to_period('M'))} to "
        f"{period_label(last.to_period('M'))}, whatever **Month** is set to. Contributed "
        "uses each bucket's current monthly contribution for every one of those months, "
        "so changing it changes the whole figure. Use **Adjustment** to account for "
        "anything the data does not show."
    )


def bucket_editor() -> None:
    """Add, change or remove buckets, saved to buckets.yaml."""
    st.subheader("Edit buckets", anchor=False)
    choices = sorted(
        set(category_breakdown(transactions)["display_label"]) | set(budgets) | set(buckets)
    )
    frame = pd.DataFrame(
        [
            {
                "Category": category,
                "Monthly contribution": float(bucket["contribution"]),
                "Adjustment": float(bucket["adjustment"]),
                "Goal": float(bucket["goal"]),
            }
            for category, bucket in sorted(buckets.items())
        ],
        columns=["Category", "Monthly contribution", "Adjustment", "Goal"],
    )
    # The key carries a version that moves on every save, so rows added before a
    # save are not replayed on top of the saved file afterwards.
    version = st.session_state.setdefault("bucket_editor_version", 0)
    edited = st.data_editor(
        frame,
        hide_index=True,
        width="stretch",
        num_rows="dynamic",
        key=f"bucket_editor_{version}",
        column_config={
            "Category": st.column_config.SelectboxColumn(
                "Category",
                options=choices,
                required=True,
                help="The spend category this bucket saves for, and that draws it down.",
            ),
            "Monthly contribution": st.column_config.NumberColumn(
                "Monthly contribution", min_value=0.0, step=10.0, format="dollar"
            ),
            "Adjustment": st.column_config.NumberColumn(
                "Adjustment",
                step=10.0,
                format="dollar",
                help="Added to the balance: savings already held, or a negative amount "
                "for money taken out that the data does not show.",
            ),
            "Goal": st.column_config.NumberColumn(
                "Goal",
                min_value=0.0,
                step=100.0,
                format="dollar",
                help="Target balance. Leave at $0 for no goal.",
            ),
        },
    )
    if st.button("Save buckets", type="primary", key="save_buckets"):
        parsed, problems = parse_bucket_rows(edited.to_dict("records"))
        if problems:
            for problem in problems:
                st.error(problem)
            return
        save_buckets(parsed, path=BUCKETS_PATH)
        st.cache_data.clear()
        st.session_state["bucket_editor_version"] = version + 1
        st.session_state["savings_saved"] = (
            f"Saved {len(parsed)} bucket(s) to {BUCKETS_PATH.name}."
        )
        st.rerun()


def savings_view() -> None:
    """Buckets, cash reconciliation and what is left after saving."""
    st.header("Savings", anchor=False)
    saved = st.session_state.pop("savings_saved", None)
    if saved:
        st.success(saved)
    savings_left_over()
    st.divider()
    savings_cash_check()
    st.divider()
    savings_buckets_report()
    st.divider()
    bucket_editor()


if view == "Savings":
    savings_view()
    st.stop()

if view == "Summary":
    summary_view()
else:
    st.header(window, anchor=False)
    transactions_view(scoped, window, selected_months)
    # The transactions page is the raw table and nothing else: it fills the
    # screen rather than being followed by the trend and the budget editor,
    # which belong to the summary. The summary keeps both.
    st.stop()

st.divider()
st.subheader("Trend", anchor=False)
if len(month_options) < 3:
    st.caption(
        f"Only {len(month_options)} month(s) of data so far. The trend fills in as "
        "more history is loaded."
    )
# The trend, the month table and the category-by-month grid follow the Categories
# filter, so the trend of one category (or a handful) can be read on its own. They
# still cover every month whatever Month is set to: they are how another month
# gets chosen, and filtering them to one month would leave a single point.
trend_frame = category_filter(transactions, selected_categories)
# The marked row narrows the trend on top of the slicer, so the trend answers
# to the same choice the transactions do. The mark is dropped earlier if the
# slicer no longer allows it, so this cannot empty the frame.
if table_selection:
    trend_frame = trend_frame[trend_frame["display_label"] == table_selection]
st.caption(
    "Trends follow the **Categories** filter but cover every month loaded, "
    "whatever **Month** is set to, so any month can be picked from here."
    + (
        f" Showing: {', '.join(narrowed_categories)}."
        if narrowed_categories
        else ""
    )
)
trend_months = monthly_summary(trend_frame)
# The budget line is the monthly targets of the categories in scope: all of
# them with no category filter, only the chosen ones with one.
trend_budget_cents = 100 * sum(
    target
    for label, target in budgets.items()
    if not narrowed_categories or label in narrowed_categories
)
# Income is drawn whenever it is in scope: always with no category filter, and
# otherwise only if the categories in scope actually took some in. The amount's
# sign is not enough, since a refund in Groceries is positive and is not income.
income_taken = (trend_frame["amount_cents"] > 0) & trend_frame[
    "display_label"
].isin(INCOME_CATEGORIES)
show_income = not narrowed_categories or bool(income_taken.any())
# The same row as the summary: the months table on the left, the line that
# draws them on the right, in a pair of columns that share the page rather than
# a fixed-width pair that would scroll sideways. The grid below covers every
# month of every category, so it keeps the whole width.
# The table body is told its height explicitly - enough for a full year of
# rows plus its header, and however many months are loaded past that - so the
# months never scroll; the line chart beside it is given that very same
# height, so the two match.
trend_body_height = (
    TREND_TABLE_HEADER_PX
    + TREND_TABLE_ROW_PX * max(len(trend_months), TREND_TABLE_MAX_ROWS)
    + TREND_TABLE_FLOOR_PX
)
months_col, trend_col = st.columns([1, 1.7], gap=SECTION_GAP_PX)
with months_col:
    month_overview_table(trend_months, trend_body_height)
with trend_col:
    trend_chart(
        trend_months, trend_budget_cents, show_income, height=trend_body_height
    )
category_month_grid(category_month_matrix(trend_frame))

st.divider()
budget_editor(transactions, budgets)
