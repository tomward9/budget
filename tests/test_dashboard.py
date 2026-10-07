"""Tests for the dashboard's drill-down behaviour, filters and table formatting.

The failure this guards against: Streamlit forbids assigning to a widget's
session_state key, and it forbids writing a widget's state before that widget
has been created in the current run. Both mistakes surface as a crash on the
first click rather than at startup, so the app would appear healthy and then
break the moment it was used.

The month multiselect leans on the same rule from the other side. Its 'All
months' option and the Clear filters button both write the widget's key from a
callback, which is the only place a write is legal, so both are tested for the
value they leave behind and not merely for not crashing.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from budget.ingest import ingest
from budget.presentation import (
    ALL_MONTHS,
    LABEL_OFF_BAR,
    LABEL_ON_BAR,
    OVER_FILL,
    SELECT_FILL,
    UNDER_FILL,
    bar_label_plan,
    budget_months,
    budget_style,
    budget_totals,
    category_filter,
    color_status,
    dollars,
    month_filter_label,
    months_label,
    net_style,
    percent_bar,
    period_label,
    resolve_period,
    spend_bar_labels,
)
from budget.reports import load_budgets, variance_table

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

AUGUST = "2026-08"
SEPTEMBER = "2026-09"
JANUARY = "2026-01"

# The project sample now covers the whole of 2026, with activity in every
# budget category every month.
MONTHS_2026 = [f"2026-{m:02d}" for m in range(1, 13)]
MONTH_LABELS_2026 = [period_label(p) for p in MONTHS_2026]


def make_csv(*rows: str) -> bytes:
    header = "Date,Description,Institution,Account,Category,Is Hidden,Is Pending,Amount"
    return ("\n".join([header, *rows, ""])).encode("utf-8")


def loaded(*rows: str) -> pd.DataFrame:
    frame, _ = ingest(io.BytesIO(make_csv(*rows)))
    return frame


def render() -> AppTest:
    """Start the dashboard and fail loudly if it raised."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    if at.exception:
        raise AssertionError(f"app raised: {at.exception[0].message}")
    return at


# --- page structure helpers ----------------------------------------------
#
# Element order is the layout. AppTest keeps the rendered tree, and a block's
# children arrive as a dict keyed by their position, so walking it in key order
# gives the order the user sees.


def main_children(at: AppTest) -> list:
    """Top-level elements of the page, in the order they render."""
    return [child for _, child in sorted(at.main.children.items())]


def top_level_index(at: AppTest, kind: str, key: str | None = None) -> int | None:
    """Where an element sits among the page's top-level elements."""
    for position, child in enumerate(main_children(at)):
        if child.type != kind:
            continue
        if key is None or getattr(child, "key", None) == key:
            return position
    return None


def node_keys(node) -> set[str]:
    """Every widget key in a block's subtree."""
    keys = set()
    if getattr(node, "key", None) is not None:
        keys.add(node.key)
    children = getattr(node, "children", None)
    if isinstance(children, dict):
        for child in children.values():
            keys |= node_keys(child)
    return keys


def row_holding(at: AppTest, *keys: str):
    """The narrowest block holding every key: the row they share.

    The summary and the trend each put a table and a chart side by side in one
    horizontal row, so the row is what has to be found rather than either of
    its panes.
    """
    wanted = set(keys)
    found = [0, None]

    def walk(node, depth: int) -> None:
        if wanted <= node_keys(node):
            found[:] = [depth, node]
        children = getattr(node, "children", None)
        if isinstance(children, dict):
            for child in children.values():
                walk(child, depth + 1)

    walk(at.main, 0)
    assert found[1] is not None, f"no block holds {sorted(wanted)}"
    return found[1]


def panes(row) -> list:
    """The direct children of a row, in the order they are drawn."""
    return [child for _, child in sorted(row.children.items())]


def _pixel_height(node) -> int | None:
    """The pixel height a fixed-height block was given, if it was given one."""
    match = re.search(
        r"height_config\s*{\s*pixel_height:\s*(\d+)\s*}",
        str(getattr(node, "proto", "")),
    )
    return int(match.group(1)) if match else None


def scroll_box(parent, *keys: str):
    """The fixed-height pane holding `keys`: the box a chart reads in.

    The chart's scroll pane is an anonymous container given the table body's
    height, so in the tree it is a flex_container carrying a pixel height; the
    chart canvas inside can be taller than the pane, and the pane scrolls to
    it rather than the page growing.
    """
    wanted = set(keys)

    def walk(node):
        if _pixel_height(node) is not None and wanted <= node_keys(node):
            return node
        children = getattr(node, "children", None)
        if isinstance(children, dict):
            for child in children.values():
                hit = walk(child)
                if hit is not None:
                    return hit
        return None

    pane = walk(parent)
    assert pane is not None, f"no fixed-height pane holds {sorted(wanted)}"
    return pane


def find_key(at: AppTest, key: str):
    """The block or element carrying `key`, wherever on the page it sits."""
    def walk(node):
        if getattr(node, "key", None) == key:
            return node
        children = getattr(node, "children", None)
        if isinstance(children, dict):
            for child in children.values():
                hit = walk(child)
                if hit is not None:
                    return hit
        return None

    for _, child in sorted(at.main.children.items()):
        hit = walk(child)
        if hit is not None:
            return hit
    raise AssertionError(f"nothing on the page carries the key {key!r}")


def difference_css(shown: pd.DataFrame, status) -> str:
    """The CSS the styler puts on the single Difference cell of `shown`.

    Reading it out of the computed styler, rather than testing colour_status on
    its own, is what proves the colour lands on Difference. A subset still
    naming Status would leave this cell unstyled and fail.
    """
    styles = budget_style(shown, status)._compute().ctx
    column = list(shown.columns).index("Difference")
    return " ".join(
        f"{name}: {value}" for name, value in styles.get((0, column), [])
    )


def shown_budget(frame: pd.DataFrame, budgets: dict[str, int], month: str) -> pd.DataFrame:
    """The variance table as the app hands it to the styler."""
    table = variance_table(frame, month=month, budgets=budgets)
    return pd.DataFrame(
        {
            "Category": table["display_label"],
            "Difference": dollars(table["variance_cents"]),
        },
        index=table.index,
    )


def figure(at: AppTest, key: str) -> dict:
    """The parsed plotly figure behind a keyed chart."""
    element = at.get_by_key(key)
    return json.loads(element.proto.spec)


def trace_values(trace: dict, field: str) -> list:
    """One field of a trace as plain Python values.

    Plotly 7 hands a numpy array to the browser as base64 keyed by dtype and
    shape rather than as a JSON list. The browser decodes that, so a test
    reading the same payload has to as well; without this, every numeric field
    arrives as a dict and comparisons against numbers raise.
    """
    value = trace.get(field)
    if not isinstance(value, dict) or "bdata" not in value:
        return list(value or [])
    raw = value["bdata"]
    if isinstance(raw, list):
        raw = "".join(raw)
    buffer = np.frombuffer(base64.b64decode(raw), dtype=np.dtype(value["dtype"]))
    shape = value.get("shape")
    return buffer.reshape(shape).tolist() if shape else buffer.tolist()


def chart_titles(at: AppTest) -> list[str]:
    return [
        json.loads(chart.proto.spec)["layout"].get("title", {}).get("text", "")
        for chart in at.get("plotly_chart")
    ]


def chart_traces(at: AppTest) -> list[dict]:
    """Every trace on every chart, whichever kind it is."""
    return [
        trace
        for chart in at.get("plotly_chart")
        for trace in json.loads(chart.proto.spec)["data"]
    ]


def go_to(at: AppTest, view: str) -> None:
    """Switch the view, the way the segmented control does."""
    for control in at.segmented_control:
        if control.label == "View":
            control.set_value(view)
    at.run()


def press(at: AppTest, label: str) -> None:
    """Click a button by its visible label and let the app react.

    AppTest's click only sets the value, so the run is what actually fires the
    button's callback. A button that changes state and is never re-run looks
    like a button that does nothing.
    """
    for button in at.button:
        if button.label == label:
            button.click()
            at.run()
            return
    raise AssertionError(f"no button labelled {label!r}")


def transaction_table(at: AppTest) -> pd.DataFrame:
    """The transaction table, the one with a Description column."""
    for frame in at.dataframe:
        if "Description" in frame.value.columns:
            return frame.value
    raise AssertionError("no transaction table found")


def budget_table(at: AppTest) -> pd.DataFrame:
    """The budget against actual table on the summary view."""
    return at.get_by_key("variance_table").value


# --- period labels and click resolution ---------------------------------


def test_period_label_is_human_readable() -> None:
    assert period_label("2026-09") == "Sep 2026"
    assert period_label("2026-01") == "Jan 2026"


def test_period_label_tolerates_junk() -> None:
    """A malformed period must not raise; it renders as given."""
    assert period_label("not-a-month") == "not-a-month"


def test_resolve_period_maps_a_clicked_label_back() -> None:
    assert resolve_period(["2026-08", "2026-09"], "Aug 2026") == "2026-08"


def test_resolve_period_ignores_an_unknown_label() -> None:
    """Returns None rather than a wrong month, so a stray click is a no-op."""
    assert resolve_period(["2026-08"], "Mar 1999") is None


def test_resolve_period_handles_a_none_label() -> None:
    assert resolve_period(["2026-08"], None) is None


# --- naming a month selection --------------------------------------------


def test_one_month_is_named_outright() -> None:
    assert months_label([SEPTEMBER]) == "Sep 2026"


def test_two_months_are_named_as_a_span() -> None:
    assert months_label([AUGUST, SEPTEMBER]) == "Aug 2026 to Sep 2026"


def test_every_month_available_is_called_all_months() -> None:
    assert months_label([AUGUST, SEPTEMBER], [AUGUST, SEPTEMBER]) == "All months"


def test_a_long_selection_is_counted_rather_than_listed() -> None:
    """Nine month names under a heading is worse than a count."""
    months = [f"2026-{number:02d}" for number in range(1, 10)]
    assert months_label(months) == "9 months"


def test_no_selection_is_named_as_such() -> None:
    assert months_label([]) == "No months selected"


# --- the open question of the build: does the app actually start? ---------


def test_app_opens_on_the_summary_view() -> None:
    at = render()
    assert at.session_state["view"] == "Summary"


def test_the_views_are_summary_transactions_and_savings() -> None:
    """The month and year tabs were replaced by the month filter; Savings is
    its own view because buckets are a separate model from the budget."""
    at = render()
    views = next(
        control for control in at.segmented_control if control.label == "View"
    )
    assert list(views.options) == ["Summary", "Transactions", "Savings"]


def month_select(at: AppTest):
    """The month slicer: a checkbox dropdown that holds one answer."""
    return next(control for control in at.multiselect if control.label == "Month")


def category_select(at: AppTest):
    """The categories multiselect."""
    return next(control for control in at.multiselect if control.label == "Categories")


def set_month(at: AppTest, value: str) -> None:
    """Pick a month (or ALL_MONTHS) and let the app react."""
    month_select(at).set_value([value])
    at.run()


def set_categories(at: AppTest, values: list[str]) -> None:
    """Pick a set of categories and let the app react."""
    category_select(at).set_value(values)
    at.run()


def test_the_month_option_reads_as_its_scope() -> None:
    """The 'all' entry is named for what it covers, so the control says what it
    is reporting on rather than leaving it to a separate button."""
    assert month_filter_label(ALL_MONTHS) == "All months"
    assert month_filter_label(SEPTEMBER) == "Sep 2026"


def test_the_month_filter_is_a_single_select() -> None:
    """Reported: the month multiselect's chips fought with its 'All months'
    entry, so choosing all and then dropping one left the two disagreeing. The
    slicer is a checkbox dropdown that holds one answer: ticking a second month
    replaces the first, so the two can never disagree."""
    at = render()
    labels = [str(control.label) for control in at.multiselect]
    assert "Month" in labels
    # No 'Select all' entry either: 'All months' is an option, not the sum of
    # ticking every month in the file.
    assert not month_select(at).proto.select_all

    month_select(at).set_value(["Aug 2026", "Sep 2026"])
    at.run()
    assert at.session_state["month_filter"] == SEPTEMBER
    assert month_select(at).value == [SEPTEMBER]


def test_the_month_starts_on_all_months() -> None:
    """Unfiltered on open: 'All months' covers the whole file until a month is
    chosen."""
    at = render()
    assert at.session_state["month_filter"] == ALL_MONTHS
    assert at.header[0].value == "All months"


def test_all_months_reports_every_month_of_spend() -> None:
    """The 'all' option has to mean every month in the report, not 'no month'."""
    at = render()
    go_to(at, "Transactions")
    assert len(transaction_table(at)) == 480


def test_the_month_filter_offers_every_month_in_the_file() -> None:
    at = render()
    assert list(month_select(at).options) == ["All months", *MONTH_LABELS_2026]


def test_there_are_no_select_all_buttons() -> None:
    """The scope lives inside each filter now, so the ambiguous shared buttons
    are gone."""
    at = render()
    labels = [str(b.label) for b in at.sidebar.button]
    assert not any(label in ("Select all", "Deselect all") for label in labels)
    assert month_select(at).options[0] == "All months"


def test_category_filter_keeps_the_listed_categories() -> None:
    """The multiselect hands over a list now, so the filter has to take one.
    Empty means no filter, one or many means keep exactly those."""
    frame = loaded(
        "05/08/2026,A,INST,MAX CHECKING,Groceries_,No,No,($10)",
        "05/08/2026,B,INST,MAX CHECKING,Gasoline,No,No,($20)",
        "05/08/2026,C,INST,MAX CHECKING,Tithing,No,No,($30)",
    )
    assert set(category_filter(frame, [])["display_label"]) == {
        "Groceries",
        "Gasoline",
        "Tithing",
    }
    assert set(category_filter(frame, ["Groceries"])["display_label"]) == {"Groceries"}
    assert set(category_filter(frame, ["Groceries", "Tithing"])["display_label"]) == {
        "Groceries",
        "Tithing",
    }
    # A bare string is tolerated as a single category rather than matched
    # character by character.
    assert set(category_filter(frame, "Gasoline")["display_label"]) == {"Gasoline"}


def test_the_category_filter_is_a_multiselect_starting_empty() -> None:
    """Categories are picked as a set now, and an empty selection means all of
    them, so there is no 'All categories' entry to conflict with the chips."""
    at = render()
    category = category_select(at)
    assert at.session_state["category_filter"] == []
    assert "Groceries" in category.options and "Tithing" in category.options
    go_to(at, "Transactions")
    assert len(set(transaction_table(at)["Category"])) > 1


def test_choosing_a_month_covers_only_that_month() -> None:
    at = render()
    set_month(at, SEPTEMBER)
    assert at.session_state["month_filter"] == SEPTEMBER
    assert at.header[0].value == "Sep 2026"

    set_month(at, ALL_MONTHS)
    assert at.session_state["month_filter"] == ALL_MONTHS
    assert at.header[0].value == "All months"


def test_app_renders_without_exception() -> None:
    at = render()
    assert not at.exception


@pytest.mark.parametrize("view", ["Summary", "Transactions"])
def test_every_view_renders(view: str) -> None:
    at = render()
    go_to(at, view)
    assert not at.exception, f"{view} view raised: {at.exception[0].message}"


def test_no_partial_period_banner_is_shown() -> None:
    """The banner was removed by request; coverage fields stay in the data
    layer but must not surface as a warning."""
    at = render()
    messages = " ".join(str(w.value) for w in at.warning)
    assert "Partial period" not in messages


def test_transfers_toggle_is_gone() -> None:
    at = render()
    labels = [str(s.label) for s in at.sidebar.toggle] + [
        str(s.label) for s in at.toggle
    ]
    assert not any("transfer" in label.lower() for label in labels)


def test_a_category_missing_from_the_file_is_dropped() -> None:
    """A stale label from an earlier upload would make the categories widget
    raise on a value outside its options, so it is swept out first. The month
    selectbox falls back the same way; it just cannot be exercised through the
    test harness, which validates an injected value before the sweep runs."""
    at = render()
    at.session_state["category_filter"] = ["Groceries", "Not A Category"]
    at.run()
    assert at.session_state["category_filter"] == ["Groceries"]


# --- the month filter -----------------------------------------------------


def test_choosing_a_month_narrows_the_transactions_table() -> None:
    at = render()
    set_month(at, AUGUST)
    go_to(at, "Transactions")
    assert len(transaction_table(at)) == 40


def test_a_single_month_still_offers_the_trends() -> None:
    """The trends are how another month gets chosen, so they stay on screen
    whatever the month filter is set to."""
    at = render()
    set_month(at, SEPTEMBER)
    headings = [s.value for s in at.subheader]
    assert "Spend by category and month" in headings
    assert at.get_by_key("category_months") is not None


def test_a_single_month_summary_names_that_month() -> None:
    at = render()
    set_month(at, SEPTEMBER)
    assert at.header[0].value == "Sep 2026"


def test_a_chosen_month_is_reported_in_the_sidebar() -> None:
    """Whatever is filtered has to be visible in the sidebar, or a summary that
    quietly covers one month looks like a summary of everything."""
    at = render()
    set_month(at, AUGUST)
    assert "Active: Aug 2026" in " ".join(c.value for c in at.sidebar.caption)


def test_the_budget_is_charged_for_every_month_in_scope() -> None:
    """Targets are monthly: one month compares against one month of target, and
    'All months' against a full year of it. The target is read from the live
    budgets file, since that is what the page reads too."""
    target = float(load_budgets()["Tithing"])
    at = render()
    set_month(at, SEPTEMBER)
    one_month = float(budget_table(at).set_index("Category").loc["Tithing", "Budget"])

    set_month(at, ALL_MONTHS)
    all_months = float(budget_table(at).set_index("Category").loc["Tithing", "Budget"])

    assert one_month == pytest.approx(target)
    assert all_months == pytest.approx(target * 12)


# --- the categories filter ------------------------------------------------


def test_choosing_a_category_narrows_the_transactions_table() -> None:
    at = render()
    set_categories(at, ["Tithing"])
    assert at.session_state["category_filter"] == ["Tithing"]
    go_to(at, "Transactions")
    assert set(transaction_table(at)["Category"]) == {"Tithing"}


def test_choosing_several_categories_keeps_them_all() -> None:
    """A multiselect is worth having only if more than one category can be
    compared at once."""
    at = render()
    set_categories(at, ["Tithing", "Groceries"])
    go_to(at, "Transactions")
    assert set(transaction_table(at)["Category"]) == {"Tithing", "Groceries"}


def test_an_empty_category_selection_means_every_category() -> None:
    at = render()
    set_categories(at, ["Tithing"])
    set_categories(at, [])
    go_to(at, "Transactions")
    assert len(set(transaction_table(at)["Category"])) > 1


def test_a_yearly_budget_says_how_it_was_scaled() -> None:
    """'All months' compares against a full year of budget, so the page has to
    say so, or a year in progress reads as mysteriously under budget. The note
    rides behind the question mark beside the table's title: hover, not click."""
    at = render()
    hints = [
        markdown.value
        for markdown in at.markdown
        if 'title="' in str(markdown.value)
    ]
    assert any("times 12 (a full year)" in hint for hint in hints), (
        "the hover hint does not explain the yearly scaling"
    )


def test_all_months_charges_a_full_year_of_budget() -> None:
    """Reported: the yearly budget was the monthly target times the months
    loaded so far. It is the monthly target times twelve."""
    assert budget_months(ALL_MONTHS, [AUGUST, SEPTEMBER]) == 12
    assert budget_months(SEPTEMBER, [AUGUST, SEPTEMBER]) == 1


def test_data_spanning_two_years_charges_two_years_of_budget() -> None:
    assert budget_months(ALL_MONTHS, ["2025-12", "2026-01"]) == 24


# --- the budget table, and where its colour lives -------------------------


def test_the_status_column_is_gone() -> None:
    """The verdict now rides on the Difference cell. A separate column repeated
    what the sign already said and had to be read across the row."""
    at = render()
    assert list(budget_table(at).columns) == [
        "Category",
        "Budget",
        "Actual",
        "Difference",
        "Used",
    ]


def test_an_under_budget_difference_is_green() -> None:
    frame = loaded("05/08/2026,ALDI,INST,MAX CHECKING,Groceries_,No,No,($50)")
    table = variance_table(frame, month="2026-08", budgets={"Groceries": 100})
    shown = shown_budget(frame, {"Groceries": 100}, "2026-08")
    assert UNDER_FILL in difference_css(shown, table["status"])


def test_an_over_budget_difference_is_red() -> None:
    frame = loaded("05/08/2026,ALDI,INST,MAX CHECKING,Groceries_,No,No,($150)")
    table = variance_table(frame, month="2026-08", budgets={"Groceries": 100})
    shown = shown_budget(frame, {"Groceries": 100}, "2026-08")
    assert OVER_FILL in difference_css(shown, table["status"])


def test_spending_the_whole_target_is_red() -> None:
    """A zero difference is a target used up in full, not money left over. This
    is the case that would break if the colour were driven by the sign alone."""
    frame = loaded("05/08/2026,ALDI,INST,MAX CHECKING,Groceries_,No,No,($100)")
    table = variance_table(frame, month="2026-08", budgets={"Groceries": 100})
    shown = shown_budget(frame, {"Groceries": 100}, "2026-08")
    assert table["variance_cents"].iloc[0] == 0
    assert OVER_FILL in difference_css(shown, table["status"])


def test_a_category_with_no_target_is_left_uncoloured() -> None:
    frame = loaded("05/08/2026,ALDI,INST,MAX CHECKING,Groceries_,No,No,($50)")
    table = variance_table(frame, month="2026-08", budgets={})
    shown = shown_budget(frame, {}, "2026-08")
    assert difference_css(shown, table["status"]) == ""


def test_over_budget_shows_red_and_under_shows_green() -> None:
    assert OVER_FILL in color_status("over")
    assert UNDER_FILL in color_status("under")


def test_no_budget_row_is_left_uncoloured() -> None:
    assert color_status("no budget") == ""


def test_difference_style_returns_one_css_string_per_cell() -> None:
    """Streamlit's marshaller unpacks each style entry as (property, value),
    so a list here raises 'too many values to unpack' at render time."""
    table = pd.DataFrame({"Difference": [1.0, -1.0, 0.0]})
    budget_style(table, ["under", "over", "no budget"])._compute()


def marked_row_css(shown: pd.DataFrame, status, selected: str) -> dict:
    """The computed CSS of every cell, keyed by (row, column name)."""
    styles = budget_style(shown, status, selected=selected)._compute().ctx
    return {
        (row, list(shown.columns)[column]): " ".join(
            f"{name}: {value}" for name, value in declarations
        )
        for (row, column), declarations in styles.items()
    }


MARKED_TABLE = pd.DataFrame(
    {
        "Category": ["Tithing", "Groceries"],
        "Budget": [50.0, 400.0],
        "Actual": [52.3, 380.0],
        "Difference": [2.3, -20.0],
        "Used": [100.0, 95.0],
    }
)


def test_the_marked_row_is_tinted_and_the_others_are_not() -> None:
    """The pick has to be visible on the row itself: the trend and the
    transactions are on other pages, and the KPIs deliberately keep everything."""
    css = marked_row_css(MARKED_TABLE, ["over", "under"], selected="Groceries")
    assert SELECT_FILL in css.get((1, "Category"), "")
    assert SELECT_FILL not in css.get((0, "Category"), ""), (
        "a row that was not picked got the tint"
    )
    assert "font-weight: bold" in css.get((1, "Budget"), "")


def test_the_marked_row_keeps_its_budget_verdict() -> None:
    """The tint marks where the pick is; Difference still has to say whether it
    is under or over, so the tint is kept off that cell."""
    css = marked_row_css(MARKED_TABLE, ["over", "under"], selected="Groceries")
    marked = css.get((1, "Difference"), "")
    assert "font-weight: bold" in marked
    assert f"background-color: {UNDER_FILL}" in marked
    other = css.get((0, "Difference"), "")
    assert f"background-color: {OVER_FILL}" in other
    assert SELECT_FILL not in other


# --- the spend bar labels -------------------------------------------------


def test_a_bar_label_carries_the_amount_and_the_share() -> None:
    labels = spend_bar_labels(pd.Series([-10000, -30000]), -40000)
    assert labels.tolist() == ["$100  ·  25%", "$300  ·  75%"]


def test_bar_label_shares_survive_a_negative_total() -> None:
    """Spend is stored negative, so a caller passes the sum of a spend column
    straight in. Dividing by that raw sum hits a negative denominator and prints
    0% everywhere, which looks like a plausible chart, not a failure."""
    labels = spend_bar_labels(pd.Series([-10000, -30000]), -40000)
    assert all("0%" not in label for label in labels)


def test_bar_label_share_is_of_the_whole_period() -> None:
    """A chart already filtered to one category still has to report that
    category's true size; dividing by its own amount would say 100%."""
    labels = spend_bar_labels(pd.Series([-10000]), -40000)
    assert labels.tolist() == ["$100  ·  25%"]


def test_bar_labels_survive_a_zero_total() -> None:
    labels = spend_bar_labels(pd.Series([0, 0]), 0)
    assert labels.tolist() == ["$0  ·  0%", "$0  ·  0%"]


def test_every_bar_on_the_chart_is_labelled() -> None:
    """The label replaces the donut, so a bar with no label would leave the
    share question unanswered for that category."""
    at = render()
    chart = figure(at, "category_spend")
    labels = chart["data"][0]["text"]
    assert labels, "the chart has no bar labels at all"
    assert all("%" in str(label) and "$" in str(label) for label in labels)


def test_a_long_bar_holds_its_label_inside() -> None:
    """A bar with room keeps the label on the bar, where it reads as part of it
    rather than as a number floating nearby."""
    at = render()
    chart = figure(at, "category_spend")
    trace = chart["data"][0]
    assert "inside" in trace["textposition"]
    # The longest bar, which has the most room of any.
    assert trace["textposition"][-1] == "inside"


def test_a_label_that_cannot_fit_is_drawn_outside_instead() -> None:
    """A label squashed into a short bar stops being readable, and the amount is
    the one thing the label is for. It goes past the tip instead."""
    plan = bar_label_plan(
        pd.Series([500.0, 1.0]), pd.Series(["$500  ·  99%", "$1  ·  0%"])
    )
    positions, colors, _ = plan
    assert positions[0] == "inside"
    assert positions[1] == "outside"


def test_a_label_outside_the_bar_is_a_colour_that_reads_on_the_background() -> None:
    """White text outside the bar would be white on the plot background and
    invisible; the colour has to follow the label's position."""
    _, colors, _ = bar_label_plan(
        pd.Series([500.0, 1.0]), pd.Series(["$500  ·  99%", "$1  ·  0%"])
    )
    assert colors[0] == LABEL_ON_BAR
    assert colors[1] == LABEL_OFF_BAR
    assert LABEL_OFF_BAR != LABEL_ON_BAR


def test_every_bar_label_is_readable_on_the_chart() -> None:
    """The check that matters on the rendered chart: whichever side a label is
    on, its colour matches, and no bar is left unlabelled or unpositioned."""
    at = render()
    trace = figure(at, "category_spend")["data"][0]
    positions = list(trace["textposition"])
    colors = list(trace["textfont"]["color"])
    assert len(positions) == len(trace_values(trace, "x")) == len(colors)
    for position, color in zip(positions, colors):
        assert position in ("inside", "outside")
        assert color == (LABEL_ON_BAR if position == "inside" else LABEL_OFF_BAR)


def test_the_axis_leaves_room_for_a_label_drawn_outside() -> None:
    """An outside label runs past the end of its bar, so the axis has to reach
    past the longest bar or those labels are clipped at the edge of the plot.
    Left to plotly, the axis stops at the longest bar and they are cut off."""
    at = render()
    chart = figure(at, "category_spend")
    longest = max(trace_values(chart["data"][0], "x"))
    axis_max = chart["layout"]["xaxis"]["range"][1]
    assert axis_max > longest


def test_a_bar_label_plan_with_nothing_to_plot_is_all_outside() -> None:
    """No bar means no bar to put a label inside, and the colours still have to
    be the ones that read on the background."""
    positions, colors, axis_max = bar_label_plan(pd.Series([0.0]), pd.Series(["$0"]))
    assert positions == ["outside"]
    assert colors == [LABEL_OFF_BAR]
    assert axis_max == 0.0


# --- the spend and budget charts ------------------------------------------


def test_the_donut_is_gone() -> None:
    """The share of spend is now printed on every bar, so the donut repeated it
    while taking a whole column and a whole chart slot to itself."""
    at = render()
    assert not [trace for trace in chart_traces(at) if trace.get("type") == "pie"]
    assert not any("Share of spend" in title for title in chart_titles(at))


def test_the_budget_table_and_the_chart_are_side_by_side() -> None:
    """Requested: the table on the left with the exact numbers, the chart on
    the right with the shape of them, in one row rather than one above the
    other. The row is a pair of page-stretching columns rather than a pair of
    fixed-width panes, so nothing reads as sitting inside a frame and nothing
    forces the row to scroll sideways."""
    at = render()
    row = row_holding(at, "variance_table", "category_spend")
    assert row.type == "flex_container", "the table and chart are not in one row"
    assert "direction: HORIZONTAL" in str(row.proto)
    assert "use_stretch: true" in str(row.proto), "the row does not fill the page"
    table_col, chart_col = panes(row)
    assert "variance_table" in node_keys(table_col), "the table is not the left column"
    assert "category_spend" in node_keys(chart_col), "the chart is not the right column"
    for column in (table_col, chart_col):
        assert column.type == "column", "a side of the row is not a column"
        assert "border: true" not in str(column.proto), "a column is still boxed"
        assert "pixel_width:" not in str(column.proto), (
            "a column is fixed-width, so the row scrolls sideways"
        )


def test_each_column_of_the_summary_row_takes_a_share_of_the_page() -> None:
    """A share of the page is what squeezed both: the table needs room for its
    five columns and the chart for its labels, so each column takes a share of
    the page rather than one squeezing the other. The chart's switch rides in
    the chart column beside the chart it switches, not in a control line of its
    own above the row."""
    at = render()
    table_col, chart_col = panes(
        row_holding(at, "variance_table", "category_spend")
    )
    for column in (table_col, chart_col):
        proto = str(column.proto)
        assert "weight:" in proto, "a column takes no share of the page"
        assert "pixel_width:" not in proto, "a column is fixed-width again"
        assert "use_stretch: true" not in proto, "a column ignores its share"
    assert "summary_chart" in node_keys(chart_col), (
        "the chart's switch no longer sits beside the chart"
    )
    assert "category_spend" in node_keys(chart_col)


def test_the_table_and_chart_gaps_match_the_sidebar_margin() -> None:
    """Requested: the gap between a table and its chart, in both the budget and
    the trend sections, is the very same size as the margin between the sidebar
    and the report content. One shared number sets both, so a table and its
    chart sit together as closely as the section sits to the sidebar, and the
    two sections read with the same spacing."""
    at = render()
    for row in (
        row_holding(at, "variance_table", "category_spend"),
        row_holding(at, "month_overview", "trend"),
    ):
        proto = str(row.proto)
        assert "gap_config" in proto
        match = re.search(r"pixel_gap:\s*(\d+)", proto)
        assert match, f"the row still uses a named gap: {proto}"
        assert int(match.group(1)) == 24, (
            f"the row's {match.group(1)}px gap is not the sidebar margin's 24px"
        )


def test_the_budget_table_holds_17_categories_before_it_scrolls() -> None:
    """Requested: 17 categories fit before the table needs to scroll, the rest
    reached by scrolling inside the table, and the budget v actual and spend by
    category charts are as long as the table. Streamlit draws a dataframe at
    its own default height of about ten rows unless it is handed one, so the
    table's dataframe is given the 17-row length explicitly and the pane the
    chart reads in gets that very same length - header, 17 rows and a little
    floor - so the two always line up exactly. Inside the pane the chart's
    canvas is one double-thick row per category, so it runs past the pane
    rather than squeezing the bars, and the pane scrolls."""
    at = render()
    table_col, chart_col = panes(
        row_holding(at, "variance_table", "category_spend")
    )
    body = 60 + 40 * 17 + 8
    assert _pixel_height(scroll_box(chart_col, "category_spend")) == body, (
        "the chart's pane is not the table body's 17-row length"
    )
    spend = figure(at, "category_spend")
    bars = len(trace_values(spend["data"][0], "x"))
    assert int(spend["layout"]["height"]) == 60 + 80 * bars + 8, (
        "the spend canvas is not one double-thick row per category"
    )
    assert any(s.value == "Budget by Category" for s in at.subheader), (
        "the table has no in-line title"
    )
    assert any(control.label == "Show" for control in at.segmented_control), (
        "the chart's switch no longer sits beside the chart"
    )
    assert "variance_table" in node_keys(table_col), "the table left its column"
    assert "summary_chart" in node_keys(chart_col), "the switch left the chart's column"
    switch = next(
        control for control in at.segmented_control if control.label == "Show"
    )
    switch.set_value("Budget vs actual")
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    comparison = figure(at, "budget_vs_actual")
    comparison_bars = len(trace_values(comparison["data"][0], "x"))
    assert int(comparison["layout"]["height"]) == 60 + 80 * comparison_bars + 8, (
        "the budget comparison canvas is not one double-thick row per category"
    )
    _, chart_col = panes(row_holding(at, "variance_table", "budget_vs_actual"))
    assert _pixel_height(scroll_box(chart_col, "budget_vs_actual")) == body, (
        "the budget comparison's pane left the table's length"
    )


def test_the_two_charts_are_swapped_by_a_control() -> None:
    """Half a chart's width each squeezed the category names, the labels and the
    scale. One chart at a time gives both the full width, and the switch costs
    one click."""
    at = render()
    switch = next(
        control for control in at.segmented_control if control.label == "Show"
    )
    assert list(switch.options) == ["Spend by category", "Budget vs actual"]


def test_only_one_of_the_two_charts_is_drawn_at_a_time() -> None:
    at = render()
    assert "Spend by category" in chart_titles(at)
    assert "Budget vs actual by category" not in chart_titles(at)


def test_the_switch_draws_the_budget_comparison_instead() -> None:
    at = render()
    switch = next(
        control for control in at.segmented_control if control.label == "Show"
    )
    switch.set_value("Budget vs actual")
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    titles = chart_titles(at)
    assert "Budget vs actual by category" in titles
    assert "Spend by category" not in titles


def test_the_switch_draws_the_spend_bars_again() -> None:
    at = render()
    # Re-found each time: a rerun rebuilds the element tree, so a control
    # captured before it would be setting a value on a widget that no longer
    # exists.
    for choice in ("Budget vs actual", "Spend by category"):
        switch = next(
            control for control in at.segmented_control if control.label == "Show"
        )
        switch.set_value(choice)
        at.run()
        assert not at.exception, at.exception[0].message if at.exception else ""
    assert "Spend by category" in chart_titles(at)


def test_the_summary_never_scrolls_sideways_when_the_sidebar_is_open() -> None:
    """Requested: once the sidebar is open there is still no sideways scrolling
    for the budget table and chart section. A pair of fixed-width panes had to
    scroll sideways when they did not fit; the columns share whatever the page
    gives them and wrap instead, so there is never a horizontal scrollbar."""
    at = render()
    row = row_holding(at, "variance_table", "category_spend")
    proto = str(row.proto)
    assert "direction: HORIZONTAL" in proto
    assert "wrap: true" in proto, "the columns will not wrap on a narrow page"
    assert "use_stretch: true" in proto, "the row is not tied to the page's width"
    for column in panes(row):
        assert "pixel_width:" not in str(column.proto), (
            "a column is fixed-width, which pushes the row sideways"
        )


# --- the summary and trend layouts ----------------------------------------


def test_the_months_table_and_the_trend_share_a_row() -> None:
    """The same row as the summary: the months on the left, the line drawing
    them on the right, in a pair of columns that share the page rather than a
    fixed-width pair that scrolls sideways."""
    at = render()
    row = row_holding(at, "month_overview", "trend")
    assert row.type == "flex_container", "the months and trend are not in one row"
    months_col, trend_col = panes(row)
    assert "month_overview" in node_keys(months_col), "the table is not the left column"
    assert "trend" in node_keys(trend_col), "the chart is not the right column"
    for column in (months_col, trend_col):
        assert column.type == "column"
        assert "pixel_width:" not in str(column.proto), (
            "a column is fixed-width, which pushes the row sideways"
        )


def test_the_months_table_fits_every_loaded_month_without_scrolling() -> None:
    """Requested: the months table fits however many months are loaded, holding
    a full year before it needs to scroll, and the line chart beside it is as
    long as the table. The months table's dataframe is handed that height
    explicitly and the trend's canvas gets the same number, so the two read as
    one length whatever the month count."""
    at = render()
    months = len(at.get_by_key("month_overview").value)
    body = 60 + 40 * max(months, 12) + 8
    assert int(figure(at, "trend")["layout"]["height"]) == body, (
        "the trend canvas is not the months table's length"
    )
    months_col, trend_col = panes(row_holding(at, "month_overview", "trend"))
    assert "month_overview" in node_keys(months_col)
    assert "trend" in node_keys(trend_col)


def test_the_category_grid_keeps_the_whole_width() -> None:
    """The grid covers every month of every category, so it is not one of the
    row's panes and takes the page rather than half of it."""
    at = render()
    assert top_level_index(at, "plotly_chart", "category_months") is not None


def test_the_summary_keeps_the_trend_and_the_budget_editor() -> None:
    at = render()
    headings = [s.value for s in at.subheader]
    assert "Trend" in headings
    assert "Budgets" in headings


def test_the_budget_comparison_is_still_on_the_summary() -> None:
    at = render()
    switch = next(
        control for control in at.segmented_control if control.label == "Show"
    )
    switch.set_value("Budget vs actual")
    at.run()
    assert "Budget vs actual by category" in chart_titles(at)


def test_the_budget_comparison_draws_horizontal_bars() -> None:
    """Requested: the two bars of a category stretch sideways rather than up a
    narrow column of height, and the category name reads along its own bars.
    Every bar carries its amount in a label like the spend chart does, and the
    scale along the front of the rows is plain numbers - the $ belongs to the
    bars' labels, not to a set of ticks reading in front of the categories."""
    at = render()
    switch = next(
        control for control in at.segmented_control if control.label == "Show"
    )
    switch.set_value("Budget vs actual")
    at.run()
    chart = figure(at, "budget_vs_actual")
    traces = chart["data"]
    assert traces, "the switch drew no budget comparison at all"
    assert all(trace["orientation"] == "h" for trace in traces)
    assert {"Budget", "Actual"} <= {trace["name"] for trace in traces}
    for trace in traces:
        assert any(str(label).startswith("$") for label in trace_values(trace, "text")), (
            f"the {trace['name']} bars carry no amount labels"
        )
    assert not str(chart["layout"]["xaxis"].get("tickprefix", "") or ""), (
        "the amount scale still shows a $ before its ticks"
    )
    bars = len(trace_values(traces[0], "x"))
    assert int(chart["layout"]["height"]) == 60 + 80 * bars + 8, (
        "the budget comparison canvas is not one double-thick row per category"
    )


def test_the_budget_comparison_lists_the_largest_actual_first() -> None:
    """Requested: the categories are ordered by actual spend, largest at the
    top, and the toggle flips the list to smallest first."""
    at = render()
    switch = next(
        control for control in at.segmented_control if control.label == "Show"
    )
    switch.set_value("Budget vs actual")
    at.run()

    def drawn_amounts() -> list[float]:
        chart = figure(at, "budget_vs_actual")
        actual = next(trace for trace in chart["data"] if trace["name"] == "Actual")
        return trace_values(actual, "x")

    # Horizontal bars paint the first data row at the bottom, so the trace runs
    # bottom-to-top: smallest first, largest last. The top row, the one the eye
    # lands on first, is therefore the biggest spender.
    amounts = drawn_amounts()
    assert amounts == sorted(amounts), "the largest spender is not drawn on top"
    assert amounts[-1] == max(amounts), "the top row is not the largest actual"
    at.toggle(key="comparison_asc").set_value(True).run()
    flipped = drawn_amounts()
    assert flipped == sorted(amounts, reverse=True), "the toggle did not flip the order"
    assert flipped != amounts, "the toggle left the order unchanged"


def test_the_budget_comparison_labels_keep_one_font_and_leave_short_bars() -> None:
    """Reported: the actual bars' labels shrank to fit, and a label stayed
    inside a bar too short to hold it. The two bars of a pair are each a share
    of the plot, not the whole of it, so the plan judges each bar at that width;
    a short bar's label moves past the tip, and no trace drops to a smaller
    font. Every position gets the colour that reads on its background."""
    at = render()
    switch = next(
        control for control in at.segmented_control if control.label == "Show"
    )
    switch.set_value("Budget vs actual")
    at.run()
    chart = figure(at, "budget_vs_actual")
    for trace in chart["data"]:
        size = trace["textfont"]["size"]
        assert isinstance(size, int) and size >= 12, (
            f"the {trace['name']} labels shrank to fit: {size!r}"
        )
        positions = trace["textposition"]
        colors = trace["textfont"]["color"]
        assert len(positions) == len(trace_values(trace, "x")) == len(colors)
        for position, color in zip(positions, colors):
            assert position in ("inside", "outside")
            assert color == (
                LABEL_ON_BAR if position == "inside" else LABEL_OFF_BAR
            )
    # The shortest actual bar has the least room of any, so its label is the one
    # that must be proved to sit outside rather than be squeezed in.
    actual = next(trace for trace in chart["data"] if trace["name"] == "Actual")
    amounts = trace_values(actual, "x")
    shortest = min(range(len(amounts)), key=lambda index: amounts[index])
    assert actual["textposition"][shortest] == "outside", (
        "the shortest bar's label does not leave the bar"
    )


# --- the category by month grid --------------------------------------------


def grid_trace(at: AppTest) -> dict:
    """The single heatmap-style trace behind the category-by-month grid."""
    return figure(at, "category_months")["data"][0]


def test_the_grid_is_not_a_bar_chart() -> None:
    """Reported: the grouped bar chart looked horrendous. It reads as a table of
    amounts now, laid out on a grid rather than as bars to compare by length."""
    at = render()
    trace = grid_trace(at)
    assert trace["type"] == "heatmap"
    assert not [t for t in chart_traces(at) if t.get("type") == "bar" and t is trace]


def test_the_grid_puts_categories_down_and_months_across() -> None:
    """Swapped from the old heatmap, which had categories across and months
    down: categories are the rows now and months are the columns."""
    at = render()
    trace = grid_trace(at)
    assert "Groceries" in trace["y"] and "Tithing" in trace["y"]
    assert list(trace["x"]) == [*MONTH_LABELS_2026, "Monthly avg", "Total"]


def test_the_grid_ends_with_an_average_and_a_total() -> None:
    """Every row carries its average per month and its total after the months.
    A month with nothing spent counts as $0 in the average, so a category that
    only spends in some months averages less than its same total."""
    at = render()
    trace = grid_trace(at)
    grid = {
        category: dict(zip(trace["x"], row))
        for category, row in zip(trace["y"], trace_values(trace, "z"))
    }
    tithing = grid["Tithing"]
    assert tithing["Total"] == pytest.approx(sum(tithing[m] for m in MONTH_LABELS_2026))
    assert tithing["Monthly avg"] == pytest.approx(
        tithing["Total"] / len(MONTH_LABELS_2026)
    )
    for row in grid.values():
        assert row["Total"] == pytest.approx(sum(row[m] for m in MONTH_LABELS_2026))


def test_the_month_headings_run_along_the_top() -> None:
    """Column headings belong at the top of a table, and on a grid tall enough
    to scroll they would otherwise sit at the bottom out of sight."""
    at = render()
    assert figure(at, "category_months")["layout"]["xaxis"]["side"] == "top"


def test_the_grid_is_not_heat_coloured() -> None:
    """The value is the printed number, not the shade, so every cell is the same
    pale wash whatever its amount and there is no colour bar to read."""
    at = render()
    trace = grid_trace(at)
    scale = trace["colorscale"]
    fills = {tuple(stop[1] for stop in scale)}
    assert len(scale) >= 2
    assert all(stop[1] == scale[0][1] for stop in scale), (
        f"the cells are still shaded by amount: {fills}"
    )
    assert trace.get("showscale") is False


def test_every_cell_shows_its_amount() -> None:
    """A grid of blank cells would be a table with no figures in it."""
    at = render()
    trace = grid_trace(at)
    printed = [cell for row in trace["text"] for cell in row]
    assert printed, "the grid has no cell labels at all"
    assert all(str(cell).startswith("$") for cell in printed)


def test_the_cell_amounts_are_not_rounded() -> None:
    """Reported: the grid rounded to whole dollars. The cents are shown, so a
    $523.47 cell reads as $523.47 rather than $523."""
    at = render()
    trace = grid_trace(at)
    printed = [cell for row in trace["text"] for cell in row]
    assert all(re.fullmatch(r"\$[\d,]+\.\d{2}", str(cell)) for cell in printed), (
        f"a cell is not showing cents: {printed}"
    )
    # The sample's largest debit carries real cents, so at least one cell must
    # end in something other than .00 for this to be a genuine 'not rounded'.
    assert any(not str(cell).endswith(".00") for cell in printed)


def test_the_cell_amounts_are_spend_magnitudes() -> None:
    """Spend is stored negative; the grid shows the amount, so a signed value
    would print a debit as a negative number in every cell."""
    at = render()
    values = [value for row in trace_values(grid_trace(at), "z") for value in row]
    assert min(values) >= 0
    assert max(values) > 0


def test_the_label_font_is_a_little_larger() -> None:
    """Reported as too small to read. The bars and the grid cells are bumped
    past plotly's default so the amounts read at a glance."""
    at = render()
    assert grid_trace(at)["textfont"]["size"] >= 12
    assert figure(at, "category_spend")["data"][0]["textfont"]["size"] >= 12


def test_the_doubled_bars_carry_a_bigger_label_font() -> None:
    """Requested: the bars doubled, and the labels grew with them. A label on a
    twice-thick bar has the room to be read without squinting, so both bar
    charts carry the bumped size while the grid's cells keep the base one."""
    at = render()
    assert figure(at, "category_spend")["data"][0]["textfont"]["size"] == 16
    assert grid_trace(at)["textfont"]["size"] == 13, (
        "the grid's cell font moved off its own size"
    )
    switch = next(
        control for control in at.segmented_control if control.label == "Show"
    )
    switch.set_value("Budget vs actual")
    at.run()
    for trace in figure(at, "budget_vs_actual")["data"]:
        assert trace["textfont"]["size"] == 16, (
            f"the {trace['name']} labels are not the doubled size"
        )


def test_the_grid_holds_every_month_even_when_one_is_chosen() -> None:
    """A single-month filter must not hide the other months from the grid, or
    the month it hid could not be clicked back on."""
    at = render()
    set_month(at, SEPTEMBER)
    assert list(grid_trace(at)["x"])[:2] == MONTH_LABELS_2026[:2]


def test_the_spend_chart_matches_the_table_length() -> None:
    """Requested: the chart reads as long as the budget table it reads against,
    and the bars are twice as thick rather than being squeezed. The pane the
    chart sits in is given the body's very length - header, 17 rows and a
    floor - and the canvas inside keeps one fixed double-thick row per bar,
    scrolling within the pane when the categories outrun it."""
    at = render()
    _, chart_col = panes(row_holding(at, "variance_table", "category_spend"))
    box = scroll_box(chart_col, "category_spend")
    assert _pixel_height(box) == 60 + 40 * 17 + 8, (
        "the chart's pane is not the table body's 17-row length"
    )
    chart = figure(at, "category_spend")
    bars = len(trace_values(chart["data"][0], "x"))
    canvas = 60 + 80 * bars + 8
    assert int(chart["layout"]["height"]) == canvas, (
        f"the {bars} bars did not each get a double-thick {80}px row"
    )


def test_the_spend_bars_match_the_budget_comparison_pair() -> None:
    """Requested: the spend chart's bars are as wide again as the budget v
    actual bars taken together, with the same room between categories. A spend
    bar and the budget-and-actual pair both fill their row's band up to the
    same 12% gaps, so the two charts read like the same ruler, and each canvas
    keeps one double-thick row per category instead of shrinking to fit."""
    at = render()
    spend = figure(at, "category_spend")
    assert spend["layout"]["bargap"] == pytest.approx(0.12), (
        "the spend chart's spacing is not the budget comparison's"
    )
    assert 1 - spend["layout"]["bargap"] == pytest.approx(0.88), (
        "a spend bar is no wider than the budget and actual pair"
    )
    spend_bars = len(trace_values(spend["data"][0], "x"))
    assert int(spend["layout"]["height"]) == 60 + 80 * spend_bars + 8, (
        "the spend canvas is not one double-thick row per category"
    )
    switch = next(
        control for control in at.segmented_control if control.label == "Show"
    )
    switch.set_value("Budget vs actual")
    at.run()
    comparison = figure(at, "budget_vs_actual")
    assert comparison["layout"]["bargap"] == pytest.approx(spend["layout"]["bargap"]), (
        "the two charts do not space their categories alike"
    )
    comparison_bars = len(trace_values(comparison["data"][0], "x"))
    assert int(comparison["layout"]["height"]) == 60 + 80 * comparison_bars + 8, (
        "the budget comparison canvas is not one double-thick row per category"
    )


def test_the_budget_cards_total_the_table() -> None:
    """Budgeted, spent and difference, over exactly the rows in the table."""
    at = render()
    table = budget_table(at)
    metrics = {m.label: m.value for m in at.metric}
    budgeted = float(table["Budget"].sum())
    spent = -float(table["Actual"].sum())
    assert metrics["Total budgeted"] == f"${budgeted:,.2f}"
    assert metrics["Total spend"] == f"${spent:,.2f}"
    assert metrics["Difference"].replace("-", "") == f"${abs(budgeted - spent):,.2f}"


def test_the_budget_cards_follow_the_category_filter() -> None:
    at = render()
    set_month(at, SEPTEMBER)
    set_categories(at, ["Tithing"])
    metrics = {m.label: m.value for m in at.metric}
    target = load_budgets()["Tithing"]
    assert metrics["Total budgeted"] == f"${target:,.2f}"
    assert metrics["Total spend"] == "$285.21"


def test_budget_totals_count_unbudgeted_spend_against_the_difference() -> None:
    table = pd.DataFrame(
        {
            "budget_cents": [10_000, 0],
            "actual_cents": [-4_000, -2_500],
            "variance_cents": [6_000, -2_500],
        }
    )
    assert budget_totals(table) == (10_000, 6_500, 3_500)


def trend_traces(at: AppTest) -> dict:
    return {trace["name"]: trace for trace in figure(at, "trend")["data"]}


def test_the_trend_has_a_budget_line() -> None:
    """The monthly budget, flat and dashed, so each month's spend reads against it."""
    at = render()
    budget = trend_traces(at)["Budget"]
    total = sum(load_budgets().values())
    assert trace_values(budget, "y") == pytest.approx([float(total)] * len(MONTHS_2026))
    assert budget["line"]["dash"] == "dash"


def test_the_budget_line_covers_only_the_chosen_categories() -> None:
    at = render()
    set_categories(at, ["Tithing"])
    budget = trend_traces(at)["Budget"]
    assert set(trace_values(budget, "y")) == {float(load_budgets()["Tithing"])}


def test_a_spending_category_draws_no_income_line() -> None:
    """Reported: Income appeared when no income category was selected."""
    at = render()
    assert "Income" in trend_traces(at)
    set_categories(at, ["Tithing"])
    assert "Income" not in trend_traces(at)
    assert "Spend" in trend_traces(at)


def test_net_is_green_when_positive_and_red_when_negative() -> None:
    styles = net_style([125.0, -40.0, 0.0])
    assert UNDER_FILL in styles[0]
    assert OVER_FILL in styles[1]
    assert styles[2] == ""


def test_the_month_table_colours_its_net_column() -> None:
    at = render()
    assert OVER_FILL in str(at.get_by_key("month_overview").proto) or UNDER_FILL in str(
        at.get_by_key("month_overview").proto
    )


def test_the_month_table_has_no_transactions_column() -> None:
    """Reported: the transaction count is not necessary on the trend's months
    table, which exists to compare the money a month took in and spent. The
    count belongs to the transactions page, where every row is one."""
    at = render()
    columns = list(at.get_by_key("month_overview").value.columns)
    assert "Transactions" not in columns
    assert set(columns) == {"Month", "Income", "Spend", "Net"}


def test_select_all_categories_fills_the_box() -> None:
    """Reported: excluding one or two categories meant adding every other one
    by hand. The checkbox list inside the dropdown ticks every category in one
    go, ready for a chip or two to come back out."""
    at = render()
    everything = list(category_select(at).options)
    set_categories(at, everything)
    assert at.session_state["category_filter"] == everything

    set_categories(at, [c for c in everything if c != "Tithing"])
    go_to(at, "Transactions")
    shown = set(transaction_table(at)["Category"])
    assert "Tithing" not in shown
    assert len(shown) > 1


def test_every_category_selected_is_treated_as_no_filter() -> None:
    """All of them is the same as none of them, so the sidebar does not list
    fifteen categories as though it were a narrowing."""
    at = render()
    set_categories(at, list(category_select(at).options))
    assert not any("Active:" in c.value for c in at.sidebar.caption)


def test_the_trends_follow_the_category_filter() -> None:
    """Reported: the trend could not show one category on its own."""
    at = render()
    set_categories(at, ["Tithing"])
    assert grid_trace(at)["y"] == ["Tithing"]
    months = next(df.value for df in at.dataframe if "Income" in df.value.columns)
    assert months["Income"].sum() == 0
    assert months["Spend"].abs().sum() == pytest.approx(3418.4)


def test_the_trends_still_cover_every_month_when_one_is_chosen() -> None:
    at = render()
    set_month(at, SEPTEMBER)
    months = next(df.value for df in at.dataframe if "Income" in df.value.columns)
    assert list(months["Month"]) == MONTH_LABELS_2026


def test_the_grid_grows_with_the_number_of_categories() -> None:
    """A fixed height with a dozen categories would squeeze every row into a
    sliver, which is the same illegibility the redesign set out to fix."""
    at = render()
    trace = grid_trace(at)
    height = figure(at, "category_months")["layout"]["height"]
    assert height >= 34 * len(trace["y"])


def test_the_largest_category_is_at_the_top() -> None:
    """Plotly draws the first row at the foot of a heatmap, so the rows run
    smallest to largest and the biggest spender ends up on top."""
    at = render()
    trace = grid_trace(at)
    totals = [sum(row) for row in trace_values(trace, "z")]
    assert totals == sorted(totals), "rows are not ordered smallest to largest"


def test_the_grid_is_square_with_no_holes() -> None:
    """Every category meets every month, so each row carries a number in every
    column: no cell is skipped and the grid stays square."""
    at = render()
    trace = grid_trace(at)
    rows = trace_values(trace, "z")
    assert len(rows) == len(trace["y"]), "a category without a full row is a hole"
    for row in rows:
        assert len(row) >= len(MONTH_LABELS_2026), "a month with no cell is a hole"
        assert all(math.isfinite(float(v)) for v in row[: len(MONTH_LABELS_2026)])


# --- money is shown in dollars, not cents ---------------------------------


def test_dollars_converts_cents() -> None:
    """The reported bug: a $503.00 tithing entry displayed as $50300.00.

    Streamlit's NumberColumn only prefixes a symbol, so a column left in cents
    renders with two extra digits of magnitude.
    """
    assert dollars(pd.Series([-50300])).tolist() == [-503.0]


def test_dollars_keeps_a_float_dtype() -> None:
    """Integer cents divided by an int 100 would land on float anyway, but the
    '$%.2f' format is only defined for floats."""
    assert dollars(pd.Series([100])).dtype.kind == "f"


def test_transaction_amounts_are_displayed_in_dollars() -> None:
    at = render()
    go_to(at, "Transactions")
    amounts = transaction_table(at)["Amount"]
    assert amounts.max() < 100_000, "amounts are still in cents"
    assert amounts.min() > -100_000, "amounts are still in cents"
    # The largest single debit in the sample is the mortgage at about $1,150,
    # so the row count alone cannot distinguish dollars from cents.
    assert -5000 < amounts.min(), "amounts are far too large to be dollars"


def test_month_overview_is_displayed_in_dollars() -> None:
    at = render()
    table = next(df.value for df in at.dataframe if "Income" in df.value.columns)
    assert table["Income"].max() < 10_000, "amounts are still in cents"


# --- percent bars ---------------------------------------------------------


def test_percent_bar_scales_a_fraction_to_percentage_points() -> None:
    """Streamlit applies a ProgressColumn format with printf semantics, so
    '%.0f%%' against 0.75 prints '1%'. The value must arrive as 75."""
    values, _ = percent_bar(pd.Series([0.75]))
    assert values.tolist() == [75.0]


def test_percent_bar_maximum_is_a_hundred() -> None:
    """The bar is drawn as value / max, so the bound has to be 100 for the
    bar's length to be the percentage."""
    _, maximum = percent_bar(pd.Series([0.5, 1.8]))
    assert maximum == pytest.approx(100.0)


def test_percent_bar_does_not_rescale_to_the_worst_overspend() -> None:
    """The reported bug: the percentage was right but the bar was not.

    A ProgressColumn draws its bar as value / max_value. With the bound
    stretched to a 174% worst case, a 48% category drew at 48/174 of the
    track, about 28%, so the bar contradicted the '48%' printed beside it.
    """
    values, maximum = percent_bar(pd.Series([0.4835, 1.7433]))
    drawn_at_fixed_bound = values.iloc[0] / maximum * 100

    # The scale that caused the report: the worst overspend, not 100.
    stretched_bound = float(values.max())
    drawn_when_stretched = values.iloc[0] / stretched_bound * 100

    assert drawn_at_fixed_bound == pytest.approx(48.35, abs=0.01)
    assert drawn_when_stretched == pytest.approx(27.7, abs=0.1)


def test_percent_bar_keeps_the_default_maximum() -> None:
    _, maximum = percent_bar(pd.Series([0.2, 0.5]))
    assert maximum == pytest.approx(100.0)


def test_budget_table_used_column_is_in_percent() -> None:
    """An unscaled fraction would put every value below 1.0 here, which is the
    visible symptom of the percent bug."""
    at = render()
    used = budget_table(at)["Used"].dropna()
    if used.empty:
        pytest.skip("no budgeted categories in the sample")
    assert max(used) > 1.0, "'Used' is still a fraction, so the bar reads as 1%"


# --- a click sets the global filter ---------------------------------------


def test_a_trend_click_selects_that_month() -> None:
    at = render()
    at.session_state["trend"] = {"selection": {"points": [{"x": "Aug 2026"}]}}
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    assert at.session_state["month_filter"] == AUGUST


def test_a_trend_click_does_not_change_the_view() -> None:
    """The filter is global, so yanking the user to another view is unnecessary;
    the filter applies wherever they are."""
    at = render()
    go_to(at, "Transactions")
    at.session_state["trend"] = {"selection": {"points": [{"x": "Aug 2026"}]}}
    at.run()
    assert at.session_state["view"] == "Transactions"
    assert at.session_state["month_filter"] == AUGUST


def test_a_click_on_the_category_by_month_grid_selects_that_month() -> None:
    """The grid puts the months across the top, so a clicked cell reports its
    month on 'x'. The category on 'y' is deliberately ignored: a cell click
    narrows the months, not the category."""
    at = render()
    at.session_state["category_months"] = {
        "selection": {"points": [{"x": "Aug 2026", "y": "Tithing", "z": 0.0}]}
    }
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    assert at.session_state["month_filter"] == AUGUST


def test_a_category_click_marks_that_category() -> None:
    """A horizontal bar chart puts the category on the y axis, so the click
    arrives under 'y'. Reading 'x' would pick up the dollar amount instead.

    A click marks the category rather than slicing the dashboard: the budget
    table and the spend chart keep every category in view, and only the trend,
    the transactions and the budget cards narrow to the one picked."""
    at = render()
    at.session_state["category_spend"] = {
        "selection": {"points": [{"x": 123.45, "y": "Groceries"}]}
    }
    at.run()
    assert not at.exception
    assert at.session_state["table_selection"] == "Groceries"
    assert at.session_state["category_filter"] == []
    assert len(budget_table(at)) > 1, "the budget table lost the other categories"
    go_to(at, "Transactions")
    assert set(transaction_table(at)["Category"]) == {"Groceries"}


def test_a_category_click_falls_back_to_custom_data() -> None:
    """The category chart populates custom_data, so a click that omits the
    trace-specific key is still resolved rather than ignored."""
    at = render()
    at.session_state["category_spend"] = {
        "selection": {"points": [{"customdata": ["Groceries"]}]}
    }
    at.run()
    assert at.session_state["table_selection"] == "Groceries"


def test_a_click_does_not_reapply_on_a_later_rerun() -> None:
    """Chart selections persist in session state, so without a handled-marker
    the same click would re-filter on every subsequent rerun."""
    at = render()
    at.session_state["trend"] = {"selection": {"points": [{"x": "Aug 2026"}]}}
    at.run()
    assert at.session_state["month_filter"] == AUGUST

    at.run()
    assert at.session_state["month_filter"] == AUGUST


def test_a_second_different_click_is_honoured() -> None:
    """The handled-marker must not block a genuinely new click."""
    at = render()
    at.session_state["trend"] = {"selection": {"points": [{"x": "Aug 2026"}]}}
    at.run()
    at.session_state["trend"] = {"selection": {"points": [{"x": "Sep 2026"}]}}
    at.run()
    assert at.session_state["month_filter"] == SEPTEMBER


def test_a_click_on_an_unknown_month_is_ignored() -> None:
    at = render()
    at.session_state["trend"] = {"selection": {"points": [{"x": "Mar 1999"}]}}
    at.run()
    assert at.session_state["month_filter"] == ALL_MONTHS


def test_read_click_returns_none_without_a_selection() -> None:
    """A chart's state exists even with no click; what matters is that its
    point list is empty, so read_click must not treat that as a click."""
    at = render()
    state = at.session_state.get("trend")
    assert state is None or not state.get("selection", {}).get("points")


def test_clicking_a_budget_row_marks_that_category() -> None:
    """A picked row marks that category: the table keeps every other row in
    view with the pick highlighted, and the trend and the transactions narrow
    to it."""
    at = render()
    at.session_state["variance_table"] = {"selection": {"rows": [0]}}
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    assert at.session_state["table_selection"] == "Mortgage"
    assert at.session_state["category_filter"] == []
    assert len(budget_table(at)) > 1, "the budget table lost the other categories"
    assert grid_trace(at)["y"] == ["Mortgage"]


def test_clicking_a_budget_row_narrows_the_budget_cards_to_it() -> None:
    """Requested: a row picked in the budget table filters the spend-vs-budget
    cards the same way a chosen month does. The cards read the picked
    category's own budget, spend and difference instead of the page's totals,
    and a bar click marks the self-same row so it narrows them the same way.
    Nothing else moves: the table still shows every category, since the
    comparison is the point of it."""
    at = render()
    at.session_state["variance_table"] = {"selection": {"rows": [0]}}
    at.run()
    assert at.session_state["table_selection"] == "Mortgage"
    row = budget_table(at).iloc[0]
    metrics = {m.label: m.value for m in at.metric}
    budgeted = float(row["Budget"])
    spent = -float(row["Actual"])
    assert metrics["Total budgeted"] == f"${budgeted:,.2f}"
    assert metrics["Total spend"] == f"${spent:,.2f}"
    assert metrics["Difference"].replace("-", "") == f"${abs(budgeted - spent):,.2f}"

    # A bar click marks the same category, so the cards follow the same way.
    at.session_state["category_spend"] = {
        "selection": {"points": [{"x": 123.45, "y": "Groceries"}]}
    }
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    metrics = {m.label: m.value for m in at.metric}
    groceries = budget_table(at)
    groceries = groceries[groceries["Category"] == "Groceries"].iloc[0]
    assert metrics["Total budgeted"] == f"${float(groceries['Budget']):,.2f}"
    assert metrics["Total spend"] == f"${-float(groceries['Actual']):,.2f}"


def test_clicking_the_marked_row_again_clears_it() -> None:
    """The mark is a choice, not a trap: an empty selection takes the dashboard
    back to everything."""
    at = render()
    at.session_state["variance_table"] = {"selection": {"rows": [0]}}
    at.run()
    assert at.session_state["table_selection"] == "Mortgage"

    at.session_state["variance_table"] = {"selection": {"rows": []}}
    at.run()
    assert at.session_state["table_selection"] is None
    assert grid_trace(at)["y"] != ["Mortgage"]


def test_clicking_a_month_row_selects_that_month() -> None:
    at = render()
    at.session_state["month_overview"] = {"selection": {"rows": [0]}}
    at.run()
    assert at.session_state["month_filter"] == JANUARY


def test_the_see_transactions_button_opens_the_transactions_view() -> None:
    """The summary reports totals; the button is how the reader gets from a
    number to the transactions that made it, without hunting for the view."""
    at = render()
    press(at, "See transactions behind this")
    assert not at.exception, at.exception[0].message if at.exception else ""
    assert at.session_state["view"] == "Transactions"


def test_the_see_transactions_button_carries_the_mark_across() -> None:
    """A category marked on the summary has to survive the jump, or the button
    opens every transaction and the reader is back to hunting."""
    at = render()
    at.session_state["category_spend"] = {"selection": {"points": [{"y": "Tithing"}]}}
    at.run()
    press(at, "See the Tithing transactions behind this")
    assert at.session_state["view"] == "Transactions"
    assert at.session_state["table_selection"] == "Tithing"
    categories = set(transaction_table(at)["Category"])
    assert categories == {"Tithing"}


# --- filters persist across views ------------------------------------------


def test_a_month_selection_survives_a_view_change() -> None:
    """The reported requirement: a filter is maintained across views until it
    is cleared, rather than resetting when the view changes."""
    at = render()
    set_month(at, AUGUST)
    for view in ("Transactions", "Summary"):
        go_to(at, view)
        assert at.session_state["month_filter"] == AUGUST


def test_a_marked_category_survives_a_view_change() -> None:
    at = render()
    at.session_state["category_spend"] = {
        "selection": {"points": [{"y": "Tithing"}]}
    }
    at.run()
    for view in ("Transactions", "Summary"):
        go_to(at, view)
        assert at.session_state["table_selection"] == "Tithing"


def test_a_month_selection_narrows_the_transactions_table() -> None:
    at = render()
    set_month(at, AUGUST)
    go_to(at, "Transactions")
    assert len(transaction_table(at)) == 40


def test_a_category_filter_narrows_the_transactions_table() -> None:
    at = render()
    at.session_state["category_spend"] = {
        "selection": {"points": [{"y": "Tithing"}]}
    }
    at.run()
    go_to(at, "Transactions")
    assert set(transaction_table(at)["Category"]) == {"Tithing"}


def test_both_filters_apply_together() -> None:
    at = render()
    set_month(at, AUGUST)
    set_categories(at, ["Groceries"])
    go_to(at, "Transactions")
    table = transaction_table(at)
    assert set(table["Category"]) == {"Groceries"}
    assert table["Date"].map(lambda d: d.month).eq(8).all()


def test_a_filter_combination_with_no_rows_says_so() -> None:
    """The sample never files 'Auto' income, so the category with the Income
    kind is empty. The view must say so rather than rendering an empty table."""
    at = render()
    at.session_state["txn_kind"] = "Income"
    at.run()
    set_categories(at, ["Auto"])
    go_to(at, "Transactions")
    messages = [i.value for i in at.info]
    assert any("No transactions match" in m for m in messages)


# --- clear filters --------------------------------------------------------


def test_clear_filters_resets_the_month_and_the_categories() -> None:
    at = render()
    set_month(at, AUGUST)
    set_categories(at, ["Tithing"])
    assert at.session_state["month_filter"] == AUGUST

    press(at, "Clear filters")
    assert at.session_state["month_filter"] == ALL_MONTHS
    assert at.session_state["category_filter"] == []


def test_clear_filters_also_resets_the_local_transaction_filters() -> None:
    at = render()
    go_to(at, "Transactions")
    at.session_state["txn_kind"] = "Spend"
    at.session_state["txn_search"] = "ALDI"
    at.run()
    press(at, "Clear filters")
    at.run()
    assert at.session_state["txn_kind"] == "Everything"
    assert at.session_state["txn_search"] == ""


def test_clear_filters_restores_every_row() -> None:
    at = render()
    set_month(at, AUGUST)
    go_to(at, "Transactions")
    filtered = len(transaction_table(at))

    press(at, "Clear filters")
    at.run()
    assert len(transaction_table(at)) > filtered


# --- the transactions view ------------------------------------------------


def test_transactions_default_to_every_row() -> None:
    """Defaulting to spending only, or to one month, would break reconciliation
    against the uploaded file."""
    at = render()
    go_to(at, "Transactions")
    table = transaction_table(at)
    assert len(table) == 480
    assert (table["Amount"] < 0).any(), "spending must be present"
    assert (table["Amount"] > 0).any(), "income must be present"


def test_transactions_are_newest_first() -> None:
    at = render()
    go_to(at, "Transactions")
    dates = transaction_table(at)["Date"].tolist()
    assert dates == sorted(dates, reverse=True)


def test_the_transactions_view_names_its_months() -> None:
    at = render()
    set_month(at, SEPTEMBER)
    go_to(at, "Transactions")
    assert at.header[0].value == "Sep 2026"


def test_the_redundant_category_table_is_gone() -> None:
    """Spend by category duplicated the bar chart's numbers, so it was removed."""
    at = render()
    headings = [h.value for h in at.subheader]
    assert "Spend by category" not in headings


def test_the_transactions_table_reaches_down_the_page() -> None:
    """Requested: the table runs down to where the month section of the Summary
    page starts, instead of ending a couple of rows in. A month of transactions
    scrolls inside the table from there."""
    at = render()
    go_to(at, "Transactions")
    pane = find_key(at, "txn_pane")
    match = re.search(r"height_config\s*{\s*pixel_height:\s*(\d+)\s*}", str(pane.proto))
    assert match, f"the pane is still a stretch height: {pane.proto}"
    assert int(match.group(1)) >= 1000, (
        f"the pane reaches only {match.group(1)}px down the page"
    )


def test_the_transactions_page_ends_at_the_table() -> None:
    """Requested: no trend and no budget editor below the table here. The
    summary keeps both."""
    at = render()
    go_to(at, "Transactions")
    assert [s.value for s in at.subheader] == ["Transactions"]
    assert not at.get("plotly_chart"), "the trend was left on the transactions page"
    assert len(at.dataframe) == 1, "the budget editor followed the table"


def test_there_is_no_account_filter() -> None:
    """Requested: the account multiselect went, with everything else that sat
    below the table."""
    at = render()
    go_to(at, "Transactions")
    assert [m.key for m in at.multiselect] == ["month_picks", "category_filter"]


# --- savings buckets -----------------------------------------------------

import budget.reports as reports_module  # noqa: E402
from budget.presentation import parse_bucket_rows, unallocated_status  # noqa: E402
from budget.reports import load_buckets as read_bucket_file  # noqa: E402
from budget.reports import load_savings_pool, save_buckets  # noqa: E402

# The sample covers the whole of 2026: twelve months loaded.
TEST_BUCKETS = {
    "Auto": {"contribution": 100, "adjustment": 0, "goal": 0},
    "Vacation": {"contribution": 150, "adjustment": 0, "goal": 600},
}


@pytest.fixture
def buckets_file(tmp_path, monkeypatch) -> Path:
    """A private buckets file, so no test reads or writes the real one.

    The app imports BUCKETS_PATH from budget.reports on every run, so pointing
    the module attribute elsewhere redirects both reading and saving.
    """
    path = tmp_path / "buckets.yaml"
    save_buckets(TEST_BUCKETS, savings_pool=0, path=path)
    monkeypatch.setattr(reports_module, "BUCKETS_PATH", path)
    return path


def savings_page() -> AppTest:
    at = render()
    go_to(at, "Savings")
    assert not at.exception, [e.message for e in at.exception]
    return at


def metric_values(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def savings_table(at: AppTest) -> pd.DataFrame:
    return next(df.value for df in at.dataframe if "Spent" in df.value.columns)


def bucket_editor_frame(at: AppTest) -> pd.DataFrame:
    return next(
        df.value
        for df in at.dataframe
        if "Adjustment" in df.value.columns and "Spent" not in df.value.columns
    )


def sample_transactions() -> pd.DataFrame:
    frame, _ = ingest(Path(__file__).resolve().parents[1] / "sample_transactions.csv")
    return frame


def sample_spend(category: str) -> float:
    frame = sample_transactions()
    return -frame.loc[frame["display_label"] == category, "amount_cents"].sum() / 100


def format_money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}${abs(cents) / 100:,.2f}"


def test_the_savings_page_is_a_view(buckets_file) -> None:
    at = savings_page()
    assert at.header[0].value == "Savings"
    headings = [s.value for s in at.subheader]
    for heading in ["Left after spending and saving", "Cash in the bank", "Buckets", "Edit buckets"]:
        assert heading in headings


def test_the_savings_page_leaves_out_the_budget_sections(buckets_file) -> None:
    """Trends and the budget editor belong to the budget views."""
    headings = [s.value for s in savings_page().subheader]
    assert "Trend" not in headings
    assert "Budgets" not in headings


def test_the_balance_by_month_chart_is_gone(buckets_file) -> None:
    """Requested: the per-month balance chart was not wanted."""
    assert not savings_page().get("plotly_chart")


def test_the_bucket_table_columns(buckets_file) -> None:
    assert list(savings_table(savings_page()).columns) == [
        "Category",
        "Monthly contribution",
        "Contributed",
        "Adjustment",
        "Spent",
        "Balance",
        "Goal",
        "Progress",
        "Left to save",
    ]


def test_contributed_covers_every_month_loaded(buckets_file) -> None:
    """Twelve months loaded, so twelve contributions each, the month in
    progress included."""
    table = savings_table(savings_page()).set_index("Category")
    assert table.loc["Vacation", "Contributed"] == pytest.approx(12 * 150.0)
    assert table.loc["Auto", "Contributed"] == pytest.approx(12 * 100.0)


def test_spent_is_all_category_spending_loaded(buckets_file) -> None:
    """Reported: every bucket showed $0 spent. It is now all spending in the
    category across the data, and the balance takes it out."""
    table = savings_table(savings_page()).set_index("Category")
    auto_spend = sample_spend("Auto")
    assert auto_spend > 0
    assert table.loc["Auto", "Spent"] == pytest.approx(auto_spend)
    assert table.loc["Auto", "Balance"] == pytest.approx(12 * 100.0 - auto_spend)


def test_the_adjustment_is_added_to_the_balance(buckets_file) -> None:
    save_buckets(
        {**TEST_BUCKETS, "Vacation": {"contribution": 150, "adjustment": -50, "goal": 600}},
        path=buckets_file,
    )
    table = savings_table(savings_page()).set_index("Category")
    assert table.loc["Vacation", "Adjustment"] == pytest.approx(-50.0)
    assert table.loc["Vacation", "Balance"] == pytest.approx(
        12 * 150.0 - 50.0 - sample_spend("Vacation")
    )


def test_goal_progress_and_left_to_save(buckets_file) -> None:
    """Progress and left-to-save are worked out from the balance and the goal,
    with the share capped at a full bar. Auto has no goal, so its goal columns
    are blank rather than $0."""
    table = savings_table(savings_page()).set_index("Category")
    balance = float(table.loc["Vacation", "Balance"])
    assert table.loc["Vacation", "Goal"] == pytest.approx(600.0)
    assert table.loc["Vacation", "Progress"] == pytest.approx(
        min(100.0, 100.0 * balance / 600.0)
    )
    assert table.loc["Vacation", "Left to save"] == pytest.approx(
        max(0.0, 600.0 - balance)
    )
    assert pd.isna(table.loc["Auto", "Goal"])
    assert pd.isna(table.loc["Auto", "Progress"])
    assert pd.isna(table.loc["Auto", "Left to save"])


def test_progress_is_capped_at_a_full_bar(buckets_file) -> None:
    # Contribution set to nothing and a lump adjustment big enough to exceed
    # even a year's Vacation spending, so the share must stop at a full bar.
    save_buckets(
        {"Vacation": {"contribution": 0, "adjustment": 2000, "goal": 600}},
        path=buckets_file,
    )
    table = savings_table(savings_page()).set_index("Category")
    assert table.loc["Vacation", "Progress"] == pytest.approx(100.0)
    assert table.loc["Vacation", "Left to save"] == pytest.approx(0.0)


def test_an_overdrawn_bucket_balance_is_red(buckets_file) -> None:
    save_buckets({"Auto": {"contribution": 10, "adjustment": 0, "goal": 0}}, path=buckets_file)
    at = savings_page()
    assert savings_table(at).set_index("Category").loc["Auto", "Balance"] < 0
    styled = [str(df.proto) for df in at.dataframe if "Spent" in df.value.columns]
    assert OVER_FILL in styled[0]


def test_left_over_is_income_minus_spend_minus_savings(buckets_file) -> None:
    at = savings_page()
    set_month(at, SEPTEMBER)
    go_to(at, "Savings")
    values = metric_values(at)
    frame = sample_transactions()
    september = frame[frame["date"].dt.to_period("M").astype(str) == SEPTEMBER]
    income = september.loc[september["amount_cents"] > 0, "amount_cents"].sum()
    spend = -september.loc[september["amount_cents"] < 0, "amount_cents"].sum()
    # The month in progress still sets aside its savings: $100 + $150.
    assert values["Set aside for savings"] == "$250.00"
    assert values["Left over"] == format_money(income - spend - 25_000)


def test_the_cash_check_compares_the_pool_with_the_buckets(buckets_file) -> None:
    at = savings_page()
    in_buckets = float(savings_table(at)["Balance"].sum())
    at.text_input(key="savings_pool_input").set_value(f"{in_buckets + 500:.2f}").run()
    values = metric_values(at)
    assert values["In buckets"] == f"${in_buckets:,.2f}"
    assert values["Unallocated"] == "$500.00"


def test_saving_the_cash_total_writes_the_file(buckets_file) -> None:
    at = savings_page()
    at.text_input(key="savings_pool_input").set_value("4200").run()
    press(at, "Save cash total")
    assert load_savings_pool(buckets_file) == 4200
    # Saving the cash must not disturb the buckets.
    assert read_bucket_file(buckets_file) == TEST_BUCKETS


def test_a_bad_cash_total_is_refused(buckets_file) -> None:
    """A total that is not a number is shown as an error and never written."""
    before = load_savings_pool(buckets_file)
    at = savings_page()
    at.text_input(key="savings_pool_input").set_value("eleventy").run()
    press(at, "Save cash total")
    assert at.error, "an invalid cash total produced no error"
    assert load_savings_pool(buckets_file) == before, "the bad total was saved"


def test_the_cash_input_shows_dollars(buckets_file) -> None:
    """Requested: the input reads like the amounts around it, $ and cents. The
    field is text rather than a number, because the browser re-renders a numeric
    input's value in the machine's own locale, comma and all; a text value of
    '$1,500.00' is shown exactly as the app put it there."""
    at = savings_page()
    field = at.text_input(key="savings_pool_input")
    assert field.value.startswith("$")
    assert field.value.endswith(".00")


def test_the_cash_input_accepts_a_typed_dollar_figure(buckets_file) -> None:
    """Prefixed and comma-separated entries, the way money is usually written
    or pasted, are read as the same total as a plain number."""
    at = savings_page()
    at.text_input(key="savings_pool_input").set_value("$137,000.00").run()
    press(at, "Save cash total")
    assert load_savings_pool(buckets_file) == 137000
    assert any(
        "Saved cash total of $137,000.00." in s.value for s in at.success
    ), "saving the cash total showed no confirmation"


def test_the_editor_columns(buckets_file) -> None:
    assert list(bucket_editor_frame(savings_page()).columns) == [
        "Category",
        "Monthly contribution",
        "Adjustment",
        "Goal",
    ]


def test_the_editor_shows_the_saved_adjustment_not_the_balance(buckets_file) -> None:
    """The adjustment is an input; the balance is worked out from it."""
    editor = bucket_editor_frame(savings_page()).set_index("Category")
    assert list(editor["Adjustment"]) == [0.0, 0.0]


def test_saving_the_editor_unchanged_writes_the_same_buckets(buckets_file) -> None:
    at = savings_page()
    press(at, "Save buckets")
    assert not at.exception
    assert read_bucket_file(buckets_file) == TEST_BUCKETS


def test_changing_a_contribution_changes_contributed_for_every_month(buckets_file) -> None:
    at = savings_page()
    save_buckets(
        {**TEST_BUCKETS, "Vacation": {"contribution": 0, "adjustment": 0, "goal": 600}},
        path=buckets_file,
    )
    at.run()
    assert savings_table(at).set_index("Category").loc["Vacation", "Contributed"] == 0.0


def test_editor_rows_become_buckets_keyed_by_category() -> None:
    buckets, problems = parse_bucket_rows(
        [
            {"Category": "Vacation", "Monthly contribution": 150.0, "Adjustment": -40.0, "Goal": 2_000.0},
            {"Category": "Gifts", "Monthly contribution": 25.0, "Adjustment": None, "Goal": None},
        ]
    )
    assert problems == []
    assert buckets == {
        "Vacation": {"contribution": 150, "adjustment": -40, "goal": 2_000},
        "Gifts": {"contribution": 25, "adjustment": 0, "goal": 0},
    }


def test_a_category_listed_twice_is_refused() -> None:
    """Its spending would come out of both buckets and be counted twice."""
    _, problems = parse_bucket_rows([{"Category": "Vacation"}, {"Category": "Vacation"}])
    assert problems and "listed twice" in problems[0]


@pytest.mark.parametrize("field", ["Monthly contribution", "Goal"])
def test_negative_contributions_and_goals_are_refused(field) -> None:
    buckets, problems = parse_bucket_rows([{"Category": "Auto", field: -5}])
    assert not buckets and "negative" in problems[0]


def test_a_negative_adjustment_is_allowed() -> None:
    buckets, problems = parse_bucket_rows([{"Category": "Auto", "Adjustment": -5}])
    assert problems == [] and buckets["Auto"]["adjustment"] == -5


def test_a_row_with_amounts_but_no_category_is_refused() -> None:
    _, problems = parse_bucket_rows([{"Category": None, "Goal": 50}])
    assert problems and "needs a category" in problems[0]


def test_blank_editor_rows_are_ignored() -> None:
    buckets, problems = parse_bucket_rows(
        [{"Category": None, "Monthly contribution": None, "Adjustment": None, "Goal": None}]
    )
    assert buckets == {} and problems == []


def test_unallocated_status_wording() -> None:
    assert unallocated_status(0) == "Every dollar assigned"
    assert "Not yet" in unallocated_status(1)
    assert "Over-allocated" in unallocated_status(-1)


# --- number formatting ---------------------------------------------------


def test_every_money_column_uses_thousands_separators() -> None:
    """Requested: numbers formatted with commas across the report. Streamlit's
    'dollar' format renders $1,234.56; a printf '$%.2f' renders $1234.56."""
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'format="$%' not in source
    assert source.count('format="dollar"') >= 10


# --- a server left running across an edit --------------------------------


@pytest.fixture
def restore_budget_modules():
    """Put the budget modules back exactly as they were after a test reloads them.

    A reload rebinds classes such as AmountParseError, and tests elsewhere hold
    the originals, so the reloaded copies must not leak past this test.
    """
    import sys

    names = ["budget.schema", "budget.ingest", "budget.reports", "budget.presentation"]
    snapshots = {name: dict(sys.modules[name].__dict__) for name in names}
    yield
    for name, snapshot in snapshots.items():
        module = sys.modules[name]
        module.__dict__.clear()
        module.__dict__.update(snapshot)


def test_a_stale_reports_module_is_reloaded(restore_budget_modules) -> None:
    """Reported: KeyError 'goal_cents'. A server started before goals existed
    kept the old reports module, and the new page asked it for a goal column.
    The app now reloads any budget module whose file is newer than its copy."""

    def old_savings_summary(transactions, buckets):
        return pd.DataFrame(columns=["category", "monthly_cents", "total_cents", "spent_cents", "balance_cents"])

    reports_module.savings_summary = old_savings_summary
    reports_module.LOADED_MTIME = 0.0  # loaded long before the file's last edit

    at = render()
    go_to(at, "Savings")
    assert not at.exception, [e.message for e in at.exception]
    assert reports_module.savings_summary is not old_savings_summary
    assert "Goal" in savings_table(at).columns


def test_an_unchanged_module_is_not_reloaded(buckets_file, restore_budget_modules) -> None:
    """Reloading every run would throw away state for nothing, and would undo
    the test redirection of the buckets file back to the real one."""
    render()
    assert reports_module.BUCKETS_PATH == buckets_file
