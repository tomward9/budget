"""Tests for the chart layer.

The failure this guards against: passing pre-formatted currency strings to
plotly. Plotly infers the axis type from the data, so a column of
'-$1,466.14' makes the axis categorical and the scale disappears. The charts
themselves render, so nothing crashes and the problem is only visible by
looking at the output.
"""

from __future__ import annotations

import plotly.express as px

from budget.ingest import ingest
from budget.reports import (
    category_breakdown,
    monthly_summary,
    variance_table,
)
from budget.schema import format_cents


def build_line_like_trend_chart() -> tuple:
    """Reproduce the trend chart's data path with a numeric axis."""
    summary = monthly_summary(
        ingest_frame(
            "05/08/2026,SPEND,INST,MAX CHECKING,Groceries_,No,No,($100)",
            "06/08/2026,EARN,INST,MAX CHECKING,Paycheck/Salary,No,No,$500",
            "05/09/2026,SPEND,INST,MAX CHECKING,Groceries_,No,No,($200)",
        )
    )
    melted = summary.melt(
        id_vars=["period"],
        value_vars=["income_cents", "spend_cents"],
        var_name="kind",
        value_name="cents",
    )
    melted["kind"] = melted["kind"].map({"income_cents": "Income", "spend_cents": "Spend"})
    melted["dollars"] = melted["cents"].abs() / 100
    return px.line(melted, x="period", y="dollars", color="kind", markers=True)


def ingest_frame(*rows: str):
    import io

    header = "Date,Description,Institution,Account,Category,Is Hidden,Is Pending,Amount"
    text = "\n".join([header, *rows, ""])
    frame, _ = ingest(io.BytesIO(text.encode("utf-8")))
    return frame


def test_formatted_strings_would_break_the_axis() -> None:
    """Documents the bug: strings make the axis categorical."""
    summary = monthly_summary(ingest_frame("05/08/2026,SPEND,INST,C,Groceries_,No,No,($100)"))
    melted = summary.melt(
        id_vars=["period"],
        value_vars=["income_cents", "spend_cents"],
        value_name="cents",
    )
    melted["Amount"] = melted["cents"].map(lambda c: format_cents(c))
    figure = px.line(melted, x="period", y="Amount")
    assert figure.layout.yaxis.type != "linear"


def test_numeric_axis_is_not_categorical() -> None:
    """A linear axis is either declared or left to auto-detection; the bug
    this guards against is plotly inferring 'category' from string data."""
    figure = build_line_like_trend_chart()
    assert figure.layout.yaxis.type != "category"
    for trace in figure.data:
        assert all(isinstance(value, (int, float)) for value in trace.y)


def test_trend_chart_y_values_are_numeric() -> None:
    figure = build_line_like_trend_chart()
    for trace in figure.data:
        for value in trace.y:
            assert isinstance(value, (int, float))
            assert not isinstance(value, str)


def test_trend_chart_plots_spend_as_a_real_series() -> None:
    """Spend must appear as a named trace with values, not vanish."""
    figure = build_line_like_trend_chart()
    names = {trace.name for trace in figure.data}
    assert names == {"Income", "Spend"}
    spend = next(trace for trace in figure.data if trace.name == "Spend")
    assert len(spend.y) == 2
    assert all(value > 0 for value in spend.y)


def test_category_chart_has_one_bar_per_category() -> None:
    frame = ingest_frame(
        "05/08/2026,ALDI,INST,MAX CHECKING,Groceries_,No,No,($50)",
        "06/08/2026,BP,INST,MAX CHECKING,Gasoline,No,No,($40)",
        "07/08/2026,CHURCH,INST,MAX CHECKING,Tithing,No,No,($300)",
    )
    breakdown = category_breakdown(frame, month="2026-08")
    plotted = breakdown.assign(dollars=breakdown["spend_cents"].abs() / 100)
    figure = px.bar(plotted, x="dollars", y="display_label", orientation="h")
    assert len(figure.data[0].y) == 3
    assert all(isinstance(v, (int, float)) for v in figure.data[0].x)


def test_budget_chart_only_plots_budgeted_categories() -> None:
    frame = ingest_frame("05/08/2026,ALDI,INST,MAX CHECKING,Groceries_,No,No,($50)")
    table = variance_table(frame, month="2026-08", budgets={"Groceries": 400})
    budgeted = table[table["budget_cents"] > 0]
    assert list(budgeted["display_label"]) == ["Groceries"]


def test_no_budgets_means_nothing_to_compare() -> None:
    frame = ingest_frame("05/08/2026,ALDI,INST,MAX CHECKING,Groceries_,No,No,($50)")
    table = variance_table(frame, month="2026-08", budgets={})
    assert table[table["budget_cents"] > 0].empty
