"""Charts: build a Plotly figure from an ExecutionResult.

Charts are optional and only built for result shapes where a chart makes sense
(grouped aggregations and missing-value results). The figure is built only from
the computed table. No LLM text is turned into code.
"""

import html
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from analyst.executor import ExecutionResult
from analyst.schemas import AnalysisPlan

MAX_CHART_POINTS = 50


@dataclass(frozen=True, eq=False)
class ChartOutcome:
    figure: go.Figure | None = None
    note: str | None = None  # why there is no chart, or a caveat about the chart


def _label(value) -> str:
    """Text for an axis label. HTML-escaped because Plotly renders some HTML tags."""
    if pd.isna(value):
        return "(missing)"
    return html.escape(str(value), quote=False)


def _safe(text: str) -> str:
    return html.escape(str(text), quote=False)


def _numbers(series: pd.Series) -> list[float | None]:
    return [None if pd.isna(v) else float(v) for v in series]


def _plottable_number(series: pd.Series) -> bool:
    return is_numeric_dtype(series) and not is_bool_dtype(series)


def _cap_note(shown: int, total: int, what: str) -> str | None:
    if total > shown:
        return f"The chart shows the first {shown} of {total} {what}."
    return None


def _grouped_chart(plan: AnalysisPlan, result: ExecutionResult) -> ChartOutcome:
    table = result.table
    if len(table) == 0:
        return ChartOutcome(note="Chart not available: the result is empty.")

    numeric_cols = [
        a.output_name
        for a in plan.aggregations
        if a.output_name in table.columns and _plottable_number(table[a.output_name])
    ]
    if not numeric_cols:
        return ChartOutcome(note="Chart not available: no numeric value to plot.")

    shown = table.head(MAX_CHART_POINTS)
    group_cols = list(plan.group_by)
    labels = [
        " / ".join(_label(v) for v in values)
        for values in shown[group_cols].itertuples(index=False, name=None)
    ]
    note = _cap_note(len(shown), result.total_result_rows, "groups")
    group_title = _safe(" / ".join(group_cols))

    if plan.chart_type == "scatter":
        if len(numeric_cols) < 2:
            return ChartOutcome(
                note="Chart not available: a scatter chart needs two numeric aggregations."
            )
        x_name, y_name = numeric_cols[0], numeric_cols[1]
        fig = go.Figure(
            go.Scatter(
                x=_numbers(shown[x_name]),
                y=_numbers(shown[y_name]),
                text=labels,
                mode="markers",
            )
        )
        fig.update_layout(
            title=f"{_safe(y_name)} vs {_safe(x_name)}",
            xaxis_title=_safe(x_name),
            yaxis_title=_safe(y_name),
        )
        return ChartOutcome(figure=fig, note=note)

    y_name = numeric_cols[0]
    values = _numbers(shown[y_name])
    if plan.chart_type == "line":
        trace = go.Scatter(x=labels, y=values, mode="lines+markers")
    else:  # "bar"
        trace = go.Bar(x=labels, y=values)
    fig = go.Figure(trace)
    fig.update_xaxes(type="category")  # keep labels like "2021" as text, not numbers
    fig.update_layout(
        title=f"{_safe(y_name)} by {group_title}",
        xaxis_title=group_title,
        yaxis_title=_safe(y_name),
    )
    return ChartOutcome(figure=fig, note=note)


def _missing_chart(plan: AnalysisPlan, result: ExecutionResult) -> ChartOutcome:
    if plan.chart_type != "bar":
        return ChartOutcome(note="Chart not available: missing values can only be shown as a bar chart.")
    table = result.table
    if len(table) == 0:
        return ChartOutcome(note="Chart not available: the result is empty.")

    shown = table.head(MAX_CHART_POINTS)
    fig = go.Figure(
        go.Bar(
            x=[_label(c) for c in shown["column"]],
            y=_numbers(shown["missing_count"]),
        )
    )
    fig.update_xaxes(type="category")
    fig.update_layout(
        title="Missing values per column",
        xaxis_title="column",
        yaxis_title="missing count",
    )
    return ChartOutcome(figure=fig, note=_cap_note(len(shown), result.total_result_rows, "columns"))


def build_chart(plan: AnalysisPlan, result: ExecutionResult) -> ChartOutcome:
    """Return a figure if a chart was requested AND makes sense for this result."""
    if plan.chart_type is None:
        return ChartOutcome()  # no chart requested: nothing to show, nothing to explain
    if plan.operation == "aggregate" and plan.group_by:
        return _grouped_chart(plan, result)
    if plan.operation == "missing_values":
        return _missing_chart(plan, result)
    return ChartOutcome(
        note="Chart not available: charts are supported for grouped aggregations "
        "and missing-value results."
    )