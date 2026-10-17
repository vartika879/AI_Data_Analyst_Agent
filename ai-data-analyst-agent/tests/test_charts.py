import pandas as pd
import pytest

from analyst.charts import MAX_CHART_POINTS, build_chart
from analyst.executor import execute_plan
from analyst.schemas import AnalysisPlan


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Asha", "Ravi", "Charu", "Dev"],
            "age": [30, 25, 40, 35],
            "city": ["X", "Y", "X", None],
            "salary": [10.0, 20.0, 30.0, None],
        }
    )


def chart_for(data, **plan_kwargs):
    plan = AnalysisPlan.model_validate(plan_kwargs)
    return build_chart(plan, execute_plan(plan, data))


def mean_salary_by_city(**extra):
    return {
        "operation": "aggregate",
        "group_by": ["city"],
        "aggregations": [{"function": "mean", "column": "salary"}],
        **extra,
    }


def test_bar_chart_for_grouped_aggregate(df):
    out = chart_for(df, **mean_salary_by_city(chart_type="bar"))
    trace = out.figure.data[0]
    assert trace.type == "bar"
    assert list(trace.x) == ["X", "Y", "(missing)"]  # missing group gets a label
    assert list(trace.y) == [20.0, 20.0, None]  # NaN becomes a gap, not a number
    assert out.note is None


def test_line_chart(df):
    out = chart_for(df, **mean_salary_by_city(chart_type="line"))
    trace = out.figure.data[0]
    assert trace.type == "scatter"
    assert "lines" in trace.mode


def test_scatter_chart_with_two_aggregations(df):
    out = chart_for(
        df,
        operation="aggregate",
        group_by=["city"],
        aggregations=[
            {"function": "mean", "column": "salary"},
            {"function": "mean", "column": "age"},
        ],
        chart_type="scatter",
    )
    trace = out.figure.data[0]
    assert trace.mode == "markers"
    assert len(trace.x) == 3


def test_scatter_with_one_aggregation_is_not_available(df):
    out = chart_for(df, **mean_salary_by_city(chart_type="scatter"))
    assert out.figure is None
    assert "two numeric" in out.note


def test_no_chart_requested_returns_nothing(df):
    out = chart_for(df, **mean_salary_by_city())
    assert out.figure is None
    assert out.note is None


def test_ungrouped_aggregate_has_no_chart(df):
    out = chart_for(
        df,
        operation="aggregate",
        aggregations=[{"function": "sum", "column": "salary"}],
        chart_type="bar",
    )
    assert out.figure is None
    assert "not available" in out.note


def test_filter_operation_has_no_chart(df):
    out = chart_for(
        df,
        operation="filter",
        filters=[{"column": "age", "operator": ">", "value": 30}],
        chart_type="bar",
    )
    assert out.figure is None
    assert "not available" in out.note


def test_missing_values_bar_chart(df):
    out = chart_for(df, operation="missing_values", chart_type="bar")
    trace = out.figure.data[0]
    assert trace.type == "bar"
    assert list(trace.x)[:2] == ["city", "salary"]
    assert list(trace.y) == [1, 1, 0, 0]


def test_missing_values_line_chart_is_not_available(df):
    out = chart_for(df, operation="missing_values", chart_type="line")
    assert out.figure is None
    assert "bar chart" in out.note


def test_text_valued_aggregation_cannot_be_plotted(df):
    out = chart_for(
        df,
        operation="aggregate",
        group_by=["city"],
        aggregations=[{"function": "min", "column": "name"}],
        chart_type="bar",
    )
    assert out.figure is None
    assert "numeric" in out.note


def test_html_in_labels_is_escaped():
    data = pd.DataFrame({"city": ["<b>x</b>", "Y"]})
    out = chart_for(
        data,
        operation="aggregate",
        group_by=["city"],
        aggregations=[{"function": "row_count"}],
        chart_type="bar",
    )
    labels = list(out.figure.data[0].x)
    assert "&lt;b&gt;x&lt;/b&gt;" in labels
    assert all("<b>" not in label for label in labels)


def test_chart_is_capped_and_says_so():
    data = pd.DataFrame({"id": [f"u{i}" for i in range(60)]})
    out = chart_for(
        data,
        operation="aggregate",
        group_by=["id"],
        aggregations=[{"function": "row_count"}],
        chart_type="bar",
    )
    assert len(out.figure.data[0].x) == MAX_CHART_POINTS
    assert "first 50 of 60" in out.note


def test_empty_grouped_result_has_no_chart(df):
    out = chart_for(
        df,
        operation="aggregate",
        filters=[{"column": "age", "operator": ">", "value": 100}],
        group_by=["city"],
        aggregations=[{"function": "row_count"}],
        chart_type="bar",
    )
    assert out.figure is None
    assert "empty" in out.note


def test_multiple_group_columns_are_joined_in_labels(df):
    out = chart_for(
        df,
        operation="aggregate",
        group_by=["city", "name"],
        aggregations=[{"function": "row_count"}],
        chart_type="bar",
    )
    assert "X / Asha" in list(out.figure.data[0].x)