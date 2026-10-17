from dataclasses import replace

import pandas as pd
import pytest

from analyst.critic import CriticReport, review_execution
from analyst.executor import execute_plan
from analyst.reporter import _fmt, build_report
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


def report_for(data, **plan_kwargs) -> str:
    plan = AnalysisPlan.model_validate(plan_kwargs)
    result = execute_plan(plan, data)
    critic = review_execution(plan, result, data)
    return build_report(plan, result, critic)


def test_sort_report_mentions_order_and_top_row(df):
    text = report_for(df, operation="sort", sort_by="age", ascending=False, limit=2)
    assert "ordered by age (descending)" in text
    assert "Showing 2 of 4" in text
    assert "name = Charu" in text
    assert "age = 40" in text


def test_ungrouped_aggregate_reports_computed_values(df):
    text = report_for(
        df,
        operation="aggregate",
        aggregations=[{"function": "sum", "column": "salary"}, {"function": "row_count"}],
    )
    assert "salary_sum = 60.00" in text
    assert "row_count = 4" in text


def test_grouped_aggregate_lists_groups(df):
    text = report_for(
        df,
        operation="aggregate",
        group_by=["city"],
        aggregations=[{"function": "mean", "column": "salary"}],
        sort_by="salary_mean",
        ascending=False,
    )
    assert "3 group(s) by city" in text
    assert "city = X, salary_mean = 20.00" in text


def test_filter_report_counts_matching_rows(df):
    text = report_for(
        df, operation="filter", filters=[{"column": "city", "operator": "==", "value": "X"}]
    )
    assert "2 row(s) matched" in text
    assert "Analysed 2 of 4 rows" in text


def test_filter_with_no_match_says_so(df):
    text = report_for(
        df, operation="filter", filters=[{"column": "age", "operator": ">", "value": 100}]
    )
    assert "No rows matched the filter" in text
    assert "result is empty" in text  # the Critic's warning is passed along


def test_missing_values_report_lists_worst_column(df):
    text = report_for(df, operation="missing_values")
    assert "2 of 4 checked column(s) have missing values" in text
    assert "city: 1 missing (25.00%)" in text


def test_no_missing_values_report():
    data = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert "No missing values" in report_for(data, operation="missing_values")


def test_correlation_report_names_strongest_pair():
    data = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 9], "z": [4, 1, 3, 2]})
    text = report_for(data, operation="correlation")
    assert "x and y" in text
    assert "cause and effect" in text


def test_truncation_is_mentioned():
    data = pd.DataFrame({"n": list(range(150))})
    text = report_for(
        data, operation="filter", filters=[{"column": "n", "operator": ">=", "value": 0}]
    )
    assert "first 100 of 150" in text


def test_report_does_not_claim_guaranteed_correctness(df):
    text = report_for(df, operation="column_summary")
    assert "consistency checks run" in text
    assert "do not guarantee" in text


def test_executor_notes_are_included(df):
    text = report_for(
        df,
        operation="aggregate",
        columns=["name"],
        aggregations=[{"function": "row_count"}],
    )
    assert "ignored" in text


def test_failed_critic_report_is_refused(df):
    plan = AnalysisPlan.model_validate({"operation": "sort", "sort_by": "age"})
    result = execute_plan(plan, df)
    failed = CriticReport(passed=False, issues=("something is wrong",), checks_run=3)
    with pytest.raises(ValueError, match="failed"):
        build_report(plan, result, failed)


def test_markdown_in_cell_values_is_neutralised():
    evil = "![x](http://evil.test/a.png)"
    data = pd.DataFrame({"name": [evil], "age": [1]})
    text = report_for(data, operation="sort", sort_by="age")
    assert "![" not in text
    assert "](" not in text


def test_number_formatting():
    assert _fmt(1234567) == "1,234,567"
    assert _fmt(20.0) == "20.00"
    assert _fmt(0.00012) == "0.00012"
    assert _fmt(float("nan")) == "missing"
    assert _fmt(True) == "true"