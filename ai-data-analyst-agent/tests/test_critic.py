from dataclasses import replace

import pandas as pd
import pytest

from analyst.critic import review_execution
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


def run_and_review(data, **plan_kwargs):
    plan = AnalysisPlan.model_validate(plan_kwargs)
    result = execute_plan(plan, data)
    return plan, result, review_execution(plan, result, data)


VALID_PLANS = [
    {"operation": "sort", "sort_by": "age", "ascending": False, "limit": 2},
    {
        "operation": "aggregate",
        "group_by": ["city"],
        "aggregations": [{"function": "mean", "column": "salary"}, {"function": "row_count"}],
        "sort_by": "salary_mean",
    },
    {"operation": "filter", "filters": [{"column": "age", "operator": ">", "value": 30}]},
    {"operation": "missing_values"},
    {"operation": "column_summary"},
    {"operation": "correlation"},
]


@pytest.mark.parametrize("plan_kwargs", VALID_PLANS)
def test_genuine_results_pass_all_checks(df, plan_kwargs):
    _, _, report = run_and_review(df, **plan_kwargs)
    assert report.passed is True
    assert report.issues == ()
    assert report.checks_run > 0


def test_operation_mismatch_is_caught(df):
    plan, result, _ = run_and_review(df, operation="sort", sort_by="age")
    report = review_execution(plan, replace(result, operation="filter"), df)
    assert report.passed is False
    assert any("different operation" in i for i in report.issues)


def test_wrong_sort_order_is_caught(df):
    plan, result, _ = run_and_review(df, operation="sort", sort_by="age", ascending=False)
    flipped = result.table.iloc[::-1].reset_index(drop=True)
    report = review_execution(plan, replace(result, table=flipped), df)
    assert any("not sorted" in i for i in report.issues)


def test_wrong_aggregate_columns_are_caught(df):
    plan, result, _ = run_and_review(
        df,
        operation="aggregate",
        group_by=["city"],
        aggregations=[{"function": "mean", "column": "salary"}],
    )
    renamed = result.table.rename(columns={"salary_mean": "x"})
    report = review_execution(plan, replace(result, table=renamed), df)
    assert any("columns" in i for i in report.issues)


def test_row_counts_that_do_not_add_up_are_caught(df):
    plan, result, _ = run_and_review(
        df,
        operation="aggregate",
        group_by=["city"],
        aggregations=[{"function": "row_count"}],
    )
    report = review_execution(plan, replace(result, table=result.table.assign(row_count=999)), df)
    assert any("row counts" in i for i in report.issues)


def test_violated_filter_is_caught(df):
    plan, result, _ = run_and_review(
        df,
        operation="filter",
        filters=[{"column": "age", "operator": ">", "value": 30}],
    )
    report = review_execution(plan, replace(result, table=df), df)
    assert any("violated" in i for i in report.issues)


def test_inconsistent_truncated_flag_is_caught(df):
    plan, result, _ = run_and_review(df, operation="sort", sort_by="age")
    report = review_execution(plan, replace(result, truncated=True), df)
    assert any("truncated" in i for i in report.issues)


def test_empty_filter_result_is_a_warning_not_a_failure(df):
    _, _, report = run_and_review(
        df,
        operation="filter",
        filters=[{"column": "age", "operator": ">", "value": 100}],
    )
    assert report.passed is True
    assert len(report.warnings) == 1
    assert "empty" in report.warnings[0]


def test_empty_result_for_structural_operation_fails(df):
    plan, result, _ = run_and_review(df, operation="missing_values")
    report = review_execution(plan, replace(result, table=result.table.iloc[0:0]), df)
    assert report.passed is False
    assert any("no rows" in i for i in report.issues)


def test_correlation_value_out_of_range_is_caught(df):
    plan, result, _ = run_and_review(df, operation="correlation")
    bad = result.table.copy()
    bad.iloc[0, 1] = 5.0
    report = review_execution(plan, replace(result, table=bad), df)
    assert any("range" in i for i in report.issues)


def test_result_larger_than_limit_is_caught(df):
    plan, result, _ = run_and_review(df, operation="sort", sort_by="age", limit=2)
    report = review_execution(plan, replace(result, table=df.reset_index(drop=True)), df)
    assert any("limit" in i for i in report.issues)


def test_result_for_different_data_is_caught(df):
    plan, result, _ = run_and_review(df, operation="sort", sort_by="age")
    report = review_execution(plan, result, df.head(2))
    assert any("input rows" in i for i in report.issues)


def test_rows_removed_without_filters_is_caught(df):
    plan, result, _ = run_and_review(df, operation="sort", sort_by="age")
    report = review_execution(plan, replace(result, rows_after_filters=1), df)
    assert any("No filters" in i for i in report.issues)


def test_impossible_missing_count_is_caught(df):
    plan, result, _ = run_and_review(df, operation="missing_values")
    tampered = result.table.assign(missing_count=999)
    report = review_execution(plan, replace(result, table=tampered), df)
    assert any("more missing values" in i for i in report.issues)


def test_ungrouped_aggregate_on_zero_matching_rows_warns(df):
    _, _, report = run_and_review(
        df,
        operation="aggregate",
        filters=[{"column": "city", "operator": "==", "value": "Nowhere"}],
        aggregations=[{"function": "sum", "column": "salary"}],
    )
    assert report.passed
    assert any("matched 0 rows" in w for w in report.warnings)