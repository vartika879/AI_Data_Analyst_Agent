import pandas as pd
import pytest

from analyst.plan_validation import PlanValidationError, validate_plan_against_dataframe
from analyst.schemas import AnalysisPlan


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "age": [30, 25, 40],
            "city": ["X", "Y", "X"],
            "salary": [10.0, 20.0, 30.0],
        }
    )


def make_plan(**kwargs) -> AnalysisPlan:
    return AnalysisPlan.model_validate(kwargs)


def test_valid_aggregate_plan_passes(df):
    plan = make_plan(
        operation="aggregate",
        group_by=["city"],
        aggregations=[{"function": "mean", "column": "salary"}],
    )
    validate_plan_against_dataframe(plan, df)  # should not raise


def test_unknown_filter_column(df):
    plan = make_plan(
        operation="filter",
        filters=[{"column": "bonus", "operator": "==", "value": 1}],
    )
    with pytest.raises(PlanValidationError, match="Unknown column"):
        validate_plan_against_dataframe(plan, df)


def test_unknown_group_by_column(df):
    plan = make_plan(
        operation="aggregate",
        group_by=["country"],
        aggregations=[{"function": "row_count"}],
    )
    with pytest.raises(PlanValidationError, match="Unknown column"):
        validate_plan_against_dataframe(plan, df)


def test_unknown_aggregation_column(df):
    plan = make_plan(
        operation="aggregate",
        aggregations=[{"function": "sum", "column": "bonus"}],
    )
    with pytest.raises(PlanValidationError, match="Unknown column"):
        validate_plan_against_dataframe(plan, df)


def test_sum_on_text_column_rejected(df):
    plan = make_plan(
        operation="aggregate",
        aggregations=[{"function": "sum", "column": "name"}],
    )
    with pytest.raises(PlanValidationError, match="numeric"):
        validate_plan_against_dataframe(plan, df)


def test_count_on_text_column_allowed(df):
    plan = make_plan(
        operation="aggregate",
        aggregations=[{"function": "count", "column": "name"}],
    )
    validate_plan_against_dataframe(plan, df)


def test_ordering_filter_on_text_column_rejected(df):
    plan = make_plan(
        operation="filter",
        filters=[{"column": "city", "operator": ">", "value": 5}],
    )
    with pytest.raises(PlanValidationError, match="numeric column"):
        validate_plan_against_dataframe(plan, df)


def test_contains_on_numeric_column_rejected(df):
    plan = make_plan(
        operation="filter",
        filters=[{"column": "age", "operator": "contains", "value": "3"}],
    )
    with pytest.raises(PlanValidationError, match="text column"):
        validate_plan_against_dataframe(plan, df)


def test_correlation_with_text_column_rejected(df):
    plan = make_plan(operation="correlation", columns=["age", "name"])
    with pytest.raises(PlanValidationError, match="numeric"):
        validate_plan_against_dataframe(plan, df)


def test_correlation_default_needs_two_numeric_columns(df):
    plan = make_plan(operation="correlation")
    with pytest.raises(PlanValidationError, match="at least two numeric"):
        validate_plan_against_dataframe(plan, df[["name", "age"]])


def test_correlation_default_with_numeric_columns_passes(df):
    plan = make_plan(operation="correlation")
    validate_plan_against_dataframe(plan, df)


def test_aggregate_sort_by_must_be_group_or_output(df):
    plan = make_plan(
        operation="aggregate",
        group_by=["city"],
        aggregations=[{"function": "mean", "column": "salary"}],
        sort_by="age",
    )
    with pytest.raises(PlanValidationError, match="sort_by"):
        validate_plan_against_dataframe(plan, df)


def test_aggregate_sort_by_output_name_passes(df):
    plan = make_plan(
        operation="aggregate",
        group_by=["city"],
        aggregations=[{"function": "mean", "column": "salary"}],
        sort_by="salary_mean",
    )
    validate_plan_against_dataframe(plan, df)


def test_sort_by_unknown_column(df):
    plan = make_plan(operation="sort", sort_by="height")
    with pytest.raises(PlanValidationError, match="Unknown column"):
        validate_plan_against_dataframe(plan, df)


def test_sort_by_rejected_for_missing_values(df):
    plan = make_plan(operation="missing_values", sort_by="name")
    with pytest.raises(PlanValidationError, match="sort_by"):
        validate_plan_against_dataframe(plan, df)


def test_unsupported_plan_raises(df):
    plan = make_plan(operation="unsupported", unsupported_reason="Needs forecasting.")
    with pytest.raises(PlanValidationError, match="Needs forecasting"):
        validate_plan_against_dataframe(plan, df)


def test_error_message_lists_available_columns(df):
    plan = make_plan(operation="column_summary", columns=["bonus"])
    with pytest.raises(PlanValidationError) as exc_info:
        validate_plan_against_dataframe(plan, df)
    assert "name, age, city, salary" in str(exc_info.value)


def test_missing_values_unknown_column(df):
    plan = make_plan(operation="missing_values", columns=["bonus"])
    with pytest.raises(PlanValidationError, match="Unknown column"):
        validate_plan_against_dataframe(plan, df)


def test_filter_and_sort_plan_passes(df):
    plan = make_plan(
        operation="sort",
        filters=[{"column": "age", "operator": ">=", "value": 30}],
        sort_by="salary",
        ascending=False,
        limit=2,
    )
    validate_plan_against_dataframe(plan, df)