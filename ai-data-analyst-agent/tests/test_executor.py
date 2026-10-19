import math

import pandas as pd
import pytest

from analyst.executor import DEFAULT_ROW_CAP, ExecutionError, execute_plan
from analyst.plan_validation import PlanValidationError
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


@pytest.fixture
def num_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"label": ["a", "b", "c"], "x": [1, 2, 3], "y": [2, 4, 6], "z": [3, 2, 1]}
    )


def run(data, **kwargs):
    return execute_plan(AnalysisPlan.model_validate(kwargs), data)


def cond(column, operator, value=None):
    return {"column": column, "operator": operator, "value": value}


# ---------- filters ----------


def test_filter_equals_text(df):
    result = run(df, operation="filter", filters=[cond("city", "==", "X")])
    assert result.table["name"].tolist() == ["Asha", "Charu"]
    assert result.rows_in == 4
    assert result.rows_after_filters == 2


def test_filter_greater_than(df):
    result = run(df, operation="filter", filters=[cond("age", ">", 30)])
    assert result.table["name"].tolist() == ["Charu", "Dev"]


def test_numeric_string_value_is_coerced_for_numeric_column(df):
    result = run(df, operation="filter", filters=[cond("age", "==", "30")])
    assert result.table["name"].tolist() == ["Asha"]


def test_text_column_rejects_number_value(df):
    with pytest.raises(ExecutionError, match="not text"):
        run(df, operation="filter", filters=[cond("city", "==", 5)])


def test_numeric_column_rejects_non_numeric_string(df):
    with pytest.raises(ExecutionError, match="not a number"):
        run(df, operation="filter", filters=[cond("age", "==", "abc")])


def test_numeric_column_rejects_bool_value(df):
    with pytest.raises(ExecutionError, match="not a number"):
        run(df, operation="filter", filters=[cond("age", "==", True)])


def test_contains_is_case_insensitive(df):
    result = run(df, operation="filter", filters=[cond("name", "contains", "AR")])
    assert result.table["name"].tolist() == ["Charu"]


def test_contains_treats_regex_characters_literally():
    data = pd.DataFrame({"t": ["a(b", "cd"]})
    paren = run(data, operation="filter", filters=[cond("t", "contains", "(")])
    assert paren.table["t"].tolist() == ["a(b"]
    dot = run(data, operation="filter", filters=[cond("t", "contains", ".")])
    assert dot.table["t"].tolist() == []


def test_is_null_and_not_null(df):
    nulls = run(df, operation="filter", filters=[cond("city", "is_null")])
    assert nulls.table["name"].tolist() == ["Dev"]
    present = run(df, operation="filter", filters=[cond("salary", "not_null")])
    assert len(present.table) == 3


def test_in_operator(df):
    result = run(df, operation="filter", filters=[cond("city", "in", ["X", "Y"])])
    assert len(result.table) == 3


def test_multiple_filters_are_combined_with_and(df):
    result = run(
        df,
        operation="filter",
        filters=[cond("city", "==", "X"), cond("age", ">=", 35)],
    )
    assert result.table["name"].tolist() == ["Charu"]


def test_filter_with_no_matches_returns_empty_table(df):
    result = run(df, operation="filter", filters=[cond("age", ">", 100)])
    assert len(result.table) == 0
    assert result.rows_after_filters == 0
    assert result.truncated is False


# ---------- sort ----------


def test_sort_descending_with_limit(df):
    result = run(df, operation="sort", sort_by="age", ascending=False, limit=2)
    assert result.table["age"].tolist() == [40, 35]
    assert result.total_result_rows == 4
    assert result.truncated is True


def test_sort_puts_missing_values_last_in_both_directions(df):
    for ascending in (True, False):
        result = run(df, operation="sort", sort_by="salary", ascending=ascending)
        assert result.table["name"].iloc[-1] == "Dev"


def test_filter_with_column_selection(df):
    result = run(
        df, operation="filter", filters=[cond("age", ">", 30)], columns=["name"]
    )
    assert list(result.table.columns) == ["name"]


# ---------- aggregate ----------


def test_grouped_mean_keeps_missing_group(df):
    result = run(
        df,
        operation="aggregate",
        group_by=["city"],
        aggregations=[{"function": "mean", "column": "salary"}],
    )
    assert len(result.table) == 3  # X, Y and the missing-city group
    t = result.table.set_index("city")
    assert t.loc["X", "salary_mean"] == 20.0
    assert t.loc["Y", "salary_mean"] == 20.0


def test_grouped_row_count(df):
    result = run(
        df,
        operation="aggregate",
        group_by=["city"],
        aggregations=[{"function": "row_count"}],
    )
    t = result.table.set_index("city")
    assert t.loc["X", "row_count"] == 2
    assert t.loc["Y", "row_count"] == 1


def test_ungrouped_aggregate_returns_one_row(df):
    result = run(
        df,
        operation="aggregate",
        aggregations=[
            {"function": "sum", "column": "salary"},
            {"function": "row_count"},
        ],
    )
    assert list(result.table.columns) == ["salary_sum", "row_count"]
    row = result.table.iloc[0]
    assert row["salary_sum"] == 60.0
    assert row["row_count"] == 4


def test_count_and_nunique(df):
    result = run(
        df,
        operation="aggregate",
        aggregations=[
            {"function": "count", "column": "city"},
            {"function": "nunique", "column": "city"},
        ],
    )
    row = result.table.iloc[0]
    assert row["city_count"] == 3
    assert row["city_nunique"] == 2


def test_aggregate_after_filter(df):
    result = run(
        df,
        operation="aggregate",
        filters=[cond("city", "==", "X")],
        aggregations=[{"function": "mean", "column": "salary"}],
    )
    assert result.table["salary_mean"].iloc[0] == 20.0
    assert result.rows_after_filters == 2


def test_aggregate_sort_by_output_name(df):
    result = run(
        df,
        operation="aggregate",
        group_by=["city"],
        aggregations=[{"function": "sum", "column": "salary"}],
        sort_by="salary_sum",
        ascending=False,
    )
    assert result.table["city"].iloc[0] == "X"  # X sums to 40, the highest


def test_ungrouped_aggregate_on_empty_filter_result(df):
    result = run(
        df,
        operation="aggregate",
        filters=[cond("age", ">", 100)],
        aggregations=[{"function": "sum", "column": "salary"}],
    )
    assert len(result.table) == 1
    assert result.table["salary_sum"].iloc[0] == 0


def test_grouped_aggregate_on_empty_filter_result(df):
    result = run(
        df,
        operation="aggregate",
        filters=[cond("age", ">", 100)],
        group_by=["city"],
        aggregations=[{"function": "mean", "column": "salary"}],
    )
    assert len(result.table) == 0
    assert list(result.table.columns) == ["city", "salary_mean"]


def test_aggregate_output_name_collision_raises():
    data = pd.DataFrame({"row_count": [1, 2], "v": [1, 2]})
    with pytest.raises(ExecutionError, match="collide"):
        run(
            data,
            operation="aggregate",
            group_by=["row_count"],
            aggregations=[{"function": "row_count"}],
        )


# ---------- missing values, summary, correlation ----------


def test_missing_values_sorted_by_count(df):
    result = run(df, operation="missing_values")
    assert list(result.table.columns) == ["column", "missing_count", "missing_percent"]
    assert result.table["column"].tolist()[:2] == ["city", "salary"]
    assert result.table["missing_percent"].iloc[0] == 25.0


def test_missing_values_for_selected_column(df):
    result = run(df, operation="missing_values", columns=["age"])
    assert len(result.table) == 1
    assert result.table["missing_count"].iloc[0] == 0


def test_column_summary(df):
    result = run(df, operation="column_summary")
    t = result.table.set_index("column")
    assert len(t) == 4
    assert t.loc["age", "min"] == 25
    assert t.loc["age", "max"] == 40
    assert t.loc["age", "mean"] == 32.5
    assert math.isnan(t.loc["name", "mean"])
    assert t.loc["salary", "non_null"] == 3
    assert t.loc["salary", "missing"] == 1
    assert t.loc["city", "unique"] == 2


def test_correlation_default_uses_numeric_columns_only(num_df):
    result = run(num_df, operation="correlation")
    assert list(result.table.columns) == ["column", "x", "y", "z"]
    t = result.table.set_index("column")
    assert t.loc["x", "y"] == pytest.approx(1.0)
    assert t.loc["x", "z"] == pytest.approx(-1.0)


def test_correlation_with_given_columns(num_df):
    result = run(num_df, operation="correlation", columns=["x", "z"])
    assert result.table.shape == (2, 3)


def test_correlation_with_too_few_rows_raises(num_df):
    with pytest.raises(ExecutionError, match="at least 2 rows"):
        run(num_df, operation="correlation", filters=[cond("x", ">", 2)])


# ---------- limits, safety ----------


def test_default_row_cap_truncates_large_results():
    data = pd.DataFrame({"n": list(range(150))})
    result = run(data, operation="filter", filters=[cond("n", ">=", 0)])
    assert len(result.table) == DEFAULT_ROW_CAP
    assert result.total_result_rows == 150
    assert result.truncated is True


def test_small_results_are_not_truncated(df):
    result = run(df, operation="filter", filters=[cond("age", ">", 0)])
    assert result.truncated is False
    assert result.total_result_rows == 4


def test_input_dataframe_is_not_modified(df):
    before = df.copy()
    run(df, operation="sort", sort_by="age", filters=[cond("age", ">", 25)])
    pd.testing.assert_frame_equal(df, before)


def test_plan_with_unknown_column_is_rejected_before_running(df):
    with pytest.raises(PlanValidationError, match="Unknown column"):
        run(df, operation="filter", filters=[cond("bonus", "==", 1)])


def test_unsupported_plan_is_rejected(df):
    with pytest.raises(PlanValidationError, match="Unsupported"):
        run(df, operation="unsupported", unsupported_reason="Needs forecasting.")


def test_text_equality_ignores_case_and_spaces(df):
    lower = run(df, operation="filter", filters=[cond("city", "==", "x")])
    assert lower.table["name"].tolist() == ["Asha", "Charu"]
    spaced = run(df, operation="filter", filters=[cond("city", "==", " X ")])
    assert spaced.table["name"].tolist() == ["Asha", "Charu"]


def test_text_not_equal_ignores_case_and_keeps_missing(df):
    result = run(df, operation="filter", filters=[cond("city", "!=", "x")])
    assert result.table["name"].tolist() == ["Ravi", "Dev"]  # Dev has no city


def test_text_in_ignores_case(df):
    result = run(df, operation="filter", filters=[cond("city", "in", ["x", "y"])])
    assert len(result.table) == 3


def test_aggregate_with_lowercase_filter_value_uses_the_rows():
    data = pd.DataFrame({"product": ["Pen", "Book", "Pen"], "net": [90, 230, 180]})
    result = run(
        data,
        operation="aggregate",
        filters=[cond("product", "==", "pen")],
        aggregations=[{"function": "sum", "column": "net"}],
    )
    assert result.rows_after_filters == 2
    assert result.table["net_sum"].tolist() == [270]