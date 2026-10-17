import pytest
from pydantic import ValidationError

from analyst.schemas import AggregationSpec, AnalysisPlan, FilterCondition


def make_plan(**kwargs) -> AnalysisPlan:
    return AnalysisPlan.model_validate(kwargs)


def test_valid_sort_plan():
    plan = make_plan(operation="sort", sort_by="age", ascending=False, limit=5)
    assert plan.is_supported
    assert plan.sort_by == "age"
    assert plan.ascending is False
    assert plan.limit == 5


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        make_plan(operation="sort", sort_by="age", code="import os")


def test_unknown_operation_is_rejected():
    with pytest.raises(ValidationError):
        make_plan(operation="delete_everything")


def test_filter_operation_requires_filters():
    with pytest.raises(ValidationError, match="requires at least one filter"):
        make_plan(operation="filter")


def test_sort_operation_requires_sort_by():
    with pytest.raises(ValidationError, match="requires sort_by"):
        make_plan(operation="sort")


def test_aggregate_requires_aggregations():
    with pytest.raises(ValidationError, match="at least one aggregation"):
        make_plan(operation="aggregate")


def test_unsupported_requires_reason():
    with pytest.raises(ValidationError, match="requires unsupported_reason"):
        make_plan(operation="unsupported")


def test_unsupported_plan_with_reason_is_valid():
    plan = make_plan(operation="unsupported", unsupported_reason="Needs forecasting.")
    assert plan.is_supported is False


def test_group_by_only_allowed_for_aggregate():
    with pytest.raises(ValidationError, match="only allowed with the 'aggregate'"):
        make_plan(operation="sort", sort_by="age", group_by=["city"])


def test_row_count_must_not_have_column():
    with pytest.raises(ValidationError, match="must not have a column"):
        AggregationSpec(function="row_count", column="age")


def test_non_row_count_function_needs_column():
    with pytest.raises(ValidationError, match="needs a column"):
        AggregationSpec(function="mean")


def test_aggregation_output_names():
    assert AggregationSpec(function="mean", column="salary").output_name == "salary_mean"
    assert AggregationSpec(function="row_count").output_name == "row_count"


def test_duplicate_aggregation_outputs_rejected():
    with pytest.raises(ValidationError, match="duplicate aggregation"):
        make_plan(
            operation="aggregate",
            aggregations=[
                {"function": "mean", "column": "age"},
                {"function": "mean", "column": "age"},
            ],
        )


def test_in_operator_needs_non_empty_list():
    with pytest.raises(ValidationError, match="non-empty list"):
        FilterCondition(column="city", operator="in", value="Delhi")
    with pytest.raises(ValidationError, match="non-empty list"):
        FilterCondition(column="city", operator="in", value=[])


def test_ordering_operator_rejects_string_value():
    with pytest.raises(ValidationError, match="numeric value"):
        FilterCondition(column="age", operator=">", value="thirty")


def test_ordering_operator_rejects_bool_value():
    with pytest.raises(ValidationError, match="numeric value"):
        FilterCondition(column="age", operator=">", value=True)


def test_null_operators_reject_value():
    with pytest.raises(ValidationError, match="does not take a value"):
        FilterCondition(column="age", operator="is_null", value=1)


def test_contains_needs_string():
    with pytest.raises(ValidationError, match="non-empty text value"):
        FilterCondition(column="name", operator="contains", value=5)


def test_limit_bounds():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        make_plan(operation="sort", sort_by="age", limit=0)
    with pytest.raises(ValidationError, match="less than or equal to"):
        make_plan(operation="sort", sort_by="age", limit=10_000)


def test_correlation_needs_two_columns_if_given():
    with pytest.raises(ValidationError, match="at least 2 columns"):
        make_plan(operation="correlation", columns=["age"])


def test_duplicate_columns_rejected():
    with pytest.raises(ValidationError, match="duplicate names in columns"):
        make_plan(operation="column_summary", columns=["age", "age"])


def test_column_names_are_stripped():
    plan = make_plan(operation="column_summary", columns=["  age  "])
    assert plan.columns == ["age"]


def test_invalid_json_is_rejected():
    with pytest.raises(ValidationError):
        AnalysisPlan.model_validate_json("this is not json")


def test_filter_value_types_are_preserved():
    assert FilterCondition(column="a", operator="==", value=5).value == 5
    assert isinstance(FilterCondition(column="a", operator="==", value=5).value, int)
    assert FilterCondition(column="a", operator="==", value="5").value == "5"