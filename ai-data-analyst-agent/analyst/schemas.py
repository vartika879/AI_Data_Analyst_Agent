"""Pydantic schemas for the structured analysis plan.

The LLM never writes code. It only fills in this plan. Every value that can
reach the executor is limited to the fixed choices (Literals) defined here.
"""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Operation = Literal[
    "column_summary",
    "filter",
    "sort",
    "aggregate",  # with optional group_by
    "missing_values",
    "correlation",
    "unsupported",  # the planner uses this when the question cannot be answered
]
FilterOperator = Literal[
    "==", "!=", ">", ">=", "<", "<=", "contains", "in", "is_null", "not_null"
]
AggFunction = Literal[
    "sum", "mean", "median", "min", "max", "count", "nunique", "row_count"
]
ChartType = Literal["bar", "line", "scatter"]

ORDERING_OPERATORS = {">", ">=", "<", "<="}

MAX_COLUMNS = 50
MAX_FILTERS = 5
MAX_GROUP_BY = 3
MAX_AGGREGATIONS = 5
MAX_IN_VALUES = 50
MAX_LIMIT = 1000

ColumnName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
]
Scalar = str | int | float | bool


class _PlanModel(BaseModel):
    # extra="forbid": unknown fields from the LLM are rejected, not ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class FilterCondition(_PlanModel):
    column: ColumnName
    operator: FilterOperator
    value: Scalar | list[Scalar] | None = None

    @model_validator(mode="after")
    def _check_value_matches_operator(self) -> "FilterCondition":
        op, value = self.operator, self.value

        if op in ("is_null", "not_null"):
            if value is not None:
                raise ValueError(f"operator '{op}' does not take a value")
        elif op == "in":
            if not isinstance(value, list) or len(value) == 0:
                raise ValueError("operator 'in' needs a non-empty list of values")
            if len(value) > MAX_IN_VALUES:
                raise ValueError(f"operator 'in' allows at most {MAX_IN_VALUES} values")
        elif op in ORDERING_OPERATORS:
            # bool is a subclass of int in Python, so exclude it explicitly.
            is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not is_number or not math.isfinite(value):
                raise ValueError(f"operator '{op}' needs a numeric value")
        elif op == "contains":
            if not isinstance(value, str) or value == "":
                raise ValueError("operator 'contains' needs a non-empty text value")
        else:  # "==" and "!="
            if value is None or isinstance(value, list):
                raise ValueError(f"operator '{op}' needs a single value")
        return self


class AggregationSpec(_PlanModel):
    function: AggFunction
    column: ColumnName | None = None

    @model_validator(mode="after")
    def _check_column(self) -> "AggregationSpec":
        if self.function == "row_count":
            if self.column is not None:
                raise ValueError("'row_count' must not have a column")
        elif self.column is None:
            raise ValueError(f"'{self.function}' needs a column")
        return self

    @property
    def output_name(self) -> str:
        """Name of the result column this aggregation will produce."""
        if self.function == "row_count":
            return "row_count"
        return f"{self.column}_{self.function}"


class AnalysisPlan(_PlanModel):
    operation: Operation
    columns: list[ColumnName] = Field(default_factory=list, max_length=MAX_COLUMNS)
    filters: list[FilterCondition] = Field(default_factory=list, max_length=MAX_FILTERS)
    group_by: list[ColumnName] = Field(default_factory=list, max_length=MAX_GROUP_BY)
    aggregations: list[AggregationSpec] = Field(
        default_factory=list, max_length=MAX_AGGREGATIONS
    )
    sort_by: ColumnName | None = None
    ascending: bool = True
    limit: int | None = Field(default=None, ge=1, le=MAX_LIMIT)
    chart_type: ChartType | None = None
    unsupported_reason: str | None = Field(default=None, max_length=300)

    @property
    def is_supported(self) -> bool:
        return self.operation != "unsupported"

    @model_validator(mode="after")
    def _check_operation_rules(self) -> "AnalysisPlan":
        op = self.operation

        if len(set(self.columns)) != len(self.columns):
            raise ValueError("duplicate names in columns")
        if len(set(self.group_by)) != len(self.group_by):
            raise ValueError("duplicate names in group_by")

        if self.group_by and op != "aggregate":
            raise ValueError("group_by is only allowed with the 'aggregate' operation")
        if self.aggregations and op != "aggregate":
            raise ValueError("aggregations are only allowed with the 'aggregate' operation")

        if op == "filter" and not self.filters:
            raise ValueError("operation 'filter' requires at least one filter")
        if op == "sort" and self.sort_by is None:
            raise ValueError("operation 'sort' requires sort_by")
        if op == "aggregate":
            if not self.aggregations:
                raise ValueError("operation 'aggregate' requires at least one aggregation")
            names = [agg.output_name for agg in self.aggregations]
            if len(set(names)) != len(names):
                raise ValueError("duplicate aggregation output names")
        if op == "correlation" and self.columns and len(self.columns) < 2:
            raise ValueError("correlation needs at least 2 columns when columns are given")
        if op == "unsupported" and not (self.unsupported_reason or "").strip():
            raise ValueError("operation 'unsupported' requires unsupported_reason")
        return self