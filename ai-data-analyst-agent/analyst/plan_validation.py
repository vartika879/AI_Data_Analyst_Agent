"""Checks an AnalysisPlan against the actual DataFrame.

Pydantic (schemas.py) checks the shape of the plan. This module checks that
the plan makes sense for this specific dataset.
"""

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_string_dtype

from analyst.schemas import ORDERING_OPERATORS, AnalysisPlan, FilterCondition

NUMERIC_AGG_FUNCTIONS = {"sum", "mean", "median"}
SORT_SUPPORTED_OPERATIONS = {"filter", "sort", "aggregate"}


class PlanValidationError(ValueError):
    """Raised when a plan does not fit the uploaded data."""


def _require_columns(df: pd.DataFrame, names: list[str], where: str) -> None:
    missing = [name for name in dict.fromkeys(names) if name not in df.columns]
    if missing:
        available = ", ".join(str(c) for c in list(df.columns)[:20])
        raise PlanValidationError(
            f"Unknown column(s) in {where}: {', '.join(missing)}. "
            f"Available columns: {available}"
        )


def _check_filter(df: pd.DataFrame, condition: FilterCondition) -> None:
    series = df[condition.column]
    if condition.operator in ORDERING_OPERATORS:
        if not is_numeric_dtype(series) or is_bool_dtype(series):
            raise PlanValidationError(
                f"Operator '{condition.operator}' needs a numeric column, "
                f"but '{condition.column}' is not numeric."
            )
    if condition.operator == "contains" and not is_string_dtype(series):
        raise PlanValidationError(
            f"Operator 'contains' needs a text column, "
            f"but '{condition.column}' is not text."
        )


def validate_plan_against_dataframe(plan: AnalysisPlan, df: pd.DataFrame) -> None:
    """Raise PlanValidationError if the plan cannot be run on this DataFrame."""
    if not plan.is_supported:
        raise PlanValidationError(f"Unsupported question: {plan.unsupported_reason}")

    _require_columns(df, plan.columns, "columns")

    for condition in plan.filters:
        _require_columns(df, [condition.column], "filters")
        _check_filter(df, condition)

    _require_columns(df, plan.group_by, "group_by")

    for agg in plan.aggregations:
        if agg.column is None:  # row_count
            continue
        _require_columns(df, [agg.column], "aggregations")
        if agg.function in NUMERIC_AGG_FUNCTIONS and not is_numeric_dtype(df[agg.column]):
            raise PlanValidationError(
                f"'{agg.function}' needs a numeric column, "
                f"but '{agg.column}' is not numeric."
            )

    if plan.operation == "correlation":
        if plan.columns:
            for name in plan.columns:
                if not is_numeric_dtype(df[name]):
                    raise PlanValidationError(
                        f"Correlation needs numeric columns, but '{name}' is not numeric."
                    )
        elif df.select_dtypes(include="number").shape[1] < 2:
            raise PlanValidationError(
                "Correlation needs at least two numeric columns in the data."
            )

    if plan.sort_by is not None:
        if plan.operation not in SORT_SUPPORTED_OPERATIONS:
            raise PlanValidationError(
                f"sort_by is not supported for the '{plan.operation}' operation."
            )
        if plan.operation == "aggregate":
            allowed = list(plan.group_by) + [a.output_name for a in plan.aggregations]
            if plan.sort_by not in allowed:
                raise PlanValidationError(
                    f"sort_by must be a group_by column or an aggregation output. "
                    f"Allowed: {', '.join(allowed)}"
                )
        else:
            _require_columns(df, [plan.sort_by], "sort_by")