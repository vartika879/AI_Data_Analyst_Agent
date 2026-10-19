"""Executor: runs a validated AnalysisPlan using fixed, pre-written Pandas code.

No exec/eval, and no user or LLM text is ever turned into code. The plan only
selects WHICH of our own functions run, and with which (validated) values.
"""

import math
import operator
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from analyst.plan_validation import validate_plan_against_dataframe
from analyst.schemas import AnalysisPlan, FilterCondition

DEFAULT_ROW_CAP = 100  # rows kept in the result when the plan gives no limit
MAX_CORRELATION_COLUMNS = 20

# Whitelists: plan strings are looked up here. A name not in the dict cannot run.
ORDERING_FUNCS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}
AGG_FUNCS = {
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "min": "min",
    "max": "max",
    "count": "count",
    "nunique": "nunique",
}


class ExecutionError(ValueError):
    """Raised when a valid plan cannot be executed on the data."""


@dataclass(frozen=True, eq=False)
class ExecutionResult:
    operation: str
    table: pd.DataFrame  # the (possibly truncated) result
    rows_in: int  # rows in the uploaded data
    rows_after_filters: int  # rows left after the plan's filters
    total_result_rows: int  # result rows before truncation
    truncated: bool  # True if table has fewer rows than total_result_rows
    notes: tuple[str, ...] = ()


def _coerce_value(series: pd.Series, column: str, value):
    """Make a filter value compatible with the column's type, or raise."""
    if is_bool_dtype(series):
        if isinstance(value, bool):
            return value
        raise ExecutionError(
            f"Column '{column}' is true/false, so the value must be true or false."
        )
    if is_numeric_dtype(series):
        if isinstance(value, bool):
            raise ExecutionError(
                f"Column '{column}' is numeric, but the value {value!r} is not a number."
            )
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                number = float(value)
            except ValueError:
                number = math.nan
            if math.isfinite(number):
                return number
        raise ExecutionError(
            f"Column '{column}' is numeric, but the value {value!r} is not a number."
        )
    if isinstance(value, str):
        return value
    raise ExecutionError(
        f"Column '{column}' is text, but the value {value!r} is not text."
    )


def _normalise_text(series: pd.Series) -> pd.Series:
    """Text compared without caring about case or surrounding spaces ("pen" == "Pen ")."""
    return series.astype("string").str.strip().str.casefold()


def _is_text_column(series: pd.Series) -> bool:
    return not is_numeric_dtype(series) and not is_bool_dtype(series)


def _build_mask(df: pd.DataFrame, cond: FilterCondition) -> pd.Series:
    series = df[cond.column]
    op = cond.operator

    if op == "is_null":
        return series.isna()
    if op == "not_null":
        return series.notna()
    if op == "contains":
        # regex=False: the text is matched literally, never as a regex pattern.
        return series.str.contains(cond.value, case=False, regex=False, na=False)
    if op in ORDERING_FUNCS:
        return ORDERING_FUNCS[op](series, cond.value)
    if op == "in":
        values = [_coerce_value(series, cond.column, v) for v in cond.value]
        if _is_text_column(series):
            wanted = [v.strip().casefold() for v in values]
            return _normalise_text(series).isin(wanted).fillna(False).astype(bool)
        return series.isin(values)

    value = _coerce_value(series, cond.column, cond.value)
    if op in ("==", "!=") and _is_text_column(series):
        # Users (and the Planner in privacy mode, which sees no sample values) write
        # "pen" for "Pen". An exact match would silently return 0 rows.
        equal = (_normalise_text(series) == value.strip().casefold()).fillna(False).astype(bool)
        return equal if op == "==" else ~equal  # missing values count as "not equal"
    if op == "==":
        return series == value
    if op == "!=":
        return series != value
    raise ExecutionError(f"Unsupported filter operator: {op}")


def _apply_filters(df: pd.DataFrame, filters: list[FilterCondition]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for cond in filters:  # all filters are combined with AND
        mask = mask & _build_mask(df, cond)
    return df[mask]


def _column_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for name in columns or list(df.columns):
        series = df[name]
        non_null = int(series.notna().sum())
        row = {
            "column": name,
            "dtype": str(series.dtype),
            "non_null": non_null,
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
        }
        if is_numeric_dtype(series) and not is_bool_dtype(series) and non_null > 0:
            row["min"] = float(series.min())
            row["max"] = float(series.max())
            row["mean"] = float(series.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _filter_or_sort(df: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
    table = df
    if plan.sort_by is not None:
        table = table.sort_values(
            by=plan.sort_by, ascending=plan.ascending, kind="stable", na_position="last"
        )
    if plan.columns:  # select columns AFTER sorting, since sort_by may not be selected
        table = table[plan.columns]
    return table.reset_index(drop=True)


def _aggregate(df: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
    names = list(plan.group_by) + [agg.output_name for agg in plan.aggregations]
    if len(set(names)) != len(names):
        raise ExecutionError(
            "Result column names collide (a group_by column has the same name "
            "as an aggregation output)."
        )

    if plan.group_by:
        # dropna=False: rows with a missing group value are kept as their own
        # group instead of silently disappearing.
        grouped = df.groupby(plan.group_by, dropna=False, sort=True)
        parts = {}
        for agg in plan.aggregations:
            if agg.function == "row_count":
                parts[agg.output_name] = grouped.size()
            else:
                parts[agg.output_name] = grouped[agg.column].agg(AGG_FUNCS[agg.function])
        table = pd.concat(parts, axis=1).reset_index()
    else:
        values = {}
        for agg in plan.aggregations:
            if agg.function == "row_count":
                values[agg.output_name] = len(df)
            else:
                values[agg.output_name] = df[agg.column].agg(AGG_FUNCS[agg.function])
        table = pd.DataFrame([values])

    if plan.sort_by is not None:
        table = table.sort_values(
            by=plan.sort_by, ascending=plan.ascending, kind="stable", na_position="last"
        )
    return table.reset_index(drop=True)


def _missing_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    names = columns or list(df.columns)
    counts = df[names].isna().sum()
    rows = len(df)
    table = pd.DataFrame(
        {
            "column": names,
            "missing_count": counts.to_numpy(),
            "missing_percent": [
                round(int(c) / rows * 100, 2) if rows else 0.0 for c in counts
            ],
        }
    )
    return table.sort_values("missing_count", ascending=False, kind="stable").reset_index(
        drop=True
    )


def _correlation(df: pd.DataFrame, columns: list[str], notes: list[str]) -> pd.DataFrame:
    if columns:
        names = list(columns)
    else:
        names = [
            c for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])
        ]
        if len(names) > MAX_CORRELATION_COLUMNS:
            notes.append(
                f"Only the first {MAX_CORRELATION_COLUMNS} numeric columns were used."
            )
            names = names[:MAX_CORRELATION_COLUMNS]

    if len(df) < 2:
        raise ExecutionError("Correlation needs at least 2 rows after filtering.")

    corr = df[names].corr(method="pearson", numeric_only=True)
    if list(corr.columns) != names:
        raise ExecutionError("Correlation needs numeric columns.")
    corr.index.name = "column"
    return corr.reset_index()


def _run(plan: AnalysisPlan, df: pd.DataFrame) -> ExecutionResult:
    notes: list[str] = []
    filtered = _apply_filters(df, plan.filters)
    op = plan.operation

    if op == "column_summary":
        table = _column_summary(filtered, plan.columns)
    elif op in ("filter", "sort"):
        table = _filter_or_sort(filtered, plan)
    elif op == "aggregate":
        if plan.columns:
            notes.append("The 'columns' field is ignored for aggregate operations.")
        table = _aggregate(filtered, plan)
    elif op == "missing_values":
        table = _missing_values(filtered, plan.columns)
    elif op == "correlation":
        table = _correlation(filtered, plan.columns, notes)
    else:
        raise ExecutionError(f"Unsupported operation: {op}")

    total = len(table)
    if op == "correlation":
        # A correlation matrix must stay square, so it is never truncated.
        truncated = False
    else:
        cap = plan.limit if plan.limit is not None else DEFAULT_ROW_CAP
        truncated = total > cap
        table = table.head(cap)

    return ExecutionResult(
        operation=op,
        table=table.reset_index(drop=True),
        rows_in=len(df),
        rows_after_filters=len(filtered),
        total_result_rows=total,
        truncated=truncated,
        notes=tuple(notes),
    )


def execute_plan(plan: AnalysisPlan, df: pd.DataFrame) -> ExecutionResult:
    """Validate the plan against the data, then run it.

    Raises PlanValidationError if the plan does not fit the data, and
    ExecutionError if the plan is valid but cannot be run.
    """
    # Validate again here (defense in depth): the executor never trusts its caller.
    validate_plan_against_dataframe(plan, df)
    try:
        return _run(plan, df)
    except ExecutionError:
        raise
    except Exception as exc:
        # Only the exception TYPE goes to the user message, never raw data.
        raise ExecutionError(
            f"Could not run the analysis ({type(exc).__name__}). "
            "Try rephrasing the question."
        ) from exc