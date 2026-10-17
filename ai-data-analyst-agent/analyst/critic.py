"""Critic: consistency checks on an execution result.

IMPORTANT: this is NOT proof that the answer is correct. The Critic cannot know
what the user really meant. It only re-checks facts that can be verified from
the plan, the result and the original data (shape, counts, ordering, filters).
"""

from dataclasses import dataclass

import pandas as pd

from analyst.executor import ORDERING_FUNCS, ExecutionResult
from analyst.schemas import AnalysisPlan

MISSING_VALUES_COLUMNS = ["column", "missing_count", "missing_percent"]
STRUCTURAL_OPERATIONS = {"column_summary", "missing_values", "correlation"}
CORRELATION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CriticReport:
    passed: bool  # False if any hard check failed
    issues: tuple[str, ...] = ()  # hard failures: the result is inconsistent
    warnings: tuple[str, ...] = ()  # soft notes: result is consistent but notable
    checks_run: int = 0


def _is_sorted(series: pd.Series, ascending: bool) -> bool:
    values = series.dropna()
    if len(values) < 2:
        return True
    try:
        if ascending:
            return bool(values.is_monotonic_increasing)
        return bool(values.is_monotonic_decreasing)
    except TypeError:
        return True  # values cannot be compared, so we cannot verify the order


def review_execution(
    plan: AnalysisPlan, result: ExecutionResult, df: pd.DataFrame
) -> CriticReport:
    issues: list[str] = []
    warnings: list[str] = []
    checks = 0

    def check(ok: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not ok:
            issues.append(message)

    table = result.table
    op = plan.operation
    names = list(plan.columns) or list(df.columns)

    # ---- bookkeeping checks (apply to every operation) ----
    check(result.operation == op, "The result is for a different operation than planned.")
    check(
        result.rows_in == len(df),
        f"Result reports {result.rows_in} input rows but the data has {len(df)}.",
    )
    check(
        0 <= result.rows_after_filters <= result.rows_in,
        "Row count after filtering is impossible.",
    )
    if not plan.filters:
        check(
            result.rows_after_filters == result.rows_in,
            "No filters were planned, but some rows were removed.",
        )
    check(
        len(table) <= result.total_result_rows,
        "The table has more rows than the reported total.",
    )
    check(
        result.truncated == (len(table) < result.total_result_rows),
        "The 'truncated' flag does not match the row counts.",
    )
    if plan.limit is not None:
        check(len(table) <= plan.limit, "The result has more rows than the requested limit.")

    # ---- output shape matches the operation ----
    if op == "aggregate":
        expected = list(plan.group_by) + [a.output_name for a in plan.aggregations]
        check(
            list(table.columns) == expected,
            f"Aggregate result columns are {list(table.columns)}, expected {expected}.",
        )
        if not result.truncated and any(a.function == "row_count" for a in plan.aggregations):
            if "row_count" in table.columns:
                check(
                    int(table["row_count"].sum()) == result.rows_after_filters,
                    "The row counts do not add up to the number of filtered rows.",
                )
    elif op in ("filter", "sort"):
        expected = names
        check(
            list(table.columns) == expected,
            f"Result columns are {list(table.columns)}, expected {expected}.",
        )
        check(
            result.total_result_rows == result.rows_after_filters,
            "Filter/sort result size does not match the filtered row count.",
        )
    elif op == "missing_values":
        check(
            list(table.columns) == MISSING_VALUES_COLUMNS,
            "Missing-value table has unexpected columns.",
        )
        if list(table.columns) == MISSING_VALUES_COLUMNS:
            check(
                bool((table["missing_count"] <= result.rows_after_filters).all()),
                "A column reports more missing values than there are rows.",
            )
            if not result.truncated:
                check(
                    set(table["column"]) == set(names),
                    "Missing-value table does not cover the expected columns.",
                )
    elif op == "column_summary":
        has_column = "column" in table.columns
        check(has_column, "Column summary has no 'column' field.")
        if has_column and not result.truncated:
            check(
                list(table["column"]) == names,
                "Column summary does not cover the expected columns.",
            )
    elif op == "correlation":
        has_column = "column" in table.columns
        check(has_column, "Correlation table has no 'column' field.")
        if has_column:
            matrix_cols = [c for c in table.columns if c != "column"]
            check(
                list(table["column"]) == matrix_cols,
                "Correlation result is not a square matrix.",
            )
            if plan.columns:
                check(
                    matrix_cols == list(plan.columns),
                    "Correlation columns differ from the requested columns.",
                )
            raw = pd.Series(table[matrix_cols].to_numpy().ravel())
            values = pd.to_numeric(raw, errors="coerce").dropna()
            check(
                bool(
                    (
                        (values >= -1 - CORRELATION_TOLERANCE)
                        & (values <= 1 + CORRELATION_TOLERANCE)
                    ).all()
                ),
                "A correlation value is outside the range -1 to 1.",
            )

    # ---- sort order (checked on the result itself) ----
    if plan.sort_by is not None and plan.sort_by in table.columns:
        check(
            _is_sorted(table[plan.sort_by], plan.ascending),
            f"The result is not sorted by '{plan.sort_by}' as requested.",
        )

    # ---- filters re-checked on the output rows (only possible for filter/sort) ----
    if op in ("filter", "sort"):
        for cond in plan.filters:
            if cond.column not in table.columns:
                continue  # the column was not selected, so it cannot be checked
            column = table[cond.column]
            ok = True
            if cond.operator == "not_null":
                ok = bool(column.notna().all())
            elif cond.operator == "is_null":
                ok = bool(column.isna().all())
            elif cond.operator in ORDERING_FUNCS:
                ok = bool(ORDERING_FUNCS[cond.operator](column, cond.value).all())
            check(ok, f"Filter on '{cond.column}' ({cond.operator}) is violated by rows in the result.")

    # ---- non-empty when expected ----
    must_have_rows = op in STRUCTURAL_OPERATIONS or (op == "aggregate" and not plan.group_by)
    if must_have_rows:
        check(len(table) > 0, "The result has no rows, but this operation should always return rows.")
    elif len(table) == 0:
        warnings.append(
            "The result is empty: no rows matched. Check the spelling and "
            "capitalisation of the values you filtered on."
        )

    return CriticReport(
        passed=not issues,
        issues=tuple(issues),
        warnings=tuple(warnings),
        checks_run=checks,
    )