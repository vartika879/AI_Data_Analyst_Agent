"""Scoring: compare a pipeline outcome with the expected answer of a case."""

import math
from dataclasses import dataclass

from analyst.charts import build_chart
from analyst.graph import AnalysisOutcome
from evals.cases import EvalCase


@dataclass(frozen=True)
class CaseScore:
    passed: bool
    reason: str | None  # None when passed
    asked: bool  # the Clarifier asked the user a question


def _same(cell, expected) -> bool:
    """Numbers are compared approximately, everything else as text."""
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(cell), float(expected), rel_tol=1e-6, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return str(cell) == str(expected)


def _check_table(expect: dict, outcome: AnalysisOutcome) -> str | None:
    """Return None if the result matches, else a short reason."""
    result = outcome.result
    table = result.table
    rows = list(table.itertuples(index=False, name=None))
    kind = expect["type"]

    if kind == "scalar":
        max_rows = expect.get("max_rows", 1)
        if result.total_result_rows > max_rows:
            return f"expected at most {max_rows} row(s), got {result.total_result_rows}"
        value = expect["value"]
        for row in rows:
            for cell in row:
                try:
                    if math.isclose(float(cell), value, rel_tol=1e-6, abs_tol=1e-6):
                        return None
                except (TypeError, ValueError):
                    continue
        return f"no numeric cell equals {value:.6g}"

    if kind == "mapping":
        wanted = expect["values"]
        if result.total_result_rows != len(wanted):
            return f"expected {len(wanted)} groups, got {result.total_result_rows}"
        for label, value in wanted.items():
            if not any(
                any(_same(c, label) for c in row) and any(_same(c, value) for c in row)
                for row in rows
            ):
                return f"no row matches {label!r} = {value:.6g}"
        return None

    if kind == "first_row_has":
        if not rows or not any(_same(c, expect["value"]) for c in rows[0]):
            return f"first row does not contain {expect['value']!r}"
        return None

    if kind == "column_values":
        values = expect["values"]
        if len(table) != len(values):
            return f"expected {len(values)} rows, got {len(table)}"
        for name in table.columns:
            if all(_same(a, b) for a, b in zip(table[name], values)):
                return None
        return f"no column starts with the expected values {values}"
    if kind == "row_ids":
        if "order_id" not in table.columns:
            return "result has no order_id column, so the rows cannot be identified"
        got = sorted(int(v) for v in table["order_id"])
        if got != expect["values"]:
            return f"expected order_id values {expect['values']}, got {got}"
        return None
    if kind == "row_count":
        if result.total_result_rows != expect["value"]:
            return f"expected {expect['value']} rows, got {result.total_result_rows}"
        return None

    return f"unknown check type {kind!r}"


def check_outcome(case: EvalCase, outcome: AnalysisOutcome) -> CaseScore:
    asked = outcome.status == "clarify"

    if case.kind == "unsupported":
        ok = outcome.status == "unsupported"
        return CaseScore(ok, None if ok else f"status {outcome.status}, expected unsupported", asked)

    if case.kind == "ambiguous":
        return CaseScore(asked, None if asked else f"status {outcome.status}, expected clarify", asked)

    # answerable
    if asked:
        return CaseScore(False, "asked a clarification question", asked)
    if outcome.status != "success":
        return CaseScore(False, f"status {outcome.status}: {outcome.message[:100]}", asked)

    reason = _check_table(case.expect, outcome)
    if reason is None and case.expect.get("chart"):
        if build_chart(outcome.plan, outcome.result).figure is None:
            reason = "no chart was produced"
    return CaseScore(reason is None, reason, asked)