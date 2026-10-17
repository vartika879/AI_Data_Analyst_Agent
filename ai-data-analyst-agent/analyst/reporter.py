"""Reporter: writes the answer text from the computed table only.

No LLM is used here, so no number can be invented: every number in the text is
read from the ExecutionResult. Values are rounded to 2 decimals for display.
"""

from numbers import Integral, Real

import pandas as pd

from analyst.critic import CriticReport
from analyst.executor import ExecutionResult
from analyst.schemas import AnalysisPlan

MAX_LISTED_ROWS = 5
MAX_TOP_ROW_COLUMNS = 6

# Characters that could turn data into markdown links, images or code.
_MD_SPECIAL = set("\\[]()<>!`*")


def _escape(text, max_len: int = 60) -> str:
    """Make untrusted text (cell values, LLM strings) safe to show as markdown."""
    text = " ".join(str(text).split())  # newlines/tabs become single spaces
    if len(text) > max_len:
        text = text[:max_len] + "…"
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in text)


def _fmt(value) -> str:
    try:
        if pd.isna(value):
            return "missing"
    except (TypeError, ValueError):
        pass  # not a scalar
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Integral):
        return f"{int(value):,}"
    if isinstance(value, Real):
        number = float(value)
        if number != 0 and abs(number) < 0.01:
            return f"{number:.3g}"
        return f"{number:,.2f}"
    return _escape(value)


def _row_text(columns, values, max_columns: int | None = None) -> str:
    pairs = list(zip(columns, values))
    if max_columns is not None:
        pairs = pairs[:max_columns]
    return ", ".join(f"{_escape(col)} = {_fmt(val)}" for col, val in pairs)


def _describe_filters(plan: AnalysisPlan) -> str:
    parts = []
    for cond in plan.filters:
        if cond.operator in ("is_null", "not_null"):
            parts.append(f"{cond.column} {cond.operator}")
        else:
            parts.append(f"{cond.column} {cond.operator} {cond.value!r}")
    return " and ".join(parts)


def _order_text(plan: AnalysisPlan) -> str:
    direction = "ascending" if plan.ascending else "descending"
    return f"ordered by {_escape(plan.sort_by)} ({direction})"


def _scope_sentence(plan: AnalysisPlan, result: ExecutionResult) -> str:
    text = f"Analysed {result.rows_after_filters:,} of {result.rows_in:,} rows"
    if plan.filters:
        text += f" (filter: {_escape(_describe_filters(plan), max_len=120)})"
    return text + "."


def _report_aggregate(plan: AnalysisPlan, result: ExecutionResult) -> list[str]:
    table = result.table
    if not plan.group_by:
        if len(table) == 0:
            return ["No result was produced."]
        values = next(table.itertuples(index=False, name=None))
        return ["Result: " + _row_text(table.columns, values) + "."]

    total = result.total_result_rows
    if total == 0:
        return ["No rows matched, so there are no groups."]
    groups = ", ".join(_escape(c) for c in plan.group_by)
    lines = [f"The data splits into {total:,} group(s) by {groups}."]
    if plan.sort_by is not None:
        lines.append(f"Groups are {_order_text(plan)}.")
    shown = table.head(MAX_LISTED_ROWS)
    rows = [f"- {_row_text(table.columns, v)}" for v in shown.itertuples(index=False, name=None)]
    lines.append("\n".join(rows))
    if total > len(shown):
        lines.append(f"(Only the first {len(shown)} groups are listed here.)")
    return lines


def _report_filter(plan: AnalysisPlan, result: ExecutionResult) -> list[str]:
    total = result.total_result_rows
    if total == 0:
        return ["No rows matched the filter."]
    lines = [f"{total:,} row(s) matched the filter."]
    if plan.sort_by is not None:
        lines.append(f"Rows are {_order_text(plan)}.")
    return lines


def _report_sort(plan: AnalysisPlan, result: ExecutionResult) -> list[str]:
    table = result.table
    if len(table) == 0:
        return ["No rows to sort."]
    lines = [
        f"Rows are {_order_text(plan)}. "
        f"Showing {len(table):,} of {result.total_result_rows:,} row(s)."
    ]
    first = next(table.itertuples(index=False, name=None))
    lines.append("Top row: " + _row_text(table.columns, first, MAX_TOP_ROW_COLUMNS) + ".")
    return lines


def _report_missing(plan: AnalysisPlan, result: ExecutionResult) -> list[str]:
    table = result.table
    if len(table) == 0:
        return ["No columns were checked."]
    bad = table[table["missing_count"] > 0]
    if len(bad) == 0:
        return ["No missing values in the checked column(s)."]
    lines = [f"{len(bad)} of {len(table)} checked column(s) have missing values."]
    top = bad.head(3)
    rows = [
        f"- {_escape(c)}: {_fmt(n)} missing ({_fmt(p)}%)"
        for c, n, p in zip(top["column"], top["missing_count"], top["missing_percent"])
    ]
    lines.append("\n".join(rows))
    return lines


def _report_summary(plan: AnalysisPlan, result: ExecutionResult) -> list[str]:
    return [f"Summary of {result.total_result_rows:,} column(s) is shown in the table."]


def _report_correlation(plan: AnalysisPlan, result: ExecutionResult) -> list[str]:
    matrix = result.table.set_index("column")
    names = list(matrix.columns)
    best = None  # (abs_r, r, a, b)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = matrix.loc[a, b]
            if pd.isna(r):
                continue
            if best is None or abs(r) > best[0]:
                best = (abs(r), float(r), a, b)
    if best is None:
        return ["No correlation could be computed (a column may be constant)."]
    _, r, a, b = best
    return [
        f"Strongest linear relationship: {_escape(a)} and {_escape(b)} "
        f"(Pearson r = {r:.2f}). Correlation does not show cause and effect."
    ]


_REPORTERS = {
    "aggregate": _report_aggregate,
    "filter": _report_filter,
    "sort": _report_sort,
    "missing_values": _report_missing,
    "column_summary": _report_summary,
    "correlation": _report_correlation,
}


def build_report(plan: AnalysisPlan, result: ExecutionResult, critic: CriticReport) -> str:
    if not critic.passed:
        raise ValueError("Cannot report on a result that failed the consistency checks.")
    reporter = _REPORTERS.get(plan.operation)
    if reporter is None:
        raise ValueError(f"No report available for operation '{plan.operation}'.")

    parts = [_scope_sentence(plan, result)]
    parts.extend(reporter(plan, result))
    if result.truncated:
        parts.append(
            f"Only the first {len(result.table):,} of {result.total_result_rows:,} "
            "result rows are shown."
        )
    parts.extend(result.notes)
    parts.extend(critic.warnings)
    parts.append(
        f"Automated consistency checks run: {critic.checks_run}, all passed. "
        "These checks catch internal inconsistencies. They do not guarantee the "
        "answer is correct or that the question was understood as you meant it."
    )
    return "\n\n".join(parts)