"""Evaluation cases. Expected answers are computed with plain Python."""

import statistics
from dataclasses import dataclass

from evals.dataset import load_rows

ROWS = load_rows()


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    kind: str  # "answerable" | "unsupported" | "ambiguous"
    category: str
    expect: dict
    reference_plan: dict | None = None  # what a perfect Planner would answer


def _nums(column: str, rows: list[dict] | None = None) -> list[float]:
    rows = ROWS if rows is None else rows
    return [float(r[column]) for r in rows if r[column] != ""]


def _by(key: str, column: str | None = None, func=sum) -> dict[str, float]:
    """Group ROWS by `key`; apply `func` to a numeric column, or count rows."""
    groups: dict[str, list[dict]] = {}
    for row in ROWS:
        groups.setdefault(row[key], []).append(row)
    if column is None:
        return {k: len(v) for k, v in groups.items()}
    return {k: func(_nums(column, v)) for k, v in groups.items()}


def _count(predicate) -> int:
    return sum(1 for r in ROWS if predicate(r))


def _ids(predicate) -> list[int]:
    """Sorted order_id values of the rows that satisfy `predicate`."""
    return sorted(int(r["order_id"]) for r in ROWS if predicate(r))


def _top_ids(column: str, n: int, descending: bool) -> list[int]:
    valid = [r for r in ROWS if r[column] != ""]
    ordered = sorted(valid, key=lambda r: float(r[column]), reverse=descending)
    return [int(r["order_id"]) for r in ordered[:n]]


def _correlation(a: str, b: str) -> float:
    pairs = [(float(r[a]), float(r[b])) for r in ROWS if r[a] != "" and r[b] != ""]
    return statistics.correlation(*zip(*pairs))


_GROSS_BY_REGION = _by("region", "gross_revenue", sum)
_BEST_REGION = max(_GROSS_BY_REGION, key=_GROSS_BY_REGION.get)

CASES: list[EvalCase] = [
    # ---------------- aggregation ----------------
    EvalCase(
        "total_gross",
        "What is the total gross revenue?",
        "answerable",
        "aggregate",
        {"type": "scalar", "value": sum(_nums("gross_revenue"))},
        {"operation": "aggregate", "aggregations": [{"function": "sum", "column": "gross_revenue"}]},
    ),
    EvalCase(
        "avg_net",
        "What is the average net revenue per order?",
        "answerable",
        "aggregate",
        {"type": "scalar", "value": statistics.mean(_nums("net_revenue"))},
        {"operation": "aggregate", "aggregations": [{"function": "mean", "column": "net_revenue"}]},
    ),
    EvalCase(
        "gross_by_region",
        "Total gross revenue by region",
        "answerable",
        "group",
        {"type": "mapping", "values": _GROSS_BY_REGION},
        {
            "operation": "aggregate",
            "group_by": ["region"],
            "aggregations": [{"function": "sum", "column": "gross_revenue"}],
        },
    ),
    EvalCase(
        "bar_gross_by_region",
        "Bar chart of total gross revenue by region",
        "answerable",
        "chart",
        {"type": "mapping", "values": _GROSS_BY_REGION, "chart": True},
        {
            "operation": "aggregate",
            "group_by": ["region"],
            "aggregations": [{"function": "sum", "column": "gross_revenue"}],
            "chart_type": "bar",
        },
    ),
    EvalCase(
        "orders_per_product",
        "How many orders are there for each product?",
        "answerable",
        "group",
        {"type": "mapping", "values": _by("product")},
        {
            "operation": "aggregate",
            "group_by": ["product"],
            "aggregations": [{"function": "row_count"}],
        },
    ),
    EvalCase(
        "avg_units_by_channel",
        "Average units by channel",
        "answerable",
        "group",
        {"type": "mapping", "values": _by("channel", "units", statistics.mean)},
        {
            "operation": "aggregate",
            "group_by": ["channel"],
            "aggregations": [{"function": "mean", "column": "units"}],
        },
    ),
    EvalCase(
        "top_region",
        "Which region has the highest total gross revenue?",
        "answerable",
        "group",
        {"type": "first_row_has", "value": _BEST_REGION},
        {
            "operation": "aggregate",
            "group_by": ["region"],
            "aggregations": [{"function": "sum", "column": "gross_revenue"}],
            "sort_by": "gross_revenue_sum",
            "ascending": False,
            "limit": 1,
        },
    ),
    EvalCase(
        "distinct_products",
        "How many different products are there?",
        "answerable",
        "aggregate",
        {"type": "scalar", "value": len({r["product"] for r in ROWS})},
        {"operation": "aggregate", "aggregations": [{"function": "nunique", "column": "product"}]},
    ),
    EvalCase(
        "median_units",
        "What is the median number of units per order?",
        "answerable",
        "aggregate",
        {"type": "scalar", "value": statistics.median(_nums("units"))},
        {"operation": "aggregate", "aggregations": [{"function": "median", "column": "units"}]},
    ),
    # ---------------- filtering ----------------
    EvalCase(
        "west_orders",
        "Show the orders from the West region",
        "answerable",
        "filter",
        {"type": "row_ids", "values": _ids(lambda r: r["region"] == "West")},
        {"operation": "filter", "filters": [{"column": "region", "operator": "==", "value": "West"}]},
    ),
    EvalCase(
        "big_orders",
        "Show orders with gross revenue above 200",
        "answerable",
        "filter",
        {"type": "row_ids", "values": _ids(lambda r: float(r["gross_revenue"]) > 200)},
        {
            "operation": "filter",
            "filters": [{"column": "gross_revenue", "operator": ">", "value": 200}],
        },
    ),
    EvalCase(
        "online_gross",
        "What is the total gross revenue of online orders?",
        "answerable",
        "filter",
        {
            "type": "scalar",
            "value": sum(float(r["gross_revenue"]) for r in ROWS if r["channel"] == "Online"),
        },
        {
            "operation": "aggregate",
            "filters": [{"column": "channel", "operator": "==", "value": "Online"}],
            "aggregations": [{"function": "sum", "column": "gross_revenue"}],
        },
    ),
    EvalCase(
        "book_units5",
        "Show Book orders with at least 5 units",
        "answerable",
        "filter",
        {
            "type": "row_ids",
            "values": _ids(
                lambda r: r["product"] == "Book" and r["units"] != "" and float(r["units"]) >= 5
            ),
        },
        {
            "operation": "filter",
            "filters": [
                {"column": "product", "operator": "==", "value": "Book"},
                {"column": "units", "operator": ">=", "value": 5},
            ],
        },
    ),
    # ---------------- sorting / top-N ----------------
    EvalCase(
        "top3_net",
        "Show the top 3 orders by net revenue",
        "answerable",
        "sort",
        {"type": "column_values", "values": _top_ids("net_revenue", 3, True)},
        {"operation": "sort", "sort_by": "net_revenue", "ascending": False, "limit": 3},
    ),
    EvalCase(
        "cheapest2",
        "Show the 2 orders with the lowest gross revenue",
        "answerable",
        "sort",
        {"type": "column_values", "values": _top_ids("gross_revenue", 2, False)},
        {"operation": "sort", "sort_by": "gross_revenue", "ascending": True, "limit": 2},
    ),
    EvalCase(
        "most_units",
        "Which order has the most units?",
        "answerable",
        "sort",
        {"type": "column_values", "values": _top_ids("units", 1, True)},
        {"operation": "sort", "sort_by": "units", "ascending": False, "limit": 1},
    ),
    # ---------------- missing values ----------------
    EvalCase(
        "missing_cols",
        "Which columns have missing values?",
        "answerable",
        "missing",
        {"type": "first_row_has", "value": "units"},
        {"operation": "missing_values"},
    ),
    EvalCase(
        "missing_units_count",
        "How many orders have no units value?",
        "answerable",
        "missing",
        # Weak check on purpose: any cell equal to the count is accepted, in a table of up to 7 rows.
        {"type": "scalar", "value": _count(lambda r: r["units"] == ""), "max_rows": 7},
        {"operation": "missing_values", "columns": ["units"]},
    ),
    # ---------------- correlation / summary ----------------
    EvalCase(
        "corr_units_gross",
        "How are units and gross revenue correlated?",
        "answerable",
        "correlation",
        {"type": "scalar", "value": _correlation("units", "gross_revenue"), "max_rows": 2},
        {"operation": "correlation", "columns": ["units", "gross_revenue"]},
    ),
    EvalCase(
        "column_summary",
        "Give me a summary of every column",
        "answerable",
        "summary",
        {"type": "row_count", "value": len(ROWS[0])},
        {"operation": "column_summary"},
    ),
    # ---------------- unsupported ----------------
    EvalCase(
        "forecast",
        "Predict the gross revenue for next quarter",
        "unsupported",
        "unsupported",
        {"type": "status", "value": "unsupported"},
    ),
    EvalCase(
        "join_files",
        "Join this data with the customers file and show customer names",
        "unsupported",
        "unsupported",
        {"type": "status", "value": "unsupported"},
    ),
    # ---------------- ambiguous ----------------
    EvalCase(
        "revenue_by_region",
        "Total revenue by region",
        "ambiguous",
        "ambiguous",
        {"type": "status", "value": "clarify"},
    ),
    EvalCase(
        "best_product",
        "Which product earns the most revenue?",
        "ambiguous",
        "ambiguous",
        {"type": "status", "value": "clarify"},
    ),
]