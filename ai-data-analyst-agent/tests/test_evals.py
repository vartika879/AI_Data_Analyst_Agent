import json

import pytest

from analyst.data_loader import load_csv
from analyst.graph import run_analysis
from analyst.llm import LLMError
from evals.cases import CASES
from evals.dataset import csv_bytes
from evals.runner import CaseRecord, CountingClient, format_report, run_config, summarize
from evals.scoring import check_outcome

DF = load_csv(csv_bytes(), "eval.csv", 1024 * 1024)
BY_ID = {c.id: c for c in CASES}
BY_QUESTION = {c.question: c for c in CASES}
ANSWERABLE = [c for c in CASES if c.kind == "answerable"]
UNSUPPORTED = [c for c in CASES if c.kind == "unsupported"]
AMBIGUOUS = [c for c in CASES if c.kind == "ambiguous"]

UNSUPPORTED_REPLY = json.dumps({"operation": "unsupported", "unsupported_reason": "Not supported."})
ASK_REPLY = json.dumps({"action": "ask", "question": "Which column?", "options": ["gross_revenue", "net_revenue"]})
PROCEED_REPLY = json.dumps({"action": "proceed"})


class ScriptedClient:
    def __init__(self, *replies):
        self.replies = list(replies)

    def complete_json(self, system_prompt, user_prompt):
        if not self.replies:
            raise AssertionError("unexpected extra LLM call")
        return self.replies.pop(0)


class OracleClient:
    """A perfect fake LLM for every role: looks the question up in the cases."""

    def __init__(self, role):
        self.role = role

    def complete_json(self, system_prompt, user_prompt):
        case = BY_QUESTION[json.loads(user_prompt)["question"]]
        if self.role == "planner":
            if case.kind == "answerable":
                return json.dumps(case.reference_plan)
            return UNSUPPORTED_REPLY
        if self.role == "clarifier":
            return ASK_REPLY if case.kind == "ambiguous" else PROCEED_REPLY
        if self.role == "reviewer":
            return json.dumps({"verdict": "answers", "reason": "Looks right."})
        return json.dumps({"answer": "The result is shown in the table below."})


def run_with(case, *planner_replies, clarifier=None):
    return run_analysis(
        case.question,
        DF,
        ScriptedClient(*planner_replies),
        clarifier_client=ScriptedClient(clarifier) if clarifier else None,
    )


def rec(case_id="x", kind="answerable", **kwargs):
    return CaseRecord(config="c", case_id=case_id, kind=kind, **kwargs)


# --------------------------------------------------- the harness checks itself


@pytest.mark.parametrize("case", ANSWERABLE, ids=lambda c: c.id)
def test_reference_plan_passes_its_own_check(case):
    outcome = run_with(case, json.dumps(case.reference_plan))
    score = check_outcome(case, outcome)
    assert score.passed, score.reason


@pytest.mark.parametrize("case", UNSUPPORTED, ids=lambda c: c.id)
def test_unsupported_case_passes_when_planner_says_unsupported(case):
    assert check_outcome(case, run_with(case, UNSUPPORTED_REPLY)).passed


@pytest.mark.parametrize("case", AMBIGUOUS, ids=lambda c: c.id)
def test_ambiguous_case_passes_when_clarifier_asks(case):
    outcome = run_with(case, clarifier=ASK_REPLY)
    score = check_outcome(case, outcome)
    assert score.passed and score.asked


def test_dataset_structure():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))
    assert len(CASES) >= 20
    for case in CASES:
        assert case.kind in {"answerable", "unsupported", "ambiguous"}
        assert (case.reference_plan is not None) == (case.kind == "answerable")


def test_expected_values_match_hand_calculation():
    assert BY_ID["total_gross"].expect["value"] == 2170
    assert BY_ID["west_orders"].expect["values"] == [7, 8, 12]
    assert BY_ID["big_orders"].expect["values"] == [2, 4, 8, 11]
    assert BY_ID["book_units5"].expect["values"] == [2, 8, 11]
    assert BY_ID["missing_units_count"].expect["value"] == 1
    assert BY_ID["median_units"].expect["value"] == 8
    assert BY_ID["top_region"].expect["value"] == "South"
    assert BY_ID["top3_net"].expect["values"] == [8, 11, 4]
    assert BY_ID["cheapest2"].expect["values"] == [12, 5]
    assert BY_ID["most_units"].expect["values"] == [3]
    assert BY_ID["column_summary"].expect["value"] == 7


# --------------------------------------------------- wrong answers are caught


def test_wrong_aggregation_function_is_caught():
    case = BY_ID["gross_by_region"]
    wrong = {**case.reference_plan, "aggregations": [{"function": "mean", "column": "gross_revenue"}]}
    score = check_outcome(case, run_with(case, json.dumps(wrong)))
    assert not score.passed
    assert "no row matches" in score.reason


def test_wrong_sort_direction_is_caught_for_top_n():
    case = BY_ID["top3_net"]
    wrong = {**case.reference_plan, "ascending": True}
    score = check_outcome(case, run_with(case, json.dumps(wrong)))
    assert not score.passed
    assert "expected values" in score.reason


def test_wrong_filter_value_is_caught():
    case = BY_ID["west_orders"]
    wrong = {"operation": "filter", "filters": [{"column": "region", "operator": "==", "value": "East"}]}
    score = check_outcome(case, run_with(case, json.dumps(wrong)))
    assert not score.passed
    assert "order_id" in score.reason

def test_filter_result_without_order_id_cannot_be_verified():
    case = BY_ID["west_orders"]
    reply = {**case.reference_plan, "columns": ["region", "gross_revenue"]}
    score = check_outcome(case, run_with(case, json.dumps(reply)))
    assert not score.passed
    assert "no order_id column" in score.reason
    
def test_wrong_scalar_is_caught():
    case = BY_ID["total_gross"]
    wrong = {"operation": "aggregate", "aggregations": [{"function": "mean", "column": "gross_revenue"}]}
    score = check_outcome(case, run_with(case, json.dumps(wrong)))
    assert not score.passed
    assert "no numeric cell" in score.reason


def test_wrong_first_row_is_caught():
    case = BY_ID["top_region"]
    wrong = {**case.reference_plan, "ascending": True}
    score = check_outcome(case, run_with(case, json.dumps(wrong)))
    assert not score.passed
    assert "first row" in score.reason


def test_failed_run_is_scored_as_failure_with_status():
    case = BY_ID["total_gross"]
    bad = json.dumps({"operation": "sort", "sort_by": "no_such_column"})
    score = check_outcome(case, run_with(case, bad, bad))
    assert not score.passed
    assert score.reason.startswith("status failed")


def test_missing_chart_is_caught():
    case = BY_ID["bar_gross_by_region"]
    no_chart = {k: v for k, v in case.reference_plan.items() if k != "chart_type"}
    score = check_outcome(case, run_with(case, json.dumps(no_chart)))
    assert not score.passed
    assert "chart" in score.reason


def test_answerable_answer_to_an_unsupported_case_fails():
    case = BY_ID["forecast"]
    reply = json.dumps(BY_ID["total_gross"].reference_plan)
    assert not check_outcome(case, run_with(case, reply)).passed


def test_false_ask_on_a_clear_question_is_recorded_and_not_counted_correct():
    case = BY_ID["total_gross"]
    outcome = run_with(case, clarifier=ASK_REPLY)
    score = check_outcome(case, outcome)
    assert score.asked is True
    assert score.passed is False


# --------------------------------------------------- runner and summary


def test_summarize_counts():
    records = [
        rec("a", passed=True),
        rec("b", passed=False, status="failed"),
        rec("c", passed=False, asked=True, status="clarify"),  # false ask
        rec("d", kind="unsupported", passed=True, status="unsupported"),
        rec("e", kind="ambiguous", passed=True, asked=True, status="clarify"),
        rec("f", repair_attempts=1, passed=True, llm_calls=3, wall_seconds=2.0),
    ]
    s = summarize(records)
    assert s["cases_run"] == 6
    assert s["answerable"] == {"n": 4, "correct": 2}
    assert s["unsupported"] == {"n": 1, "correct": 1}
    assert s["ambiguous"] == {"n": 1, "asked": 1}
    assert s["false_asks"] == 1
    assert s["failed_runs"] == 1
    assert s["repairs_used"] == 1
    assert s["cost"]["llm_calls"] == 3


def test_summarize_reviewer_cross_table():
    records = [
        rec("a", passed=True, review_verdict="answers"),
        rec("b", passed=False, review_verdict="answers"),
        rec("c", passed=True, review_verdict="mismatch"),
    ]
    assert summarize(records)["reviewer"] == {
        "answers": {"n": 2, "correct": 1},
        "mismatch": {"n": 1, "correct": 1},
    }


def test_summarize_narrator_counts():
    records = [
        rec("a", narrative_ok=True),
        rec("b", narrative_error="The AI summary was discarded because it contained numbers that are not in the result: 5."),
        rec("c", narrative_error="Groq rate limit reached."),
    ]
    assert summarize(records)["narrator"] == {
        "n": 3, "accepted": 1, "grounding_rejected": 1, "other": 1,
    }


def test_summarize_empty_records_does_not_crash():
    s = summarize([])
    assert s["cases_run"] == 0
    assert s["answerable"]["n"] == 0
    assert s["reviewer"] is None and s["narrator"] is None
    assert "config" in format_report({"empty": s})


def test_counting_client_counts_calls_even_when_they_fail():
    class Flaky:
        def complete_json(self, system_prompt, user_prompt):
            raise LLMError("boom")

    client = CountingClient(Flaky())
    with pytest.raises(LLMError):
        client.complete_json("s", "u")
    assert client.calls == 1
    assert client.seconds >= 0


def test_counting_client_returns_the_reply():
    client = CountingClient(ScriptedClient("hello"))
    assert client.complete_json("s", "u") == "hello"
    assert client.calls == 1


def test_baseline_config_with_a_perfect_llm_scores_everything():
    records = run_config("baseline", CASES, DF, OracleClient)
    s = summarize(records)
    assert s["cases_run"] == len(ANSWERABLE) + len(UNSUPPORTED)  # ambiguous cases are skipped
    assert s["answerable"]["correct"] == len(ANSWERABLE)
    assert s["unsupported"]["correct"] == len(UNSUPPORTED)
    assert s["cost"]["llm_calls"] == s["cases_run"]  # one Planner call each


def test_full_config_with_a_perfect_llm_scores_everything():
    records = run_config("full", CASES, DF, OracleClient)
    s = summarize(records)
    assert s["cases_run"] == len(CASES)
    assert s["answerable"]["correct"] == len(ANSWERABLE)
    assert s["unsupported"]["correct"] == len(UNSUPPORTED)
    assert s["ambiguous"] == {"n": len(AMBIGUOUS), "asked": len(AMBIGUOUS)}
    assert s["false_asks"] == 0
    assert s["failed_runs"] == 0
    assert s["reviewer"]["answers"]["n"] == len(ANSWERABLE)
    assert s["narrator"]["accepted"] == len(ANSWERABLE)
    # answerable: 4 calls; unsupported: clarifier + planner; ambiguous: clarifier only
    assert s["cost"]["llm_calls"] == 4 * len(ANSWERABLE) + 2 * len(UNSUPPORTED) + len(AMBIGUOUS)