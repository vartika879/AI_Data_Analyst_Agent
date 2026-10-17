"""Evaluation runner: runs the cases through the real pipeline and reports.

    python -m evals.runner --configs baseline,full

This makes REAL Groq API calls (see the cost notes in the README).
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from analyst.config import load_settings
from analyst.data_loader import load_csv
from analyst.graph import run_analysis
from analyst.llm import GroqClient, LLMClient
from evals.cases import CASES, EvalCase
from evals.dataset import csv_bytes
from evals.scoring import check_outcome

ROLES = ("clarifier", "reviewer", "narrator")
CONFIGS: dict[str, frozenset[str]] = {
    "baseline": frozenset(),
    "clarifier": frozenset({"clarifier"}),
    "reviewer": frozenset({"reviewer"}),
    "narrator": frozenset({"narrator"}),
    "full": frozenset(ROLES),
}


class CountingClient:
    """Wraps an LLM client and counts calls and time (failed calls count too)."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.calls = 0
        self.seconds = 0.0

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        start = time.perf_counter()
        self.calls += 1
        try:
            return self._inner.complete_json(system_prompt, user_prompt)
        finally:
            self.seconds += time.perf_counter() - start


@dataclass(frozen=True)
class CaseRecord:
    config: str
    case_id: str
    kind: str
    category: str = ""
    repeat: int = 0
    passed: bool = False
    reason: str | None = None
    asked: bool = False
    status: str = "success"
    repair_attempts: int = 0
    review_verdict: str | None = None
    narrative_ok: bool = False
    narrative_error: str | None = None
    llm_calls: int = 0
    llm_seconds: float = 0.0
    wall_seconds: float = 0.0


def run_case(
    case: EvalCase, df, roles: frozenset[str], client_factory, config: str = "", repeat: int = 0
) -> CaseRecord:
    planner = CountingClient(client_factory("planner"))
    helpers = {role: CountingClient(client_factory(role)) for role in roles}

    start = time.perf_counter()
    outcome = run_analysis(
        case.question,
        df,
        planner,
        reviewer_client=helpers.get("reviewer"),
        narrator_client=helpers.get("narrator"),
        clarifier_client=helpers.get("clarifier"),
    )
    wall = time.perf_counter() - start

    score = check_outcome(case, outcome)
    clients = [planner, *helpers.values()]
    return CaseRecord(
        config=config,
        case_id=case.id,
        kind=case.kind,
        category=case.category,
        repeat=repeat,
        passed=score.passed,
        reason=score.reason,
        asked=score.asked,
        status=outcome.status,
        repair_attempts=outcome.repair_attempts,
        review_verdict=outcome.review.verdict if outcome.review else None,
        narrative_ok=outcome.narrative is not None,
        narrative_error=(outcome.narrative_error or "")[:120] or None,
        llm_calls=sum(c.calls for c in clients),
        llm_seconds=sum(c.seconds for c in clients),
        wall_seconds=wall,
    )


def run_config(
    config: str,
    cases: list[EvalCase],
    df,
    client_factory,
    delay: float = 0.0,
    repeats: int = 1,
    on_record=None,
) -> list[CaseRecord]:
    roles = CONFIGS[config]
    records = []
    for repeat in range(repeats):
        for case in cases:
            # Ambiguous cases can only be scored if a Clarifier exists.
            if case.kind == "ambiguous" and "clarifier" not in roles:
                continue
            record = run_case(case, df, roles, client_factory, config, repeat)
            records.append(record)
            if on_record:
                on_record(record)
            if delay:
                time.sleep(delay)
    return records


def summarize(records: list[CaseRecord]) -> dict:
    answerable = [r for r in records if r.kind == "answerable"]
    unsupported = [r for r in records if r.kind == "unsupported"]
    ambiguous = [r for r in records if r.kind == "ambiguous"]
    n = len(records)

    summary = {
        "cases_run": n,
        "answerable": {"n": len(answerable), "correct": sum(r.passed for r in answerable)},
        "unsupported": {"n": len(unsupported), "correct": sum(r.passed for r in unsupported)},
        "ambiguous": {"n": len(ambiguous), "asked": sum(r.asked for r in ambiguous)},
        "false_asks": sum(r.asked for r in answerable + unsupported),
        "failed_runs": sum(r.status == "failed" for r in records),
        "repairs_used": sum(r.repair_attempts > 0 for r in records),
        "reviewer": None,
        "narrator": None,
        "cost": {
            "llm_calls": sum(r.llm_calls for r in records),
            "calls_per_case": (sum(r.llm_calls for r in records) / n) if n else 0.0,
            "seconds_per_case": (sum(r.wall_seconds for r in records) / n) if n else 0.0,
        },
    }

    if any(r.review_verdict for r in answerable):
        by_verdict: dict[str, dict] = {}
        for r in answerable:
            if r.review_verdict:
                entry = by_verdict.setdefault(r.review_verdict, {"n": 0, "correct": 0})
                entry["n"] += 1
                entry["correct"] += int(r.passed)
        summary["reviewer"] = by_verdict

    if any(r.narrative_ok or r.narrative_error for r in records):
        successes = [r for r in answerable if r.status == "success"]
        accepted = sum(r.narrative_ok for r in successes)
        rejected = sum(
            1 for r in successes if r.narrative_error and "not in the result" in r.narrative_error
        )
        summary["narrator"] = {
            "n": len(successes),
            "accepted": accepted,
            "grounding_rejected": rejected,
            "other": len(successes) - accepted - rejected,
        }
    return summary


def _frac(part: int, whole: int) -> str:
    return "-" if whole == 0 else f"{part}/{whole} ({part / whole:.0%})"


def format_report(summaries: dict[str, dict]) -> str:
    header = [
        "config", "answerable ok", "unsupported ok", "ambiguous asked",
        "false asks", "failed", "repairs", "calls/case", "sec/case",
    ]
    rows = [header]
    for name, s in summaries.items():
        rows.append(
            [
                name,
                _frac(s["answerable"]["correct"], s["answerable"]["n"]),
                _frac(s["unsupported"]["correct"], s["unsupported"]["n"]),
                _frac(s["ambiguous"]["asked"], s["ambiguous"]["n"]),
                str(s["false_asks"]),
                str(s["failed_runs"]),
                str(s["repairs_used"]),
                f"{s['cost']['calls_per_case']:.1f}",
                f"{s['cost']['seconds_per_case']:.1f}",
            ]
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths)) for r in rows]

    for name, s in summaries.items():
        if s["reviewer"]:
            lines += ["", f"[{name}] Reviewer verdicts on answerable cases (n, final result correct):"]
            for verdict, e in sorted(s["reviewer"].items()):
                lines.append(f"  {verdict:<9} n={e['n']}  correct={e['correct']}")
        if s["narrator"]:
            nar = s["narrator"]
            lines += [
                "",
                f"[{name}] Narrator on {nar['n']} successful answerable cases: "
                f"accepted={nar['accepted']}  grounding_rejected={nar['grounding_rejected']}  "
                f"other(api/format/skipped)={nar['other']}",
            ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the analyst pipeline with real LLM calls.")
    parser.add_argument("--configs", default="baseline,full", help=f"comma list of: {', '.join(CONFIGS)}")
    parser.add_argument("--repeats", type=int, default=1, help="run every case this many times")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to wait between cases")
    parser.add_argument("--limit", type=int, default=None, help="only the first N cases")
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        parser.error(f"unknown config(s): {', '.join(unknown)}")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    settings = load_settings()
    if not settings.has_api_key:
        print("GROQ_API_KEY is not set. Add it to your .env file.")
        return 1

    cases = CASES[: args.limit] if args.limit else CASES
    df = load_csv(csv_bytes(), "eval.csv", settings.max_upload_bytes)
    models = {
        "planner": settings.groq_model,
        "clarifier": settings.clarifier_model,
        "reviewer": settings.reviewer_model,
        "narrator": settings.narrator_model,
    }

    def factory(role: str) -> LLMClient:
        return GroqClient(settings.groq_api_key, models[role])

    estimate = sum(
        sum(1 + len(CONFIGS[n]) for c in cases if not (c.kind == "ambiguous" and "clarifier" not in CONFIGS[n]))
        for n in names
    ) * args.repeats
    print(f"Models: {models}")
    print(f"About {estimate} LLM calls (more if repairs happen). Check your Groq rate limits.\n")

    summaries, all_records = {}, []
    for name in names:
        def show(record: CaseRecord, name=name) -> None:
            mark = "PASS" if record.passed else "FAIL"
            extra = "" if record.passed else f"  ({record.reason})"
            print(f"[{name}] {record.case_id:<22} {mark}{extra}", flush=True)

        records = run_config(name, cases, df, factory, args.delay, args.repeats, show)
        summaries[name] = summarize(records)
        all_records += records

    print("\n" + format_report(summaries))

    out_dir = Path("evals/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(
        json.dumps({"summaries": summaries, "records": [asdict(r) for r in all_records]}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved details to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())