"""Privacy policy: decides what data the LLM agents are allowed to see.

Default `PrivacyPolicy()` keeps the old behaviour (nothing hidden), so existing
code keeps working. `PrivacyPolicy.strict()` is the privacy mode:

- no sample values are sent (only column names, types and counts),
- no result rows are sent to the Reviewer / Narrator (the Narrator is skipped),
- optional "hidden" columns are not shown to the LLM at all, and a question
  that names one is rejected locally, BEFORE any LLM call.

The real calculation always runs locally on the full DataFrame. The user
always sees the real table on their own screen.

What this does NOT do (see README "Privacy mode"):
- The question text is sent to the LLM as typed. It is not masked or edited.
- Column NAMES that are not hidden are sent, because the question uses them.
- Hidden-column detection matches the column NAME in the question (any case,
  spaces/underscores/hyphens treated alike, simple plural "s"). It does not
  understand synonyms ("pay" for "salary"), typos or abbreviations.
- Column names that contain no letter or digit (for example "%" or "#") are
  not matched in text, because they would match almost every question.
"""

import re
import unicodedata
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache

from analyst.schemas import AnalysisPlan

HIDDEN_PLACEHOLDER = "[hidden]"

_SEPARATORS = r"[\s_\-]+"  # "first name", "first_name" and "first-name" are the same name
_BRACKETS = re.compile(r"\s*[\(\[\{][^\)\]\}]*[\)\]\}]")  # "salary (USD)" -> "salary"


class HiddenColumnError(ValueError):
    """A question names a hidden column, so it must not be sent to the LLM.

    The names are in `.columns` (for the user's screen only). `str(exc)` is
    generic on purpose, so it is safe even if it is logged or forwarded.
    """

    def __init__(self, columns) -> None:
        super().__init__("The question names a column that is hidden from the AI.")
        self.columns = tuple(columns)


def _clean(text: str) -> str:
    """Compatibility-normalise and drop invisible format characters (e.g. zero-width)."""
    text = unicodedata.normalize("NFKC", str(text))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _variants(name: str) -> list[str]:
    base = _clean(name).strip()
    variants = [base]
    without_brackets = _BRACKETS.sub("", base).strip()
    if without_brackets and without_brackets != base:
        variants.append(without_brackets)
    return variants


def _pattern(variant: str) -> str | None:
    """Regex for one name variant, or None if it cannot be matched safely."""
    tokens = [t for t in re.split(_SEPARATORS, variant) if t]
    if not tokens or not any(ch.isalnum() for ch in variant):
        return None  # empty, or only punctuation such as "%"
    body = _SEPARATORS.join(re.escape(t) for t in tokens)
    # Word boundaries only where the name itself starts/ends with a word
    # character: "age" must not match inside "average" or "manage".
    start = r"(?<!\w)" if re.match(r"\w", tokens[0][0]) else ""
    last = tokens[-1][-1]
    end = ""
    if re.match(r"\w", last):
        end = (r"s?" if last.isalpha() else "") + r"(?!\w)"
    return start + body + end


@lru_cache(maxsize=64)
def _compiled(names: frozenset[str]) -> tuple:
    """(name, regex) per hidden column, plus one regex for redaction (longest first)."""
    per_name = []
    alternatives: list[tuple[int, str]] = []
    for name in sorted(names):
        patterns = [p for v in _variants(name) if (p := _pattern(v)) is not None]
        for v in _variants(name):
            p = _pattern(v)
            if p is not None:
                alternatives.append((len(v), p))
        if patterns:
            per_name.append((name, re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)))
    alternatives.sort(key=lambda item: item[0], reverse=True)
    combined = (
        re.compile("|".join(f"(?:{p})" for _, p in alternatives), re.IGNORECASE)
        if alternatives
        else None
    )
    return tuple(per_name), combined


def find_hidden_mentions(text: str, hidden_columns) -> list[str]:
    """Hidden column names that `text` refers to (sorted). Empty list = none."""
    names = frozenset(str(c) for c in hidden_columns)
    if not names:
        return []
    per_name, _ = _compiled(names)
    cleaned = _clean(text)
    return sorted(name for name, rx in per_name if rx.search(cleaned))


@dataclass(frozen=True)
class PrivacyPolicy:
    share_samples: bool = True
    share_result_rows: bool = True
    hidden_columns: frozenset[str] = frozenset()

    @classmethod
    def strict(cls, hidden_columns=()) -> "PrivacyPolicy":
        return cls(
            share_samples=False,
            share_result_rows=False,
            hidden_columns=frozenset(str(c) for c in hidden_columns),
        )

    @property
    def rows_allowed(self) -> bool:
        """Result rows may go to the LLM only if nothing is hidden and sharing is on."""
        return self.share_result_rows and not self.hidden_columns

    @property
    def limits_sharing(self) -> bool:
        """True if this policy holds back anything from the LLM."""
        return not (self.share_samples and self.rows_allowed)

    def visible(self, column) -> bool:
        return str(column) not in self.hidden_columns

    def redact(self, text: str) -> str:
        """Replace hidden column names in `text` with a placeholder.

        Case-insensitive, whole-name matching (so "age" is not replaced inside
        "average"). Longer names win over shorter overlapping ones. The text is
        compatibility-normalised first, so it may change slightly.
        """
        if not self.hidden_columns:
            return text
        _, combined = _compiled(self.hidden_columns)
        if combined is None:
            return text
        return combined.sub(HIDDEN_PLACEHOLDER, _clean(text))


def ensure_no_hidden_mentions(text: str, policy: "PrivacyPolicy | None") -> None:
    """Raise HiddenColumnError if `text` names a hidden column. Call before any LLM call."""
    if policy is None or not policy.hidden_columns:
        return
    found = find_hidden_mentions(text, policy.hidden_columns)
    if found:
        raise HiddenColumnError(found)


def hidden_columns_used(plan: AnalysisPlan, policy: PrivacyPolicy) -> list[str]:
    """Hidden columns that the plan refers to (the LLM guessed them from the question)."""
    if not policy.hidden_columns:
        return []
    used = list(plan.columns) + list(plan.group_by)
    used += [c.column for c in plan.filters]
    used += [a.column for a in plan.aggregations if a.column]
    if plan.sort_by:
        used.append(plan.sort_by)
    return sorted({c for c in used if c in policy.hidden_columns})


def tracing_disabled(policy: "PrivacyPolicy | None"):
    """Context manager: switch LangSmith tracing off while a limiting policy is active.

    langgraph installs `langsmith`. If LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 is set
    in the environment, the graph state, which includes the whole DataFrame, could be
    uploaded to LangSmith. Under a limiting policy we turn tracing off for the run.
    """
    if policy is None or not policy.limits_sharing:
        return nullcontext()
    try:
        from langsmith import tracing_context

        return tracing_context(enabled=False)
    except (ImportError, TypeError) as exc:
        # Fail closed: the graph state contains the DataFrame, so continuing
        # without confirmed tracing suppression could disclose uploaded data.
        raise RuntimeError(
            "Privacy mode blocked this analysis because LangSmith tracing "
            "could not be safely disabled."
        ) from exc
