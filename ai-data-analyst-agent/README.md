# AI Data Analyst Agent

Upload a CSV, ask a question in plain language, get a table, a chart and a short written answer.
The LLM never writes code. It only picks an operation from a fixed list, and my own
Pandas functions do the real calculation.

> **Live demo:** _add your Streamlit link here_ (limited to 10 analyses per session)
>
> Add screenshots here, e.g. `![demo](docs/demo.png)`

## How it works

```
question -> [Clarifier] -> Planner -> Executor -> Critic -> [Reviewer] -> Reporter -> [Narrator]
```

| Step | What it does | LLM? |
|---|---|---|
| Clarifier | Asks ONE question (with clickable options) if the question is truly ambiguous, e.g. "revenue" when both `gross_revenue` and `net_revenue` exist | yes |
| Planner | Turns the question into a small JSON plan (`aggregate`, `filter`, `sort`, `correlation`, ...) or says it is `unsupported` | yes |
| Executor | Runs the validated plan with fixed, pre-written Pandas code. No `exec`/`eval` | no |
| Critic | Re-checks facts about the result (row counts, sort order, filters) | no |
| Reviewer | Judges whether the result answers the question. Advisory, can trigger one repair | yes |
| Reporter | Builds the table, chart and template report | no |
| Narrator | Writes a plain-language summary. It is dropped if any number in it cannot be traced to the data | yes |

Built with LangGraph (workflow), Pydantic (plan validation), Pandas, Plotly and Streamlit.
Each agent can use a different model and provider, with optional fallbacks and two API keys.

## Design choices

- **No generated code.** Plans are validated, and operations and functions come from whitelists, so a prompt injection cannot run anything.
- **Fail open and fail safe.** A broken Clarifier never blocks a question. A Narrator summary with an untraceable number is thrown away.
- **Privacy mode.** Sends only column names and types (no sample values, no result rows), supports hidden columns, and rejects a question that names a hidden column before any LLM call. The calculation always runs locally. Limits: the question text is sent as typed, and hidden-column detection matches names only, not synonyms.
- **One repair budget** shared by the Critic, Reviewer and planner errors.

## Supported operations

Column summary, filter, sort, aggregate (sum, mean, median, min, max, count, nunique, row count, with optional group by), missing values, correlation, and bar/line/scatter charts on one table.
Not supported: forecasting, joins with other files, date arithmetic, changing data, outside knowledge.

## Evaluation

A test set in `evals/` has 24 questions: answerable, unsupported and ambiguous.
Final run (`--configs full --repeats 1`, 24 cases) with `gpt-oss-120b` as planner and `gpt-oss-20b` for the other agents, via Groq:

| Metric | Result |
|---|---|
| Answerable questions correct | 20/20 |
| Unsupported correctly refused | 2/2 |
| Ambiguous questions correctly clarified | 2/2 |
| Unnecessary clarification questions | 0 |
| Reviewer agreed with the result | 20/20 |
| Narrator summaries accepted | 20/20 |
| LLM calls / time per case | 3.6 calls, 2.8 s |

**Bug found by the eval:** early runs had the Clarifier asking questions about unsupported requests ("predict next quarter", "join with customers file") instead of letting the Planner refuse them (0/6 unsupported). The Clarifier did not know the tool's limits. Telling it what the tool can and cannot do fixed it (6/6, no false asks).

**Bug found by manual testing:** with Privacy mode on, the Planner sees no sample values, so "net revenue of pen" filtered on `pen` while the data says `Pen`. Text filters were case-sensitive, so 0 rows matched and the answer was a plausible-looking `0` (the real answer is 695). Fixes: text equality now ignores case and spaces, the Critic warns when a filter matches 0 rows, and the Planner prompt treats "X of Y" as one aggregated number. All three have unit tests.

**Honest note:** a baseline with only the Planner also scored 100% on the answerable questions, so this test set is too easy to show that the Reviewer or Narrator improve accuracy. They add explanation and grounding checks, at about 3.5x the calls. Harder cases are the next step.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # then add GROQ_API_KEY
streamlit run app.py
```

Tests: `python -m pytest -q`
Evals: `python -m evals.runner --configs baseline,full --repeats 3 --delay 10`

## Limitations

- One table at a time, fixed list of operations.
- Free-tier Groq limits apply per organization and per model, so a second key only helps if it comes from a different organization. The app retries temporary errors. On a heavy testing day the limit was hit (155 retries in one eval run) and the Clarifier and Reviewer became unavailable; the app then still answers the question using the Planner alone, but without clarification or review. Eval numbers above come from a clean run with no retries.
- The Critic and Reviewer check consistency, not truth. The Reviewer is advisory.
- Tested on a small synthetic sales dataset.

## Project layout

```
analyst/   clarifier, planner, executor, critic, reviewer, narrator, reporter, graph, privacy, llm
evals/     test cases, scoring, runner, provider check
tests/     unit and end-to-end tests
app.py     Streamlit UI
```