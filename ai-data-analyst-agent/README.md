# AI Data Analyst Agent

Upload a CSV, ask a question in plain English, and get a data-backed answer,
a result table and (optionally) a chart.

Built to learn LangGraph and agentic patterns, with safety as a design goal:
**the LLM never writes or runs code.** It only fills in a structured plan.

## How it works

```
question ──> Planner ──> Executor ──> Critic ──> Reporter ──> answer + table + chart
              (LLM)      (fixed       (checks)    (template,
                          Pandas)                  no LLM)
                 ^            |
                 └── repair ──┘   (at most one attempt, recoverable errors only)
```

1. **Planner** (Groq LLM): turns the question into an `AnalysisPlan` (Pydantic).
   Output that does not match the schema is rejected.
2. **Executor**: runs the plan with pre-written Pandas functions chosen from a
   whitelist. No `exec`, no `eval`. Plans are validated against the real
   columns and dtypes first.
3. **Critic**: re-checks what can be verified (result shape, row counts,
   sort order, filters, correlation range). It catches internal
   inconsistencies. **It does not guarantee the answer is correct** or that
   the question was understood as you meant it.
4. **Reporter**: writes the answer from the computed table with a template, so
   it cannot invent numbers.

If the plan is rejected for a recoverable reason (unknown column, wrong value
type, malformed LLM output), the problem is sent back to the Planner once.
API failures and Critic failures are not retried.

## Supported operations

Column summary, filtering, sorting, grouping + aggregation (sum, mean, median,
min, max, count, nunique, row count), missing-value analysis, correlation.
Charts (bar, line, scatter) for grouped aggregations; bar for missing values.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then edit .env and add your Groq API key
streamlit run app.py
```

Configuration (`.env`): `GROQ_API_KEY`, `GROQ_MODEL`, `MAX_UPLOAD_MB`.
Model names change over time, so check Groq's current model list if you get a
"model not found" error.

## Tests

```powershell
pytest -v
```

The tests never call the real API. The LLM is replaced by a scripted fake.

## Project layout

```
analyst/
  config.py           settings from environment variables
  data_loader.py      CSV validation and loading
  schemas.py          Pydantic plan schemas
  plan_validation.py  plan vs. real DataFrame checks
  executor.py         whitelisted Pandas operations
  llm.py              Groq client wrapper
  planner.py          question -> validated plan
  critic.py           consistency checks
  reporter.py         answer text from the result
  charts.py           Plotly figures
  graph.py            LangGraph workflow + repair
app.py                Streamlit UI
tests/                pytest suite
```

## Security notes

- API key comes from the environment only; it is never logged or shown, and
  `Settings` hides it from `repr`.
- No LLM-generated code is executed. Plans are limited to fixed choices.
- `contains` filters are literal text, never regex.
- Cell values and LLM text are escaped before being shown as markdown or in
  chart labels.
- Uploaded data lives in memory for the session only.

## Privacy

To plan a question, the app sends the **column names, types, missing counts
and up to 5 sample values per column** (40 characters each) to Groq's API.
The full file is not sent. Do not use sensitive data you are not allowed to
share with that provider.

## Known limitations

- One CSV at a time; no joins.
- Dates are treated as text.
- Filters are combined with AND only. `!=` also matches missing values.
- Results show at most 100 rows; charts at most 50 points.
- The answer text is template-based, so it is plain rather than conversational.
- The Planner can still misunderstand a question and produce a valid but
  wrong plan. Check the "How this was computed" panel.
- No authentication, database or deployment setup.