# Budget dashboard

A local-first budgeting dashboard in Python, Streamlit, and Plotly. It turns
bank/credit-card CSV exports into a monthly budget check: how much came in,
what went out by category, whether each category stayed on target, what was
set aside for savings, and what is left over.

## Screens

- **Budget** — the month at a glance: income, spend, and net; spend vs. target
  per category; the selected month's transactions.
- **Trends** — a category x month grid and per-category spend trends with the
  budget line overlaid, for all loaded months at once.
- **Transactions** — every row filtered by month, category, and spend/income
  kind, with CSV export.
- **Savings** — monthly contributions per bucket, drawn down by actual category
  spending, with goal progress and a cash-pool reconciliation.

## Quickstart

```bash
git clone <this repo>
cd budget
conda env create -f environment.yml
conda activate budget
streamlit run app.py
```

Then open http://localhost:8501 in a browser. On Windows you can also
double-click `run_budget.bat`, which opens the browser for you (edit the
`BUDGET_PYTHON` path at the top of that file if your conda env lives
elsewhere).

## Data

The repo ships with `sample_transactions.csv`: a full synthetic year
(January 1 – December 31, 2026) covering every budget category, with no real
personal data. The dashboard runs on it out of the box.

To use your own data:

1. Export a CSV export from your bank or credit card (see
   `budget/ingest.py` for the expected columns).
2. Drop it into `data/raw/` (picked up on the next run) or upload it in the
   sidebar.
3. Budget targets and savings buckets are edited in the app (Budgets and
   Savings pages, saved to `budget/rules/*.yaml`) or by hand in those files.

Everything runs locally — no data leaves the machine.

## Tests

```bash
conda run -n budget python -m pytest tests -q --no-header -p no:cacheprovider
```

The suite (298 tests) covers parsing/ingest edge cases, the monthly and
yearly rollups, chart behavior, and the full app via Streamlit's `AppTest`.

## Layout

- `app.py` — the Streamlit app (views, filters, chart wire-up).
- `budget/` — the logic: `ingest.py` (CSV parsing), `reports.py`
  (summaries, budget variance, savings buckets), `presentation.py`
  (chart/tool primitives), `rules/` (budget targets and buckets as YAML).
- `tests/` — pytest suite for all of the above.

## License

MIT. See [LICENSE](LICENSE).