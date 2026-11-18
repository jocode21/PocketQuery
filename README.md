# PocketQuery — a tiny transformer trained from scratch that turns English into SQL

Type a question in plain English, get the SQL — and the real result.
Built as a LinkedIn / portfolio project: the model is trained from
scratch on this machine, no pretrained weights, no API, no paid service.

> **Training — DONE on RTX 3050, 2026-11-18**
>
> - Full training: 12,800 examples, 30 epochs, device cuda, 1,034s (~17 min)
> - Model: 12,231,691 parameters
> - Loss: 2.7602 → 0.0065
> - Exact match: 90.5% (181 / 200)
> - SELECT execution: 99.4% (156 / 157)
> - Writes valid and runs: 100% (43 / 43)
> - Smoke, for comparison: exact 0%, SELECT 1.3%, writes 27.9%


## What it does

Web UI (`python app.py` → http://127.0.0.1:5000). Type free English
about the college database — there is deliberately **no suggested
prompts section** — and the app shows the generated SQL plus the real
result.

- **SELECT** runs immediately: SQL box (with Copy) + result table +
  row count + time + Download CSV.
- **INSERT / UPDATE / DELETE** are two-step: Generate → you see the
  SQL plus a preview of how many rows it would affect → you click
  **Confirm** → only then it runs, and you see affected rows.
- **Blocked always:** DROP TABLE / DROP DATABASE, TRUNCATE, ALTER,
  CREATE TABLE from a prompt. UPDATE / DELETE without a WHERE clause
  are blocked, so a wrong prompt can never wipe a table. INSERT and
  UPDATE values for marks / attendance must be 0–100.
- **Wrong prompts never crash the app:** a typo or vague question
  shows what the model assumed (the SQL it produced); an out-of-schema
  question is blocked with the list of valid tables; bad SQL shows the
  SQL tried plus the SQLite error.

## The database

SQLite file `data/college.db`, created automatically on first run.
No server, no password, no internet. Fixed schema only:

- `students(id, name, branch, year, email)`
- `courses(id, name, credits, faculty_id)`
- `marks(student_id, course_id, marks)`
- `faculty(id, name, department)`
- `attendance(student_id, course_id, percent)`

Seed data: 50 fake students (CSE, ECE, ME, CE), 8 courses, 6 faculty,
~276 marks and ~276 attendance rows. The **Reset database** button
deletes `college.db` and recreates it with the same seed data.

## The model

A small encoder-decoder Transformer, trained from random
initialisation — no pretrained weights, no lfm3, no Ollama:

- d_model 256, 8 heads, 3 encoder + 3 decoder layers (6 layers),
  feed-forward 3072, dropout 0.1, context 256 tokens
- **12,231,691 parameters** (measured, see `TEST_REPORT.md`)
- Word-level tokenizer built from the dataset (vocab 1,291 tokens)
- English → SQL, greedy decoding

## The dataset

`python generate_data.py` builds ~16,000 examples from templates +
paraphrases + slot filling (not hand-written one by one):

| Operation | Examples |
|---|---|
| SELECT | 12,000 |
| INSERT | 1,500 |
| UPDATE | 1,500 |
| DELETE | 1,000 |
| **Total** | **16,000** |

Split (shuffled, seed 42): 12,800 train / 1,600 val / 1,600 test.
Format: JSONL `{"input": "...", "sql": "...", "op": "..."}`.
Every SELECT in the dataset was verified to execute on the real
`college.db`, and the tokenizer round-trips 16,000/16,000 SQL strings
exactly.

## Run it on Joel's Windows PC (RTX 3050)

```bat
:: 1. Python 3.12, then in the project folder:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

:: 2. Full training (GPU). About 17 minutes, measured on the
::    RTX 3050 (1,034s for 30 epochs).
python train.py --full

:: 3. Check the metrics it measured:
::    open data\training_metrics.json, or visit /metrics in the app

:: 4. Run the web app:
python app.py
::    then open http://127.0.0.1:5000 in a browser
```

For a quick check without the GPU wait, `python train.py --smoke`
trains 2,000 rows for 3 epochs on CPU. The CLI mode is
`python main.py` (interactive) or `python main.py "show CSE students"`.

## Tests

```bat
python -m pytest tests/ -q
```

33 tests (validator, DB, dataset, model, Flask app) plus 1 inference
test that runs only when `data/model.pt` exists. See `TEST_REPORT.md`
for the exact counts and commands run during verification.

## Project layout

```
app.py             Flask web app (routes, confirm flow, CSV, reset)
main.py            CLI mode (interactive + one-shot)
model.py           Tiny transformer + tokenizer + greedy decoding
train.py           Training (--smoke / --full / --eval-only)
generate_data.py   Dataset generator (16,000 examples, 80/10/10 split)
db.py              SQLite creation, seeding, run_sql
validate.py        Safety rules: allowed ops, WHERE-required, ranges
requirements.txt   torch (CPU) + flask + pytest + numpy
templates/         index.html, metrics.html (no suggested prompts)
static/style.css   Plain styling, no framework
tests/             pytest suite (validator, db, dataset, model, app)
data/              college.db, dataset/train/val/test.jsonl, vocab.json
                   model.pt + training_metrics.json (after training)
TEST_REPORT.md     Exact verification results for this build
```

## Limits (honest)

- Fixed schema only. It cannot answer about another database, and it
  cannot create a database from a prompt. That would need a large
  pretrained model that reads the schema at run time — a different
  project, and not from scratch.
- Accuracy is measured on held-out validation questions. Complex
  3-table joins are the weakest pattern. Vague questions (e.g. "good
  students") make the model guess a threshold; the app shows you the
  SQL it assumed so you can see the guess.
- Full training measured on the RTX 3050, 2026-11-18: exact match
  90.5% (181 / 200), SELECT execution 99.4% (156 / 157), writes
  100% (43 / 43). The 80–85% SELECT / 75–80% writes targets were
  exceeded.
