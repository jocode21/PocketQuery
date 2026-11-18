# PocketQuery — Test Report (v1, verified 2026-11-18 on this VM)

How to reproduce everything below (venv used on this VM):
```
~/workspace/.pocketquery-venv/bin/python generate_data.py
~/workspace/.pocketquery-venv/bin/python db.py
~/workspace/.pocketquery-venv/bin/python -m pytest tests/ -q
~/workspace/.pocketquery-venv/bin/python train.py --smoke
```
On Joel's PC the same commands run with the project venv
(`python -m venv .venv`, `pip install -r requirements.txt`).

## 1. Dataset generation — PASS

Command: `python generate_data.py`

| Check | Result |
|---|---|
| Total examples | 16,000 |
| SELECT / INSERT / UPDATE / DELETE | 12,000 / 1,500 / 1,500 / 1,000 |
| Split train / val / test (80/10/10) | 12,800 / 1,600 / 1,600, no overlap between splits |
| Vocab size | 1,291 tokens (1,287 distinct + 4 specials) |
| Every SELECT executes on the real `college.db` | 12,000 / 12,000, 0 errors |
| No DROP / TRUNCATE / ALTER / CREATE in dataset | 0 found |
| Every UPDATE / DELETE has a WHERE clause | 100% |

## 2. Database creation — PASS

Command: `python db.py`

| Table | Rows |
|---|---|
| students | 50 |
| courses | 8 |
| marks | 276 |
| faculty | 6 |
| attendance | 276 |

Branches are exactly CSE, ECE, ME, CE. `data/college.db` is
recreated automatically if deleted.

## 3. Tokenizer round-trip — PASS

`encode` then `decode` on every SQL string in the dataset:
**16,000 / 16,000 exact.** (Four detokenize bugs were found and
fixed on the way there: dots in `students.name`, emails like
`joseph487@aisat.ac.in`, `Prof. George` vs `students.name`, and
`>=` split into `> =`.)

## 4. Model — PASS (structure)

- Architecture: d_model 256, 8 heads, 3 encoder + 3 decoder layers,
  feed-forward 3072, dropout 0.1, context 256.
- **Parameters: 12,231,691** (in the locked 10–20M range).
- Forward pass shape check: `(2, 8, 1291)` logits — PASS.
- `get_model()` without a checkpoint raises a friendly
  `FileNotFoundError` telling the user to run `train.py` — PASS.

## 5. Validator self-test — PASS (15/15)

Command: `python validate.py` — 15 cases, 0 failures. Covers:
allows SELECT/INSERT/UPDATE/DELETE; blocks DROP, TRUNCATE, ALTER,
CREATE, multi-statement input; blocks UPDATE/DELETE without WHERE;
blocks unknown tables/columns; blocks marks outside 0–100 (and
allows exactly 0 and 100); blocks empty input.

## 6. Pytest suite — 34 PASSED, 0 failed

Command: `python -m pytest tests/ -q`
Result: **34 passed, 0 failed** (one warning only: a harmless
PyTorch nested-tensor prototype warning from `nn.Transformer`).

| File | Tests | What is covered |
|---|---|---|
| test_validate.py | 8 | All validator rules above |
| test_db.py | 3 | DB creation, counts, branches, run_sql SELECT + write |
| test_dataset.py | 7 | Size, op counts, splits, format, no DROP, WHERE rule, all SELECTs run, vocab coverage |
| test_model.py | 4 | Param count in range, forward shapes, round-trip, missing checkpoint |
| test_app.py | 11 | Home page (200, no suggested-prompts text), SELECT returns SQL+rows, DROP blocked (intent and SQL), unknown table blocked, DELETE-without-WHERE blocked, write needs confirmation then executes, confirm-with-nothing-pending fails, CSV export, reset recreates DB, history recorded |
| test_inference.py | 1 | Real-model generate() for one prompt of each operation (runs only when `data/model.pt` exists) |

## 7. Smoke training — PASS (pipeline), accuracy LOW by design

Command: `python train.py --smoke` (2,000 rows, 3 epochs, batch 32,
AdamW lr 3e-4, CPU, 782 s)

| Epoch | Loss |
|---|---|
| 1 | 11.4856 |
| 2 | 2.1545 |
| 3 | 1.3924 |

Loss decreases every epoch, `data/model.pt` (49 MB) saved.
Evaluation on 200 held-out validation rows (greedy decoding):

| Metric | Measured (smoke) |
|---|---|
| Exact match | **0 / 200 (0.0%)** |
| SELECT execution accuracy | **2 / 157 (1.3%)** |
| Writes producing valid SQL that runs | **12 / 43 (27.9%)** |

These numbers are the honest smoke result: 3 epochs on 2,000 rows
on CPU. The model has learned SQL *structure* (correct tables,
JOINs, ORDER BY) but drops values (e.g. `student_id = AND`,
wrong numbers). **Do not present these as full-training accuracy.**
Full training (`python train.py --full`: 12,800 rows, 30 epochs,
RTX 3050) has not been run; its accuracy is unmeasured.

## 8. Real-model end-to-end check (smoke model, Flask test client)

Five prompts through `POST /ask`, after the EXPLAIN dry-run fix:

| Prompt | Result |
|---|---|
| SELECT "show CSE students" | Friendly error: "The SQL the model produced does not run: near CSE" — no crash, nothing executed |
| INSERT "add a new student ..." | Validator blocks it: unknown column `aisat` — no crash |
| UPDATE "update marks of student 1 in course 2 to 88" | Valid SQL, preview says 1 row would be affected, waits for Confirm (values are wrong — smoke-model accuracy, not an app bug) |
| DELETE "delete student with id 77" | Blocked by EXPLAIN dry-run: "does not run: near AND" — **not** offered for confirmation |
| BAD "drop the students table" | Blocked before the model: "That operation is blocked" |

## 9. Live server check — PASS

Started `python app.py`, curled it, stopped it:
- `GET /` → 200, page contains "PocketQuery", 0 occurrences of
  "suggested".
- `POST /ask` (JSON) → 200, friendly JSON error for the smoke
  model's bad SQL.
- `GET /metrics` → 200.

## 10. CLI check — PASS

`python main.py "show CSE students"` → prints the SQL the model
produced plus "SQLite error: near CSE", exits 0, no crash, no
traceback. DB row counts unchanged afterwards (50 / 8 / 276 / 6 / 276).

## Bugs found and fixed during this build

1. Prior partial work: dataset was only 5,141 rows with empty
   val/test files (hard-coded split slices) → generator rewritten,
   now 16,000 with computed 80/10/10 splits.
2. Prior partial work: 5 branches including EEE, against the locked
   4-branch scope → fixed in `db.py` and `generate_data.py`.
3. Validator accepted `SELECT foo FROM students` (no unqualified
   column check) → added.
4. Validator allowed UPDATE/DELETE without WHERE → now blocked.
5. Validator did not check marks 0–100 → added for SET and INSERT.
6. Generator targets summed to 12,000, not 16,000 → fixed to
   12,000 / 1,500 / 1,500 / 1,000.
7. Detokenize round-trip failed on dots, emails, `Prof.`/`Dr.`,
   and `>=` → fixed; now 16,000/16,000.
8. `pip install` failed twice: the PyTorch index has no Flask, and
   `/tmp` is a 512 MB tmpfs (torch did not fit, Errno 28) → venv
   moved to the home disk, torch and Flask installed separately.
9. `model.py` raised a mask-type UserWarning → boolean causal mask.
10. Vocab read 1,041 in an early run vs 1,291 on re-run → investigated:
    the generator is deterministic (re-run produces byte-identical
    files); the correct values are vocab 1,291 and 12,231,691
    parameters. README corrected.
11. `app.py` CSV export unpacked 3 values from a 4-tuple → fixed.
12. App offered invalid DELETE SQL for confirmation (preview failed
    silently) → added an EXPLAIN dry-run for all generated SQL plus
    a preview-failure error.
13. One bad edit duplicated a code block in `app.py` → caught by
    reading the file, region rewritten cleanly; pytest re-run green.
14. `test_app.py` posted `/confirm` as a form and expected JSON →
    fixed the test to post JSON (form posts correctly return HTML).

## Known limits (not bugs)

- Fixed schema only; cannot answer about another database or create
  one from a prompt (needs a large pretrained model — different
  project, not from scratch).
- Smoke-model accuracy is low (see §7). Full-training accuracy on
  the RTX 3050 is unmeasured until `train.py --full` runs.
- `data/model.pt` (49 MB) is a smoke checkpoint. Delete it before
  full training or let `--full` overwrite it.
- The venv used for verification lives at
  `~/workspace/.pocketquery-venv` (outside the repo). On Joel's PC,
  create a fresh venv inside the project folder as the README says.
