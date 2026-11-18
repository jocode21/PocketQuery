# PocketQuery — Training Log

Evidence that PocketQuery was trained from scratch, in two runs.
Final numbers below were filled in after the full run finished
(~12:26 IST, 2026-11-18) and Step 8 eval ran. Only the live app-test
screenshots 11–14 are still unconfirmed in chat.
Do not post target numbers — only the measured numbers below.

## Setup (Joel's PC, 2026-11-18)

- Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), Ryzen 7 7735HS, 16 GB RAM
- Driver 621.14, CUDA UMD 13.5 (from `nvidia-smi`, 2026-11-18)
- Python 3.12, Windows, venv `.venv`
- Model: encoder-decoder Transformer, 6 layers, d_model 256, vocab 1,291
- **Parameters: 12,231,691** (measured — see `data/training_metrics.json`
  and `TEST_REPORT.md`)
- Dataset: 16,000 examples — 12,000 SELECT / 1,500 INSERT / 1,500 UPDATE /
  1,000 DELETE. Split 12,800 train / 1,600 val / 1,600 test (seed 42).
- Database: `data/college.db` — 50 students, 8 courses, 6 faculty,
  276 marks, 276 attendance rows.

### CUDA fix — read this before trusting the full run

The shipped `requirements.txt` installs CPU-only `torch==2.15.0+cpu`,
and `train.py` silently falls back to CPU. The first full run on
2026-11-18 was started on CPU by mistake (RTX 0%, CPU ~61%) and stopped
with Ctrl+C. Fix applied in the same venv:

```bat
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Verified after the fix:

```text
2.15.0+cu132
13.2
True
NVIDIA GeForce RTX 3050 Laptop GPU
```

Do NOT re-run `pip install -r requirements.txt` for torch — it would
reinstall the CPU build. Full training was restarted only after the
check above printed `True`, and its first log line says `device=cuda`.

## Run 1 — Smoke (CPU, VM values carried over; re-checked on PC)

- Command: `python train.py --smoke` (2,000 rows, 3 epochs, batch 32, lr 3e-4)
- Code tests on PC first: **34 passed, 1 warning in 18.24s**
  (2026-11-18). The warning is PyTorch's nested-tensor prototype
  warning, not a PocketQuery error.
- Smoke metrics saved before the full run overwrote them:
  `data/training_metrics_smoke.json` (copy of `data/training_metrics.json`)

| | Smoke |
|---|---|
| Epochs / train rows | 3 / 2,000 |
| Device | cpu |
| Loss history | 11.4856, 2.1545, 1.3924 |
| Time | 782 s |
| Exact match (200 eval rows) | 0.0% (0/200) |
| SELECT execution accuracy | 1.3% (2/157) |
| Writes valid and runs | 27.9% (12/43) |

## Run 2 — Full (GPU, RTX 3050, 2026-11-18)

- Command: `python train.py --full > training_log_full.txt 2>&1`
- Started: ~12:10 IST, 2026-11-18 (after the CUDA fix above)
- First log line: `device=cuda train_rows=12800 val_rows=1600 vocab=1291`
- Early epochs (from `training_log_full.txt`):

| Epoch | Loss | Cumulative time |
|---|---|---|
| 1/30 | 2.7602 | 38 s |
| 2/30 | 0.2756 | 74 s |
| 3/30 | 0.1621 | 110 s |
| 4/30 | 0.1156 | 146 s |
| 5–30/30 | 0.09, 0.0724, 0.0607, 0.0524, 0.046, 0.0413, 0.0381, 0.0349, 0.031, 0.0293, 0.0259, 0.0243, 0.0215, 0.02, 0.0173, 0.0153, 0.0133, 0.0121, 0.0102, 0.0091, 0.0082, 0.0077, 0.007, 0.0068, 0.0064, 0.0065 | 931 s total |

- Pace: ~37 s/epoch, so ~18 minutes total; expected finish ~12:28–12:30 IST.
- Finished: epoch 30/30, ~12:26 IST (data/model.pt modified 12:25:41), 931 s training (~15.5 min; 1,034 s in data/training_metrics.json including eval)
- Final loss: 0.0065
- Model saved: `data/model.pt` — do NOT commit/push this file
  (47 MB, already in `.gitignore`).

### Full-run measured accuracy (fill in after Step 8)

Run first:

```bat
python train.py --eval-only
```

Then open `data/training_metrics.json` and copy the numbers here:

| | Full |
|---|---|
| Epochs / train rows | 30 / 12,800 |
| Device | cuda |
| Final loss | 0.0065 |
| Exact match (200 eval rows) | 90.5% (181/200) |
| SELECT execution accuracy | 99.4% (156/157) |
| Writes valid and runs | 100% (43/43) |

## App test (Step 9 — after the eval above)

- `python app.py` → http://127.0.0.1:5000
- Screenshots in `screenshots/`: 11-app-home, 12-app-select,
  13-app-write-confirm, 14-app-blocked, 15-app-metrics
- Metrics page verified (15-app-metrics.png): exact 90.5%, SELECT 99.4%,
  writes 100%. Live SELECT / INSERT / UPDATE / DELETE / blocked tests
  tried after that — Joel reported "works good!"; individual
  screenshots 11–14 not yet confirmed in chat.
- SELECT test: `show CSE students with marks above 80` — [TBD: worked / failed, SQL shown]
- Write test: update marks, confirm preview — [TBD]
- Blocked test: `drop the students table` — [TBD: blocked message shown]

## Before GitHub / LinkedIn

- [ ] README updated with the measured full numbers above (smoke vs full),
      not the 80–85% target.
- [x] Fix: README's honest-status box still says "12,167,441 parameters"
      in one line; the measured value everywhere else is **12,231,691**.
      Done 2026-11-18: box replaced with the full-training box.
- [ ] Loss curve (smoke 11.49 → 1.39; full 2.76 → [TBD]) added.
- [ ] Best 3 screenshots in the README (app home, app SELECT, full metrics).
- [ ] Commit code + metrics + this log. Do NOT push `data/model.pt`.
- [ ] LinkedIn post only after the app test above is done.
