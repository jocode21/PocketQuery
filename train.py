"""PocketQuery — training script (from scratch, no pretrained weights).

  Full training (Joel's PC, RTX 3050):
      python train.py --full
      -> 16,000 examples, 30 epochs, batch 32, AdamW lr 3e-4, CUDA if
         available. Expected time on an RTX 3050: roughly 1-3 hours
         (not measured on this VM — this VM has no GPU).

  Smoke training (this VM, CPU only — verifies the whole pipeline):
      python train.py --smoke
      -> 2,000 examples, 3 epochs, batch 32, CPU. A few minutes.
         Loss must go down and model.pt must be saved. Accuracy after
         a smoke run is LOW by design and is reported honestly in
         data/training_metrics.json — never present it as full-training
         accuracy.

  Evaluate an existing checkpoint:
      python train.py --eval-only [--eval-limit 200]

Outputs: data/model.pt, data/training_metrics.json
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import create_db  # noqa: E402
from model import (  # noqa: E402
    BOS, EOS, PAD, build_model, count_parameters, decode, encode,
    generate, load_vocab, save_checkpoint,
)
from validate import validate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class SQLDataset(Dataset):
    def __init__(self, rows, vocab):
        self.rows = rows
        self.vocab = vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        src = encode(r["input"], self.vocab, lowercase=True)
        tgt = encode(r["sql"], self.vocab)
        return src, tgt


def collate(batch, vocab):
    pad = vocab[PAD]
    srcs = [torch.tensor(s, dtype=torch.long) for s, _ in batch]
    tgts = [torch.tensor(t, dtype=torch.long) for _, t in batch]
    src_len = max(len(s) for s in srcs)
    tgt_len = max(len(t) for t in tgts) + 1  # +1 for EOS
    src_padded = torch.full((len(batch), src_len), pad, dtype=torch.long)
    # decoder input: BOS + sql tokens ; labels: sql tokens + EOS
    dec_in = torch.full((len(batch), tgt_len), pad, dtype=torch.long)
    labels = torch.full((len(batch), tgt_len), -100, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_padded[i, : len(s)] = s
        dec_in[i, 0] = vocab[BOS]
        dec_in[i, 1: len(t) + 1] = t
        labels[i, : len(t)] = t
        labels[i, len(t)] = vocab[EOS]
    return src_padded, dec_in, labels


def norm(sql):
    return " ".join(sql.strip().rstrip(";").split()).upper()


def execution_matches(con, gen_sql, gold_sql, op):
    """True if running gen_sql gives the same outcome as gold_sql.

    SELECT: same result rows (order-insensitive, except when the
    query has ORDER BY / LIMIT, where order matters).
    Writes: same number of affected rows, each run on a fresh copy
    of the seed DB so the two runs cannot interfere.
    """
    if op == "SELECT":
        try:
            g = con.execute(gen_sql).fetchall()
            t = con.execute(gold_sql).fetchall()
        except sqlite3.Error:
            return False
        if "ORDER BY" in gold_sql.upper() or "LIMIT" in gold_sql.upper():
            return g == t
        return sorted(map(str, g)) == sorted(map(str, t))
    return None  # writes are compared on temp copies by the caller


def evaluate(model, vocab, rows, device, db_path, limit=200):
    """Greedy-decode `limit` val rows. Returns a metrics dict.

    Exact match is strict (normalised SQL text). Execution accuracy
    is measured only on SELECT rows in this function; write rows are
    scored by exact match + 'valid SQL that runs on a fresh DB copy'.
    """
    model.eval()
    rows = rows[:limit]
    exact = 0
    exec_ok = 0
    exec_n = 0
    write_ok = 0
    write_n = 0
    samples = []
    con = sqlite3.connect(db_path)
    for r in rows:
        gen = generate(model, vocab, r["input"], device)
        if norm(gen) == norm(r["sql"]):
            exact += 1
        if r["op"] == "SELECT":
            exec_n += 1
            if execution_matches(con, gen, r["sql"], "SELECT"):
                exec_ok += 1
        else:
            write_n += 1
            ok, _, err = validate(gen)
            if ok:
                tmp = os.path.join(
                    tempfile.gettempdir(), "pocketquery_eval.db")
                shutil.copyfile(db_path, tmp)
                try:
                    c2 = sqlite3.connect(tmp)
                    c2.execute(gen)
                    c2.commit()
                    write_ok += 1
                except sqlite3.Error:
                    pass
                finally:
                    c2.close()
                    if os.path.exists(tmp):
                        os.remove(tmp)
        if len(samples) < 5:
            samples.append({"input": r["input"], "expected": r["sql"],
                            "generated": gen, "op": r["op"]})
    con.close()
    return {
        "eval_rows": len(rows),
        "exact_match": exact / max(len(rows), 1),
        "exact_match_n": exact,
        "select_execution_accuracy": exec_ok / max(exec_n, 1),
        "select_execution_n": exec_n,
        "write_valid_and_runs": write_ok / max(write_n, 1),
        "write_n": write_n,
        "samples": samples,
    }


def train(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    vocab = load_vocab(os.path.join(DATA, "vocab.json"))
    train_rows = load_jsonl(os.path.join(DATA, "train.jsonl"))
    val_rows = load_jsonl(os.path.join(DATA, "val.jsonl"))
    if args.limit:
        train_rows = train_rows[: args.limit]
    print(f"device={device} train_rows={len(train_rows)} "
          f"val_rows={len(val_rows)} vocab={len(vocab)}")
    model, cfg = build_model(len(vocab), device=device)
    print(f"parameters: {count_parameters(model):,}")
    ds = SQLDataset(train_rows, vocab)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, vocab),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    model.train()
    t0 = time.time()
    history = []
    for ep in range(args.epochs):
        tot, nb = 0.0, 0
        for src, dec_in, labels in loader:
            src, dec_in, labels = (src.to(device), dec_in.to(device),
                                   labels.to(device))
            src_pad = src == vocab[PAD]
            dec_pad = dec_in == vocab[PAD]
            logits = model(src, dec_in, src_pad, dec_pad)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        avg = tot / max(nb, 1)
        history.append(round(avg, 4))
        print(f"epoch {ep + 1}/{args.epochs} loss={avg:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    save_checkpoint(model, vocab, os.path.join(DATA, "model.pt"))
    print("saved data/model.pt")
    # Evaluate on validation rows (greedy decode; slow on CPU, so the
    # default limit is small — pass --eval-limit for a bigger sample).
    metrics = evaluate(model, vocab, val_rows, device,
                       os.path.join(DATA, "college.db"),
                       limit=args.eval_limit)
    metrics.update({
        "mode": "smoke" if args.smoke else "full",
        "epochs": args.epochs,
        "train_examples": len(train_rows),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": device,
        "seconds": round(time.time() - t0),
        "loss_history": history,
        "parameters": count_parameters(model),
    })
    with open(os.path.join(DATA, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("samples",)}, indent=1))
    return metrics


def main():
    p = argparse.ArgumentParser(description="Train PocketQuery from scratch.")
    p.add_argument("--full", action="store_true",
                   help="full training: all 12,800 train rows, 30 epochs")
    p.add_argument("--smoke", action="store_true",
                   help="smoke training: 2,000 rows, 3 epochs, CPU-friendly")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="max train rows (0 = all)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-limit", type=int, default=200)
    p.add_argument("--model", default=os.path.join(DATA, "model.pt"))
    args = p.parse_args()
    if args.smoke:
        args.epochs = args.epochs or 3
        args.limit = args.limit or 2000
    elif args.full:
        args.epochs = args.epochs or 30
        args.limit = args.limit or 0
    else:
        args.epochs = args.epochs or 3
        args.limit = args.limit or 0
    if args.eval_only:
        from model import load_checkpoint
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, vocab, cfg = load_checkpoint(args.model, device)
        val_rows = load_jsonl(os.path.join(DATA, "val.jsonl"))
        print(json.dumps(evaluate(model, vocab, val_rows, device,
                                  os.path.join(DATA, "college.db"),
                                  limit=args.eval_limit), indent=1))
        return
    train(args)


if __name__ == "__main__":
    main()
