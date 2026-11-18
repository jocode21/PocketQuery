"""Dataset tests: size, splits, format, and SQL that really runs."""
import json
import os
import sqlite3

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")


def _load(name):
    with open(os.path.join(DATA, name)) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_dataset_size_and_op_counts():
    rows = _load("dataset.jsonl")
    assert len(rows) == 16000
    from collections import Counter
    c = Counter(r["op"] for r in rows)
    assert c["SELECT"] == 12000
    assert c["INSERT"] == 1500
    assert c["UPDATE"] == 1500
    assert c["DELETE"] == 1000


def test_splits_cover_dataset_without_overlap():
    all_rows = _load("dataset.jsonl")
    train, val, test = (_load("train.jsonl"), _load("val.jsonl"),
                        _load("test.jsonl"))
    assert (len(train), len(val), len(test)) == (12800, 1600, 1600)
    assert len(train) + len(val) + len(test) == len(all_rows)
    keys = lambda rs: {(r["input"], r["sql"]) for r in rs}
    assert not (keys(train) & keys(val))
    assert not (keys(train) & keys(test))
    assert not (keys(val) & keys(test))


def test_rows_have_required_fields_and_valid_ops():
    for r in _load("dataset.jsonl"):
        assert set(r) >= {"input", "sql", "op"}
        assert r["op"] in ("SELECT", "INSERT", "UPDATE", "DELETE")
        assert isinstance(r["input"], str) and r["input"].strip()
        assert isinstance(r["sql"], str) and r["sql"].strip().endswith(";")
        assert r["sql"].strip().split(None, 1)[0].upper() == r["op"]


def test_no_drop_or_create_in_dataset():
    import re
    for r in _load("dataset.jsonl"):
        assert not re.search(r"\b(DROP|TRUNCATE|ALTER|CREATE)\b",
                             r["sql"], re.IGNORECASE), r["sql"]


def test_writes_all_have_where_or_are_insert():
    import re
    for r in _load("dataset.jsonl"):
        if r["op"] in ("UPDATE", "DELETE"):
            assert re.search(r"\bWHERE\b", r["sql"], re.IGNORECASE), r["sql"]


def test_all_select_sql_runs_on_real_db():
    con = sqlite3.connect(os.path.join(DATA, "college.db"))
    bad = 0
    for r in _load("dataset.jsonl"):
        if r["op"] != "SELECT":
            continue
        try:
            con.execute(r["sql"]).fetchall()
        except sqlite3.Error:
            bad += 1
    con.close()
    assert bad == 0


def test_vocab_covers_dataset_tokens():
    from model import tokenize
    with open(os.path.join(DATA, "vocab.json")) as f:
        vocab = json.load(f)
    assert len(vocab) >= 800
    missing = set()
    for r in _load("dataset.jsonl")[:2000]:
        for tok in tokenize(r["input"].lower()) + tokenize(r["sql"]):
            if tok not in vocab:
                missing.add(tok)
    assert not missing, sorted(missing)[:10]
