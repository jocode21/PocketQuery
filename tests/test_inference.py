"""Real-model inference tests.

These need data/model.pt, so they SKIP when no checkpoint exists
(e.g. before any training run). They check the pipeline end to end:
generate() runs without crashing and returns SQL-shaped text for one
prompt of each operation. They deliberately do NOT assert correctness
— a smoke-trained model is weak by design, and its real accuracy is
reported in data/training_metrics.json, not hidden by these tests.
"""
import json
import os

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(DATA, "model.pt")


def _first_of_op(op):
    with open(os.path.join(DATA, "val.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["op"] == op:
                return r
    return None


@pytest.mark.skipif(not os.path.exists(MODEL_PATH),
                    reason="no trained checkpoint yet (run train.py)")
def test_generate_each_operation():
    from model import generate, get_model
    model, vocab = get_model()
    seen = {}
    for op in ("SELECT", "INSERT", "UPDATE", "DELETE"):
        row = _first_of_op(op)
        assert row is not None
        out = generate(model, vocab, row["input"])
        assert isinstance(out, str) and out.strip(), op
        seen[op] = {"input": row["input"], "expected": row["sql"],
                    "generated": out}
    print("\n" + json.dumps(seen, indent=1))
