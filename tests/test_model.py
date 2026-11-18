"""Model tests: size, tokenizer round-trip, forward pass."""
import json
import os

import pytest
import torch

from model import (build_model, count_parameters, decode, encode,
                   get_model, load_vocab)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")


def test_parameter_count_in_spec_range():
    vocab = load_vocab(os.path.join(DATA, "vocab.json"))
    model, _ = build_model(len(vocab))
    n = count_parameters(model)
    assert 10_000_000 <= n <= 20_000_000, f"params={n:,}"


def test_forward_pass_shapes():
    vocab = load_vocab(os.path.join(DATA, "vocab.json"))
    model, _ = build_model(len(vocab))
    model.eval()
    x = torch.randint(0, len(vocab), (2, 10))
    y = torch.randint(0, len(vocab), (2, 8))
    with torch.no_grad():
        out = model(x, y)
    assert tuple(out.shape) == (2, 8, len(vocab))


def test_tokenizer_round_trip_on_dataset_sample():
    vocab = load_vocab(os.path.join(DATA, "vocab.json"))
    inv = {i: t for t, i in vocab.items()}
    rows = [json.loads(l) for l in open(os.path.join(DATA, "dataset.jsonl"))]
    for r in rows[:200]:
        assert decode(encode(r["sql"], vocab), inv) == r["sql"]


def test_get_model_raises_without_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_model(model_path=str(tmp_path / "nope.pt"))
