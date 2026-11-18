"""PocketQuery — tiny transformer, trained from scratch.

Architecture (locked scope): a small encoder-decoder Transformer.
  d_model=256, 8 heads, 3 encoder layers + 3 decoder layers (= 6 layers),
  dim_feedforward=3072, dropout 0.1, context 256 tokens, word-level
  tokenizer built from the dataset (see generate_data.py).
No pretrained weights, no external model, no API: every weight starts
from a random initialisation and is learned only from data/*.jsonl.

Checkpoint format (model.pt): a dict with
  {"config": {...}, "vocab": {...}, "state_dict": {...}}
so the app can load it with no training code and no extra files.
"""
import json
import math
import os
import re

import torch
import torch.nn as nn

CONFIG = {
    "d_model": 256,
    "nhead": 8,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 3072,
    "dropout": 0.1,
    "max_len": 256,
    "vocab_size": 1291,  # replaced by the real vocab size at load time
}

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"


def tokenize(text):
    """Same tokeniser as generate_data.py: words, numbers, punctuation."""
    return re.findall(r"[A-Za-z_]+|[0-9]+|[^\sA-Za-z0-9_]", text)


def load_vocab(path):
    with open(path) as f:
        return json.load(f)


def encode(text, vocab, lowercase=False, add_bos=False, add_eos=False):
    if lowercase:
        text = text.lower()
    ids = [vocab.get(t, vocab[UNK]) for t in tokenize(text)]
    if add_bos:
        ids = [vocab[BOS]] + ids
    if add_eos:
        ids = ids + [vocab[EOS]]
    return ids


def decode(ids, inv_vocab, skip_special=True):
    out = []
    for i in ids:
        tok = inv_vocab.get(int(i), UNK)
        if skip_special and tok in (PAD, BOS, EOS):
            continue
        if tok == EOS:
            break
        out.append(tok)
    return detokenize(out)


def detokenize(tokens):
    """Re-join tokens into SQL / English text: join with spaces, then
    tidy the spacing around punctuation, dots and quotes."""
    text = " ".join(tokens)
    # Inside quotes, join dots that start a lowercase word (emails:
    # aisat.ac.in). 'Prof. George' keeps its space because George
    # starts uppercase. This runs before the generic rules, and it
    # loops over the whole segment so chained dots all join.
    def _fix_quoted(m):
        seg = m.group(0)
        seg = re.sub(r"\s*\.\s*(?=[a-z])", ".", seg)
        seg = re.sub(r"([A-Za-z])\s+(\d)", r"\1\2", seg)
        return seg
    text = re.sub(r"'[^']*'", _fix_quoted, text)
    # Qualified names only: students.name, marks.marks.
    text = re.sub(
        r"\b(students|marks|courses|faculty|attendance)\s*\.\s*([A-Za-z_]+)",
        r"\1.\2", text)
    text = re.sub(r"\s+([,;.)])", r"\1", text)   # no space before , ; . )
    text = re.sub(r"\s*@\s*", "@", text)         # emails: a@b, not a @ b
    text = re.sub(r">\s+=", ">=", text)          # >=, not > =
    text = re.sub(r"<\s+=", "<=", text)          # <=, not < =
    text = re.sub(r"!\s+=", "!=", text)          # !=, not ! =
    text = re.sub(r"\b(COUNT|AVG|SUM|MIN|MAX)\s+\(", r"\1(", text)
    text = re.sub(r"\(\s+", "(", text)           # no space after (
    text = re.sub(r"\s+\)", ")", text)           # no space before )
    text = re.sub(r"'\s+([^']*?)\s+'", r"'\1'", text)  # 'CSE', not ' CSE '
    text = re.sub(r"-\s+(\d)", r"-\1", text)     # -5, not - 5
    return text.strip()


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, config=None):
        super().__init__()
        cfg = dict(CONFIG)
        cfg.update(config or {})
        self.cfg = cfg
        d = cfg["d_model"]
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos_embedding = nn.Embedding(cfg["max_len"], d)
        self.transformer = nn.Transformer(
            d_model=d,
            nhead=cfg["nhead"],
            num_encoder_layers=cfg["num_encoder_layers"],
            num_decoder_layers=cfg["num_decoder_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            batch_first=True,
        )
        # Tie the output projection to the input embedding: fewer
        # parameters, and it helps a small model learn the vocabulary.
        self.out = nn.Linear(d, vocab_size)
        self.out.weight = self.embedding.weight
        self.scale = math.sqrt(d)

    def _embed(self, ids):
        positions = torch.arange(
            ids.size(1), device=ids.device
        ).unsqueeze(0)
        return self.embedding(ids) * self.scale + self.pos_embedding(positions)

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        # Boolean causal mask (same dtype family as the padding masks,
        # otherwise torch warns about mismatched mask types).
        tgt_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool,
                       device=tgt.device), diagonal=1,
        )
        out = self.transformer(
            self._embed(src),
            self._embed(tgt),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.out(out)

    def encode_src(self, src, src_pad_mask=None):
        return self.transformer.encoder(
            self._embed(src), src_key_padding_mask=src_pad_mask
        )

    def decode_step(self, tgt, memory, tgt_pad_mask=None,
                    src_pad_mask=None):
        tgt_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool,
                       device=tgt.device), diagonal=1,
        )
        out = self.transformer.decoder(
            self._embed(tgt), memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.out(out)


def build_model(vocab_size, config=None, device="cpu"):
    cfg = dict(CONFIG)
    cfg.update(config or {})
    cfg["vocab_size"] = vocab_size
    model = TinyTransformer(vocab_size, cfg)
    return model.to(device), cfg


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, vocab, path):
    torch.save(
        {
            "config": model.cfg,
            "vocab": vocab,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    model, cfg = build_model(len(vocab), ckpt.get("config"), device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab, cfg


@torch.no_grad()
def generate(model, vocab, text, device="cpu", max_len=64):
    """Greedy decode English -> SQL. Returns the SQL string."""
    inv = {i: t for t, i in vocab.items()}
    src = torch.tensor(
        [encode(text, vocab, lowercase=True)],
        dtype=torch.long, device=device,
    )
    src_pad = src == vocab[PAD]
    memory = model.encode_src(src, src_pad)
    tgt = torch.tensor([[vocab[BOS]]], dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model.decode_step(tgt, memory, src_pad_mask=src_pad)
        nxt = int(logits[0, -1].argmax())
        tgt = torch.cat(
            [tgt, torch.tensor([[nxt]], device=device)], dim=1
        )
        if nxt == vocab[EOS]:
            break
    return decode(tgt[0].tolist()[1:], inv)


# ---- Lazy singleton for the web app / CLI: load once, reuse. ----
_MODEL_CACHE = {}


def get_model(model_path=None, device="cpu"):
    """Return (model, vocab). model_path defaults to data/model.pt
    next to this file. Raises FileNotFoundError if no checkpoint
    exists yet — run train.py first."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = model_path or os.path.join(here, "data", "model.pt")
    if path not in _MODEL_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained model at {path}. Run: python train.py --smoke "
                "(quick check) or python train.py --full (full training)."
            )
        _MODEL_CACHE[path] = load_checkpoint(path, device)
    return _MODEL_CACHE[path][0], _MODEL_CACHE[path][1]
