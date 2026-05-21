"""data_prep.py — Dataset utilities for fine-tuning homework (Week 5)."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ChatML helpers
# ---------------------------------------------------------------------------

CHATML_ROLES = {"system", "user", "assistant"}

_ROLE_TOKENS = {
    "system": "<|im_start|>system\n{content}<|im_end|>\n",
    "user": "<|im_start|>user\n{content}<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
}


def format_chatml(messages: list[dict]) -> str:
    """Convert [{role, content},...] to a ChatML string."""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        template = _ROLE_TOKENS.get(role, "<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append(template.format(role=role, content=content))
    # Append generation prompt stub for the last turn if it's not already assistant
    return "".join(parts)


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict],
    add_generation_prompt: bool = True,
) -> str:
    """Apply tokenizer.apply_chat_template, falling back to format_chatml."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            result = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            print(f"[data_prep] Applied tokenizer chat template (add_generation_prompt={add_generation_prompt})")
            return result
        except Exception as exc:
            print(f"[data_prep] tokenizer.apply_chat_template failed ({exc}), falling back to format_chatml")
    else:
        print("[data_prep] Tokenizer has no chat_template — using format_chatml fallback")
    text = format_chatml(messages)
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load a newline-delimited JSON file into a list of dicts."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"[data_prep] Loaded {len(records)} records from {path}")
    return records


def save_jsonl(records: list[dict], path: str) -> None:
    """Save a list of dicts to a JSONL file, creating parent directories."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[data_prep] Saved {len(records)} records to {path}")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def train_val_test_split(
    records: list[dict],
    ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list, list, list]:
    """Shuffle and split records into (train, val, test) by ratio."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    print(
        f"[data_prep] Split {n} records -> train={len(train)}, val={len(val)}, test={len(test)} "
        f"(ratios={ratios}, seed={seed})"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Sequence packing
# ---------------------------------------------------------------------------

_DEFAULT_EOS = "<|endoftext|>"


def pack_sequences(
    records: list[dict],
    max_len: int = 512,
    text_field: str = "text",
) -> list[dict]:
    """Concatenate short records with EOS until max_len would be exceeded."""
    packed: list[dict] = []
    buffer = ""
    eos = _DEFAULT_EOS

    for rec in records:
        text = rec[text_field]
        candidate = buffer + text + eos if buffer else text + eos
        if len(candidate) > max_len and buffer:
            packed.append({"text": buffer})
            buffer = text + eos
        else:
            buffer = candidate

    if buffer:
        packed.append({"text": buffer})

    print(
        f"[data_prep] Packed {len(records)} records into {len(packed)} sequences "
        f"(max_len={max_len} chars, text_field='{text_field}')"
    )
    return packed


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def show_dataset_stats(records: list[dict], text_field: str = "text") -> dict:
    """Print and return count/avg/min/max char length and a sample entry."""
    lengths = [len(r[text_field]) for r in records]
    stats: dict = {
        "count": len(records),
        "avg_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "sample": records[0] if records else {},
    }
    print(f"[data_prep] Dataset stats:")
    print(f"  count       : {stats['count']}")
    print(f"  avg chars   : {stats['avg_chars']}")
    print(f"  min chars   : {stats['min_chars']}")
    print(f"  max chars   : {stats['max_chars']}")
    print(f"  sample entry: {json.dumps(stats['sample'],indent=2)[:200]}")
    return stats
