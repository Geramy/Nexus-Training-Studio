#!/usr/bin/env python3
"""Shared helpers for the canonical training dataset.

The studio keeps ONE editable corpus at workspace/data/dataset.jsonl — one JSON
object per line:
    {"id": "...", "source": "...", "messages": [ ...OpenAI shape... ]}

The UI table, the Excel importer, the HF-dataset importer, and the synthetic
generator all append here; prepare_data.py reads it (plus live agent traces).
"""
import hashlib
import json
from pathlib import Path

DATA = Path("workspace/data")
DATASET = DATA / "dataset.jsonl"

VALID_ROLES = {"system", "user", "assistant", "tool"}


def conv_hash(messages):
    """Stable hash of a conversation's messages (for dedupe + ids)."""
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def valid_conversation(messages):
    """A usable conversation: a non-empty messages[] with valid roles and at
    least one assistant turn to learn from."""
    if not isinstance(messages, list) or not messages:
        return False
    if not all(isinstance(m, dict) and m.get("role") in VALID_ROLES
               for m in messages):
        return False
    return any(m.get("role") == "assistant" for m in messages)


def read_dataset():
    out = []
    if not DATASET.exists():
        return out
    for line in DATASET.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def write_dataset(rows):
    DATA.mkdir(parents=True, exist_ok=True)
    DATASET.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                               for r in rows))


def append_conversations(convos, source, dedupe=True):
    """Append [{'messages':[...]}] (or bare message-lists) to the dataset.
    Returns (added, skipped)."""
    rows = read_dataset()
    seen = {r.get("id") for r in rows}
    added = skipped = 0
    for c in convos:
        msgs = c.get("messages") if isinstance(c, dict) else c
        if not valid_conversation(msgs):
            skipped += 1
            continue
        cid = conv_hash(msgs)
        if dedupe and cid in seen:
            skipped += 1
            continue
        seen.add(cid)
        row = {"id": cid, "source": source, "messages": msgs}
        # Carry the per-example tool schemas (mlx_lm "tools" format) when present.
        if isinstance(c, dict) and c.get("tools"):
            row["tools"] = c["tools"]
        rows.append(row)
        added += 1
    write_dataset(rows)
    return added, skipped
