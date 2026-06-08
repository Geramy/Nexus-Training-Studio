#!/usr/bin/env python3
"""Turn the agents' raw posted traces into an mlx-lm `messages` dataset.

Reads workspace/data/raw.jsonl (one JSON object per line, each
`{"messages": [...]}` in OpenAI shape, tool_calls allowed). Validates, dedupes,
optionally blends workspace/data/general.jsonl (general instruction/tool-use data
to prevent forgetting), shuffles deterministically, and writes train.jsonl +
valid.jsonl. mlx-lm applies the model's chat template AND completion-only loss
masking for `messages` datasets, so train == serve format.
"""
import json, hashlib, random, sys
from pathlib import Path

DATA = Path("workspace/data")
RAW = DATA / "raw.jsonl"
GENERAL = DATA / "general.jsonl"
DATASET = DATA / "dataset.jsonl"          # canonical curated corpus (table/imports/gen)
CONVERSATIONS = DATA / "conversations"   # one {"messages":[...]} json per id
VAL_FRACTION = 0.1
SEED = 7


def _normalize_args(messages):
    """Nemotron's chat template calls `.items()` on tool_call arguments, so it
    needs them as DICTS. Our canonical dataset (and live agent traces) store them
    as JSON STRINGS (OpenAI wire format) — convert here so train rendering works.
    Behavior is unchanged; only the on-disk training shape is adjusted."""
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                except json.JSONDecodeError:
                    fn["arguments"] = {}
    return messages


def _load(path: Path):
    out = []
    if not path.exists():
        return out
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  skip {path.name}:{ln} (bad json: {e})")
            continue
        msgs = obj.get("messages")
        if not isinstance(msgs, list) or not msgs:
            print(f"  skip {path.name}:{ln} (no messages[])")
            continue
        # Must contain at least one assistant turn to learn from.
        if not any(m.get("role") == "assistant" for m in msgs):
            print(f"  skip {path.name}:{ln} (no assistant turn)")
            continue
        item = {"messages": _normalize_args(msgs)}
        if isinstance(obj.get("tools"), list) and obj["tools"]:
            item["tools"] = obj["tools"]   # mlx_lm "tools" format passthrough
        out.append(item)
    return out


def _load_conversations(dirp: Path):
    """One {"messages":[...]} JSON object per file (the per-conversation traces
    the studio keeps the longest of)."""
    out = []
    if not dirp.exists():
        return out
    for p in sorted(dirp.glob("*.json")):
        try:
            obj = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"  skip {p.name} (bad json: {e})")
            continue
        msgs = obj.get("messages")
        if (isinstance(msgs, list) and msgs
                and any(m.get("role") == "assistant" for m in msgs)):
            out.append({"messages": _normalize_args(msgs)})
        else:
            print(f"  skip {p.name} (no usable messages)")
    return out


def _iter_jsonl(path: Path):
    """Stream valid, normalized items from a JSONL file (memory-bounded)."""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = obj.get("messages")
            if not isinstance(msgs, list) or not msgs:
                continue
            if not any(m.get("role") == "assistant" for m in msgs):
                continue
            item = {"messages": _normalize_args(msgs)}
            if isinstance(obj.get("tools"), list) and obj["tools"]:
                item["tools"] = obj["tools"]
            yield item


def _iter_conversations(dirp: Path):
    if not dirp.exists():
        return
    for p in sorted(dirp.glob("*.json")):
        try:
            obj = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        msgs = obj.get("messages")
        if (isinstance(msgs, list) and msgs
                and any(m.get("role") == "assistant" for m in msgs)):
            yield {"messages": _normalize_args(msgs)}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    seen = set()
    n = {"train": 0, "val": 0}
    ftrain = open(DATA / "train.jsonl", "w")
    fvalid = open(DATA / "valid.jsonl", "w")

    def emit(item, force_train=False):
        h = hashlib.sha256(
            json.dumps(item, sort_keys=True).encode()).hexdigest()
        if h in seen:
            return False
        seen.add(h)
        line = json.dumps(item)
        if not force_train and rng.random() < VAL_FRACTION:
            fvalid.write(line + "\n"); n["val"] += 1
        else:
            ftrain.write(line + "\n"); n["train"] += 1
        return True

    try:
        # Canonical corpus + live agent traces (deduped, streamed).
        for it in _iter_jsonl(DATASET):
            emit(it)
        for it in _iter_jsonl(RAW):
            emit(it)
        for it in _iter_conversations(CONVERSATIONS):
            emit(it)
        unique = n["train"] + n["val"]
        # Blend ~20% general replay into TRAIN to limit forgetting.
        cap = int(0.2 * max(1, unique))
        g = 0
        for it in _iter_jsonl(GENERAL):
            if g >= cap:
                break
            if emit(it, force_train=True):
                g += 1
        if g:
            print(f"  blended {g} general replay examples")
    finally:
        ftrain.close()
        fvalid.close()

    if n["train"] + n["val"] == 0:
        print("No usable examples. Generate or import data first.")
        sys.exit(1)
    print(f"Prepared {n['train']} train / {n['val']} valid "
          f"({len(seen)} unique).")
    if n["train"] < 50:
        print("WARNING: <50 training examples — generate/import more first.")


if __name__ == "__main__":
    main()
