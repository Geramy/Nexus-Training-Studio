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
CONVERSATIONS = DATA / "conversations"   # one {"messages":[...]} json per id
VAL_FRACTION = 0.1
SEED = 7


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
        out.append({"messages": msgs})
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
            out.append({"messages": msgs})
        else:
            print(f"  skip {p.name} (no usable messages)")
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    items = _load(RAW) + _load_conversations(CONVERSATIONS)
    general = _load(GENERAL)
    if not items:
        print(f"No usable examples in {RAW}. Have the agents POST to "
              f"/training-data first.")
        sys.exit(1)

    # Dedupe by content hash.
    seen, deduped = set(), []
    for it in items:
        h = hashlib.sha256(json.dumps(it, sort_keys=True).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        deduped.append(it)

    # Blend general data (~20%) to prevent catastrophic forgetting.
    blended = list(deduped)
    if general:
        take = max(1, int(0.2 * len(deduped)))
        random.Random(SEED).shuffle(general)
        blended += general[:take]
        print(f"  blended {min(take, len(general))} general examples")

    random.Random(SEED).shuffle(blended)
    n_val = max(1, int(VAL_FRACTION * len(blended)))
    valid, train = blended[:n_val], blended[n_val:]

    (DATA / "train.jsonl").write_text(
        "\n".join(json.dumps(x) for x in train) + "\n")
    (DATA / "valid.jsonl").write_text(
        "\n".join(json.dumps(x) for x in valid) + "\n")
    print(f"Prepared {len(train)} train / {len(valid)} valid "
          f"(from {len(deduped)} unique traces).")
    if len(train) < 50:
        print("WARNING: <50 training examples — collect more (distill validated "
              "agent runs) before a real run, or it will overfit.")


if __name__ == "__main__":
    main()
