#!/usr/bin/env python3
"""Import a Hugging Face dataset into the training corpus.

Auto-maps the common community schemas to our OpenAI `messages` shape:
  - chat:        a "messages"/"conversation" column already in role/content form
  - ShareGPT:    "conversations" = [{"from":"human|gpt|system","value":...}]
  - Alpaca:      "instruction" (+ optional "input") / "output"
  - prompt/resp: "prompt"/"response", "question"/"answer", "text"/"label"
  - tool-call sets that already ship "messages" with tool_calls pass through.

Usage:
  import_hf_dataset.py <dataset_id> [--split train] [--config NAME]
                       [--limit N] [--text-field F --label-field G]
Honors HF_TOKEN for gated/private datasets.
"""
import argparse
import os
import sys

from data_common import append_conversations

SYS_KEYS = ("system", "system_prompt")


def _from_sharegpt(conv):
    role_map = {"human": "user", "user": "user", "gpt": "assistant",
                "assistant": "assistant", "system": "system", "tool": "tool",
                "function": "tool"}
    out = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = role_map.get(str(turn.get("from", "")).lower())
        val = turn.get("value", turn.get("content", ""))
        if role and val:
            out.append({"role": role, "content": val})
    return out


def _row_to_messages(row, args):
    # 1) already-chat columns
    for key in ("messages", "conversation"):
        if isinstance(row.get(key), list) and row[key] and \
                isinstance(row[key][0], dict) and "role" in row[key][0]:
            return row[key]
    # 2) ShareGPT
    if isinstance(row.get("conversations"), list):
        msgs = _from_sharegpt(row["conversations"])
        if msgs:
            return msgs
    # 3) explicit overrides
    if args.text_field and args.label_field:
        u = str(row.get(args.text_field, "")).strip()
        v = str(row.get(args.label_field, "")).strip()
        if u and v:
            return _wrap(row, u, v)
    # 4) Alpaca / prompt-response families
    for u_key, v_key in (("instruction", "output"), ("prompt", "response"),
                         ("question", "answer"), ("input", "output"),
                         ("text", "label")):
        if u_key in row and v_key in row:
            u = str(row.get(u_key, "")).strip()
            if u_key == "instruction" and str(row.get("input", "")).strip():
                u = f"{u}\n\n{str(row['input']).strip()}"
            v = str(row.get(v_key, "")).strip()
            if u and v:
                return _wrap(row, u, v)
    return None


def _wrap(row, user, assistant):
    msgs = []
    for sk in SYS_KEYS:
        if str(row.get(sk, "")).strip():
            msgs.append({"role": "system", "content": str(row[sk]).strip()})
            break
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": assistant})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--text-field", default=None)
    ap.add_argument("--label-field", default=None)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: `datasets` not installed ({e}) — run Setup env.")
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Loading {args.dataset} (split={args.split}"
          f"{', config=' + args.config if args.config else ''}) …")
    try:
        ds = load_dataset(args.dataset, args.config, split=args.split,
                          token=token)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR loading dataset: {e}")
        return 1

    convos, miss = [], 0
    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    for i in range(n):
        msgs = _row_to_messages(ds[i], args)
        if msgs:
            convos.append({"messages": msgs})
        else:
            miss += 1
    if not convos:
        cols = list(ds.features.keys())
        print(f"No rows mapped. Dataset columns: {cols}. "
              f"Pass --text-field/--label-field to map manually.")
        return 1
    added, skipped = append_conversations(
        convos, source=f"hf:{args.dataset}")
    print(f"Imported {added} example(s) from {args.dataset} "
          f"(skipped {skipped} dup/invalid, {miss} unmappable rows).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
