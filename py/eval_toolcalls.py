#!/usr/bin/env python3
"""BFCL-style tool-call eval: replay held-out conversations and score whether the
model emits the SAME tool call the ground-truth assistant did.

For each assistant turn that carries tool_calls, we feed the preceding messages
(+ the example's tools) through the chat template, generate, parse the model's
tool call, and compare:
  - name exact-match  (did it call the right function?)
  - args exact-match  (identical JSON arguments?)
  - args key-match    (same argument keys, values ignored — partial credit)

Usage: eval_toolcalls.py --model workspace/fused [--data workspace/data/valid.jsonl]
                         [--limit 100]
"""
import argparse
import json
import re
import sys
from pathlib import Path


def _norm(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def parse_tool_call(text):
    """Best-effort extraction of {name, arguments} from model output across the
    common formats."""
    # 0) Nemotron format: <function=NAME><parameter=k>\nVALUE\n</parameter>...</function>
    m = re.search(r"<function=([^>\s]+)\s*>(.*?)</function>", text, re.S)
    if m:
        name = m.group(1).strip()
        args = {}
        for pm in re.finditer(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>",
                              m.group(2), re.S):
            raw = pm.group(2).strip()
            try:  # values may be JSON (lists/objects) or plain strings
                args[pm.group(1).strip()] = json.loads(raw)
            except Exception:  # noqa: BLE001
                args[pm.group(1).strip()] = raw
        return {"name": name, "arguments": args}
    # 1) <tool_call>{...}</tool_call> or <tool_call> ... (Qwen/Nemotron JSON style)
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
        obj = _try_json(m.group(1))
        if obj:
            return _shape(obj)
    # 2) ```json { ... } ```
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
        obj = _try_json(m.group(1))
        if obj:
            return _shape(obj)
    # 3) first balanced bare JSON object containing "name"
    for m in re.finditer(r"\{.*?\}", text, re.S):
        obj = _try_json(m.group(0))
        if obj and ("name" in obj or "function" in obj):
            return _shape(obj)
    return None


def _try_json(s):
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return None


def _shape(obj):
    """Normalize varied shapes to {'name':..., 'arguments': {...}}."""
    if "function" in obj and isinstance(obj["function"], dict):
        fn = obj["function"]
        args = fn.get("arguments", {})
    else:
        fn = obj
        args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        args = _try_json(args) or {}
    return {"name": fn.get("name"), "arguments": args if isinstance(args, dict)
            else {}}


def gold_calls(messages):
    """Yield (prefix_messages, gold_tool_call) for each assistant tool-call turn."""
    out = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            args = _try_json(args) if isinstance(args, str) else args
            out.append((messages[:i],
                        {"name": fn.get("name"), "arguments": args or {}}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="workspace/fused")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir to load on top of --model "
                         "(skips fusing).")
    ap.add_argument("--data", default="workspace/data/valid.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"ERROR: {path} not found — run Prepare data first.")
        return 1

    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: mlx_lm not available ({e}) — run Setup env.")
        return 1

    print(f"Loading {args.model}"
          f"{' + adapter ' + args.adapter if args.adapter else ''} …")
    model, tok = (load(args.model, adapter_path=args.adapter)
                  if args.adapter else load(args.model))
    sampler = make_sampler(temp=0.0)

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    name_ok = args_ok = key_ok = total = parsed = 0
    for r in rows:
        msgs, tools = r.get("messages", []), r.get("tools")
        for prefix, gold in gold_calls(msgs):
            if total >= args.limit:
                break
            total += 1
            try:  # thinking OFF → the model emits the tool call directly
                prompt = tok.apply_chat_template(
                    prefix, add_generation_prompt=True, tokenize=False,
                    tools=tools, enable_thinking=False)
            except Exception:
                prompt = tok.apply_chat_template(
                    prefix, add_generation_prompt=True, tokenize=False)
            text = generate(model, tok, prompt=prompt,
                            max_tokens=args.max_tokens, sampler=sampler,
                            verbose=False)
            pred = parse_tool_call(text)
            if not pred:
                continue
            parsed += 1
            if pred["name"] == gold["name"]:
                name_ok += 1
                if _norm(pred["arguments"]) == _norm(gold["arguments"]):
                    args_ok += 1
                if set(pred["arguments"]) == set(gold["arguments"]):
                    key_ok += 1
        if total >= args.limit:
            break

    if total == 0:
        print("No gold tool calls found in the data.")
        return 0
    pct = lambda n: f"{100*n/total:.1f}%"  # noqa: E731
    print(f"\n=== Tool-call eval over {total} gold calls "
          f"({parsed} parsed) ===")
    print(f"  function-name exact:  {name_ok}/{total}  ({pct(name_ok)})")
    print(f"  arguments exact:      {args_ok}/{total}  ({pct(args_ok)})")
    print(f"  argument-keys match:  {key_ok}/{total}  ({pct(key_ok)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
