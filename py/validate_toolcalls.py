#!/usr/bin/env python3
"""Schema-verification QC for the corpus (research: execution/schema verification
beats volume). For every assistant tool_call, check it against the example's
tools[] schema: the function exists, required params are present, no unknown
params, and enum values are valid. Reports the invalid rate; optionally writes a
cleaned file with offending conversations dropped.

Usage:
  validate_toolcalls.py [--data workspace/data/dataset.jsonl] [--clean OUT] [--sample N]
"""
import argparse
import json
import sys
from collections import Counter


def _schema_index(tools):
    idx = {}
    for t in tools or []:
        fn = t.get("function", t)
        params = fn.get("parameters", {}) or {}
        idx[fn.get("name")] = {
            "props": set((params.get("properties") or {}).keys()),
            "required": set(params.get("required") or []),
            "enums": {k: set(v["enum"]) for k, v in
                      (params.get("properties") or {}).items()
                      if isinstance(v, dict) and "enum" in v},
        }
    return idx


def validate_conv(row):
    """Return a list of error strings for one conversation (empty = valid)."""
    errs = []
    idx = _schema_index(row.get("tools"))
    for m in row.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    errs.append(f"{name}: arguments not valid JSON")
                    continue
            if not isinstance(args, dict):
                args = {}
            if name not in idx:
                # tools[] may omit rarely-used fns; only flag if tools given
                if row.get("tools"):
                    errs.append(f"{name}: not in tools[]")
                continue
            spec = idx[name]
            missing = spec["required"] - set(args)
            if missing:
                errs.append(f"{name}: missing required {sorted(missing)}")
            unknown = set(args) - spec["props"]
            if unknown:
                errs.append(f"{name}: unknown params {sorted(unknown)}")
            for k, allowed in spec["enums"].items():
                if k in args and args[k] not in allowed:
                    errs.append(f"{name}.{k}: '{args[k]}' not in enum")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="workspace/data/dataset.jsonl")
    ap.add_argument("--clean", default=None, help="write valid rows here")
    ap.add_argument("--sample", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    total = valid = 0
    reasons = Counter()
    out = open(args.clean, "w") if args.clean else None
    try:
        with open(args.data) as f:
            for line in f:
                if not line.strip():
                    continue
                if args.sample and total >= args.sample:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                errs = validate_conv(row)
                if errs:
                    for e in errs[:1]:
                        reasons[e.split(":")[0] + ": " + e.split(":", 1)[1]
                                .strip().split(" ")[0]] += 1
                else:
                    valid += 1
                    if out:
                        out.write(line if line.endswith("\n") else line + "\n")
    finally:
        if out:
            out.close()

    bad = total - valid
    print(f"Validated {total} conversations: {valid} valid "
          f"({100*valid/total:.2f}%), {bad} invalid.")
    if reasons:
        print("Top issues:")
        for r, c in reasons.most_common(10):
            print(f"  {c:6d}  {r}")
    if args.clean:
        print(f"Wrote {valid} valid rows → {args.clean}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
