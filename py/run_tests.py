#!/usr/bin/env python3
"""Run the fine-tuned model against use/test cases and report pass/fail.

Test cases live in workspace/tests/cases.jsonl, one JSON object per line:
  {"name": "...", "prompt": "...", "expect": ["substring", ...], "system": "..."}
- "expect" (optional): all substrings (case-insensitive) must appear to PASS.
  Omit it to just eyeball the generation.
- "system" (optional): per-case system prompt.

A starter file is created on first run if none exists.

Usage: run_tests.py --model workspace/fused
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(ROOT, "workspace", "tests", "cases.jsonl")

def ensure_cases():
    if not os.path.exists(CASES):
        from seedlib import load_seed
        starter = load_seed("test_cases")        # editable JSON seed
        os.makedirs(os.path.dirname(CASES), exist_ok=True)
        with open(CASES, "w") as f:
            for c in starter:
                f.write(json.dumps(c) + "\n")
        print(f"· wrote starter test cases → {CASES} (edit these for real tests)")


def load_cases():
    ensure_cases()
    out = []
    with open(CASES) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="workspace/fused")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    cases = load_cases()
    if not cases:
        print("No test cases found."); return 0

    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: mlx_lm not available ({e}) — run Setup env first.")
        return 1

    print(f"Loading {args.model} …")
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0)

    passed = 0
    for i, c in enumerate(cases, 1):
        name = c.get("name", f"case {i}")
        msgs = []
        if c.get("system"):
            msgs.append({"role": "system", "content": c["system"]})
        msgs.append({"role": "user", "content": c.get("prompt", "")})
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)
        text = generate(model, tokenizer, prompt=prompt,
                        max_tokens=args.max_tokens, sampler=sampler, verbose=False)

        expect = [e.lower() for e in c.get("expect", []) if e]
        low = text.lower()
        missing = [e for e in expect if e not in low]
        ok = not missing
        if ok:
            passed += 1
        mark = "✓ PASS" if ok else ("✗ FAIL" if expect else "·  ran ")
        print(f"\n{mark}  [{i}/{len(cases)}] {name}")
        print(f"  prompt: {c.get('prompt','')[:120]}")
        snippet = text.strip().replace("\n", "\n          ")
        print(f"  output: {snippet[:600]}")
        if missing:
            print(f"  missing expected: {missing}")

    scored = [c for c in cases if c.get("expect")]
    print(f"\n=== {passed}/{len(scored)} scored case(s) passed "
          f"({len(cases)} total run) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
