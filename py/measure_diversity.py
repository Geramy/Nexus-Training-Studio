#!/usr/bin/env python3
"""Quantify corpus diversity (research: measure it, don't guess). Reports
distinct-n (unique n-grams / total), type-token ratio, and unique-message ratio
over the USER turns (the entropy that drives generalization) and assistant text.

Usage: measure_diversity.py [--data workspace/data/dataset.jsonl] [--sample 20000]
"""
import argparse
import json
import random
import re

WORD = re.compile(r"\w+")


def toks(s):
    return WORD.findall(s.lower())


def distinct_n(token_lists, n):
    grams, total = set(), 0
    for tl in token_lists:
        for i in range(len(tl) - n + 1):
            grams.add(tuple(tl[i:i + n]))
            total += 1
    return len(grams) / total if total else 0.0


def report(label, texts):
    tls = [toks(t) for t in texts]
    allw = [w for tl in tls for w in tl]
    vocab = set(allw)
    uniq_msgs = len(set(texts)) / len(texts) if texts else 0
    print(f"\n── {label} ({len(texts)} turns) ──")
    print(f"  unique messages:   {uniq_msgs:.3f}")
    print(f"  vocabulary size:   {len(vocab)}")
    print(f"  type-token ratio:  {len(vocab)/len(allw):.4f}" if allw else "  (empty)")
    print(f"  distinct-1/2/3:    {distinct_n(tls,1):.3f} / "
          f"{distinct_n(tls,2):.3f} / {distinct_n(tls,3):.3f}")
    print(f"  avg words/turn:    {len(allw)/len(texts):.1f}" if texts else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="workspace/data/dataset.jsonl")
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()

    rng = random.Random(7)
    users, assts = [], []
    with open(args.data) as f:
        for line in f:
            if not line.strip():
                continue
            if rng.random() > args.sample / 300000:  # rough reservoir
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for m in row.get("messages", []):
                c = m.get("content")
                if not isinstance(c, str) or not c.strip():
                    continue
                if m["role"] == "user":
                    users.append(c)
                elif m["role"] == "assistant":
                    assts.append(c)

    print(f"Diversity report for {args.data}")
    report("USER turns", users)
    report("ASSISTANT text turns", assts)


if __name__ == "__main__":
    main()
