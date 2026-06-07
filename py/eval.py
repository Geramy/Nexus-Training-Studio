#!/usr/bin/env python3
"""Behavioral smoke-eval BEFORE you export — does the fine-tune still tool-call?

Loads the fused model (workspace/fused) and runs a few Nexus-style tool-calling
prompts, printing what it produces so you can confirm it (a) emits tool calls and
(b) didn't regress into prose. Pass --base <repo> to print the base side-by-side.

Usage: python py/eval.py [--base <hf_repo_or_dir>] [--model workspace/fused]
"""
import argparse, json

SCENARIOS = [
    {
        "system": "You are the Setup host. Ask one question at a time via the "
                  "ask_question tool, then save answers with propose_tags.",
        "user": "I'm building a mobile app for a lemonade stand where people find "
                "and order.",
        "want": "should call propose_tags (Food & Beverage, iOS/Android, Ordering) "
                "or ask_question — NOT a prose paragraph.",
    },
    {
        "system": "You are the project Coordinator. Use generate_image when asked "
                  "for a picture.",
        "user": "make me a photo of the launch screen",
        "want": "should call generate_image.",
    },
]


def _run(model_path, tools_hint):
    from mlx_lm import load, generate
    model, tok = load(model_path)
    outs = []
    for s in SCENARIOS:
        msgs = [{"role": "system", "content": s["system"]},
                {"role": "user", "content": s["user"]}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False)
        text = generate(model, tok, prompt=prompt, max_tokens=256, verbose=False)
        outs.append(text)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="workspace/fused")
    ap.add_argument("--base", default=None)
    args = ap.parse_args()

    print("=== FINE-TUNED (workspace/fused) ===")
    ft = _run(args.model, None)
    base = _run(args.base, None) if args.base else [None] * len(SCENARIOS)

    for i, s in enumerate(SCENARIOS):
        print(f"\n--- scenario {i+1}: {s['user']!r}")
        print(f"    want: {s['want']}")
        print(f"    FINE-TUNED:\n{ft[i]}\n")
        if base[i] is not None:
            print(f"    BASE:\n{base[i]}\n")
    print("\nEyeball: did the fine-tuned model emit TOOL CALLS (not prose) and "
          "stay coherent? If it regressed vs base, do NOT export — lower the LR / "
          "iters, or add more/general data.")


if __name__ == "__main__":
    main()
