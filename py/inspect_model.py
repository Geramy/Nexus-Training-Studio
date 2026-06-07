#!/usr/bin/env python3
"""Verify arch + LoRA targets WITHOUT downloading weights.

Pulls just config.json (+ the safetensors index) and reports: architecture, MoE
hints, and the real attention projection module names — so you can confirm the
`keys` in lora_config.yaml match this model (the #1 thing to get right on an MoE).

Usage: python py/inspect_model.py <hf_repo_or_local_dir>
"""
import json, sys
from pathlib import Path


def _read_json_maybe(repo: str, fname: str):
    p = Path(repo) / fname
    if p.exists():
        return json.loads(p.read_text())
    try:
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(repo, fname)).read_text())
    except Exception as e:
        print(f"  (could not fetch {fname}: {e})")
        return None


def main():
    if len(sys.argv) < 2:
        print("usage: inspect_model.py <hf_repo_or_local_dir>")
        sys.exit(2)
    repo = sys.argv[1]
    cfg = _read_json_maybe(repo, "config.json") or {}
    arch = cfg.get("architectures", ["?"])
    print(f"architecture: {arch}")
    moe_keys = [k for k in cfg
                if any(t in k.lower() for t in ("expert", "moe", "router"))]
    print(f"MoE config keys: {moe_keys or '(none found)'}")
    for k in ("model_type", "num_hidden_layers", "hidden_size",
              "num_experts", "num_local_experts"):
        if k in cfg:
            print(f"  {k}: {cfg[k]}")

    idx = (_read_json_maybe(repo, "model.safetensors.index.json") or {})
    weights = list((idx.get("weight_map") or {}).keys())
    if weights:
        proj = sorted({
            ".".join(w.split(".")[-3:-1])  # e.g. self_attn.q_proj
            for w in weights if "proj" in w
        })
        print("\nprojection modules found (use the attention ones as LoRA keys):")
        for p in proj:
            print(f"  - {p}")
        if not any("self_attn" in p for p in proj):
            print("  NOTE: no 'self_attn.*_proj' — this arch names attention "
                  "differently (hybrid Mamba?). Update lora_config.yaml keys.")
    else:
        print("\n(no safetensors index — single-file model; load it to list "
              "modules)")


if __name__ == "__main__":
    main()
