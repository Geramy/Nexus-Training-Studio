#!/usr/bin/env python3
"""Remap an MLX-fused NemotronH-MoE checkpoint to HF tensor naming so llama.cpp's
convert_hf_to_gguf.py can read it.

The ONLY naming difference (verified) is the routed MoE experts: MLX stacks them
as `...mixer.switch_mlp.fc1.weight` [E, ffn, hidden] and `...fc2.weight`
[E, hidden, ffn]; HF wants per-expert `...mixer.experts.{i}.up_proj.weight`
[ffn, hidden] and `...experts.{i}.down_proj.weight` [hidden, ffn]. It's a clean
slice along dim 0 — no transpose. Everything else is copied unchanged.

Run with a torch-capable Python (the model is bf16): .venv-convert/bin/python.
Usage: remap_mlx_to_hf.py <src_dir> <dst_dir>
"""
import glob
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 5_000_000_000


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    shard, shard_bytes, idx, weight_map = {}, 0, 1, {}

    def flush():
        nonlocal shard, shard_bytes, idx
        if not shard:
            return
        name = f"model-{idx:05d}.safetensors"
        save_file(shard, os.path.join(dst, name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        print(f"  wrote {name} ({len(shard)} tensors, "
              f"{shard_bytes/1e9:.1f} GB)")
        shard, shard_bytes, idx = {}, 0, idx + 1

    def add(k, t):
        nonlocal shard_bytes
        t = t.contiguous()
        if shard_bytes + t.numel() * t.element_size() > SHARD_BYTES:
            flush()
        shard[k] = t
        shard_bytes += t.numel() * t.element_size()

    files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    n_split = 0
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k.endswith("mixer.switch_mlp.fc1.weight"):
                    base = k[:-len("switch_mlp.fc1.weight")]
                    for i in range(t.shape[0]):
                        add(f"{base}experts.{i}.up_proj.weight", t[i].clone())
                    n_split += 1
                elif k.endswith("mixer.switch_mlp.fc2.weight"):
                    base = k[:-len("switch_mlp.fc2.weight")]
                    for i in range(t.shape[0]):
                        add(f"{base}experts.{i}.down_proj.weight", t[i].clone())
                    n_split += 1
                else:
                    add(k, t)
    flush()

    json.dump({"metadata": {}, "weight_map": weight_map},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"))

    # Copy config + tokenizer + everything non-weight.
    for fn in os.listdir(src):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))

    print(f"Done: split {n_split} switch_mlp tensors → per-expert, "
          f"{len(weight_map)} tensors total → {dst}")


if __name__ == "__main__":
    sys.exit(main())
