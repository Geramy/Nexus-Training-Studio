#!/usr/bin/env bash
# Merge the LoRA adapter and export a quantized GGUF.
# Args: <base_model> <llama_cpp_dir> [quant=Q4_K_M]
set -euo pipefail
cd "$(dirname "$0")/.."   # studio root

MODEL="${1:?base model required}"
LLAMA="${2:?llama.cpp dir required}"
QUANT="${3:-Q4_K_M}"
mkdir -p workspace/gguf

echo "▶ fuse adapter → workspace/fused"
python -m mlx_lm fuse \
  --model "$MODEL" \
  --adapter-path workspace/adapters \
  --save-path workspace/fused

echo "▶ convert → GGUF (f16)"
python "$LLAMA/convert_hf_to_gguf.py" workspace/fused \
  --outfile workspace/gguf/model-f16.gguf --outtype f16

# llama-quantize lives in build/bin (cmake) or repo root depending on build.
QBIN="$LLAMA/build/bin/llama-quantize"
[ -x "$QBIN" ] || QBIN="$LLAMA/llama-quantize"
echo "▶ quantize → $QUANT"
"$QBIN" workspace/gguf/model-f16.gguf "workspace/gguf/model-$QUANT.gguf" "$QUANT"

echo "✓ GGUF ready: workspace/gguf/model-$QUANT.gguf  (load in lemonade)"
