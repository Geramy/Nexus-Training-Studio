#!/usr/bin/env bash
# LoRA fine-tune via MLX. Args: <base_model> [lora_config.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."   # studio root

MODEL="${1:?base model required}"
CONFIG="${2:-py/lora_config.yaml}"
mkdir -p workspace/adapters

# Render {base_model} into a concrete config mlx-lm can read.
RENDERED="workspace/lora_config.rendered.yaml"
sed "s|{base_model}|${MODEL//|/\\|}|g" "$CONFIG" > "$RENDERED"

echo "▶ mlx_lm.lora  model=$MODEL  config=$RENDERED"
python -m mlx_lm lora --config "$RENDERED"
echo "✓ adapter written to workspace/adapters"
