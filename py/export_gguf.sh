#!/usr/bin/env bash
# Merge the LoRA adapter and export GGUF(s) at one or more quantizations.
# Uses an importance matrix (imatrix) for the BEST-quality K-quants — this is the
# stock-llama.cpp equivalent of the "_XL" dynamic quants (Q6_K / Q4_K_M etc.).
# Args: <fuse_model> <llama_cpp_dir> <quant_csv>   e.g.  Q8_0,Q6_K,Q4_K_M
set -euo pipefail
cd "$(dirname "$0")/.."   # studio root

MODEL="${1:?fuse model required}"
LLAMA="${2:?llama.cpp dir required}"
QUANT_CSV="${3:-Q8_0,Q6_K,Q4_K_M}"
# Two interpreters: MLX_PY (mlx-lm, for fuse) and CONVERT_PY (torch, for the GGUF
# converter). They may differ — Nemotron's converter needs torch, which has no
# wheel for the mlx env's Python. Default both to `python`.
MLX_PY="${MLX_PY:-python}"
CONVERT_PY="${CONVERT_PY:-python}"
mkdir -p workspace/gguf

# Fuse the adapter into the (possibly 8-bit) base and DE-QUANTIZE to f16 so it
# converts cleanly to GGUF, then quantize down from there.
echo "▶ fuse adapter (de-quantized to f16) → workspace/fused"
"$MLX_PY" -m mlx_lm fuse \
  --model "$MODEL" \
  --adapter-path workspace/adapters \
  --save-path workspace/fused \
  --dequantize

echo "▶ convert → GGUF f16"
"$CONVERT_PY" "$LLAMA/convert_hf_to_gguf.py" workspace/fused \
  --outfile workspace/gguf/model-f16.gguf --outtype f16

bin() { [ -x "$LLAMA/build/bin/$1" ] && echo "$LLAMA/build/bin/$1" || echo "$LLAMA/$1"; }
QBIN="$(bin llama-quantize)"
IMBIN="$(bin llama-imatrix)"

# ── Build a calibration corpus from our own training data ───────────────────
# imatrix needs representative text; our training conversations are exactly that.
CAL="workspace/gguf/calibration.txt"
"$MLX_PY" - "$CAL" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
n = 0
with open(out, "w") as w:
    for path in (glob.glob("workspace/data/*.jsonl")
                 + glob.glob("workspace/data/conversations/*.json")):
        try:
            with open(path) as f:
                if path.endswith(".jsonl"):
                    rows = [json.loads(l) for l in f if l.strip()]
                else:
                    rows = [json.load(f)]
        except Exception:
            continue
        for r in rows:
            for m in r.get("messages", []):
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    w.write(c.strip() + "\n"); n += 1
print(f"calibration lines: {n}")
PY

# ── Importance matrix (skip gracefully if the binary or corpus is missing) ───
IMATRIX_ARG=""
if [ -x "$IMBIN" ] && [ -s "$CAL" ]; then
  echo "▶ compute imatrix (best-quality calibration)"
  if "$IMBIN" -m workspace/gguf/model-f16.gguf -f "$CAL" \
       -o workspace/gguf/imatrix.dat --chunks 64; then
    IMATRIX_ARG="--imatrix workspace/gguf/imatrix.dat"
    echo "  ✓ imatrix → workspace/gguf/imatrix.dat"
  else
    echo "  ⚠ imatrix failed — falling back to plain K-quants."
  fi
else
  echo "⚠ no llama-imatrix or calibration data — using plain K-quants."
fi

IFS=',' read -ra QUANTS <<< "$QUANT_CSV"
for Q in "${QUANTS[@]}"; do
  Q="$(echo "$Q" | xargs)"   # trim
  [ -z "$Q" ] && continue
  echo "▶ quantize → $Q"
  # Q8_0 doesn't benefit from an imatrix; everything else does.
  if [ "$Q" = "Q8_0" ] || [ -z "$IMATRIX_ARG" ]; then
    "$QBIN" workspace/gguf/model-f16.gguf "workspace/gguf/model-$Q.gguf" "$Q"
  else
    "$QBIN" $IMATRIX_ARG \
      workspace/gguf/model-f16.gguf "workspace/gguf/model-$Q.gguf" "$Q"
  fi
  echo "  ✓ workspace/gguf/model-$Q.gguf"
done
echo "✓ GGUF export done — load these in lemonade."
