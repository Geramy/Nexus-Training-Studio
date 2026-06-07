# Nexus Training Studio

A small **Flutter macOS app** that fine-tunes a model for the Nexus agents on
Apple Silicon (M-series), then exports a **GGUF** you can load in lemonade / the
router. Sibling to `nexus_projects_client`.

It is the **control plane**: a one-screen UI + an **HTTPS API** the Nexus agents
call to push training examples, plus a **model scanner** (LM Studio, Hugging Face
cache, lemonade). The actual training is a **Python + MLX** pipeline it runs for
you (Flutter/Dart can't run MLX directly).

```
Nexus agents ──HTTPS POST /training-data──▶  Training Studio (Flutter)
                                               │  scans models · shows logs
                                               ▼  shells out to →  py/ (MLX)
   data → mlx_lm.lora (LoRA, attention-only, completion-masked)
        → mlx_lm.fuse (merge adapter)
        → llama.cpp convert_hf_to_gguf → llama-quantize  →  *.gguf
```

## Why this design avoids "it messed up the model"
Baked into `py/lora_config.yaml` + `py/prepare_data.py`:
- **LoRA on the instruct checkpoint**, never full fine-tune.
- **Attention-only** LoRA targets — the MoE router/experts are left alone (the
  classic way an MoE gets wrecked). *Verify the key names match this arch — see
  "Architecture risk" below.*
- **Completion-only loss** + the model's **exact chat template** (mlx-lm applies
  both for a `messages` dataset, so train==serve format).
- Conservative **rank 16 / scale 32 / lr 1.5e-4 / 1–2 epochs** + a behavioral
  **eval gate** (`py/eval.py`) before you export.

## ⚠️ Architecture risk (read before a long run)
Base model: **nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** — a hybrid
Mamba/Transformer **MoE**. Two things to confirm on YOUR machine first (the UI's
**“Check support”** button does both, fast):
1. `mlx-lm` can load + LoRA this architecture. If not, training can't run on MLX
   yet — fall back to a CUDA box (PEFT/QLoRA) and use this studio only for data +
   GGUF export.
2. Your `llama.cpp` `convert_hf_to_gguf.py` supports this arch (you already run it
   as GGUF, so conversion likely works).
Also confirm the LoRA `keys` in `lora_config.yaml` match this model's attention
projection module names (run `py/inspect_model.py`).

## Run it
```bash
cd training_studio
flutter create .            # generate macOS platform scaffolding (one time)
flutter pub get
# Python side (one time):
python3 -m venv .venv && source .venv/bin/activate
pip install -r py/requirements.txt
# Edit config.yaml: set llama_cpp_dir + python path if needed.
flutter run -d macos
```
The HTTPS API starts on `https://localhost:8443` (self-signed cert auto-generated
in `workspace/certs/`). Point the Nexus agents' "training sink" at it.

## API (for the agents to feed data)
- `POST /training-data` — body `{"messages":[{role,content,tool_calls,...}, ...]}`
  (one validated conversation). Appended to `workspace/data/raw.jsonl`.
- `POST /training-data/batch` — body `{"items":[{messages:[...]}, ...]}`.
- `GET  /models` — scanned models (path, format, sizeGB, source).
- `GET  /status` — current pipeline stage + counts.
- `GET  /health` — ok.

## Workspace layout
```
workspace/
  data/raw.jsonl          # everything the agents posted
  data/train.jsonl        # prepared (messages format) — mlx-lm reads this dir
  data/valid.jsonl
  adapters/               # mlx_lm.lora output (LoRA adapter)
  fused/                  # mlx_lm.fuse output (merged safetensors)
  gguf/                   # converted + quantized GGUF (load this in lemonade)
  certs/                  # self-signed TLS cert for the HTTPS API
```

This is a v1 scaffold — run it on the Mac and iterate. It is NOT tested here (no
MLX/Flutter in the build env); expect to tweak the arch-specific `keys` and paths.
