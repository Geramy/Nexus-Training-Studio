#!/usr/bin/env python3
"""Download a model from Hugging Face into the local HF cache (so the scanner
finds it). Honors HF_TOKEN for private/gated repos.  Usage: hf_download.py <repo>"""
import os, sys
from huggingface_hub import snapshot_download

if len(sys.argv) < 2:
    print("usage: hf_download.py <org/repo>"); sys.exit(2)
repo = sys.argv[1]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
print(f"Downloading {repo} (token: {'yes' if token else 'no'}) …")
path = snapshot_download(repo_id=repo, token=token)
print(f"Done → {path}")
