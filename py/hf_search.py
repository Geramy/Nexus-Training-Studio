#!/usr/bin/env python3
"""Search Hugging Face for MODEL or DATASET repos by free text — powers the
download-field and dataset-import autocompletes. Prints a JSON list of repo ids.
Honors HF_TOKEN (so private/org repos you can see show up too).

Usage: hf_search.py "<query>" [limit] [--datasets]
"""
import json, os, sys
from huggingface_hub import HfApi

args = [a for a in sys.argv[1:] if a != "--datasets"]
is_datasets = "--datasets" in sys.argv
q = args[0] if args else ""
limit = int(args[1]) if len(args) > 1 else 25
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

if not q.strip():
    print(json.dumps([])); sys.exit(0)

api = HfApi(token=token)
try:
    if is_datasets:
        items = api.list_datasets(search=q, limit=limit, sort="downloads")
    else:
        items = api.list_models(search=q, limit=limit, sort="downloads")
    print(json.dumps([it.id for it in items]))
except Exception as e:  # noqa: BLE001
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
