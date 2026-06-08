#!/usr/bin/env python3
"""List the Hugging Face model repos the current token can write to — the user's
own namespace plus every org they belong to. Emits JSON so the studio UI can
offer "pick an existing repo OR create a new one" without the user typing IDs.

Usage: hf_repos.py            # prints {"user":..,"orgs":[..],"repos":[..]}
"""
import json, os, sys
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    print(json.dumps({"error": "no HF token — set it in the studio first."}))
    sys.exit(1)

api = HfApi(token=token)
try:
    me = api.whoami()
except Exception as e:  # noqa: BLE001
    print(json.dumps({"error": f"whoami failed: {e}"}))
    sys.exit(1)

user = me.get("name")
orgs = [o.get("name") for o in me.get("orgs", []) if o.get("name")]
namespaces = [user] + orgs

repos = []
for ns in namespaces:
    try:
        for m in api.list_models(author=ns, token=token):
            repos.append(m.id)
    except Exception:  # noqa: BLE001
        pass

print(json.dumps({
    "user": user,
    "orgs": orgs,
    "namespaces": namespaces,
    "repos": sorted(set(repos)),
}))
