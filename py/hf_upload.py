#!/usr/bin/env python3
"""Upload a local model/GGUF folder (or file) to a Hugging Face repo — your
account OR an organization, public or private. Honors HF_TOKEN.

Usage: hf_upload.py <local_path> <dest_repo> [--private]
  dest_repo: "user/name" or "org/name"
"""
import os, sys
from huggingface_hub import create_repo, upload_folder, upload_file

if len(sys.argv) < 3:
    print("usage: hf_upload.py <local_path> <dest_repo> [--private]"); sys.exit(2)
local, dest = sys.argv[1], sys.argv[2]
private = "--private" in sys.argv[3:]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    print("ERROR: no HF token — set it in the studio (needed to push)."); sys.exit(1)
if not os.path.exists(local):
    print(f"ERROR: {local} does not exist."); sys.exit(1)

print(f"Creating repo {dest} (private={private}) …")
create_repo(dest, private=private, exist_ok=True, token=token, repo_type="model")
print(f"Uploading {local} → {dest} …")
if os.path.isdir(local):
    upload_folder(folder_path=local, repo_id=dest, token=token)
else:
    upload_file(path_or_fileobj=local, path_in_repo=os.path.basename(local),
                repo_id=dest, token=token)
print(f"Done → https://huggingface.co/{dest}")
