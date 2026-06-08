#!/usr/bin/env python3
"""Seed-data loader. Application/training-specific data (industries, app types,
libraries, phrasings, tool schemas, …) lives as JSON — NOT hardcoded in the
generator — so it's editable (incl. from the UI) and the bulk stays out of git.

Layout:
  workspace/seeds/<name>.json     ← the REAL, editable data (gitignored)
  seeds/<name>.example.json       ← a small committed EXAMPLE (documents shape)

load_seed() reads the workspace copy; if it's missing it bootstraps it from the
committed example so a fresh clone still runs.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "workspace" / "seeds"
EXAMPLES = ROOT / "seeds"


def seed_path(name):
    return WORK / f"{name}.json"


def example_path(name):
    return EXAMPLES / f"{name}.example.json"


def load_seed(name):
    """Return the parsed seed JSON for [name], bootstrapping from the example."""
    p = seed_path(name)
    if not p.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        ex = example_path(name)
        if ex.exists():
            shutil.copy(ex, p)
        else:
            raise FileNotFoundError(
                f"seed '{name}' missing and no example at {ex}")
    return json.loads(p.read_text())


def save_seed(name, obj):
    WORK.mkdir(parents=True, exist_ok=True)
    seed_path(name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def list_seeds():
    """All seed names known (from workspace + examples)."""
    names = set()
    if WORK.exists():
        names |= {p.stem for p in WORK.glob("*.json")}
    if EXAMPLES.exists():
        names |= {p.name[:-len(".example.json")]
                  for p in EXAMPLES.glob("*.example.json")}
    return sorted(names)
