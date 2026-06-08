#!/usr/bin/env python3
"""Import training data from an Excel (.xlsx) or CSV file into the dataset.

Two supported layouts (auto-detected from the columns):

  LONG (one row per message — the full tool-calling shape):
    conversation_id | role | content | name | tool_calls | tool_call_id
    - role: system | user | assistant | tool
    - tool_calls: JSON array string, e.g.
        [{"id":"c1","type":"function","function":{"name":"ask_question",
          "arguments":"{\"question\":\"…\",\"options\":[\"A\",\"B\"]}"}}]
    - tool_call_id: the id this tool result answers (for role=tool rows)
    Rows are grouped by conversation_id, in file order.

  SIMPLE (one row per example):
    system | user | assistant         (or prompt/response, instruction/output)

Usage:
  import_excel.py <file.xlsx|file.csv>
  import_excel.py --template <out.xlsx>   # write a fill-in template
"""
import json
import sys
from pathlib import Path

import pandas as pd

from data_common import append_conversations

LONG_COLS = {"role", "content"}


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str).fillna("")
    return pd.read_csv(path, dtype=str).fillna("")


def _parse_long(df: pd.DataFrame):
    convos, order = {}, []
    for _, row in df.iterrows():
        cid = (row.get("conversation_id") or "conv").strip() or "conv"
        role = (row.get("role") or "").strip().lower()
        if role not in ("system", "user", "assistant", "tool"):
            continue
        msg = {"role": role}
        content = row.get("content", "")
        msg["content"] = content if content != "" else None
        if row.get("name", "").strip():
            msg["name"] = row["name"].strip()
        tc = row.get("tool_calls", "").strip()
        if tc:
            try:
                msg["tool_calls"] = json.loads(tc)
            except json.JSONDecodeError:
                print(f"  warn: bad tool_calls JSON in {cid}, skipped that field")
        if row.get("tool_call_id", "").strip():
            msg["tool_call_id"] = row["tool_call_id"].strip()
        if cid not in convos:
            convos[cid] = []
            order.append(cid)
        convos[cid].append(msg)
    return [{"messages": convos[c]} for c in order]


def _parse_simple_clean(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    convos = []
    # prompt/response style
    pairs = [("prompt", "response"), ("instruction", "output"),
             ("question", "answer"), ("user", "assistant")]
    for a, b in pairs:
        if a in cols and b in cols:
            for _, row in df.iterrows():
                msgs = []
                if "system" in cols and str(row[cols["system"]]).strip():
                    msgs.append({"role": "system",
                                 "content": str(row[cols["system"]]).strip()})
                u = str(row[cols[a]]).strip()
                v = str(row[cols[b]]).strip()
                if not u or not v:
                    continue
                msgs.append({"role": "user", "content": u})
                msgs.append({"role": "assistant", "content": v})
                convos.append({"messages": msgs})
            return convos
    return []


def write_template(out: Path):
    df = pd.DataFrame([
        {"conversation_id": "example-1", "role": "system",
         "content": "You are the Setup host…", "name": "",
         "tool_calls": "", "tool_call_id": ""},
        {"conversation_id": "example-1", "role": "user",
         "content": "A mobile app for a lemonade stand.", "name": "",
         "tool_calls": "", "tool_call_id": ""},
        {"conversation_id": "example-1", "role": "assistant",
         "content": "", "name": "",
         "tool_calls": json.dumps([{
             "id": "c1", "type": "function",
             "function": {"name": "propose_tags",
                          "arguments": json.dumps({"tags": [
                              {"category": "platforms", "value": "iOS"}]})}}]),
         "tool_call_id": ""},
        {"conversation_id": "example-1", "role": "tool",
         "content": "{\"ok\":true}", "name": "propose_tags",
         "tool_calls": "", "tool_call_id": "c1"},
    ])
    df.to_excel(out, index=False)
    print(f"Template written → {out}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--template":
        write_template(Path(sys.argv[2]))
        return 0
    if len(sys.argv) < 2:
        print("usage: import_excel.py <file.xlsx|csv> | --template <out.xlsx>")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1
    df = _read(path)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower() for c in df.columns}
    if LONG_COLS.issubset(lower):
        df.columns = [c.lower() for c in df.columns]
        convos = _parse_long(df)
        layout = "long (per-message)"
    else:
        convos = _parse_simple_clean(df)
        layout = "simple (prompt/response)"
    if not convos:
        print(f"No conversations parsed from {path.name}. Columns: "
              f"{list(df.columns)}. See --template for the expected layout.")
        return 1
    added, skipped = append_conversations(convos, source=f"excel:{path.name}")
    print(f"Imported {added} conversation(s) [{layout}] from {path.name} "
          f"(skipped {skipped} dup/invalid).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
