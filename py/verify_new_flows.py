#!/usr/bin/env python3
"""Verify the newest flows (sub-axis, refine, coordinator-tasks) generate
correctly. generate() now returns serve-shaped EXAMPLES (one per assistant
generation point, grouped by "conv"), so checks run over the union of each
conversation's examples."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from gen_training_data import generate

examples = generate(800, {"setup", "discovery", "tasks"}, 3)
print(f"total examples: {len(examples)}")

# Reassemble per-conversation views: the LAST example of a group carries the
# fullest (possibly trimmed) history; the union of all its examples covers
# everything that conversation contains.
groups = {}
for e in examples:
    groups.setdefault(e["conv"], []).append(e)
print(f"conversations: {len(groups)} (avg {len(examples)/len(groups):.1f} ex/conv)")


def texts(exs):
    out = []
    for e in exs:
        for m in e["messages"]:
            if isinstance(m.get("content"), str):
                out.append(m["content"])
            for tc in m.get("tool_calls") or []:
                out.append(json.dumps(tc))
    return "\n".join(out)


def tool_result_texts(exs):
    return "\n".join(m["content"] for e in exs for m in e["messages"]
                     if m.get("role") == "tool" and isinstance(m.get("content"), str))


convos = list(groups.values())
sub = [g for g in convos if 'NEXT: "' in tool_result_texts(g)]
print(f"sub-axis convos (NEXT: \" in an actual tool result): {len(sub)}")

refine = [g for g in convos if "read_plan" in texts(g) and "update_plan" in texts(g)]
print(f"refine convos (read_plan + update_plan): {len(refine)}")

coord = [g for g in convos if "list_agents" in texts(g) and "create_task" in texts(g)]
print(f"coordinator task convos (list_agents + create_task): {len(coord)}")

problems = []

# Serve-shape invariants on EVERY example
for e in examples:
    ms = e["messages"]
    if ms[-1]["role"] != "assistant":
        problems.append("example whose final msg is not the assistant target")
        break
boards = [e for e in examples
          if any(m["role"] == "system" and "BOARD STATE" in (m.get("content") or "")
                 for m in e["messages"][1:])]
for e in boards:
    ms = e["messages"]
    idxs = [i for i, m in enumerate(ms) if m["role"] == "system"
            and "BOARD STATE" in (m.get("content") or "")]
    if len(idxs) != 1 or idxs[0] != len(ms) - 2:
        problems.append("board-state msg not exactly once at the tail")
        break
print(f"examples with board state at tail: {len(boards)}")

if sub:
    t = texts(sub[0])
    if "ask_question" not in t:
        problems.append("sub-axis convo missing ask_question follow-up")
    if "BOARD STATE" not in t:
        problems.append("sub-axis convo missing BOARD STATE")
    for line in tool_result_texts(sub[0]).split("\n"):
        if 'NEXT: "' in line:
            print(f"  sample NEXT line: {line.strip()[:200]}")
            break
else:
    problems.append("NO sub-axis convos generated")

if refine:
    t = texts(refine[0])
    if "Updated " not in t:
        problems.append("refine convo missing plan_updated result")
    if "- [ ]" not in t:
        problems.append("refine convo missing checklist merge")
    if "BOARD STATE" in t:
        problems.append("refine convo has BOARD STATE (interview-phase only!)")
else:
    problems.append("NO refine convos generated")

if coord:
    t = texts(coord[0])
    if "agent_persona_id" not in t:
        problems.append("coordinator create_task missing agent_persona_id")
    if "Available agents" not in t:
        problems.append("coordinator convo missing agents list result")
else:
    problems.append("NO coordinator task convos generated")

if problems:
    print("PROBLEMS:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("ALL NEW FLOWS VERIFIED OK")
