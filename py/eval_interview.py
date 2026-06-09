#!/usr/bin/env python3
"""Behavioral eval of the SETUP interview, run against a SERVED GGUF (llama-server
or any OpenAI-compatible /v1/chat/completions with --jinja tool calling).

Unlike the BFCL exact-match eval, this simulates a full multi-turn interview and
checks the properties that actually matter for the product:
  • COVERAGE   — every REQUIRED topic ends up tagged (nothing skipped),
  • ONCE-ONLY  — no topic is asked more than once (no looping / re-asking),
  • COMPLETES  — the host reaches finalize_setup,
and records throughput from the server timings:
  • TPS        — generation tokens/sec  (predicted_per_second),
  • PP/s       — prompt processing/sec  (prompt_per_second).

Usage:
  eval_interview.py --endpoint http://127.0.0.1:8099/v1/chat/completions \
                    --model q4 --scenarios 8 --max-turns 16
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

SEEDS = Path("workspace/seeds")
REQUIRED = ["industries", "platforms", "objectives", "features", "languages", "frameworks"]

# Vocab → topic, to map an ask_question (which carries no category) back to a topic
# by what its options look like. Plus keyword fallbacks on the question text.
VOCAB = {
    "platforms": {"web", "ios", "android", "macos", "windows", "linux", "embedded", "cloud / server"},
    "languages": {"dart", "c", "c++", "c#", "java", "rust", "go", "python", "typescript", "sql"},
    "objectives": {"customer-facing ui", "admin dashboard", "public api", "realtime / streaming",
                   "data persistence", "offline support", "authentication", "payments", "machine learning"},
}
KEYWORDS = [
    ("platforms", ("platform", "surface", "run on", "device")),
    ("objectives", ("objective", "goal", "should do", "should it do", "what should")),
    ("features", ("feature", "capabilit")),
    ("industries", ("industry", "industries", "domain", "sector")),
    ("languages", ("language", "stack")),
    ("frameworks", ("framework",)),
    ("databases", ("database", "data store")),
]

# Test-case ideas live in an EDITABLE seed (workspace/seeds/interview_cases.json,
# example at seeds/interview_cases.example.json) so they show up in the Seeds UI
# and can be added to without touching code.
def load_cases():
    return json.loads((SEEDS / "interview_cases.json").read_text())["cases"]


def post(endpoint, body, timeout=180):
    req = urllib.request.Request(
        endpoint, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def topic_of(question, options):
    opts = {str(o).strip().lower() for o in (options or [])}
    for topic, vocab in VOCAB.items():
        if opts and len(opts & vocab) >= max(1, len(opts) // 2):
            return topic
    q = (question or "").lower()
    for topic, keys in KEYWORDS:
        if any(k in q for k in keys):
            return topic
    return None


def simulate(endpoint, model, system, tools, idea, max_turns):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": idea}]
    tagged = set()           # categories propose_tags has saved
    asked = []               # topics asked via ask_question (in order)
    repeats = []             # topics asked again after already asked/tagged
    finalized = False
    tps, pps = [], []
    turns = 0

    while turns < max_turns:
        turns += 1
        body = {"model": model, "messages": msgs, "tools": tools,
                "tool_choice": "auto", "temperature": 0.0, "stream": False}
        try:
            resp = post(endpoint, body)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        tm = resp.get("timings") or {}
        if tm.get("predicted_per_second"):
            tps.append(tm["predicted_per_second"])
        if tm.get("prompt_per_second"):
            pps.append(tm["prompt_per_second"])

        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            # Host replied in prose without a tool — a stall; nudge once then stop.
            msgs.append({"role": "assistant", "content": msg.get("content") or ""})
            msgs.append({"role": "user", "content": "Please continue setting it up."})
            continue

        msgs.append({"role": "assistant", "content": msg.get("content"),
                     "reasoning_content": msg.get("reasoning_content"),
                     "tool_calls": calls})
        done = False
        for c in calls:
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            cid = c.get("id", "c")

            if fn == "propose_tags":
                cats = {str(t.get("category", "")).strip().lower()
                        for t in (args.get("tags") or []) if t.get("category")}
                tagged |= cats
                result = {"ok": True, "added": len(args.get("tags") or [])}
            elif fn == "remove_tags":
                for t in (args.get("tags") or []):
                    tagged.discard(str(t.get("category", "")).strip().lower())
                result = {"ok": True}
            elif fn == "ask_question":
                topic = topic_of(args.get("question"), args.get("options"))
                # A repeat = asking about a topic already asked OR already tagged.
                if topic and (topic in {a for a in asked} or topic in tagged):
                    repeats.append(topic)
                asked.append(topic)
                picks = (args.get("options") or ["Yes"])[:2]
                result = {"answer": picks}
            elif fn == "finalize_setup":
                missing = [c for c in REQUIRED if c not in tagged]
                if missing:
                    result = {"ok": False, "error": "not ready", "missing": missing,
                              "message": "Add at least one tag to: " + ", ".join(missing)}
                else:
                    result = {"ok": True, "plans": ["/PLANS/Overview.md"]}
                    finalized = True
                    done = True
            else:
                result = {"ok": True}
            msgs.append({"role": "tool", "tool_call_id": cid, "name": fn,
                         "content": json.dumps(result)})
        if done:
            break

    return {"tagged": sorted(tagged), "asked": asked, "repeats": repeats,
            "finalized": finalized, "turns": turns,
            "missing": [c for c in REQUIRED if c not in tagged],
            "tps": tps, "pps": pps, "idea": idea, "messages": msgs}


def write_transcript(out_dir, model, i, r):
    """Write a scenario as raw JSON + a readable Markdown transcript so the run can
    be inspected turn-by-turn (reasoning, tool calls, simulated answers)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{model}-{i:02d}"
    base.with_suffix(".json").write_text(json.dumps(r, indent=2, ensure_ascii=False))

    L = [f"# Interview {i} — {model}", "", f"**Idea:** {r['idea']}", "",
         f"**finalized:** {r['finalized']}  |  **covered:** {not r['missing']}"
         f"  |  **repeats:** {r['repeats'] or 'none'}  |  **missing:** {r['missing'] or 'none'}",
         f"  |  **ask_questions:** {len(r['asked'])}  |  **turns:** {r['turns']}", ""]
    for m in r["messages"]:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            L += [f"### 🧑 user", m.get("content", ""), ""]
        elif role == "assistant":
            L.append("### 🤖 assistant")
            if m.get("reasoning_content"):
                L += [f"<think>{m['reasoning_content']}</think>", ""]
            if m.get("content"):
                L += [m["content"], ""]
            for c in (m.get("tool_calls") or []):
                fn = c["function"]["name"]
                L += [f"**→ {fn}**", "```json", c["function"].get("arguments", "{}"), "```", ""]
        elif role == "tool":
            L += [f"_↳ {m.get('name','tool')} result:_ `{m.get('content','')}`", ""]
    base.with_suffix(".md").write_text("\n".join(L))
    return base.with_suffix(".md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8099/v1/chat/completions")
    ap.add_argument("--model", default="gguf")
    ap.add_argument("--scenarios", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=16)
    args = ap.parse_args()

    tools = json.loads((SEEDS / "tool_schemas.json").read_text())["setup"]
    system_t = json.loads((SEEDS / "prompts.json").read_text())["setup_system"]

    cases = load_cases()
    out_dir = Path("workspace/interview_runs")
    runs = []
    for i in range(min(args.scenarios, len(cases))):
        idea = cases[i]["idea"]
        r = simulate(args.endpoint, args.model, system_t.format(name=f"App{i+1}"),
                     tools, idea, args.max_turns)
        if "error" in r:
            print(f"  scenario {i+1}: ERROR {r['error']}"); continue
        md = write_transcript(out_dir, args.model, i + 1, r)
        ok_cov, ok_once = not r["missing"], not r["repeats"]
        flag = "OK " if (ok_cov and ok_once and r["finalized"]) else "!! "
        print(f"  {flag}#{i+1} {idea[:40]!r:42} finalized={r['finalized']} "
              f"covered={ok_cov} repeats={r['repeats'] or '-'} asks={len(r['asked'])} "
              f"turns={r['turns']}  → {md}")
        runs.append(r)
    print(f"\nTranscripts (Markdown + JSON) written to: {out_dir}/")

    if not runs:
        print("no successful runs"); return 1
    n = len(runs)
    cov = sum(not r["missing"] for r in runs)
    once = sum(not r["repeats"] for r in runs)
    fin = sum(r["finalized"] for r in runs)
    clean = sum(not r["missing"] and not r["repeats"] and r["finalized"] for r in runs)
    all_tps = [x for r in runs for x in r["tps"]]
    all_pps = [x for r in runs for x in r["pps"]]
    pct = lambda k: f"{100*k/n:.0f}%"  # noqa: E731
    print(f"\n=== Interview eval over {n} scenarios ===")
    print(f"  COVERAGE (all required tagged):   {cov}/{n}  ({pct(cov)})")
    print(f"  ONCE-ONLY (no topic re-asked):    {once}/{n}  ({pct(once)})")
    print(f"  COMPLETES (reached finalize):     {fin}/{n}  ({pct(fin)})")
    print(f"  CLEAN (all three):                {clean}/{n}  ({pct(clean)})")
    print(f"  avg ask_questions / interview:    {statistics.mean(len(r['asked']) for r in runs):.1f}")
    if all_tps:
        print(f"  TPS  (gen tok/s):   mean {statistics.mean(all_tps):.1f}  "
              f"min {min(all_tps):.1f}  max {max(all_tps):.1f}")
    if all_pps:
        print(f"  PP/s (prompt tok/s): mean {statistics.mean(all_pps):.1f}  "
              f"min {min(all_pps):.1f}  max {max(all_pps):.1f}")

    # Persist a compact result the studio GUI reads (mirrors eval_result.json).
    Path("workspace").mkdir(exist_ok=True)
    Path("workspace/interview_result.json").write_text(json.dumps({
        "model": args.model, "endpoint": args.endpoint, "scenarios": n,
        "coverage_pct": round(100 * cov / n, 1),
        "once_only_pct": round(100 * once / n, 1),
        "completes_pct": round(100 * fin / n, 1),
        "clean_pct": round(100 * clean / n, 1),
        "avg_asks": round(statistics.mean(len(r["asked"]) for r in runs), 2),
        "tps_mean": round(statistics.mean(all_tps), 1) if all_tps else None,
        "pps_mean": round(statistics.mean(all_pps), 1) if all_pps else None,
        "transcripts": [f"workspace/interview_runs/{args.model}-{i+1:02d}.md"
                        for i in range(len(runs))],
        "cases": [{"idea": r["idea"], "finalized": r["finalized"],
                   "covered": not r["missing"], "repeats": r["repeats"],
                   "asks": len(r["asked"]), "turns": r["turns"]} for r in runs],
    }, indent=2))
    print("· wrote workspace/interview_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
