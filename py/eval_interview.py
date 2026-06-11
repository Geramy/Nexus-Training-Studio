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


def _tool_results():
    """Serve-exact result templates (same editable seed the generator uses)."""
    return json.loads((SEEDS / "tool_results.json").read_text())


def board_state(tr, by_cat):
    """setupStateSummary() (setup_tools.dart) — byte-identical text."""
    if not by_cat:
        return tr["board_state_empty"]
    lines = [f"- {c}: {', '.join(v)}" for c, v in by_cat.items()]
    return tr["board_state_header"] + "\n" + "\n".join(lines)


def _norm_pkg(raw):
    """SetupToolExecutor._normPkg — lowercase, keep repo segment of owner/repo."""
    s = str(raw).strip().lower()
    return s.rsplit("/", 1)[-1]


def _canonical(v):
    """LoopGuard._canonical — deterministic key-order-independent args dump."""
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_canonical(v[k])}" for k in sorted(map(str, v))) + "}"
    if isinstance(v, list):
        return "[" + ",".join(_canonical(x) for x in v) + "]"
    return json.dumps(v)


class LoopGuard:
    """core/agents/loop_guard.dart — proceed → warn (2nd) → block (3rd+) on
    CONSECUTIVE identical calls; any different call in between resets the
    streak (so a legitimate retry after the state changed is allowed)."""

    def __init__(self, warn_at=2, block_at=3):
        self.warn_at, self.block_at = warn_at, block_at
        self._last, self._streak = None, 0

    def observe(self, tool, args):
        fp = f"{tool}({_canonical(args)})"
        self._streak = self._streak + 1 if fp == self._last else 1
        self._last = fp
        if self._streak >= self.block_at:
            return "block"
        if self._streak >= self.warn_at:
            return "warn"
        return "proceed"


# Simulated user reply when the host hands the turn back with prose mid-interview
# (a NEW send() at serve: fresh board snapshot, anti-stall counters reset).
USER_CONTINUE = "Sounds good — please continue."


def simulate(endpoint, model, system, tools, idea, max_turns):
    """SERVE-FAITHFUL simulation of SetupSession.send() (setup_session.dart):

    • the BOARD STATE system message is computed ONCE per user turn (the app
      calls setupStateSummary() before the round loop) — it does NOT refresh
      as propose_tags lands mid-turn,
    • history is trimmed to the last 4 user-initiated turns (_recentHistory;
      continue-nudges don't count as turns),
    • ask_question answers come back as TOOL RESULTS in the same turn (the app
      awaits the picker inline) and set lastSelection until propose_tags runs,
    • the app's anti-stall ladder runs only when a round has NO tool calls:
      empty → continue-nudge ×2; lastSelection pending → record-nudge ×2;
      prose-question tells → ask-tool nudge ×2; undecided lookups/considers →
      reconcile ×2; otherwise the turn ENDS (prose hands back to the user) and
      we simulate the user replying = a new turn with a FRESH snapshot.
    """
    tr = _tool_results()
    hist = [{"role": "user", "content": idea}]  # working history (no system msg)
    by_cat = {}              # category → [values] (the board)
    asked = []               # topics asked via ask_question (in order)
    repeats = []             # topics asked again after already asked/tagged
    finalized = False
    tps, pps = [], []
    rounds = 0
    user_turns = 1
    # SetupToolExecutor state
    last_selection = None    # lastSelection: answer not yet saved via propose_tags
    pending = {}             # _pendingDecisions: norm name → label
    # Per-send() anti-stall counters (reset on every real user turn)
    empty_rounds = selection_nudges = ask_tool_nudges = reconcile_rounds = 0
    guard = LoopGuard()  # session-scoped, like the app's _loopGuard
    snapshot = board_state(tr, by_cat)  # frozen at turn start

    def recent(window=4):
        """_recentHistory(): last `window` user turns; continue-nudges excluded."""
        idx = [i for i, m in enumerate(hist)
               if m["role"] == "user" and m.get("content") != tr["nudge_continue"]]
        if len(idx) <= window:
            return hist
        return hist[idx[-window]:]

    while rounds < max_turns:
        rounds += 1
        # Serve-exact request shape: [system, …recent history, BOARD STATE] —
        # the board rides as a TRAILING system message so the cached
        # [system+tools] prefix stays byte-identical.
        body = {"model": model,
                "messages": [{"role": "system", "content": system}]
                            + recent()
                            + [{"role": "system", "content": snapshot}],
                "tools": tools,
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
            content = msg.get("content") or ""
            hist.append({"role": "assistant", "content": content,
                         "reasoning_content": msg.get("reasoning_content")})
            spoke = bool(content.strip())
            # 1) truly EMPTY round → continue-nudge (app: emptyRounds < 2)
            if not spoke and empty_rounds < 2:
                empty_rounds += 1
                hist.append({"role": "user", "content": tr["nudge_continue"]})
                continue
            # 2) answer acknowledged but never recorded → record-nudge
            if last_selection is not None and selection_nudges < 2:
                selection_nudges += 1
                sel, last_selection = last_selection, None
                hist.append({"role": "user",
                             "content": tr["nudge_record"].format(sel=sel)})
                continue
            # 3) asked a question in prose (no picker rendered) → ask-tool nudge
            low = content.lower()
            if ask_tool_nudges < 2 and (
                    "options:" in low or "select all" in low
                    or "(you may select" in low or "(select" in low):
                ask_tool_nudges += 1
                hist.append({"role": "user", "content": tr["nudge_ask_tool"]})
                continue
            # 4) looked-up/considered items left undecided → reconcile nudge
            if pending and reconcile_rounds < 2:
                reconcile_rounds += 1
                hist.append({"role": "user", "content": tr["nudge_reconcile"].format(
                    items=", ".join(pending.values()),
                    it_them="it" if len(pending) == 1 else "them")})
                continue
            # Turn ENDS — at serve a spoken no-tool reply hands back to the
            # user. Simulate the user's reply: a NEW turn → fresh snapshot,
            # counters reset.
            if finalized or user_turns >= 4:
                break
            user_turns += 1
            hist.append({"role": "user", "content": USER_CONTINUE})
            snapshot = board_state(tr, by_cat)
            empty_rounds = selection_nudges = ask_tool_nudges = reconcile_rounds = 0
            continue

        hist.append({"role": "assistant", "content": msg.get("content"),
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
            tagged = set(by_cat.keys())

            # Serve's LoopGuard: a 3rd identical call is REFUSED outright; a 2nd
            # one runs but carries a self-correct note appended to its result.
            action = guard.observe(fn, args)
            if action == "block":
                hist.append({"role": "tool", "tool_call_id": cid, "name": fn,
                             "content": tr["loop_block"].format(tool=fn)})
                continue

            if fn == "propose_tags":
                vals = []
                for t in (args.get("tags") or []):
                    cat = str(t.get("category", "")).strip().lower()
                    val = str(t.get("value", "")).strip()
                    if not cat or not val:
                        continue
                    vs = by_cat.setdefault(cat, [])
                    if val not in vs:
                        vs.append(val)
                    vals.append(val)
                    pending.pop(_norm_pkg(val), None)  # decision made: added
                last_selection = None  # cleared the moment propose_tags runs
                result = tr["propose"].format(values=", ".join(vals))
            elif fn == "remove_tags":
                vals = []
                for t in (args.get("tags") or []):
                    cat = str(t.get("category", "")).strip().lower()
                    val = str(t.get("value", "")).strip()
                    vs = by_cat.get(cat, [])
                    if val in vs:
                        vs.remove(val)
                        vals.append(val)
                    if not vs:
                        by_cat.pop(cat, None)
                result = tr["remove"].format(n=len(vals), values=", ".join(vals))
            elif fn == "ask_question":
                topic = topic_of(args.get("question"), args.get("options"))
                # A repeat = asking about a topic already asked OR already tagged.
                if topic and (topic in {a for a in asked} or topic in tagged):
                    repeats.append(topic)
                asked.append(topic)
                picks = [str(o) for o in (args.get("options") or ["Yes"])[:2]]
                last_selection = ", ".join(picks)  # pending until propose_tags
                result = tr["ask_selected"].format(picks=", ".join(picks))
            elif fn == "finalize_setup":
                missing = [c2 for c2 in REQUIRED if c2 not in tagged]
                if missing:
                    result = tr["finalize_missing"].format(
                        missing=", ".join(m.capitalize() for m in missing))
                else:
                    result = tr["finalize_ok"].format(
                        files="/PLANS/Overview.md, /PLANS/Client.md, "
                              "/PLANS/Server.md, /PLANS/Database.md")
                    finalized = True
                    done = True
            elif fn == "lookup_package":
                name = args.get("name", "")
                pending[_norm_pkg(name)] = name  # must be added or dismissed
                result = tr["lookup"].format(
                    name=name,
                    ecosystem=args.get("ecosystem", "pubdev"),
                    verdict="fresh", last_release="2026-04-15")
            elif fn == "consider_items":
                items = [str(i) for i in (args.get("items") or [])]
                for it in items:
                    pending[_norm_pkg(it)] = it
                result = tr["consider"].format(n=len(items),
                                               items=", ".join(items))
            elif fn == "dismiss_item":
                name = args.get("name", "")
                pending.pop(_norm_pkg(name), None)
                result = tr["dismiss"].format(
                    name=name,
                    reason=args.get("reason", "not needed"))
            elif fn == "scope_options":
                cat = args.get("category", "features")
                inds = by_cat.get("industries", ["the selected industry"])
                plat = args.get("platform")
                result = tr["scope_options"].format(
                    category=cat, scope=", ".join(inds),
                    platform=f" ({plat})" if plat else "",
                    values="(use sensible, domain-fitting options)")
            elif fn == "generate_image":
                result = tr["image_generated"]
            elif fn == "edit_image":
                result = tr["image_edited"]
            else:
                result = "ok"
            if action == "warn":
                result = f"{result}\n\n{tr['loop_warn'].format(tool=fn)}"
            hist.append({"role": "tool", "tool_call_id": cid, "name": fn,
                         "content": result})
        if done:
            break

    tagged = set(by_cat.keys())
    return {"tagged": sorted(tagged), "asked": asked, "repeats": repeats,
            "finalized": finalized, "turns": rounds,
            "missing": [c for c in REQUIRED if c not in tagged],
            "tps": tps, "pps": pps, "idea": idea, "messages": hist}


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
    ap.add_argument("--scenarios", type=int, default=0,
                    help="how many cases to run from the top (0 = ALL)")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--label", default="trained",
                    help="which side of the A/B this run is: 'base' or 'trained'")
    ap.add_argument("--case", type=int, default=None,
                    help="run ONLY this 1-based case for a CLEAN per-case TPS/PP "
                         "(no batch warmup blending); default runs all")
    args = ap.parse_args()

    # Match what the app actually SERVES: setup_session.dart passes
    # includeLibraryTools: TRUE, so the library-verification tools are part of
    # the served tool list and must be part of the eval's too — train == serve.
    # (We previously excluded them based on a stale reading of the flag.)
    tools = json.loads((SEEDS / "tool_schemas.json").read_text())["setup"]
    system_t = json.loads((SEEDS / "prompts.json").read_text())["setup_system"]
    all_cases = load_cases()
    out_dir = Path("workspace/interview_runs")

    # Which seed cases to run: a single 1-based --case, else the first --scenarios.
    if args.case is not None:
        idxs = [args.case - 1] if 1 <= args.case <= len(all_cases) else []
    elif args.scenarios <= 0:
        idxs = list(range(len(all_cases)))  # 0 = run every case
    else:
        idxs = list(range(min(args.scenarios, len(all_cases))))
    if not idxs:
        print("no cases to run (check --case range)"); return 1

    def case_result(i, r):
        tps = statistics.mean(r["tps"]) if r["tps"] else None
        pps = statistics.mean(r["pps"]) if r["pps"] else None
        return {"idea": r["idea"], "finalized": r["finalized"],
                "covered": not r["missing"], "repeats": r["repeats"],
                "asks": len(r["asked"]), "turns": r["turns"],
                "tps": round(tps, 1) if tps else None,
                "pps": round(pps, 1) if pps else None,
                "transcript": f"workspace/interview_runs/{args.label}-{i + 1:02d}.md"}

    # Load the keyed result once, keep a per-seed-index `cases` list, and re-write
    # it AFTER EVERY CASE so the studio UI updates live as the run progresses.
    out = Path("workspace/interview_result.json")
    out.parent.mkdir(exist_ok=True)
    data = {}
    if out.exists():
        try:
            data = json.loads(out.read_text())
        except Exception:  # noqa: BLE001
            data = {}
    if "coverage_pct" in data:  # migrate a legacy flat result → keyed
        data = {"trained": data}
    cases = (data.get(args.label, {}).get("cases") or [])
    while len(cases) < len(all_cases):
        cases.append(None)
    # A NEW multi-case run starts from a CLEAN SLATE: stale pass/fail results
    # from the previous run must not linger in the studio panel, and must not
    # blend into this run's percentages. A single --case spot-check still
    # updates only its own slot so the rest of the table keeps its context.
    if args.case is None:
        cases = [None] * len(all_cases)

    def flush(running: bool, ran_this: int):
        done = [c for c in cases if c]
        tpsv = [c["tps"] for c in done if c.get("tps")]
        ppsv = [c["pps"] for c in done if c.get("pps")]

        def pct(n):
            return round(100 * n / len(done), 1) if done else None

        data[args.label] = {
            "model": args.model, "endpoint": args.endpoint, "scenarios": len(done),
            # progress for THIS run (not cumulative non-null) so a re-run shows
            # "running 3/27", not "running 27/27".
            "running": running, "ran_this": ran_this, "total": len(idxs),
            "coverage_pct": pct(sum(1 for c in done if c["covered"])),
            "once_only_pct": pct(sum(1 for c in done if not c["repeats"])),
            "completes_pct": pct(sum(1 for c in done if c["finalized"])),
            "avg_asks": round(statistics.mean(c["asks"] for c in done), 2) if done else None,
            "tps_mean": round(statistics.mean(tpsv), 1) if tpsv else None,
            "pps_mean": round(statistics.mean(ppsv), 1) if ppsv else None,
            "cases": cases,
        }
        out.write_text(json.dumps(data, indent=2))

    # Write the reset state IMMEDIATELY so the panel zeroes the moment the run
    # starts instead of showing the previous run until case 1 lands.
    flush(running=True, ran_this=0)

    ran = 0
    for n_idx, i in enumerate(idxs):
        idea = all_cases[i]["idea"]
        r = simulate(args.endpoint, args.model, system_t.format(name=f"App{i + 1}"),
                     tools, idea, args.max_turns)
        if "error" in r:
            print(f"  case {i + 1}: ERROR {r['error']}"); continue
        # Name transcripts by LABEL (base/trained), not model — both sides use the
        # same model id ("gguf"), so model-named files would overwrite each other.
        write_transcript(out_dir, args.label, i + 1, r)
        cases[i] = case_result(i, r)
        ran += 1
        cr = cases[i]
        flag = "OK " if (cr["covered"] and not cr["repeats"] and cr["finalized"]) \
            else "!! "
        print(f"  {flag}#{i + 1} {idea[:38]!r:40} cov={cr['covered']} "
              f"once={not cr['repeats']} fin={cr['finalized']} "
              f"tps={cr['tps']} pps={cr['pps']}")
        flush(running=(n_idx < len(idxs) - 1), ran_this=ran)  # persist per case

    if ran == 0:
        print("no successful runs"); return 1
    flush(running=False, ran_this=ran)
    print(f"· {ran} case(s) this run — interview_result.json updated per case, "
          f"transcripts in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
