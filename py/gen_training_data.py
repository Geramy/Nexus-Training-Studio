#!/usr/bin/env python3
"""Procedural generator for the Nexus combined model: tens of thousands of
high-quality, FORMAT-CORRECT tool-calling conversations for (1) Setup interview,
(2) Discovery / user-story building, and (3) Task generation.

Diversity is the #1 quality lever (research: +7.4% BFCL from request-phrasing +
argument-value diversity), so scenarios are sampled combinatorially from large
pools (domain nouns × app types × platform sets × stacks × phrasings) and each
yields several randomized conversation variants. Every example carries the REAL
tool schemas (mlx_lm "tools" format) and the app's exact message shapes, so
train == serve. Output is appended (deduped) to workspace/data/dataset.jsonl.

Usage:
  gen_training_data.py [--target 10000] [--kinds setup,discovery,tasks] [--seed 7]
"""
import argparse
import json
import random
import sys

from data_common import append_conversations
from tool_schemas import tools_for
from taxonomy import INDUSTRIES, INFER_REFLECTIONS, opener_for

# ── Seed data (editable JSON: full in workspace/seeds/, examples in seeds/) ──
from seedlib import load_seed
DOMAIN_NOUNS = load_seed("domains")
PLATFORM_SETS = load_seed("platform_sets")
STACKS = [{**s, "surfaces": set(s["surfaces"])} for s in load_seed("stacks")]
LIBRARIES = load_seed("libraries")
NAME_SUFFIXES = load_seed("name_suffixes")
APP_TYPES = load_seed("app_types")
_PH = load_seed("phrasings")
FULL_TEMPLATES = _PH["full_templates"]
SPARSE_TEMPLATES = _PH["sparse_templates"]
OBJ_QUESTIONS = _PH["obj_questions"]
FEAT_QUESTIONS = _PH["feat_questions"]
CONSTRAINTS = _PH["constraints"]
STYLE_PREFIX = _PH["style_prefix"]
STYLE_SUFFIX = _PH["style_suffix"]
AMBIGUOUS_IDEAS = _PH["ambiguous_ideas"]
VAGUE_OPENERS = _PH["vague_openers"]
CORRECTION_TEMPLATES = _PH["correction_templates"]
_PROMPTS = load_seed("prompts")          # setup/discovery/pm system prompts
STORY_PH = load_seed("story_phrasings")  # user-terminology walkthrough phrasing
# Industry → sub-axis catalog, derived from the APP's own scoped-vocab asset
# (assets/setup/scoped_vocab.json) so training tracks whatever the catalog says:
# every industry introduces its own follow-up axis (Gaming → Genre, Healthcare →
# Care Setting, …) whose chosen value re-scopes the objectives/features vocab.
SCOPED = load_seed("scoped_vocab")


# ───────────────────────── message + tool-call helpers ─────────────────────


class Ids:
    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return f"c{self.n}"


def sys_msg(t):
    return {"role": "system", "content": t}


def user_msg(t):
    return {"role": "user", "content": t}


# Short, tool-aware THINKING the assistant emits before it acts, per tool. The
# Nemotron chat template renders `reasoning_content` as `<think>…</think>` before
# the content/tool call, so seeding every assistant turn with a brief, closed
# thought teaches think-then-act AND to close the tag. These thoughts are EDITABLE
# JSON (workspace/seeds/reasoning.json, example at seeds/reasoning.example.json) —
# the propose_tags/ask_question/finalize thoughts carry the "tagged = done, don't
# re-ask" lesson, so editing them tunes the anti-loop behaviour without code.
# Seed values may be a single string OR a list of phrasings; lists are sampled
# per use so the same lesson appears in many wordings (verbatim-repeated
# reasoning gets memorized and parroted at serve time — run-2 failure mode).
_REASON_RNG = random.Random(0)


class _ReasonSeed(dict):
    def _pick(self, v):
        return _REASON_RNG.choice(v) if isinstance(v, list) else v

    def get(self, key, default=None):
        return self._pick(super().get(key, default))

    def __getitem__(self, key):
        return self._pick(super().__getitem__(key))


_REASON = _ReasonSeed(load_seed("reasoning"))


def _reason_for(calls):
    name = calls[0]["function"]["name"] if calls else None
    return _REASON.get(name) or _REASON.get("default", "")


def asst_text(t, reasoning=None):
    return {"role": "assistant", "content": t,
            "reasoning_content": reasoning or _REASON.get("wrapup", "")}


# Serve-EXACT tool-result strings (editable JSON: workspace/seeds/tool_results.json,
# example at seeds/tool_results.example.json) — verbatim copies of what the app's
# executor (setup_tools.dart) returns. Builders keep passing small result dicts;
# tool_result() renders them through these templates so the TRAINED result format
# is byte-identical to the SERVED one (the app returns prose strings, not JSON).
_TR = load_seed("tool_results")

# cid → (tool name, args) for the current conversation, so tool_result() can
# reconstruct the serve string from the original call (the executor formats its
# reply from the request args — e.g. propose_tags echoes the proposed values).
_CALL_ARGS = {}


def tool_call(ids, name, args):
    cid = ids.next()
    _CALL_ARGS[cid] = (name, args)
    return cid, {"id": cid, "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}


def asst_calls(calls, content=None, reasoning=None):
    return {"role": "assistant", "content": content if content else None,
            "tool_calls": calls,
            "reasoning_content": reasoning or _reason_for(calls)}


def _serve_result(cid, obj):
    """Render the result the REAL executor would return for this call."""
    name, args = _CALL_ARGS.get(cid, (None, {}))
    if name == "propose_tags":
        vals = [t["value"] for t in args.get("tags", [])]
        return _TR["propose"].format(values=", ".join(vals))
    if name == "remove_tags":
        vals = [t["value"] for t in args.get("tags", [])]
        return _TR["remove"].format(n=len(vals), values=", ".join(vals))
    if name == "ask_question":
        if obj.get("skipped"):
            return _TR["ask_skipped"]
        if obj.get("text"):
            return _TR["ask_answered"].format(text=obj["text"])
        return _TR["ask_selected"].format(picks=", ".join(obj.get("answer", [])))
    if name == "finalize_setup":
        if obj.get("ok"):
            files = obj.get("plans") or ["/PLANS/Overview.md", "/PLANS/Client.md",
                                         "/PLANS/Server.md", "/PLANS/Database.md"]
            return _TR["finalize_ok"].format(files=", ".join(files))
        missing = [m.capitalize() for m in obj.get("missing", [])]
        return _TR["finalize_missing"].format(missing=", ".join(missing))
    if name == "lookup_package":
        verdict = obj.get("verdict", "fresh")
        days = int(obj.get("last_release_days", 30))
        # plausible date: fixed reference minus the staleness window
        yr, rem = 2026, 150 - days
        while rem <= 0:
            yr, rem = yr - 1, rem + 365
        return _TR["lookup"].format(
            name=args.get("name", obj.get("name", "")),
            ecosystem=args.get("ecosystem", "pubdev"), verdict=verdict,
            last_release=f"{yr}-{max(1, min(12, rem // 30)):02d}-15")
    if name == "consider_items":
        items = args.get("items", [])
        return _TR["consider"].format(n=len(items), items=", ".join(items))
    if name == "dismiss_item":
        return _TR["dismiss"].format(name=args.get("name", ""),
                                     reason=args.get("reason", "not needed"))
    if name == "generate_image":
        return _TR["image_generated"]
    if name == "edit_image":
        return _TR["image_edited"]
    if name == "add_user_story":
        par = args.get("parent_story_id")
        return _TR["story_added"].format(
            title=args.get("title", ""), id=obj.get("id"),
            parent=_TR["story_added_parent"].format(parent=par) if par else "")
    if name == "move_user_story":
        par = args.get("parent_story_id")
        dest = f"under #{par}" if par else "to root"
        if args.get("order_index") is not None:
            dest += f" at position {args['order_index']}"
        return _TR["story_moved"].format(id=args.get("story_id"), dest=dest)
    if name == "update_user_story":
        return _TR["story_updated"].format(id=args.get("story_id"))
    if name == "add_note":
        return _TR["note_added"].format(id=obj.get("id", 1),
                                        story=args.get("story_id"))
    if name == "draft_stories_from_text":
        n = obj.get("made", 0)
        par = args.get("parent_story_id")
        return _TR["stories_drafted"].format(
            n=n, plural="story" if n == 1 else "stories",
            parent=f" under #{par}" if par else "")
    if name == "list_user_stories":
        stories = obj.get("stories", [])
        if not stories:
            return _TR["stories_list_empty"]
        lines = [_TR["stories_list_header"].format(n=len(stories))]
        for s in stories:
            par = s.get("parent")
            lines.append(_TR["stories_list_item"].format(
                id=s["id"], kind=s.get("kind", "story"),
                status=s.get("status", "draft"), title=s["title"],
                parent=_TR["stories_list_child"].format(parent=par)
                if par else ""))
        return "\n".join(lines) + "\n"
    if name == "create_task":
        return _TR["task_created"].format(
            title=args.get("title", ""), id=obj.get("id"),
            who=obj.get("who", "Worker"))
    if name == "update_task":
        return _TR["task_updated"].format(title=obj.get("title",
                                                        args.get("title", "")))
    if name == "update_task_status":
        return _TR["task_status"].format(title=obj.get("title", ""),
                                         status=args.get("status", ""))
    if name == "assign_agent_to_task":
        return _TR["agent_assigned"].format(persona=obj.get("persona", ""),
                                            title=obj.get("title", ""))
    if name == "list_tasks":
        rows = obj.get("tasks", [])
        lines = [_TR["tasks_list_header"].format(n=len(rows))]
        for t in rows:
            extra = (f" agent={t['agent']}" if t.get("agent") else "")
            if t.get("parent"):
                extra = f" subtask-of={t['parent']}" + extra
            lines.append(_TR["tasks_list_item"].format(
                id=t["id"], title=t["title"],
                priority=t.get("priority", "MED"),
                status=t.get("status", "TODO"), extra=extra))
        return "\n".join(lines)
    if name == "list_agents":
        rows = obj.get("agents", [])
        lines = [_TR["agents_list_header"].format(n=len(rows))]
        for a in rows:
            lines.append(_TR["agents_list_item"].format(
                id=a["id"], name=a["name"], role=a.get("role", "")))
        lines.append(_TR["agents_list_footer"])
        return "\n".join(lines)
    if name == "list_plans":
        return _TR["plans_list"].format(items="\n".join(
            f"- {x}" for x in obj.get("plans", [])))
    if name == "read_plan":
        return obj.get("content", "(empty file)")
    if name == "update_plan":
        return _TR["plan_updated"].format(
            name=str(args.get("path", "")).split("/")[-1])
    return None


def tool_result(cid, obj, name=None):
    content = obj if isinstance(obj, str) else (
        _serve_result(cid, obj) or json.dumps(obj))
    m = {"role": "tool", "tool_call_id": cid, "content": content}
    if name:
        m["name"] = name
    return m


# Serve-request reconstruction (setup_session.dart): each example must be the
# EXACT byte sequence the app sends when the model generates — anything else
# (e.g. board state after every user msg, loss on prompt tokens) trains a
# distribution the model never sees at serve time and causes parroting/loops.
_CONTINUE_NUDGE = _TR["nudge_continue"]
_RECORD_NUDGE_PREFIX = _TR["nudge_record"].split("{sel}")[0]


def _is_real_user(m):
    """True for user messages that start a TURN (the app recomputes the board
    snapshot once per turn). Nudges are user-role but mid-turn — no recompute."""
    if m["role"] != "user" or not isinstance(m.get("content"), str):
        return False
    c = m["content"]
    return c != _CONTINUE_NUDGE and not c.startswith(_RECORD_NUDGE_PREFIX)


def _trim_history(hist, window=4):
    """Mirror _recentHistory(): keep from the 4th-last user message. Continue-
    nudges are excluded from the turn count (record-nudges DO count) — exactly
    as setup_session.dart counts them."""
    idx = [i for i, m in enumerate(hist)
           if m["role"] == "user" and m.get("content") != _CONTINUE_NUDGE]
    if len(idx) <= window:
        return hist
    return hist[idx[-window]:]


def split_serve_points(msgs, board=True, window=4):
    """Explode one scripted conversation into one training example per assistant
    GENERATION POINT, each shaped byte-exactly like the app's request:
      [system] + trimmed history + [turn-start BOARD STATE system msg] + target
    The snapshot is recomputed ONCE per user turn (setupStateSummary() runs
    before the round loop), so mid-turn generations carry the board as it stood
    at turn start — tags the assistant proposed THIS turn are not in it yet.
    Tag effects apply AFTER each generation (the executor runs the call), which
    the replay mirrors by folding propose/remove into by_cat post-emission.

    An assistant message marked "_no_target": True appears in later examples'
    CONTEXT but is never itself a generation target — used to script a mistaken
    move (e.g. re-calling a rejected finalize so the loop-guard text appears)
    without TRAINING the model to make it. The marker is metadata only and is
    stripped from every emitted message."""
    sysm, hist = msgs[0], msgs[1:]
    by_cat, snapshot, out = {}, None, []

    def clean(m):
        if isinstance(m, dict) and "_no_target" in m:
            return {k: v for k, v in m.items() if k != "_no_target"}
        return m

    def render():
        if not by_cat:
            return _TR["board_state_empty"]
        lines = [f"- {c}: {', '.join(v)}" for c, v in by_cat.items()]
        return _TR["board_state_header"] + "\n" + "\n".join(lines)

    for i, m in enumerate(hist):
        if _is_real_user(m):
            snapshot = render()          # app computes it once per turn
        if m["role"] == "assistant":
            if not m.get("_no_target"):
                ctx = [clean(x) for x in
                       (_trim_history(hist[:i], window) if board else hist[:i])]
                tail = ([sys_msg(snapshot)]
                        if (board and snapshot is not None) else [])
                out.append([sysm] + ctx + tail + [m])
            for c in (m.get("tool_calls") or []):  # executor applies AFTER gen
                fn = c["function"]["name"]
                if fn not in ("propose_tags", "remove_tags"):
                    continue
                for t in json.loads(c["function"]["arguments"]).get("tags", []):
                    cat, val = t.get("category"), t.get("value")
                    if not cat or not val:
                        continue
                    if fn == "propose_tags":
                        vs = by_cat.setdefault(cat, [])
                        if val not in vs:
                            vs.append(val)
                    else:
                        vs = by_cat.get(cat, [])
                        if val in vs:
                            vs.remove(val)
                        if not vs:
                            by_cat.pop(cat, None)
    return out


# ───────────────────────── component pools ─────────────────────────────────



# Stacks chosen to fit the surface set.

# Candidate libraries per language, with a freshness verdict (some stale, to
# teach dismiss_item). (package, ecosystem, verdict)


def _platform_bucket(plats):
    s = set(plats)
    if s & {"iOS", "Android"}:
        return "Mobile"
    if "Web" in s:
        return "Web"
    if "Desktop" in s:
        return "Desktop"
    return "Cloud/Server"



# App archetypes — each provides templated objectives/features/flow/edges/tasks.
# {d} = domain noun. Tasks use {lang}/{fw}/{db} for stack-specific instructions.

# First-message phrasings (request-phrasing diversity is a top quality lever).


# ───────────────────────── input evolution (Evol-Instruct-style) ───────────
# Raise USER-message entropy: inject varied real-world constraints and shift
# register/length. Combinatorial → far higher distinct-n + unique-message ratio,
# which is the lever against fast memorization / poor generalization.


def evolve_user(R, msg):
    """Apply a random constraint clause + light register/length shift."""
    out = msg
    if R.random() < 0.6:
        c = R.choice(CONSTRAINTS).replace("{n}", str(R.choice(
            [20, 50, 100, 200, 500, 1000])))
        joiner = R.choice([" — ", ", ", ". Also ", "; ", " and ", ". Oh and "])
        out = out.rstrip(".") + joiner + c + "."
    r = R.random()
    if r < 0.12:
        out = out.lower()                      # casual all-lowercase
    elif r < 0.18:
        out = out.replace("I want", "i wanna").replace("going to", "gonna")
    pre, suf = R.choice(STYLE_PREFIX), R.choice(STYLE_SUFFIX)
    if pre:
        out = pre + out[0].lower() + out[1:]
    return out + suf


def umsg(R, text):
    """A user turn with evolved phrasing (entropy)."""
    return user_msg(evolve_user(R, text))


# ───────────────────────── system prompts ──────────────────────────────────

# System prompts are EDITABLE JSON (workspace/seeds/prompts.json, example at
# seeds/prompts.example.json), not hardcoded here — same data-driven pattern as the
# other seeds, and the single source so train == eval == serve. Keep in sync with
# the app's served setup/coordinator prompts.
def interview_system(p):
    return _PROMPTS["setup_system"].format(name=p["name"])


def _baseline(p):
    """The app's PROJECT BASELINE block, built from this scenario's interview
    tags exactly as project_baseline.dart renders it."""
    return _PROMPTS["project_baseline"].format(
        described=f"{p['idea']} — {p['blurb']}",
        industries=", ".join(p["industries"]),
        platforms=", ".join(p["platforms"]),
        objectives=", ".join(p["objectives"]),
        features=", ".join(p["features"]),
        languages=", ".join(p["languages"]),
        frameworks=", ".join(p["frameworks"]),
        databases=", ".join(p["databases"]),
        libraries=", ".join(p.get("libraries", []) or ["(none)"]),
        services=", ".join(p["services"]))


def discovery_system(p):
    """Serve-exact discovery system prompt: PROJECT BASELINE followed by the
    coordinator's discovery instructions — the app injects the baseline ABOVE
    the prompt, which says 'The PROJECT BASELINE above…'."""
    return _baseline(p) + "\n\n" + _PROMPTS["discovery_system"].format(
        name=p["name"])


def coordinator_system(p, context):
    """Serve-exact Coordinator system prompt (coordinator_session.dart):
    behavioral preamble + PROJECT BASELINE + live project context + the tool
    catalog (incl. the CRITICAL assign-an-agent rule and the speak-after-tools
    rule)."""
    return (_PROMPTS["coordinator_system"].format(name=p["name"]) + "\n\n"
            + _baseline(p) + "\n\nLive project context:\n" + context + "\n\n"
            + _PROMPTS["coordinator_catalog"])


def refine_system(p):
    return _PROMPTS["refine_system"].format(name=p["name"])


def pm_system(p):
    return _PROMPTS["pm_system"].format(
        name=p["name"],
        langs=", ".join(p["languages"]), fws=", ".join(p["frameworks"]),
        lang0=p["languages"][0], db0=p["databases"][0])


# ───────────────────────── scenario synthesis ──────────────────────────────

def make_scenario(R):
    # Start from a real industry + a NATURAL idea phrase (drives inference).
    industry = R.choice(list(INDUSTRIES.keys()))
    tax = INDUSTRIES[industry]
    idea = R.choice(tax["ideas"])
    # Pick an app archetype this industry implies (intersect with our templates).
    app_choices = [a for a in tax["apps"] if a in APP_TYPES] or list(APP_TYPES)
    app_key = R.choice(app_choices)
    app = APP_TYPES[app_key]
    domain = R.choice(DOMAIN_NOUNS)
    plats = R.choice(tax["platforms"])
    fits = [s for s in STACKS if s["surfaces"] & set(plats)] or STACKS
    stack = R.choice(fits)
    name = (idea.split()[-1] if idea.split()[-1].isalpha() else domain.split()[0])
    name = name.capitalize() + R.choice(NAME_SUFFIXES)

    # Objectives blend the industry's and the archetype's vocab (more coverage).
    obj_pool = list(dict.fromkeys(tax["objectives"] + app["objectives"]))
    objs = R.sample(obj_pool, R.randint(4, min(6, len(obj_pool))))
    feats = R.sample(app["features"], R.randint(3, min(5, len(app["features"]))))
    flow = app["flow"][:R.randint(4, min(6, len(app["flow"])))]
    edge = R.choice(app["edges"])
    role = R.choice(app["roles"])

    def fill(t):
        return (t.replace("{lang}", stack["languages"][0])
                 .replace("{fw}", stack["frameworks"][0])
                 .replace("{db}", stack["databases"][0])
                 .replace("{d}", domain))

    tasks = [{"title": fill(t), "description": fill(desc), "layer": layer}
             for (t, desc, layer) in app["tasks"]]

    return {
        "app": app_key, "name": name, "domain": domain, "idea": idea,
        "blurb": app["blurb"].format(d=domain),
        "industries": [industry],          # the GROUND TRUTH to infer from `idea`
        "platforms": plats,
        "objectives": objs, "features": feats,
        "languages": stack["languages"], "frameworks": stack["frameworks"],
        "databases": stack["databases"], "services": stack["services"],
        "role": role, "flow": flow, "edge": edge, "tasks": tasks,
        "epic_title": f"{app_key.capitalize()} for {domain}",
        "epic_narrative": f"As a {role}, I want to use {name}, so that the "
                          f"{domain} runs smoothly.",
    }


# ───────────────────────── tag helpers ─────────────────────────────────────

def _all_tags(p):
    return ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]]
            + [{"category": "objectives", "value": v} for v in p["objectives"]]
            + [{"category": "features", "value": v} for v in p["features"]]
            + [{"category": "languages", "value": v} for v in p["languages"]]
            + [{"category": "frameworks", "value": v} for v in p["frameworks"]]
            + [{"category": "databases", "value": v} for v in p["databases"]]
            + [{"category": "services", "value": v} for v in p["services"]])


def _stack_tags(p):
    return ([{"category": "languages", "value": v} for v in p["languages"]]
            + [{"category": "frameworks", "value": v} for v in p["frameworks"]]
            + [{"category": "databases", "value": v} for v in p["databases"]]
            + [{"category": "services", "value": v} for v in p["services"]])


def _plans_result():
    return {"ok": True, "plans": ["/PLANS/Overview.md", "/PLANS/Client.md",
                                  "/PLANS/Server.md", "/PLANS/Database.md"]}


def _fmt_list(xs):
    xs = [x.lower() for x in xs]
    return ", ".join(xs)


def _human_or(xs):
    """Join labels the way a person rejects them: "logistics" / "logistics or
    ecommerce" / "logistics, ecommerce, or retail" (lowercased)."""
    xs = [x.lower() for x in xs]
    if len(xs) == 1:
        return xs[0]
    if len(xs) == 2:
        return f"{xs[0]} or {xs[1]}"
    return ", ".join(xs[:-1]) + f", or {xs[-1]}"


# ───────────────────────── conversation builders ───────────────────────────

def _libraries_phase(ids, p, R, msgs):
    """Append the LIBRARIES lookup step — every setup does this before finalizing:
    scope the candidates, verify each on the internet, add the maintained ones and
    dismiss the stale. Mutates msgs. This is what makes the step actually finish."""
    lang = p["languages"][0]
    pool = LIBRARIES.get(lang) or LIBRARIES["TypeScript"]
    cands = R.sample(pool, min(R.randint(2, 3), len(pool)))
    eco = cands[0][1]
    bucket = _platform_bucket(p["platforms"])
    cid, call = tool_call(ids, "scope_options", {
        "category": "libraries", "platform": bucket})
    msgs.append(asst_calls([call], content=(
        f"Before I finalize, let me pick {lang} libraries and verify they're "
        f"current.")))
    msgs.append(tool_result(cid, _TR["scope_options"].format(
        category="libraries", scope=", ".join(p["industries"]),
        platform=f" ({bucket})", values=", ".join(c[0] for c in cands)),
        "scope_options"))
    lookups = []
    for name, ecosystem, verdict in cands:
        cid, call = tool_call(ids, "lookup_package",
                              {"name": name, "ecosystem": ecosystem})
        lookups.append((cid, call, name, verdict))
    msgs.append(asst_calls([c for _, c, _, _ in lookups], content=(
        f"Checking each on {'pub.dev' if eco == 'pubdev' else 'GitHub'}.")))
    for cid, _, name, verdict in lookups:
        days = 25 if verdict in ("fresh", "aging") else 950
        msgs.append(tool_result(
            cid, {"name": name, "verdict": verdict, "last_release_days": days},
            "lookup_package"))
    keep = [(n, v) for _, _, n, v in lookups if v in ("fresh", "aging")]
    drop = [(n, v) for _, _, n, v in lookups if v in ("stale", "dead")]
    dec = []
    if keep:
        cid, call = tool_call(ids, "propose_tags", {
            "tags": [{"category": "libraries", "value": n, "forLanguage": lang}
                     for n, _ in keep]})
        dec.append((cid, call, "propose_tags"))
    for n, v in drop:
        cid, call = tool_call(ids, "dismiss_item",
                              {"name": n, "reason": f"{v} — not maintained"})
        dec.append((cid, call, "dismiss_item"))
    if dec:
        msgs.append(asst_calls([c for _, c, _ in dec], content=(
            "Adding the maintained ones, skipping the stale.")))
        for cid, _, nm in dec:
            msgs.append(tool_result(cid, {"ok": True}, nm))


def build_setup_full(p, R):
    ids = Ids()
    msg = R.choice(FULL_TEMPLATES).format(
        name=p["name"], blurb=p["blurb"], plats=", ".join(p["platforms"]),
        objs=_fmt_list(p["objectives"]), feats=_fmt_list(p["features"]),
        langs=", ".join(p["languages"]), fws=", ".join(p["frameworks"]),
        dbs=", ".join(p["databases"]), svcs=", ".join(p["services"]))
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, msg))]
    cid, call = tool_call(ids, "propose_tags", {"tags": _all_tags(p)})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(_all_tags(p))},
                            "propose_tags"))
    _libraries_phase(ids, p, R, msgs)
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call], content=(
        f"You gave me everything — tagged the {p['industries'][0]} domain, "
        f"platforms, objectives, features, and a {p['languages'][0]}/"
        f"{p['frameworks'][0]} stack. Finalizing now.")))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — generated the Overview, Client, "
                          "Server, and Database plans. Ready for discovery."))
    return {"messages": msgs}


def build_setup_infer(p, R):
    """Pure INFERENCE: the user states a natural idea with NO industry/tags named
    (e.g. "I want to sell lemonade"); the host infers the industry + platforms +
    objectives and tags them, reflecting the inferred industry. This is what
    teaches "sell lemonade → Food & Beverage"."""
    ids = Ids()
    opener = R.choice(opener_for(p["idea"])).format(idea=p["idea"])
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, opener))]
    ind = p["industries"][0]
    inferred = ([{"category": "industries", "value": ind}]
                + [{"category": "platforms", "value": v} for v in p["platforms"]]
                + [{"category": "objectives", "value": v}
                   for v in p["objectives"][:R.randint(2, 3)]])
    cid, call = tool_call(ids, "propose_tags", {"tags": inferred})
    msgs.append(asst_calls([call],
                content=R.choice(INFER_REFLECTIONS).format(ind=ind)))
    msgs.append(tool_result(cid, {"ok": True, "added": len(inferred)},
                            "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(FEAT_QUESTIONS), "options": p["features"],
        "multi": True})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"answer": p["features"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "features", "value": v} for v in p["features"]]
                + _stack_tags(p)})
    msgs.append(asst_calls([call], content=(
        f"Adding those features and a {p['languages'][0]}/{p['frameworks'][0]} "
        f"stack.")))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    _libraries_phase(ids, p, R, msgs)
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated. Ready for "
                          "discovery."))
    return {"messages": msgs}


def build_setup_partial(p, R):
    ids = Ids()
    # Natural opener (no industry named) so this also teaches inference.
    msg = R.choice(opener_for(p["idea"])).format(idea=p["idea"])
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, msg))]
    seed = ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]]
            + [{"category": "objectives", "value": v}
               for v in p["objectives"][:2]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(seed)}, "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(OBJ_QUESTIONS), "options": p["objectives"],
        "multi": True})
    msgs.append(asst_calls([call], content=(
        f"Recorded {p['industries'][0]} on {', '.join(p['platforms'])} — "
        f"now the rest.")))
    rest = p["objectives"][2:] or p["objectives"][:1]
    msgs.append(tool_result(cid, {"answer": rest}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "objectives", "value": v} for v in rest]})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(rest)}, "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(FEAT_QUESTIONS), "options": p["features"],
        "multi": True})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"answer": p["features"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "features", "value": v} for v in p["features"]]})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(p["features"])},
                            "propose_tags"))
    cid, call = tool_call(ids, "propose_tags", {"tags": _stack_tags(p)})
    msgs.append(asst_calls([call], content=(
        f"Proposing a {p['languages'][0]}/{p['frameworks'][0]} stack with "
        f"{p['databases'][0]}.")))
    msgs.append(tool_result(cid, {"ok": True, "added": len(_stack_tags(p))},
                            "propose_tags"))
    _libraries_phase(ids, p, R, msgs)
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated. Ready for discovery."))
    return {"messages": msgs}


# Ideas that genuinely span TWO industries — the host should ASK which apply
# (offer both), not silently pick one. Candidates must be valid INDUSTRIES keys.


def build_setup_libraries(p, R):
    """The LIBRARIES phase the model kept skipping: after the stack, look libraries
    up on the internet (scope_options → consider_items → lookup_package) and
    SELECT them — propose_tags the current ones, dismiss_item the stale/dead — then
    finalize. Teaches the verify-don't-guess, finish-every-step behavior."""
    ids = Ids()
    lang = p["languages"][0]
    pool = LIBRARIES.get(lang) or LIBRARIES["TypeScript"]
    cands = R.sample(pool, min(R.randint(3, 4), len(pool)))
    eco = cands[0][1]

    rich = (f"I'm building {p['name']}: {p['blurb']}. Runs on "
            f"{', '.join(p['platforms'])}, in {lang}/{p['frameworks'][0]} with "
            f"{p['databases'][0]}.")
    msgs = [sys_msg(interview_system(p)), user_msg(rich)]

    base = _all_tags(p)
    cid, call = tool_call(ids, "propose_tags", {"tags": base})
    msgs.append(asst_calls([call], content=(
        f"Basics tagged — now let me pick {lang} libraries and check they're "
        f"current before adding them.")))
    msgs.append(tool_result(cid, {"ok": True, "added": len(base)}, "propose_tags"))

    # Candidate vocabulary for the stack.
    bucket = _platform_bucket(p["platforms"])
    cid, call = tool_call(ids, "scope_options", {
        "category": "libraries", "platform": bucket})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _TR["scope_options"].format(
        category="libraries", scope=", ".join(p["industries"]),
        platform=f" ({bucket})", values=", ".join(c[0] for c in cands)),
        "scope_options"))

    # Register what we're weighing.
    cid, call = tool_call(ids, "consider_items", {
        "category": "libraries", "items": [c[0] for c in cands],
        "note": f"candidate {lang} libraries"})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "pending": len(cands)},
                            "consider_items"))

    # Verify each on the internet (batched), then read the verdicts.
    lookups = []
    for name, ecosystem, verdict in cands:
        cid, call = tool_call(ids, "lookup_package",
                              {"name": name, "ecosystem": ecosystem})
        lookups.append((cid, call, name, verdict))
    msgs.append(asst_calls([c for _, c, _, _ in lookups], content=(
        f"Checking each on {'pub.dev' if eco == 'pubdev' else 'GitHub'}.")))
    for cid, _, name, verdict in lookups:
        days = 25 if verdict in ("fresh", "aging") else 950
        msgs.append(tool_result(
            cid, {"name": name, "verdict": verdict, "last_release_days": days},
            "lookup_package"))

    # Decide: add the current ones, dismiss the stale/dead — every item resolved.
    keep = [(n, v) for _, _, n, v in lookups if v in ("fresh", "aging")]
    drop = [(n, v) for _, _, n, v in lookups if v in ("stale", "dead")]
    decisions = []
    if keep:
        cid, call = tool_call(ids, "propose_tags", {
            "tags": [{"category": "libraries", "value": n, "forLanguage": lang}
                     for n, _ in keep]})
        decisions.append((cid, call, "propose_tags"))
    for n, v in drop:
        cid, call = tool_call(ids, "dismiss_item",
                              {"name": n, "reason": f"{v} — not maintained"})
        decisions.append((cid, call, "dismiss_item"))
    msgs.append(asst_calls([c for _, c, _ in decisions], content=(
        "Adding the maintained ones and skipping the stale.")))
    for cid, _, nm in decisions:
        msgs.append(tool_result(cid, {"ok": True}, nm))

    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — libraries verified, plans generated."))
    return {"messages": msgs}


def build_setup_ambiguous(p, R):
    """AMBIGUOUS INDUSTRY: the idea fits more than one industry (a lemonade stand
    = Food & Beverage AND Retail). The host must ASK which apply (offer both,
    multi-select) instead of guessing one. Then it tags what the user chose."""
    ids = Ids()
    idea, candidates = R.choice(AMBIGUOUS_IDEAS)
    opener = R.choice(opener_for(idea)).format(idea=idea)
    msgs = [sys_msg(interview_system(p)), user_msg(opener)]

    # Ask which industries apply — DO NOT propose one yet.
    cid, call = tool_call(ids, "ask_question", {
        "question": "That could fit more than one industry — which apply?",
        "options": candidates, "multi": True})
    msgs.append(asst_calls([call], content=(
        f"A {idea} could span a couple of industries — which fit?")))
    msgs.append(tool_result(cid, {"answer": candidates}, "ask_question"))

    # Tag BOTH chosen industries + platforms.
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "industries", "value": c} for c in candidates]
                + [{"category": "platforms", "value": v} for v in p["platforms"]]})
    msgs.append(asst_calls([call], content=(
        f"Tagging it as {' and '.join(candidates)}.")))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))

    # Objectives, features, stack, finalize.
    cid, call = tool_call(ids, "ask_question", {
        "question": "What should it do?", "options": p["objectives"],
        "multi": True})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"answer": p["objectives"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "objectives", "value": v} for v in p["objectives"]]
                + [{"category": "features", "value": v} for v in p["features"]]
                + _stack_tags(p)})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    _libraries_phase(ids, p, R, msgs)
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated."))
    return {"messages": msgs}


def build_setup_vague(p, R):
    """AMBIGUOUS input: the user states only a platform (or nothing) with NO
    purpose ("I want to make a mobile app"). The host must tag ONLY what's stated
    and ASK what it should do — it must NOT invent objectives/features. This is
    the counter-example to over-tagging generic defaults."""
    ids = Ids()
    # (opener text, platforms it explicitly states — possibly empty)
    text, stated = R.choice(VAGUE_OPENERS)
    msgs = [sys_msg(interview_system(p)), user_msg(text)]

    if stated:
        # Tag ONLY the platforms that were actually stated — nothing else.
        cid, call = tool_call(ids, "propose_tags", {
            "tags": [{"category": "platforms", "value": v} for v in stated]})
        msgs.append(asst_calls([call], content=(
            f"Got it — a {'/'.join(stated)} app. What should it do for your "
            f"users?")))
        msgs.append(tool_result(cid, {"ok": True, "added": len(stated)},
                                "propose_tags"))
        cid, call = tool_call(ids, "ask_question", {
            "question": "What should the app do? (pick its main goals)",
            "options": p["objectives"], "multi": True})
        msgs.append(asst_calls([call]))
    else:
        # Nothing concrete stated → ask first, tag nothing yet.
        cid, call = tool_call(ids, "ask_question", {
            "question": "Happy to help — what should the app do?",
            "options": p["objectives"], "multi": True})
        msgs.append(asst_calls([call], content=(
            "Happy to help — tell me what it should do and who it's for.")))
    msgs.append(tool_result(cid, {"answer": p["objectives"]}, "ask_question"))

    # Now that the purpose is known, tag the real objectives + inferred industry.
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "industries", "value": p["industries"][0]}]
                + [{"category": "objectives", "value": v}
                   for v in p["objectives"]]})
    msgs.append(asst_calls([call], content=(
        f"That sounds like a {p['industries'][0]} app — tagging those goals.")))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))

    # Features, stack, finalize.
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(FEAT_QUESTIONS), "options": p["features"],
        "multi": True})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"answer": p["features"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "features", "value": v} for v in p["features"]]
                + _stack_tags(p)})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    _libraries_phase(ids, p, R, msgs)
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated."))
    return {"messages": msgs}


def _missing_label(missing):
    """The gate capitalizes section labels ('Languages, Frameworks')."""
    return ", ".join(m.capitalize() for m in missing)


def _mask_assistants(msgs, start):
    """Mark every assistant message from index `start` on as context-only
    (_no_target) — used when a whole scripted phase (e.g. jumping to the
    libraries lookup before the product interview is done) is the MISTAKE the
    example recovers from, so none of it ever becomes a training target."""
    for m in msgs[start:]:
        if m.get("role") == "assistant":
            m["_no_target"] = True


def build_setup_recovery(p, R, shape=None):
    """A premature finalize_setup gets REJECTED by the gate and the host
    recovers — the single most important transition the eval exposed (a model
    that ignores the rejection grinds finalize forever). Six shapes:

      early    — only industries/platforms tagged; gate lists all 4 product
                 gaps; recover by asking objectives (the original flow),
      objectives — everything but objectives tagged; gate lists ONE section;
                 recover by asking it (eval failures #10/#16),
      stack    — full product interview but languages/frameworks never tagged;
                 recover by DERIVING the stack, no question (eval #3/#21/#26),
      loopguard — like `objectives`/`stack`, but the host repeats the rejected
                 finalize once (scripted as a _no_target context-only turn, so
                 the mistake is never trained) and the loop-guard text appears;
                 the trained target is the recovery AFTER the guard message.
      compound — the run-4 live-failure trajectory: the host jumps to the
                 LIBRARIES phase early (scripted mistake, context-only), then
                 a premature finalize is rejected naming THREE-PLUS sections
                 ("Features, Languages, Frameworks", sometimes + Objectives) —
                 a rejection shape run-4 had never seen. Recovery works the
                 list: one question per open product topic, tag the answers,
                 derive the stack, finalize (run-4 eval #7/8/9/11/18/19/23).
      early_libs — NO rejection at all: trains the correct continuation AT the
                 trap state itself (libraries tagged, features/stack open) —
                 ask the features question instead of finalizing. Fills the
                 target hole that made run-4 improvise a premature finalize.
    """
    # NOTE: `features_done` exists but is NOT in the random rotation — run-7
    # showed it COLLIDES with early_libs at serve (the trap states are nearly
    # identical; the board snapshot is stale within a turn, so the model
    # tie-breaks randomly: derive-while-features-missing or re-ask loops).
    shape = shape or R.choice(["early", "objectives", "stack", "loopguard",
                               "compound", "early_libs"])
    ids = Ids()

    if shape in ("compound", "early_libs", "features_done"):
        return _build_recovery_compound(p, R, ids, shape)

    if shape == "early":
        msg = f"Let's set up {p['name']}. It's a {p['industries'][0]} app."
        msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, msg))]
        seed = ([{"category": "industries", "value": v} for v in p["industries"]]
                + [{"category": "platforms", "value": v} for v in p["platforms"]])
        cid, call = tool_call(ids, "propose_tags", {"tags": seed})
        msgs.append(asst_calls([call]))
        msgs.append(tool_result(cid, {"ok": True, "added": len(seed)},
                                "propose_tags"))
        missing = ["objectives", "features", "languages", "frameworks"]
        # The premature finalize is the MISTAKE this flow recovers from — it
        # must be context only, never a target (or we'd train the mistake).
        cid, call = tool_call(ids, "finalize_setup", {})
        bad = asst_calls([call])
        bad["_no_target"] = True
        msgs.append(bad)
        msgs.append(tool_result(cid, {"ok": False, "missing": missing},
                                "finalize_setup"))
        cid, call = tool_call(ids, "ask_question", {
            "question": R.choice(OBJ_QUESTIONS), "options": p["objectives"],
            "multi": True})
        msgs.append(asst_calls([call], reasoning=_REASON["finalize_rejected_ask"]
                               .format(missing=_missing_label(missing),
                                       topic="objectives")))
        msgs.append(tool_result(cid, {"answer": p["objectives"]}, "ask_question"))
        # Two short batches, not one giant dump (run-5 repetition-attractor fix):
        # the answers first, then the derived stack with its own rationale.
        fill = ([{"category": "objectives", "value": v} for v in p["objectives"]]
                + [{"category": "features", "value": v} for v in p["features"]])
        cid, call = tool_call(ids, "propose_tags", {"tags": fill})
        msgs.append(asst_calls([call]))
        msgs.append(tool_result(cid, {"ok": True, "added": len(fill)},
                                "propose_tags"))
        cid, call = tool_call(ids, "propose_tags", {"tags": _stack_tags(p)})
        msgs.append(asst_calls([call],
                               reasoning=_REASON["compound_derive_stack"]))
        msgs.append(tool_result(cid, {"ok": True,
                                      "added": len(_stack_tags(p))},
                                "propose_tags"))
        _libraries_phase(ids, p, R, msgs)
        cid, call = tool_call(ids, "finalize_setup", {})
        msgs.append(asst_calls([call],
                               content="Everything's tagged now — finalizing."))
        msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
        msgs.append(asst_text("Setup complete — plans generated. Ready for "
                              "discovery."))
        return {"messages": msgs}

    # The other three shapes share an opener: a mostly-complete interview where
    # exactly ONE gap slips through, so the gate's rejection names 1-2 sections
    # — the shape the eval showed the model has never seen.
    gap = "objectives" if shape == "objectives" else "stack"
    if shape == "loopguard":
        gap = R.choice(["objectives", "stack"])

    opener = R.choice(opener_for(p["idea"])).format(idea=p["idea"])
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, opener))]
    # SHORT opener + separate second batch (run-5 lesson: one giant trained
    # tag dump becomes a repetition attractor at temp-0 serve).
    seed = ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]])
    if gap == "objectives":
        # objectives never asked; features + stack tagged
        rest = ([{"category": "features", "value": v} for v in p["features"]]
                + _stack_tags(p))
        missing = ["objectives"]
    else:
        # full product interview, but languages/frameworks never tagged
        rest = ([{"category": "objectives", "value": v}
                 for v in p["objectives"]]
                + [{"category": "features", "value": v}
                   for v in p["features"]])
        missing = ["languages", "frameworks"]
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(seed)},
                            "propose_tags"))
    cid, call = tool_call(ids, "propose_tags", {"tags": rest})
    msgs.append(asst_calls([call], reasoning=_REASON["second_batch_tags"]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(rest)},
                            "propose_tags"))
    if gap == "stack":
        _libraries_phase(ids, p, R, msgs)

    # the premature finalize, rejected by the gate with the SHORT missing list
    # (context only — the mistake itself must never be a training target)
    cid, call = tool_call(ids, "finalize_setup", {})
    bad = asst_calls([call])
    bad["_no_target"] = True
    msgs.append(bad)
    msgs.append(tool_result(cid, {"ok": False, "missing": missing},
                            "finalize_setup"))

    if shape == "loopguard":
        # The host repeats the rejected call once — context only, NEVER a
        # training target — so the serve loop-guard text enters the corpus.
        cid, call = tool_call(ids, "finalize_setup", {})
        bad = asst_calls([call])
        bad["_no_target"] = True
        msgs.append(bad)
        guard = (_TR["loop_block"].format(tool="finalize_setup")
                 if R.random() < 0.5 else
                 _TR["finalize_missing"].format(
                     missing=_missing_label(missing))
                 + "\n\n" + _TR["loop_warn"].format(tool="finalize_setup"))
        msgs.append(tool_result(cid, guard, "finalize_setup"))
        recover_reason = _REASON["loop_guard_recover"].format(
            missing=_missing_label(missing))
    elif gap == "objectives":
        recover_reason = _REASON["finalize_rejected_ask"].format(
            missing=_missing_label(missing), topic="objectives")
    else:
        recover_reason = _REASON["finalize_rejected_derive"].format(
            missing=_missing_label(missing))

    # the RECOVERY — the generation point this whole builder exists to teach
    if gap == "objectives":
        cid, call = tool_call(ids, "ask_question", {
            "question": R.choice(OBJ_QUESTIONS), "options": p["objectives"],
            "multi": True})
        msgs.append(asst_calls([call], reasoning=recover_reason))
        msgs.append(tool_result(cid, {"answer": p["objectives"]},
                                "ask_question"))
        fill = [{"category": "objectives", "value": v}
                for v in p["objectives"]]
        cid, call = tool_call(ids, "propose_tags", {"tags": fill})
        msgs.append(asst_calls([call]))
        msgs.append(tool_result(cid, {"ok": True, "added": len(fill)},
                                "propose_tags"))
    else:
        cid, call = tool_call(ids, "propose_tags", {"tags": _stack_tags(p)})
        msgs.append(asst_calls([call], reasoning=recover_reason))
        msgs.append(tool_result(cid, {"ok": True,
                                      "added": len(_stack_tags(p))},
                                "propose_tags"))

    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call],
                           reasoning=_REASON["finalize_setup"]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated. Ready for "
                          "discovery."))
    return {"messages": msgs}


def _build_recovery_compound(p, R, ids, shape):
    """The `compound` and `early_libs` recovery shapes (see
    build_setup_recovery). Both replay the trajectory the run-4 eval exposed:
    seed tags land, then the host jumps ahead to the LIBRARIES phase while
    features and the stack are still open. The whole early-libraries phase is
    scripted CONTEXT (every assistant turn `_no_target`) — we teach how to get
    OUT of that state, never how to get into it.

      compound   — a premature finalize (context-only) is rejected naming
                   "Features, Languages, Frameworks" (+ Objectives ~35% of the
                   time, which adds the ask-objectives→ask-features two-question
                   chain); optionally the loop-guard fires first. The trained
                   recovery asks the open product question(s), tags answers
                   under the RIGHT categories, derives the stack, finalizes.
      early_libs — no rejection: the trained target sits AT the trap state —
                   libraries tagged, features/languages/frameworks empty —
                   and the correct move is the features question, not finalize.
      features_done — CONTRAST to early_libs (run-5 eval #1/#3): the host
                   over-dumped at the start (features included — that opener is
                   context-only, never trained) and then jumped to libraries.
                   At the trap state features are ALREADY tagged — in history,
                   not in the turn-start board snapshot — so the trained move
                   is deriving the stack, NOT re-asking the features question.
    """
    opener = R.choice(opener_for(p["idea"])).format(idea=p["idea"])
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, opener))]

    if shape == "features_done":
        # The over-dump (industries+platforms+objectives+FEATURES in one call)
        # is the mistake that sets up this state — context only, never trained.
        dump = ([{"category": "industries", "value": v} for v in p["industries"]]
                + [{"category": "platforms", "value": v} for v in p["platforms"]]
                + [{"category": "objectives", "value": v} for v in p["objectives"]]
                + [{"category": "features", "value": v} for v in p["features"]])
        cid, call = tool_call(ids, "propose_tags", {"tags": dump})
        bad = asst_calls([call])
        bad["_no_target"] = True
        msgs.append(bad)
        msgs.append(tool_result(cid, {"ok": True, "added": len(dump)},
                                "propose_tags"))
        libs_start = len(msgs)
        _libraries_phase(ids, p, R, msgs)
        _mask_assistants(msgs, libs_start)
        # Trained target AT the trap: features are tagged (the tool result in
        # history proves it; the board snapshot predates it) — derive the
        # stack directly. No question, no finalize bounce.
        cid, call = tool_call(ids, "propose_tags", {"tags": _stack_tags(p)})
        msgs.append(asst_calls([call],
                               reasoning=_REASON["features_present_derive"]))
        msgs.append(tool_result(cid, {"ok": True,
                                      "added": len(_stack_tags(p))},
                                "propose_tags"))
        cid, call = tool_call(ids, "finalize_setup", {})
        msgs.append(asst_calls([call],
                               content="Everything's tagged now — finalizing.",
                               reasoning=_REASON["finalize_setup"]))
        msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
        msgs.append(asst_text("Setup complete — plans generated. Ready for "
                              "discovery."))
        return {"messages": msgs}

    # `compound` sometimes leaves objectives open too → 4-section rejection.
    with_objectives = shape == "compound" and R.random() < 0.35
    # SHORT opener — industries+platforms ONLY. Run-5 lesson: a long opening
    # tag dump is a greedy-decoding repetition attractor (7/27 eval cases
    # never closed the array). The trained opener must END after a few tags;
    # everything else arrives in a separate, also-short batch.
    seed = ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(seed)},
                            "propose_tags"))
    if not with_objectives:
        objs = [{"category": "objectives", "value": v}
                for v in p["objectives"]]
        cid, call = tool_call(ids, "propose_tags", {"tags": objs})
        msgs.append(asst_calls([call],
                               reasoning=_REASON["second_batch_tags"]))
        msgs.append(tool_result(cid, {"ok": True, "added": len(objs)},
                                "propose_tags"))

    # The premature LIBRARIES jump — the mistake. Context only, never trained.
    libs_start = len(msgs)
    _libraries_phase(ids, p, R, msgs)
    _mask_assistants(msgs, libs_start)

    missing = ((["objectives"] if with_objectives else [])
               + ["features", "languages", "frameworks"])

    if shape == "compound":
        # premature finalize → rejection with the LONG missing list
        # (context only — the mistake itself must never be a training target)
        cid, call = tool_call(ids, "finalize_setup", {})
        bad = asst_calls([call])
        bad["_no_target"] = True
        msgs.append(bad)
        msgs.append(tool_result(cid, {"ok": False, "missing": missing},
                                "finalize_setup"))
        if R.random() < 0.4:
            # loop-guard variant: the rejected finalize repeats once (context
            # only) so recovery is also trained from behind the guard text.
            cid, call = tool_call(ids, "finalize_setup", {})
            bad = asst_calls([call])
            bad["_no_target"] = True
            msgs.append(bad)
            guard = (_TR["loop_block"].format(tool="finalize_setup")
                     if R.random() < 0.5 else
                     _TR["finalize_missing"].format(
                         missing=_missing_label(missing))
                     + "\n\n" + _TR["loop_warn"].format(tool="finalize_setup"))
            msgs.append(tool_result(cid, guard, "finalize_setup"))
            first_reason = _REASON["loop_guard_recover"].format(
                missing=_missing_label(missing))
        else:
            first_reason = _REASON["finalize_rejected_compound"].format(
                missing=_missing_label(missing))
    else:
        # early_libs: no rejection — the trained move at this exact board
        # state IS the features question.
        first_reason = _REASON["libraries_early_ask"].format(
            missing=_missing_label(missing))

    if with_objectives:
        cid, call = tool_call(ids, "ask_question", {
            "question": R.choice(OBJ_QUESTIONS), "options": p["objectives"],
            "multi": True})
        msgs.append(asst_calls([call], reasoning=first_reason))
        msgs.append(tool_result(cid, {"answer": p["objectives"]},
                                "ask_question"))
        fill = [{"category": "objectives", "value": v}
                for v in p["objectives"]]
        cid, call = tool_call(ids, "propose_tags", {"tags": fill})
        msgs.append(asst_calls([call]))
        msgs.append(tool_result(cid, {"ok": True, "added": len(fill)},
                                "propose_tags"))
        # objectives are closed now — the remaining gate list is the exact
        # 3-section shape the live eval produced ("Features, Languages,
        # Frameworks"), handled with the features question next.
        feat_reason = _REASON["finalize_rejected_compound"].format(
            missing=_missing_label(["features", "languages", "frameworks"]))
    else:
        feat_reason = first_reason

    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(FEAT_QUESTIONS), "options": p["features"],
        "multi": True})
    msgs.append(asst_calls([call], reasoning=feat_reason))
    msgs.append(tool_result(cid, {"answer": p["features"]}, "ask_question"))
    fill = [{"category": "features", "value": v} for v in p["features"]]
    cid, call = tool_call(ids, "propose_tags", {"tags": fill})
    msgs.append(asst_calls([call],
                           reasoning=_REASON["compound_save_features"]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(fill)},
                            "propose_tags"))
    cid, call = tool_call(ids, "propose_tags", {"tags": _stack_tags(p)})
    msgs.append(asst_calls([call],
                           reasoning=_REASON["compound_derive_stack"]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(_stack_tags(p))},
                            "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call],
                           content="Everything's tagged now — finalizing.",
                           reasoning=_REASON["finalize_setup"]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated. Ready for "
                          "discovery."))
    return {"messages": msgs}


def build_setup_correction(p, R):
    """The user REJECTS an industry tag the host guessed ("this is not a logistics
    or ecommerce app — it's Food & Beverage, remove those") → the host calls
    `remove_tags` for EXACTLY the disowned values, keeps the right one, and moves on.

    Teaches two things at once:
      • remove_tags is called ONLY when the user explicitly disowns a tag (every
        other builder keeps its tags, so the contrast makes "don't remove unless
        told" the default), and
      • it works for ANY industry — the wrong value is drawn from the taxonomy DB,
        so over the corpus we loop "this is not a {db industry}" across all values.
    """
    ids = Ids()
    correct = p["industries"][0]
    # 1-2 WRONG industries from the taxonomy DB (never the correct one) that the
    # host over-eagerly guessed alongside the right one.
    pool = [k for k in INDUSTRIES if k != correct]
    wrongs = R.sample(pool, R.randint(1, 2))
    guess = ([{"category": "industries", "value": v} for v in [correct] + wrongs]
             + [{"category": "platforms", "value": v} for v in p["platforms"]])

    opener = R.choice(opener_for(p["idea"])).format(idea=p["idea"])
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, opener))]
    cid, call = tool_call(ids, "propose_tags", {"tags": guess})
    msgs.append(asst_calls([call], content=(
        f"Tagging it as {_human_or([correct] + wrongs)} to start.")))
    msgs.append(tool_result(cid, {"ok": True, "added": len(guess)}, "propose_tags"))

    # The user corrects, naming the wrong industries AND the right one.
    msgs.append(user_msg(R.choice(CORRECTION_TEMPLATES).format(
        wrong=_human_or(wrongs), correct=correct.lower(), idea=p["idea"])))

    # The host removes EXACTLY the disowned tags — nothing else.
    cid, call = tool_call(ids, "remove_tags", {
        "tags": [{"category": "industries", "value": v} for v in wrongs]})
    msgs.append(asst_calls([call], content=(
        f"Got it — removing {_human_or(wrongs)} and keeping {correct.lower()}.")))
    msgs.append(tool_result(cid, {"ok": True, "removed": len(wrongs)}, "remove_tags"))

    # Continue the interview from the corrected state.
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(OBJ_QUESTIONS), "options": p["objectives"],
        "multi": True})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"answer": p["objectives"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "objectives", "value": v} for v in p["objectives"]]
                + [{"category": "features", "value": v} for v in p["features"]]
                + _stack_tags(p)})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    _libraries_phase(ids, p, R, msgs)
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call], content="Corrected and complete — finalizing."))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated. Ready for discovery."))
    return {"messages": msgs}


def build_setup_image(p, R):
    """User asks to see a mockup → generate_image, then continues setup."""
    ids = Ids()
    msg = (f"I'm building {p['name']} — {p['blurb']}. Can you show me a mockup "
           f"of the home screen first?")
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, msg))]
    cid, call = tool_call(ids, "generate_image", {
        "prompt": f"{p['app']} app home screen for a {p['domain']}, clean modern "
                  f"UI, {', '.join(p['platforms'])} style", "size": "1024x1792"})
    msgs.append(asst_calls([call], content="Here's a first look at the home "
                                           "screen."))
    msgs.append(tool_result(cid, {"ok": True,
                "image_url": "/workspace/img/mockup1.png"}, "generate_image"))
    seed = ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]]
            + [{"category": "objectives", "value": v} for v in p["objectives"][:3]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call], content="Tagging what you described while we "
                                           "look at it."))
    msgs.append(tool_result(cid, {"ok": True, "added": len(seed)}, "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(FEAT_QUESTIONS), "options": p["features"],
        "multi": True})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"answer": p["features"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "features", "value": v} for v in p["features"]]
                + _stack_tags(p)})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    _libraries_phase(ids, p, R, msgs)
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call], content="Looks good — finalizing the plan."))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated."))
    return {"messages": msgs}


def _narrative(p, R, step):
    return (f"As a {p['role']}, I want to {step.lower()}, so that "
            f"{R.choice(STORY_PH['benefits']).format(d=p['domain'])}.")


def build_discovery(p, R):
    """Big-chunk flow: the user dumps the whole flow in one paragraph (in user
    terminology) → the host either drafts via draft_stories_from_text or adds the
    stories itself, then CHAINS the steps (each the child of the previous), adds
    the edge case + note, checks the shape, and closes with confirm."""
    ids = Ids()
    opens = [
        f"I want a {p['role']} to {p['flow'][0].lower()} and go from there.",
        f"Let me walk you through how a {p['role']} uses it.",
        f"The core flow is what a {p['role']} does step by step.",
    ]
    msgs = [sys_msg(discovery_system(p)), user_msg(evolve_user(R, R.choice(opens)))]
    cid, call = tool_call(ids, "add_user_story", {
        "title": p["epic_title"], "narrative": p["epic_narrative"],
        "kind": "epic"})
    msgs.append(asst_calls([call], reasoning=_REASON.get("story_root")))
    msgs.append(tool_result(cid, {"id": 1}, "add_user_story"))
    msgs.append(asst_text(
        f"Captured \"{p['epic_title']}\" as the root. Walk me through the first "
        f"thing a {p['role']} does."))
    chunk = R.choice(STORY_PH["big_chunk"]).format(
        role=p["role"],
        steps=", ".join(s.lower() for s in p["flow"][:-1]),
        last=p["flow"][-1].lower())
    msgs.append(user_msg(chunk))
    sid = 1
    if R.random() < 0.5:
        cid, call = tool_call(ids, "draft_stories_from_text",
                              {"text": chunk, "parent_story_id": 1})
        msgs.append(asst_calls([call], reasoning=_REASON.get("story_draft")))
        msgs.append(tool_result(cid, {"made": len(p["flow"])},
                                "draft_stories_from_text"))
        # re-chain the drafted flat list into a parent→child flow chain
        chain, prev = [], 1
        for i in range(len(p["flow"])):
            cid, call = tool_call(ids, "move_user_story", {
                "story_id": i + 2, "parent_story_id": prev, "order_index": 0})
            chain.append((cid, call))
            prev = i + 2
        msgs.append(asst_calls([c for _, c in chain],
                               reasoning=_REASON.get("story_chain")))
        for cid, _ in chain:
            msgs.append(tool_result(cid, {}, "move_user_story"))
        sid = prev
    else:
        # add each step directly as the CHILD of the previous (the id from each
        # result is the parent of the next — no fix-up moves needed)
        prev = 1
        for i, s in enumerate(p["flow"]):
            cid, call = tool_call(ids, "add_user_story", {
                "title": s, "narrative": _narrative(p, R, s),
                "parent_story_id": prev, "kind": "story"})
            msgs.append(asst_calls(
                [call], reasoning=_REASON.get("story_chain") if i else
                _REASON.get("add_user_story")))
            prev = i + 2
            msgs.append(tool_result(cid, {"id": prev}, "add_user_story"))
        sid = prev
    case, answer, etitle, enarr, eaccept = p["edge"]
    msgs.append(asst_text(f"Chained the {len(p['flow'])} steps. What happens "
                          f"when {case}?"))
    msgs.append(user_msg(evolve_user(R, answer)))
    cid, call = tool_call(ids, "add_user_story", {
        "title": etitle, "narrative": enarr, "acceptance_criteria": eaccept,
        "parent_story_id": sid, "kind": "substory"})
    msgs.append(asst_calls([call]))
    edge_id = len(p["flow"]) + 2
    msgs.append(tool_result(cid, {"id": edge_id}, "add_user_story"))
    cid, call = tool_call(ids, "add_note", {
        "story_id": edge_id, "body": f"Edge case: {case}."})
    msgs.append(asst_calls([call], reasoning=_REASON.get("story_note")))
    msgs.append(tool_result(cid, {"id": 1}, "add_note"))
    cid, call = tool_call(ids, "list_user_stories", {})
    msgs.append(asst_calls([call]))
    tree = [{"id": 1, "title": p["epic_title"], "parent": None,
             "kind": "epic"}]
    prev = 1
    for i, s in enumerate(p["flow"]):
        tree.append({"id": i + 2, "title": s, "parent": prev})
        prev = i + 2
    tree.append({"id": edge_id, "title": etitle, "parent": prev,
                 "kind": "substory"})
    msgs.append(tool_result(cid, {"stories": tree}, "list_user_stories"))
    msgs.append(asst_text("That captures the full flow with the edge case "
                          "handled. If nothing's missing, press \"Generate tasks "
                          "from stories\"."))
    return {"messages": msgs}


def build_discovery_walkthrough(p, R):
    """Turn-by-turn walkthrough in USER TERMINOLOGY: the user narrates what a
    person does step by step ("then they tap…"), and the host captures each step
    as a chained story (parent = the previous step's id), reflects in one short
    sentence, and ends EVERY turn with exactly ONE focused question. Closes by
    paraphrasing the flow back for confirmation. This is the core tree-building
    behavior the app needs."""
    ids = Ids()
    flow = p["flow"]
    opener = R.choice(STORY_PH["openers"]).format(
        role=p["role"], first=flow[0].lower())
    msgs = [sys_msg(discovery_system(p)), user_msg(opener)]
    # Root epic + first step in one move, then ask what happens next.
    cid1, call1 = tool_call(ids, "add_user_story", {
        "title": p["epic_title"], "narrative": p["epic_narrative"],
        "kind": "epic"})
    msgs.append(asst_calls([call1], reasoning=_REASON.get("story_root")))
    msgs.append(tool_result(cid1, {"id": 1}, "add_user_story"))
    cid2, call2 = tool_call(ids, "add_user_story", {
        "title": flow[0], "narrative": _narrative(p, R, flow[0]),
        "parent_story_id": 1, "kind": "story"})
    msgs.append(asst_calls([call2], content=(
        f"Got it — \"{flow[0]}\" is the entry point. "
        + R.choice(STORY_PH["host_questions"][:2]).format(
            step=flow[0].lower(), extra="", edge="", role=p["role"])),
        reasoning=_REASON.get("story_chain")))
    msgs.append(tool_result(cid2, {"id": 2}, "add_user_story"))
    # Each subsequent step: user narrates in plain words → host chains it and
    # asks ONE focused question about the next.
    prev = 2
    for i, step in enumerate(flow[1:], start=1):
        msgs.append(user_msg(R.choice(STORY_PH["step_phrases"]).format(
            step=step.lower())))
        cid, call = tool_call(ids, "add_user_story", {
            "title": step, "narrative": _narrative(p, R, step),
            "parent_story_id": prev, "kind": "story"})
        prev = i + 2
        last = i == len(flow) - 1
        q = (R.choice(STORY_PH["host_questions"][-1:]).format(
                 role=p["role"], step=step.lower(), extra="", edge="")
             if last else
             R.choice(STORY_PH["host_questions"][:2]).format(
                 step=step.lower(), extra="", edge="", role=p["role"]))
        msgs.append(asst_calls([call], content=(
            f"Added \"{step}\" as the next step. {q}"),
            reasoning=_REASON.get("story_chain")))
        msgs.append(tool_result(cid, {"id": prev}, "add_user_story"))
    # A passing mention → capture + probe (EXPAND WHAT THEY MENTION IN PASSING).
    extra = R.choice(p["features"])
    msgs.append(user_msg(R.choice(STORY_PH["passing_mentions"]).format(
        extra=extra.lower())))
    cid, call = tool_call(ids, "add_user_story", {
        "title": extra, "narrative": _narrative(p, R, f"use {extra.lower()}"),
        "parent_story_id": 1, "kind": "story"})
    msgs.append(asst_calls([call], content=(
        f"Captured \"{extra}\" so it doesn't get lost. "
        + STORY_PH["host_questions"][2].format(
            extra=extra.lower(), step="", edge="", role=p["role"])),
        reasoning=_REASON.get("story_expand")))
    mention_id = prev + 1
    msgs.append(tool_result(cid, {"id": mention_id}, "add_user_story"))
    msgs.append(user_msg(R.choice(STORY_PH["mention_details"])))
    cid, call = tool_call(ids, "add_note", {
        "story_id": mention_id,
        "body": f"{extra}: keep it simple in v1 — view in one place, tap for "
                f"detail, alert on changes."})
    msgs.append(asst_calls([call], reasoning=_REASON.get("story_note")))
    msgs.append(tool_result(cid, {"id": 1}, "add_note"))
    # Closing: paraphrase the whole flow back, user confirms, point at the button.
    summary = (" → ".join(s for s in flow) + f", plus {extra.lower()}")
    msgs.append(asst_text(R.choice(STORY_PH["confirm_closes"]).format(
        summary=summary), reasoning=_REASON.get("story_confirm")))
    msgs.append(user_msg(R.choice(STORY_PH["done_phrases"])))
    msgs.append(asst_text(R.choice(STORY_PH["ready_closes"])))
    return {"messages": msgs}


def build_discovery_grouped(p, R):
    """Feature-AREA grouping: the user describes a couple of distinct areas; the
    host groups each under an intermediate parent story (area) with sub-stories
    beneath — a real nested tree, not a flat list — then verifies the shape with
    list_user_stories and fixes one nesting with move_user_story."""
    ids = Ids()
    areas = R.sample(p["features"], min(2, len(p["features"])))
    opener = (f"There are really two parts to it: {areas[0].lower()} and "
              f"{areas[1].lower()}. Want me to explain each?")
    msgs = [sys_msg(discovery_system(p)), user_msg(opener)]
    cid, call = tool_call(ids, "add_user_story", {
        "title": p["epic_title"], "narrative": p["epic_narrative"],
        "kind": "epic"})
    msgs.append(asst_calls([call], content=(
        f"Yes — let's take them one at a time. Walk me through "
        f"{areas[0].lower()} first."), reasoning=_REASON.get("story_root")))
    msgs.append(tool_result(cid, {"id": 1}, "add_user_story"))
    next_id = 2
    for area in areas:
        steps = R.sample(p["flow"], min(2, len(p["flow"])))
        msgs.append(user_msg(R.choice(STORY_PH["big_chunk"]).format(
            role=p["role"], steps=", ".join(s.lower() for s in steps[:-1])
            or steps[0].lower(), last=steps[-1].lower())))
        # area parent under the epic
        cid, call = tool_call(ids, "add_user_story", {
            "title": area, "narrative": _narrative(p, R, f"use {area.lower()}"),
            "parent_story_id": 1, "kind": "story"})
        msgs.append(asst_calls([call], reasoning=_REASON.get("story_group")))
        area_id = next_id
        next_id += 1
        msgs.append(tool_result(cid, {"id": area_id}, "add_user_story"))
        # sub-stories nested under the area
        subs = []
        for s in steps:
            cid, call = tool_call(ids, "add_user_story", {
                "title": s, "narrative": _narrative(p, R, s),
                "parent_story_id": area_id, "kind": "substory"})
            subs.append((cid, call))
            next_id += 1
        msgs.append(asst_calls([c for _, c in subs]))
        for j, (cid, _) in enumerate(subs):
            msgs.append(tool_result(cid, {"id": next_id - len(subs) + j},
                                    "add_user_story"))
        q = (STORY_PH["host_questions"][2].format(
                 extra=areas[1].lower(), step="", edge="", role=p["role"])
             if area == areas[0] else
             STORY_PH["host_questions"][-1].format(
                 role=p["role"], step="", extra="", edge=""))
        msgs.append(asst_text(f"\"{area}\" is captured with its steps nested "
                              f"under it. {q}"))
        if area == areas[0]:
            msgs.append(user_msg(f"Sure — so for {areas[1].lower()}, here's "
                                 f"how I picture it."))
    # check the shape; one sub-story landed under the wrong area → fix it
    cid, call = tool_call(ids, "list_user_stories", {})
    msgs.append(asst_calls([call]))
    tree = [{"id": 1, "title": p["epic_title"], "parent": None, "kind": "epic"}]
    msgs.append(tool_result(cid, {"stories": tree + [
        {"id": i, "title": "…", "parent": 1} for i in range(2, next_id)]},
        "list_user_stories"))
    cid, call = tool_call(ids, "move_user_story", {
        "story_id": next_id - 1, "parent_story_id": next_id - 3,
        "order_index": 1})
    msgs.append(asst_calls([call], reasoning=_REASON.get("story_reparent")))
    msgs.append(tool_result(cid, {}, "move_user_story"))
    msgs.append(asst_text(R.choice(STORY_PH["confirm_closes"]).format(
        summary=f"{areas[0]} and {areas[1]}, each with its own steps"),
        reasoning=_REASON.get("story_confirm")))
    msgs.append(user_msg(R.choice(STORY_PH["done_phrases"])))
    msgs.append(asst_text(R.choice(STORY_PH["ready_closes"])))
    return {"messages": msgs}


def _agents_for(p):
    """A plausible per-project agent roster (the serve list_agents shape: ids the
    coordinator must reuse in create_task/assign_agent_to_task)."""
    ui_name = "Flutter Dev" if "Flutter" in p["frameworks"] else "Frontend Dev"
    return {
        "ui": {"id": 1, "name": ui_name,
               "role": f"Client engineer ({p['frameworks'][0]})"},
        "api": {"id": 2, "name": "Backend Dev",
                "role": f"Server engineer ({p['languages'][0]})"},
        "db": {"id": 3, "name": "Data Engineer",
               "role": f"Database engineer ({p['databases'][0]})"},
    }


def _no_tasks_context(p):
    return (f"Project \"{p['name']}\": No tasks yet. The user is likely "
            f"starting planning.")


def build_tasks(p, R):
    """The serve pattern for creating work: list_agents FIRST (the catalog's
    CRITICAL rule — every task must carry an agent), then create_task WITH
    agent_persona_id, then a short spoken confirmation (never end on tools)."""
    ids = Ids()
    asks = [
        "Generate tasks from the setup and discovery stories.",
        "Break the plans down into tasks and assign them.",
        "Turn the user stories into concrete engineering tasks.",
        "Create the build tasks for v1 and assign each to an agent.",
    ]
    msgs = [sys_msg(coordinator_system(p, _no_tasks_context(p))),
            user_msg(evolve_user(R, R.choice(asks)))]
    agents = _agents_for(p)
    cid, call = tool_call(ids, "list_agents", {})
    msgs.append(asst_calls([call], reasoning=(
        "Every task must be assigned, so I list the agents FIRST and reuse "
        "their real ids.")))
    msgs.append(tool_result(cid, {"agents": list(agents.values())},
                            "list_agents"))
    calls, meta = [], []
    for t in p["tasks"]:
        a = agents.get(t["layer"], agents["api"])
        cid, call = tool_call(ids, "create_task", {
            "title": t["title"], "description": t["description"],
            "priority": "HIGH" if t["layer"] in ("db", "ui") else "MED",
            "agent_persona_id": a["id"]})
        calls.append(call)
        meta.append((cid, t, a))
    msgs.append(asst_calls(calls))
    for i, (cid, t, a) in enumerate(meta):
        msgs.append(tool_result(cid, {"id": i + 1, "who": a["name"]},
                                "create_task"))
    msgs.append(asst_text(
        f"Created {len(p['tasks'])} tasks from the stories — each is "
        f"stack-specific and assigned to the right agent."))
    return {"messages": msgs}


def build_tasks_breakdown(p, R):
    """Create a parent task, then sub-tasks under it (parent_task_id) — agents
    listed first, every task assigned (the serve CRITICAL rule)."""
    ids = Ids()
    feature = R.choice(p["features"])
    msgs = [sys_msg(coordinator_system(p, _no_tasks_context(p))),
            user_msg(evolve_user(R, f"Break down the \"{feature}\" feature into subtasks."))]
    agents = _agents_for(p)
    cid, call = tool_call(ids, "list_agents", {})
    msgs.append(asst_calls([call], reasoning=(
        "Every task must be assigned, so I list the agents FIRST and reuse "
        "their real ids.")))
    msgs.append(tool_result(cid, {"agents": list(agents.values())},
                            "list_agents"))
    cid, call = tool_call(ids, "create_task", {
        "title": f"Implement {feature}",
        "description": f"Objective: deliver {feature.lower()}. Parent task; "
                       f"split into UI, data, and API subtasks.",
        "priority": "HIGH", "agent_persona_id": agents["api"]["id"]})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"id": 1, "who": agents["api"]["name"]},
                            "create_task"))
    subs, scalls = p["tasks"], []
    for t in subs:
        a = agents.get(t["layer"], agents["api"])
        cid, call = tool_call(ids, "create_task", {
            "title": t["title"], "description": t["description"],
            "parent_task_id": 1,
            "priority": "HIGH" if t["layer"] == "db" else "MED",
            "agent_persona_id": a["id"]})
        scalls.append((cid, call, a))
    msgs.append(asst_calls([c for _, c, _ in scalls]))
    for i, (cid, _, a) in enumerate(scalls):
        msgs.append(tool_result(cid, {"id": i + 2, "who": a["name"]},
                                "create_task"))
    msgs.append(asst_text(f"Split \"{feature}\" into {len(subs)} subtasks under "
                          f"the parent. Ready to assign."))
    return {"messages": msgs}


def build_tasks_status(p, R):
    """List tasks then update a status — a maintenance interaction. The live
    context already shows the tasks (serve injects it), and the host still
    re-reads with list_tasks before mutating (the catalog's read-before-mutate
    rule), then speaks a short confirmation."""
    ids = Ids()
    context = (f"Project: {p['name']} (id: 1)\n"
               f"Current tasks ({len(p['tasks'])} total):\n"
               + "\n".join(f"- {t['title']} [HIGH] — TODO"
                           for t in p["tasks"]))
    msgs = [sys_msg(coordinator_system(p, context)),
            user_msg(evolve_user(R, "What is open, and mark the schema task in progress?"))]
    cid, call = tool_call(ids, "list_tasks", {})
    msgs.append(asst_calls([call]))
    agents = _agents_for(p)
    open_tasks = [{"id": i + 1, "title": t["title"], "status": "TODO",
                   "priority": "HIGH" if t["layer"] in ("db", "ui") else "MED",
                   "agent": agents.get(t["layer"], agents["api"])["id"]}
                  for i, t in enumerate(p["tasks"])]
    msgs.append(tool_result(cid, {"tasks": open_tasks}, "list_tasks"))
    db_idx = next((i for i, t in enumerate(p["tasks"])
                   if t["layer"] == "db"), 0)
    cid, call = tool_call(ids, "update_task_status", {
        "task_id": db_idx + 1, "status": "IN_PROGRESS"})
    msgs.append(asst_calls([call], content=(
        f"{len(open_tasks)} tasks are open. Marking the schema task in "
        f"progress.")))
    msgs.append(tool_result(cid, {"title": p["tasks"][db_idx]["title"]},
                            "update_task_status"))
    msgs.append(asst_text("Done — the schema task is now in progress."))
    return {"messages": msgs}


def build_refine(p, R):
    """REFINE stage (after finalize): the user DESCRIBES part of the app in
    their own words; the host routes it to the right plan file, ALWAYS
    read_plan first, then update_plan with the FULL merged Markdown (preserving
    headings + the checkbox skeleton), and confirms in 1-2 sentences."""
    ids = Ids()
    area, plan, skeleton = R.choice([
        ("ui", "/PLANS/Client.md",
         "# Client\n\n## Screens\n- [ ] Home screen\n- [ ] Settings"),
        ("server", "/PLANS/Server.md",
         "# Server\n\n## Endpoints\n- [ ] GET /health"),
        ("db", "/PLANS/Database.md",
         "# Database\n\n## Entities\n- [ ] users"),
    ])
    feat = R.choice(p["features"])
    if area == "ui":
        say = (f"So the main screen should show {feat.lower()} front and "
               f"center, with big friendly buttons — my customers aren't "
               f"techy.")
        x, detail = feat, "primary surface, large tap targets, friendly tone"
        confirm = f"Added a {feat} screen spec to Client.md."
    elif area == "server":
        say = (f"When somebody uses {feat.lower()}, the app needs to check it "
               f"on the server so two people can't grab the same thing.")
        x, detail = feat, (f"validates {feat.lower()} server-side; rejects "
                           f"conflicts atomically")
        confirm = f"Added a {feat} endpoint with conflict checking to Server.md."
    else:
        say = (f"Each {p['domain']} customer should have their history saved "
               f"— what they did, when, and how it went.")
        x, detail = "customer_history", "per-customer activity rows (what/when/outcome)"
        confirm = "Added a customer_history entity to Database.md."
    msgs = [sys_msg(refine_system(p)), user_msg(evolve_user(R, say))]
    # FULL new markdown = the existing skeleton with the new checklist item
    # ADDED (headings + existing boxes preserved — never deleted).
    new_md = skeleton + f"\n- [ ] {x} — {detail}"
    cid, call = tool_call(ids, "read_plan", {"path": plan})
    msgs.append(asst_calls([call], reasoning=_REASON.get("refine_read", "")))
    msgs.append(tool_result(cid, {"content": skeleton}, "read_plan"))
    cid, call = tool_call(ids, "update_plan", {"path": plan,
                                               "content": new_md})
    msgs.append(asst_calls([call], reasoning=_REASON.get("refine_merge", "")))
    msgs.append(tool_result(cid, {}, "update_plan"))
    msgs.append(asst_text(f"{confirm} What should we flesh out next — another "
                          f"screen, the API, or the data model?"))
    return {"messages": msgs}


# ───────────────────────── orchestration ───────────────────────────────────

def build_setup_nudge_record(p, R):
    """The app's RECORD-SELECTION nudge: the host acknowledged an answer in prose
    without calling propose_tags, the session injects the recovery nudge, and the
    host immediately records the picks. Teaches the correct response to the nudge
    (and, by contrast, that answers must be tagged before moving on)."""
    ids = Ids()
    opener = R.choice(opener_for(p["idea"])).format(idea=p["idea"])
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, opener))]
    seed = ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(FEAT_QUESTIONS), "options": p["features"],
        "multi": True})
    msgs.append(asst_calls([call]))
    picks = p["features"][:R.randint(2, max(2, len(p["features"]) - 1))]
    msgs.append(tool_result(cid, {"answer": picks}, "ask_question"))
    # The mistake: prose acknowledgement, no propose_tags. The app detects it and
    # injects the nudge as a user message.
    msgs.append(asst_text("Great choices — those will work well.",
                          reasoning=_REASON.get("wrapup", "")))
    msgs.append(user_msg(_TR["nudge_record"].format(sel=", ".join(picks))))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "features", "value": v} for v in picks]
                + _stack_tags(p)})
    msgs.append(asst_calls([call], reasoning=_REASON[
        "nudge_record_recover"].format(category="features")))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(OBJ_QUESTIONS), "options": p["objectives"],
        "multi": True})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"answer": p["objectives"][:3]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "objectives", "value": v}
                 for v in p["objectives"][:3]]})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated."))
    return {"messages": msgs}


def build_setup_nudge_continue(p, R):
    """The app's CONTINUE nudge: the host produced neither text nor a tool call
    (a stall); the session injects the continue nudge and the host takes the next
    step — asking the next OPEN topic, exactly once."""
    ids = Ids()
    opener = R.choice(opener_for(p["idea"])).format(idea=p["idea"])
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, opener))]
    seed = ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]]
            + [{"category": "objectives", "value": v}
               for v in p["objectives"][:2]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    # The stall, then the app's nudge.
    msgs.append(asst_text("", reasoning=_REASON.get("stall", "")))
    msgs.append(user_msg(_TR["nudge_continue"]))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(FEAT_QUESTIONS), "options": p["features"],
        "multi": True})
    msgs.append(asst_calls([call], reasoning=_REASON[
        "nudge_continue_recover"].format(topic="Features")))
    msgs.append(tool_result(cid, {"answer": p["features"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "features", "value": v} for v in p["features"]]
                + _stack_tags(p)})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated."))
    return {"messages": msgs}


def build_setup_skip(p, R):
    """The user SKIPS a question ("User skipped the question." result). The host
    must not re-ask it — it proposes sensible minimal defaults itself and moves
    on, so the interview still completes."""
    ids = Ids()
    opener = R.choice(opener_for(p["idea"])).format(idea=p["idea"])
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, opener))]
    seed = ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]]
            + [{"category": "objectives", "value": v}
               for v in p["objectives"][:3]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(FEAT_QUESTIONS), "options": p["features"],
        "multi": True})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"skipped": True}, "ask_question"))
    feats = p["features"][:2]
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "features", "value": v} for v in feats]
                + _stack_tags(p)})
    msgs.append(asst_calls([call], content=(
        f"No problem — I'll start it with {feats[0].lower()} and "
        f"{feats[1].lower()}; we can adjust later."), reasoning=_REASON[
        "skip_no_reask"].format(topic="features")))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated."))
    return {"messages": msgs}


def _short_label(s):
    """Compress a long scoped-vocabulary sentence into the ≤5-word tag label the
    app's propose_tags accepts (first clause, first 4 words)."""
    words = str(s).split(",")[0].split("(")[0].split()
    return " ".join(words[:4]).rstrip(".;:")


def build_setup_subaxis(p, R):
    """INDUSTRY-DEPENDENT follow-up: proposing an industry returns the serve
    'NEXT: …' instruction for that industry's sub-axis (Gaming → Genre,
    Healthcare → Care Setting, …). The host obeys it: asks the sub-axis question
    with the catalog's values, tags the pick under the sub-axis category, then
    calls scope_options — whose vocabulary now depends on that pick — and turns
    the long scoped phrasing into SHORT (≤5-word) tags. Every industry's axis +
    values come from the app's own catalog seed, so training tracks the DB."""
    ind = p["industries"][0]
    if ind not in SCOPED:
        ind = R.choice(list(SCOPED.keys()))
    sa = SCOPED[ind]
    pick = R.choice(sa["detail"])
    ids = Ids()
    opener = R.choice(opener_for(p["idea"])).format(idea=p["idea"])
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, opener))]

    seed = ([{"category": "industries", "value": ind}]
            + [{"category": "platforms", "value": v} for v in p["platforms"]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call]))
    # Serve-exact result: Proposed + the industry's NEXT sub-axis instruction.
    msgs.append(tool_result(
        cid,
        _TR["propose"].format(values=", ".join(t["value"] for t in seed))
        + _TR["propose_next_subaxis"].format(
            axis=sa["name"], industries=ind, axis_lower=sa["name"].lower(),
            options=", ".join(sa["values"]), axis_key=sa["key"]),
        "propose_tags"))

    # Obey the NEXT: ask the sub-axis question with the catalog's values.
    cid, call = tool_call(ids, "ask_question", {
        "question": f"Which {sa['name'].lower()}(s) does this target?",
        "options": sa["values"], "multi": True})
    msgs.append(asst_calls([call], reasoning=_REASON.get("subaxis_follow", "")
                           .format(axis=sa["name"])))
    msgs.append(tool_result(cid, {"answer": [pick["value"]]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": sa["key"], "value": pick["value"]}]})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))

    # Re-scope objectives to THAT sub-axis value (the vocabulary depends on it).
    cid, call = tool_call(ids, "scope_options", {"category": "objectives"})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _TR["scope_options"].format(
        category="objectives", scope=f"{ind} / {pick['value']}", platform="",
        values=", ".join(pick["objectives"])), "scope_options"))
    short_objs = [_short_label(o) for o in pick["objectives"]]
    cid, call = tool_call(ids, "ask_question", {
        "question": "Objectives — which of these matter most?",
        "options": short_objs, "multi": True})
    msgs.append(asst_calls([call], reasoning=_REASON.get("scope_compress", "")))
    chosen = short_objs[:R.randint(2, 3)]
    msgs.append(tool_result(cid, {"answer": chosen}, "ask_question"))
    short_feats = [_short_label(f) for f in pick["features"][:3]]
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "objectives", "value": v} for v in chosen]
                + [{"category": "features", "value": v} for v in short_feats]
                + _stack_tags(p)})
    msgs.append(asst_calls([call], content=(
        f"Recorded those for {pick['value']}, with fitting features and a "
        f"{p['languages'][0]}/{p['frameworks'][0]} stack.")))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated."))
    return {"messages": msgs}


def build_course_correction(spec):
    """A corrected replay of a REAL eval failure, from a compact editable spec
    (workspace/seeds/course_corrections.json): ask each open topic ONCE, never
    re-ask a tagged topic, derive the stack, finalize promptly. Built through the
    same helpers as every other builder so formats can never drift."""
    ids = Ids()
    msgs = [sys_msg(interview_system({"name": spec["name"]})),
            user_msg(spec["idea"])]
    seed = ([{"category": "industries", "value": v} for v in spec["industries"]]
            + [{"category": "platforms", "value": v} for v in spec["platforms"]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call], reasoning=_REASON["correction_seed"]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": "What should it do? (objectives)",
        "options": spec["objectives"], "multi": True})
    msgs.append(asst_calls([call],
                           reasoning=_REASON["correction_ask_objectives"]))
    msgs.append(tool_result(cid, {"answer": spec["objectives"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "objectives", "value": v}
                 for v in spec["objectives"]]})
    msgs.append(asst_calls([call],
                           reasoning=_REASON["correction_save_objectives"]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "ask_question", {
        "question": "What features matter most for v1?",
        "options": spec["features"], "multi": True})
    msgs.append(asst_calls([call], reasoning=_REASON["correction_ask_features"]))
    msgs.append(tool_result(cid, {"answer": spec["features"]}, "ask_question"))
    cid, call = tool_call(ids, "propose_tags", {
        "tags": [{"category": "features", "value": v} for v in spec["features"]]
                + [{"category": "languages", "value": v}
                   for v in spec["languages"]]
                + [{"category": "frameworks", "value": v}
                   for v in spec["frameworks"]]})
    msgs.append(asst_calls([call],
                           reasoning=_REASON["correction_save_features"]))
    msgs.append(tool_result(cid, {"ok": True}, "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call], reasoning=_REASON["correction_finalize"]))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated. Ready for "
                          "discovery."))
    return {"messages": msgs}


SETUP_VARIANTS = [build_setup_full, build_setup_partial, build_setup_recovery,
                  build_setup_image, build_setup_infer, build_setup_vague,
                  build_setup_ambiguous, build_setup_libraries,
                  build_setup_correction, build_setup_nudge_record,
                  build_setup_nudge_continue, build_setup_skip]
TASK_VARIANTS = [build_tasks, build_tasks_breakdown, build_tasks_status]


def generate(target, kinds, seed):
    """Synthesize TRAINING EXAMPLES (not whole conversations): every scripted
    conversation is split at each assistant generation point into the exact
    request the app would send there (see split_serve_points). `target` counts
    examples. Each example carries a "conv" group id so the train/valid split
    can keep sibling examples (shared prefixes!) on the same side."""
    R = random.Random(seed)
    _REASON_RNG.seed(seed * 7919 + 13)
    examples = []
    conv_n = 0

    def emit(convo, kind, board):
        nonlocal conv_n
        conv_n += 1
        cid = f"c{seed}-{conv_n}"
        for ex in split_serve_points(convo["messages"], board=board):
            examples.append({"messages": ex, "tools": tools_for(kind),
                             "conv": cid})

    # round-robin the kinds so the corpus stays balanced as it grows
    while len(examples) < target:
        p = make_scenario(R)
        if "recovery" in kinds:
            # ADDITIVE-pass corpus (post run-4): ONLY the trajectories the
            # eval got wrong, weighted toward the two new shapes. General
            # coverage comes from REPLAY of the previous corpus (sampled
            # separately into the same dataset), not from regenerating it.
            for s in ("compound", "early_libs", "compound", "early_libs",
                      "compound", "early_libs"):
                emit(build_setup_recovery(p, R, shape=s), "setup", board=True)
            # keep the original rejection shapes warm alongside the new ones
            emit(build_setup_recovery(p, R), "setup", board=True)
        if "setup" in kinds:
            # board=True: setup interview requests carry the turn-start BOARD
            # STATE system msg at the tail + the 4-turn trimmed history —
            # byte-identical to setup_session.dart.
            for b in SETUP_VARIANTS:
                emit(b(p, R), "setup", board=True)
            # Weight the ambiguous-industry case up so "ask which industry"
            # becomes a learned default for cross-industry ideas (was too rare
            # at 1/8, so the model defaulted to confident single-industry tags).
            for _ in range(2):
                emit(build_setup_ambiguous(p, R), "setup", board=True)
            # Weight the tag-correction case up too, so "user says it's NOT a
            # {industry} → remove_tags" is well-learned across every DB value (each
            # call draws fresh wrong industries from the taxonomy).
            emit(build_setup_correction(p, R), "setup", board=True)
            # Weight the finalize-rejection recovery up (2 extra draws beyond
            # the SETUP_VARIANTS one): the run-3 eval showed an unrecovered
            # premature finalize is THE dominant failure mode, and each draw
            # picks a random shape (early/objectives/stack/loopguard), so it
            # takes several per scenario to cover them.
            for _ in range(2):
                emit(build_setup_recovery(p, R), "setup", board=True)
            # Industry-DEPENDENT follow-up (NEXT: sub-axis → scoped vocabulary),
            # weighted 2x: every industry's axis/values/objectives come from the
            # app's own catalog, so the model learns that the questions CHANGE
            # with the chosen industry instead of one generic script.
            for _ in range(2):
                emit(build_setup_subaxis(p, R), "setup", board=True)
            # REFINE stage (post-finalize plan editing) — read_plan → update_plan.
            # No board state: stateSummary is interview-phase only.
            emit(build_refine(p, R), "refine", board=False)
        if "discovery" in kinds:
            # Story-tree coverage is a first-class goal: the chunk flow (draft vs
            # manual chosen inside), the turn-by-turn USER-TERMINOLOGY walkthrough
            # (weighted 2x — it teaches chaining + one-question-per-turn + expand-
            # passing-mentions + confirm-close), and the feature-area grouped tree.
            for _ in range(2):
                emit(build_discovery(p, R), "discovery", board=False)
            for _ in range(2):
                emit(build_discovery_walkthrough(p, R), "discovery", board=False)
            emit(build_discovery_grouped(p, R), "discovery", board=False)
        if "tasks" in kinds:
            for b in TASK_VARIANTS:
                emit(b(p, R), "tasks", board=False)
    # Course corrections — compact editable specs (workspace/seeds/
    # course_corrections.json), each the corrected replay of a REAL eval failure
    # (ask-once, never re-ask a tagged topic, derive the stack, finalize). Built
    # through the same helpers + serve-point splitting so formats never drift.
    if "setup" in kinds:
        for spec in load_seed("course_corrections").get("corrections", []):
            emit(build_course_correction(spec), "setup", board=True)
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=None,
                    help="how many training EXAMPLES (serve-shaped generation "
                         "points) to synthesize (pre-dedupe). Defaults to "
                         "60000, or 8000 for a recovery-only additive pass.")
    ap.add_argument("--kinds", default="setup,discovery,tasks")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    target = args.target or (8000 if kinds == {"recovery"} else 60000)

    examples = generate(target, kinds, args.seed)
    added, skipped = append_conversations(examples, source="generated")
    print(f"Synthesized {len(examples)} example(s) → added {added}, "
          f"skipped {skipped} dup. Kinds: {sorted(kinds)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
