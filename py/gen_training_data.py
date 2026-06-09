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


# Short, tool-aware THINKING the assistant emits before it acts. The Nemotron
# chat template renders `reasoning_content` as `<think>…</think>` before the
# content/tool call, so seeding every assistant turn with a brief, closed thought
# teaches the model to think-then-act AND to close the tag (preventing the leaked,
# never-closed `<think>` the model was doing). History thinking is auto-truncated
# by the template, so only the latest turn's thought is kept at serve time.
_REASON = {
    "propose_tags": "Mapping what they told me to the right tags before I move on.",
    "remove_tags": "They said those are wrong — I'll remove exactly the tags they rejected, nothing else.",
    "ask_question": "That topic is still open, so I'll ask it with concrete options.",
    "finalize_setup": "Every required topic has at least one tag now, so I can finalize.",
    "scope_options": "Before finalizing I should scope the libraries and verify them.",
    "lookup_package": "Let me check each package is still maintained before adding it.",
    "dismiss_item": "These came back stale, so I'll drop them.",
    "consider_items": "Worth weighing these options before I decide.",
    "generate_image": "They want to see it, so I'll generate a mockup.",
    "edit_image": "Applying the change they asked for to the latest image.",
    "create_task": "Turning this into a concrete, stack-specific task.",
    "assign_agent_to_task": "Assigning it to the agent that fits.",
    "update_task": "Updating the task with their change.",
    "update_task_status": "Moving the task to its new status.",
    "add_user_story": "Capturing this as a user story in the tree.",
    "update_user_story": "Refining that story with the new detail.",
    "draft_stories_from_text": "Big chunk — I'll split it into clean stories.",
}


def _reason_for(calls):
    name = calls[0]["function"]["name"] if calls else None
    return _REASON.get(name, "I'll call the tool to act on what they said.")


def asst_text(t, reasoning=None):
    return {"role": "assistant", "content": t,
            "reasoning_content": reasoning or "Wrapping up in one short sentence."}


def tool_call(ids, name, args):
    cid = ids.next()
    return cid, {"id": cid, "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}


def asst_calls(calls, content=None, reasoning=None):
    return {"role": "assistant", "content": content if content else None,
            "tool_calls": calls,
            "reasoning_content": reasoning or _reason_for(calls)}


def tool_result(cid, obj, name=None):
    m = {"role": "tool", "tool_call_id": cid,
         "content": obj if isinstance(obj, str) else json.dumps(obj)}
    if name:
        m["name"] = name
    return m


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


def discovery_system(p):
    return _PROMPTS["discovery_system"].format(name=p["name"])


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
    cid, call = tool_call(ids, "scope_options", {
        "category": "libraries", "platform": _platform_bucket(p["platforms"])})
    msgs.append(asst_calls([call], content=(
        f"Before I finalize, let me pick {lang} libraries and verify they're "
        f"current.")))
    msgs.append(tool_result(cid, {"libraries": [c[0] for c in cands]},
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
    cid, call = tool_call(ids, "scope_options", {
        "category": "libraries", "platform": _platform_bucket(p["platforms"])})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"libraries": [c[0] for c in cands]},
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


def build_setup_recovery(p, R):
    ids = Ids()
    msg = f"Let's set up {p['name']}. It's a {p['industries'][0]} app."
    msgs = [sys_msg(interview_system(p)), user_msg(evolve_user(R, msg))]
    seed = ([{"category": "industries", "value": v} for v in p["industries"]]
            + [{"category": "platforms", "value": v} for v in p["platforms"]])
    cid, call = tool_call(ids, "propose_tags", {"tags": seed})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(seed)}, "propose_tags"))
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call]))
    missing = ["objectives", "features", "languages", "frameworks"]
    msgs.append(tool_result(cid, {"ok": False, "error": "not ready",
                "missing": missing,
                "message": "Add at least one tag to each: " + ", ".join(missing)},
                "finalize_setup"))
    cid, call = tool_call(ids, "ask_question", {
        "question": R.choice(OBJ_QUESTIONS), "options": p["objectives"],
        "multi": True})
    msgs.append(asst_calls([call], content=(
        "Not ready yet — it still needs objectives and a stack. Let me gather "
        "those.")))
    msgs.append(tool_result(cid, {"answer": p["objectives"]}, "ask_question"))
    fill = ([{"category": "objectives", "value": v} for v in p["objectives"]]
            + [{"category": "features", "value": v} for v in p["features"]]
            + _stack_tags(p))
    cid, call = tool_call(ids, "propose_tags", {"tags": fill})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True, "added": len(fill)}, "propose_tags"))
    _libraries_phase(ids, p, R, msgs)
    cid, call = tool_call(ids, "finalize_setup", {})
    msgs.append(asst_calls([call], content="Everything's tagged now — finalizing."))
    msgs.append(tool_result(cid, _plans_result(), "finalize_setup"))
    msgs.append(asst_text("Setup complete — plans generated. Ready for discovery."))
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


def build_discovery(p, R):
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
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"id": "S1"}, "add_user_story"))
    msgs.append(asst_calls([], content=None) if False else asst_text(
        f"Captured \"{p['epic_title']}\" as the root. Walk me through the first "
        f"thing a {p['role']} does."))
    chunk = (f"They {', '.join(s.lower() for s in p['flow'][:-1])}, and finally "
             f"{p['flow'][-1].lower()}.")
    msgs.append(user_msg(evolve_user(R, chunk)))
    use_draft = R.random() < 0.5
    if use_draft:
        cid, call = tool_call(ids, "draft_stories_from_text",
                              {"text": chunk, "parent_story_id": "S1"})
        msgs.append(asst_calls([call]))
        drafted = [{"id": f"S{i+2}", "title": s} for i, s in enumerate(p["flow"])]
        msgs.append(tool_result(cid, {"created": drafted},
                                "draft_stories_from_text"))
    else:
        # manual add, one story per step, batched
        calls = []
        for i, s in enumerate(p["flow"]):
            cid, call = tool_call(ids, "add_user_story", {
                "title": s,
                "narrative": f"As a {p['role']}, I want to {s.lower()}, so that "
                             f"the flow continues.",
                "parent_story_id": "S1", "kind": "story"})
            calls.append((cid, call, f"S{i+2}"))
        msgs.append(asst_calls([c for _, c, _ in calls]))
        for cid, _, sid in calls:
            msgs.append(tool_result(cid, {"id": sid}, "add_user_story"))
    # chain the steps
    chain = []
    prev = "S1"
    for i in range(len(p["flow"])):
        cid, call = tool_call(ids, "move_user_story", {
            "story_id": f"S{i+2}", "parent_story_id": prev, "order_index": 0})
        chain.append((cid, call, f"S{i+2}"))
        prev = f"S{i+2}"
    msgs.append(asst_calls([c for _, c, _ in chain]))
    for cid, _, sid in chain:
        msgs.append(tool_result(cid, {"ok": True, "story_id": sid},
                                "move_user_story"))
    case, answer, etitle, enarr, eaccept = p["edge"]
    msgs.append(asst_text(f"Chained the {len(p['flow'])} steps. What happens when "
                          f"{case}?"))
    msgs.append(user_msg(evolve_user(R, answer)))
    cid, call = tool_call(ids, "add_user_story", {
        "title": etitle, "narrative": enarr, "acceptance_criteria": eaccept,
        "parent_story_id": prev, "kind": "substory"})
    msgs.append(asst_calls([call]))
    edge_id = f"S{len(p['flow'])+2}"
    msgs.append(tool_result(cid, {"id": edge_id}, "add_user_story"))
    cid, call = tool_call(ids, "add_note", {
        "story_id": edge_id, "body": f"Edge case: {case}."})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"ok": True}, "add_note"))
    cid, call = tool_call(ids, "list_user_stories", {})
    msgs.append(asst_calls([call]))
    tree = [{"id": "S1", "title": p["epic_title"], "parent": None}] + [
        {"id": f"S{i+2}", "title": s} for i, s in enumerate(p["flow"])]
    msgs.append(tool_result(cid, {"stories": tree}, "list_user_stories"))
    msgs.append(asst_text("That captures the full flow with the edge case "
                          "handled. If nothing's missing, press \"Generate tasks "
                          "from stories\"."))
    return {"messages": msgs}


def build_tasks(p, R):
    ids = Ids()
    asks = [
        "Generate tasks from the setup and discovery stories.",
        "Break the plans down into tasks and assign them.",
        "Turn the user stories into concrete engineering tasks.",
        "Create the build tasks for v1 and assign each to an agent.",
    ]
    msgs = [sys_msg(pm_system(p)), user_msg(evolve_user(R, R.choice(asks)))]
    agent_for = {"ui": "sde-frontend", "db": "sde-database", "api": "sde-backend"}
    if "Flutter" in p["frameworks"]:
        agent_for["ui"] = "sde-flutter"
    calls, meta = [], []
    for t in p["tasks"]:
        cid, call = tool_call(ids, "create_task", {
            "title": t["title"], "description": t["description"],
            "priority": "HIGH" if t["layer"] in ("db", "ui") else "MED"})
        calls.append(call)
        meta.append((cid, t["layer"]))
    msgs.append(asst_calls(calls))
    tids = []
    for i, (cid, _) in enumerate(meta):
        msgs.append(tool_result(cid, {"id": f"T{i+1}", "status": "TODO"},
                                "create_task"))
        tids.append(f"T{i+1}")
    acalls = []
    for i, (_, layer) in enumerate(meta):
        cid, call = tool_call(ids, "assign_agent_to_task", {
            "task_id": tids[i], "agent_persona_id": agent_for.get(layer,
                                                                  "sde-backend")})
        acalls.append((cid, call))
    msgs.append(asst_calls([c for _, c in acalls]))
    for cid, _ in acalls:
        msgs.append(tool_result(cid, {"ok": True}, "assign_agent_to_task"))
    msgs.append(asst_text(
        f"Created {len(p['tasks'])} tasks from the stories and assigned each to "
        f"the right agent — every one has acceptance criteria and a verify "
        f"command."))
    return {"messages": msgs}


def build_tasks_breakdown(p, R):
    """Create a parent task, then sub-tasks under it (parent_task_id)."""
    ids = Ids()
    feature = R.choice(p["features"])
    msgs = [sys_msg(pm_system(p)),
            user_msg(evolve_user(R, f"Break down the \"{feature}\" feature into subtasks."))]
    cid, call = tool_call(ids, "create_task", {
        "title": f"Implement {feature}",
        "description": f"Objective: deliver {feature.lower()}. Parent task; "
                       f"split into UI, data, and API subtasks.",
        "priority": "HIGH"})
    msgs.append(asst_calls([call]))
    msgs.append(tool_result(cid, {"id": "T1", "status": "TODO"}, "create_task"))
    subs, scalls = p["tasks"], []
    for t in subs:
        cid, call = tool_call(ids, "create_task", {
            "title": t["title"], "description": t["description"],
            "parent_task_id": "T1",
            "priority": "HIGH" if t["layer"] == "db" else "MED"})
        scalls.append((cid, call, t["layer"]))
    msgs.append(asst_calls([c for _, c, _ in scalls]))
    for i, (cid, _, _) in enumerate(scalls):
        msgs.append(tool_result(cid, {"id": f"T{i+2}", "status": "TODO"},
                                "create_task"))
    msgs.append(asst_text(f"Split \"{feature}\" into {len(subs)} subtasks under "
                          f"the parent. Ready to assign."))
    return {"messages": msgs}


def build_tasks_status(p, R):
    """List tasks then update a status — a maintenance interaction."""
    ids = Ids()
    msgs = [sys_msg(pm_system(p)),
            user_msg(evolve_user(R, "What is open, and mark the schema task in progress?"))]
    cid, call = tool_call(ids, "list_tasks", {})
    msgs.append(asst_calls([call]))
    open_tasks = [{"id": f"T{i+1}", "title": t["title"], "status": "TODO"}
                  for i, t in enumerate(p["tasks"])]
    msgs.append(tool_result(cid, {"tasks": open_tasks}, "list_tasks"))
    db_idx = next((i for i, t in enumerate(p["tasks"])
                   if t["layer"] == "db"), 0)
    cid, call = tool_call(ids, "update_task_status", {
        "task_id": f"T{db_idx+1}", "status": "IN_PROGRESS"})
    msgs.append(asst_calls([call], content=(
        f"{len(open_tasks)} tasks are open. Marking the schema task in "
        f"progress.")))
    msgs.append(tool_result(cid, {"ok": True, "status": "IN_PROGRESS"},
                            "update_task_status"))
    msgs.append(asst_text("Done — the schema task is now in progress."))
    return {"messages": msgs}


# ───────────────────────── orchestration ───────────────────────────────────

SETUP_VARIANTS = [build_setup_full, build_setup_partial, build_setup_recovery,
                  build_setup_image, build_setup_infer, build_setup_vague,
                  build_setup_ambiguous, build_setup_libraries,
                  build_setup_correction]
TASK_VARIANTS = [build_tasks, build_tasks_breakdown, build_tasks_status]


def generate(target, kinds, seed):
    R = random.Random(seed)
    convos = []
    # round-robin the kinds so the corpus stays balanced as it grows
    while len(convos) < target:
        p = make_scenario(R)
        if "setup" in kinds:
            for b in SETUP_VARIANTS:
                convos.append({**b(p, R), "tools": tools_for("setup")})
            # Weight the ambiguous-industry case up so "ask which industry"
            # becomes a learned default for cross-industry ideas (was too rare
            # at 1/8, so the model defaulted to confident single-industry tags).
            for _ in range(2):
                convos.append({**build_setup_ambiguous(p, R),
                               "tools": tools_for("setup")})
            # Weight the tag-correction case up too, so "user says it's NOT a
            # {industry} → remove_tags" is well-learned across every DB value (each
            # call draws fresh wrong industries from the taxonomy).
            convos.append({**build_setup_correction(p, R),
                           "tools": tools_for("setup")})
        if "discovery" in kinds:
            # two discovery variants (draft vs manual chosen inside)
            for _ in range(2):
                convos.append({**build_discovery(p, R),
                               "tools": tools_for("discovery")})
        if "tasks" in kinds:
            for b in TASK_VARIANTS:
                convos.append({**b(p, R), "tools": tools_for("tasks")})
    return convos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10500,
                    help="how many conversations to synthesize (pre-dedupe)")
    ap.add_argument("--kinds", default="setup,discovery,tasks")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}

    convos = generate(args.target, kinds, args.seed)
    added, skipped = append_conversations(convos, source="generated")
    print(f"Synthesized {len(convos)} conversation(s) → added {added}, "
          f"skipped {skipped} dup. Kinds: {sorted(kinds)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
