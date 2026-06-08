#!/usr/bin/env python3
"""The REAL Nexus tool schemas, grouped by agent, used to attach a per-example
`tools` array to each training conversation (mlx_lm.lora "tools" format). Keeping
the served schemas in the training data makes train == serve and is the single
biggest lever for reliable tool calling (per the diversity research).

Mirror of the app's buildToolSchemas() — keep in sync with:
  setup_tools.dart, coordinator_tools.dart.
"""


def _fn(name, desc, props, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                **({"required": required} if required else {}),
            },
        },
    }


_STR = {"type": "string"}


# ───────────────────────────── Setup interview ──────────────────────────────
SETUP_TOOLS = [
    _fn("generate_image",
        "Generate an image from a text description and show it to the user (e.g. "
        "a mock-up of a screen, a logo, a concept). Call this whenever the user "
        "asks to see / make / draw / show a picture during setup.",
        {"prompt": {**_STR, "description": "Detailed visual description."},
         "size": {**_STR, "description": "WxH — 1024x1024 (default), 1024x1792, "
                  "or 1792x1024."}},
        ["prompt"]),
    _fn("edit_image",
        "Modify the most recent image with a described change (e.g. \"make the "
        "background blue\"). The latest image is used as the source.",
        {"prompt": {**_STR, "description": "The change to apply."},
         "size": {**_STR, "description": "Optional output size."}},
        ["prompt"]),
    _fn("ask_question",
        "Ask the user ONE interview question and get their answer. It shows the "
        "options as buttons the user taps and returns their selection. Give a "
        "clear question plus 2-8 short options, and keep multi=true so the user "
        "can pick several.",
        {"question": {**_STR, "description": "A single, clear question."},
         "options": {"type": "array", "items": _STR,
                     "description": "2-8 short, selectable choices."},
         "multi": {"type": "boolean",
                   "description": "Whether the user may pick more than one. "
                                  "DEFAULTS TO TRUE; set false only for yes/no."}},
        ["question", "options"]),
    _fn("propose_tags",
        "Save the user's answer(s) to the project profile as tags. Each tag value "
        "is a SHORT label (a few words, ≤5) for ONE concept — give several items "
        "as several tag objects. Batch tags across categories in one call.",
        {"tags": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "category": {**_STR, "description": "industries, platforms, "
                             "objectives, features, languages, frameworks, "
                             "databases, libraries, services, or a sub-axis."},
                "value": {**_STR, "description": "A SHORT label (≤5 words), one "
                          "concept per tag."},
                "forLanguage": {**_STR, "description": "Libraries ONLY: the "
                                "language this package is used with."}},
            "required": ["category", "value"]}}},
        ["tags"]),
    _fn("scope_status",
        "Read the adaptive scope from the user's current selections. Call right "
        "AFTER proposing industries, and again after platforms. Reports selected "
        "industries, any sub-axis to ask NEXT, and selected platforms.", {}),
    _fn("scope_options",
        "Get vocabulary tailored to the user's industry + sub-axis for a "
        "category. Call BEFORE asking objectives or features. Pass platform for "
        "languages/frameworks/libraries.",
        {"category": {"type": "string",
                      "enum": ["objectives", "features", "languages",
                               "frameworks", "libraries", "platforms"]},
         "platform": {**_STR, "description": "Mobile, Desktop, Web, Console, "
                      "Embedded, Cloud/Server."}},
        ["category"]),
    _fn("finalize_setup",
        "Resolve the architecture from confirmed tags and generate the /PLANS "
        "layer files. PRECONDITION: every REQUIRED section must have at least one "
        "tag first; otherwise this returns what is still missing.", {}),
]

# ───────────────────────────── Discovery / stories ──────────────────────────
DISCOVERY_TOOLS = [
    _fn("draft_stories_from_text",
        "When the user says a big chunk describing several things at once, pass "
        "their RAW text here. A helper splits it and rephrases each part into a "
        "clean user story (title + As a… I want… so that…) and adds them to the "
        "tree (optionally under parent_story_id).",
        {"text": {**_STR, "description": "The user's raw description to split."},
         "parent_story_id": {**_STR, "description": "Optional parent id."}},
        ["text"]),
    _fn("add_user_story",
        "Add a user story to the project story tree during discovery. Make it a "
        "child of an epic/story via parent_story_id to build the tree "
        "(epics → stories → sub-stories).",
        {"title": {**_STR, "description": "Short node title."},
         "narrative": {**_STR, "description": "As a <role>, I want <goal>, so "
                       "that <benefit>."},
         "acceptance_criteria": {**_STR, "description": "Optional markdown bullets."},
         "parent_story_id": {**_STR, "description": "Optional parent id to nest under."},
         "kind": {"type": "string", "enum": ["epic", "story", "substory"]}},
        ["title"]),
    _fn("update_user_story",
        "Update an existing user story (title, narrative, acceptance, status).",
        {"story_id": _STR, "title": _STR, "narrative": _STR,
         "acceptance_criteria": _STR,
         "status": {"type": "string", "enum": ["draft", "confirmed", "done"]}},
        ["story_id"]),
    _fn("move_user_story",
        "Re-parent and/or re-order a story to fix the tree (nest under another, "
        "chain a flow, or pull to root). Set parent_story_id to the new parent or "
        "null for root; optional order_index (0 = first).",
        {"story_id": _STR,
         "parent_story_id": {**_STR, "description": "New parent id, or null/empty "
                             "to make it a root."},
         "order_index": {"type": "integer"}},
        ["story_id"]),
    _fn("list_user_stories",
        "List the current user-story tree (ids, titles, parents, status) so you "
        "stay grounded in what is captured.", {}),
    _fn("add_note",
        "Attach a descriptive note to a user story (a detail, decision, "
        "constraint, or open question).",
        {"story_id": _STR, "body": {**_STR, "description": "The note text."}},
        ["story_id", "body"]),
]

# ───────────────────────────── PM / task generation ─────────────────────────
PM_TOOLS = [
    _fn("create_task",
        "Create a task. Write a concrete, stack-specific instruction with a clear "
        "objective, acceptance criteria, and a runnable verification.",
        {"title": {**_STR, "description": "Short imperative task title."},
         "description": {**_STR, "description": "Objective + acceptance criteria "
                         "+ a verify command and its expected result."},
         "parent_task_id": {**_STR, "description": "Optional parent task id."},
         "priority": {"type": "string", "enum": ["HIGH", "MED", "LOW"]},
         "agent_persona_id": {**_STR, "description": "Optional agent to assign."}},
        ["title"]),
    _fn("update_task",
        "Update a task's title, description, or priority.",
        {"task_id": _STR, "title": _STR, "description": _STR,
         "priority": {"type": "string", "enum": ["HIGH", "MED", "LOW"]}},
        ["task_id"]),
    _fn("update_task_status",
        "Update a task's status.",
        {"task_id": _STR,
         "status": {"type": "string",
                    "enum": ["TODO", "IN_PROGRESS", "BLOCKED", "DONE"]}},
        ["task_id", "status"]),
    _fn("list_tasks", "List the project's tasks (ids, titles, status).", {}),
    _fn("assign_agent_to_task",
        "Assign an agent persona to work on a task.",
        {"task_id": _STR, "agent_persona_id": _STR},
        ["task_id", "agent_persona_id"]),
    _fn("list_agents", "List available agent personas to assign.", {}),
    _fn("read_plan",
        "Read a plan file's full Markdown.",
        {"path": {**_STR, "description": "e.g. /PLANS/Client.md"}}, ["path"]),
    _fn("list_plans", "List the generated plan files under /PLANS.", {}),
]


def tools_for(kind):
    return {"setup": SETUP_TOOLS, "discovery": DISCOVERY_TOOLS,
            "tasks": PM_TOOLS}.get(kind, [])
