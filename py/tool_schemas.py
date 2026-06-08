#!/usr/bin/env python3
"""The REAL Nexus tool schemas, grouped by agent, used to attach a per-example
`tools` array to each training conversation (mlx_lm.lora "tools" format). Keeping
the served schemas in the training data makes train == serve.

The schemas are editable JSON, not hardcoded: workspace/seeds/tool_schemas.json
(full, gitignored) with a committed example at seeds/tool_schemas.example.json.
Keep them in sync with the app's setup_tools.dart / coordinator_tools.dart.
"""
from seedlib import load_seed

_d = load_seed("tool_schemas")
SETUP_TOOLS = _d["setup"]
DISCOVERY_TOOLS = _d["discovery"]
PM_TOOLS = _d["tasks"]


def tools_for(kind):
    return {"setup": SETUP_TOOLS, "discovery": DISCOVERY_TOOLS,
            "tasks": PM_TOOLS}.get(kind, [])
