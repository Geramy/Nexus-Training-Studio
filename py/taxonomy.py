#!/usr/bin/env python3
"""Industry taxonomy with NATURAL-LANGUAGE triggers — teaches the model to INFER
the right industry/objectives from how a real person describes their idea
("I want to sell lemonade" → Food & Beverage), instead of being handed the tag.

The data is editable JSON, not hardcoded here: workspace/seeds/industries.json
(full, gitignored) with a committed example at seeds/industries.example.json.
"""
from seedlib import load_seed

_d = load_seed("industries")
INDUSTRIES = _d["industries"]
VERB_OPENERS = _d["verb_openers"]
NOUN_OPENERS = _d["noun_openers"]
INFER_REFLECTIONS = _d["infer_reflections"]


def opener_for(idea):
    """Pick the right opener register for an idea phrase (noun vs verb)."""
    first = idea.strip().split()[0].lower()
    return NOUN_OPENERS if first in ("a", "an", "the") else VERB_OPENERS
