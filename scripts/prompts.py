"""
Shared prompt templates and prompt-building logic, used by BOTH the
real-time pilot script and the batch request builder -- so the two paths
always generate from an identical prompt and never drift apart.

Deliberately has no dependency on google-genai so it can be imported and
unit-tested without that package installed.
"""
import json

EXTRACTION_PROMPT = """You are analyzing the STRUCTURE of a short story, not \
its content. Given the story below, output a single JSON object with \
exactly these fields, and nothing else:

- "genre_trope": a short label (e.g. "family favoritism", "hidden paternity",
"custody scheme", "financial betrayal")
- "character_roles": list of abstract roles and their relationship \
(e.g. "protagonist (adult daughter)", "antagonist (favored sibling)") -- \
no proper names
- "central_secret_or_stakes": one abstract sentence describing the hidden \
truth, scheme, or high-stakes element that drives the story (e.g. "a \
family member is secretly favored over another in a way that is only \
revealed at a major event") -- no proper nouns or specific incidents
- "beats": an ordered list of 5-7 objects, each with "beat" (one of: hook, \
inciting_incident, escalation, complication, twist, resolution) and \
"description" (one abstract sentence, no proper nouns, no specific \
settings, no verbatim phrasing from the source)
- "emotional_arc": the feeling progression a reader experiences, e.g. \
"sympathy -> frustration -> tension -> vindication"
- "hook_style": the technique of the opening line -- describe it \
abstractly (e.g. "opens with a direct quote from an antagonist" or \
"states the twist as a flat fact in the first sentence")
- "pov_and_tense": e.g. "first person, past tense"

Do not include any character names, place names, or specific incidents \
from the source story. Return ONLY the JSON object, no markdown fences, \
no preamble.

STORY:
{story_text}
"""

GENERATION_PROMPT = """Write an original, ~280-word Reddit-style short \
story (first person, conversational tone, in the style of r/{subreddit_hint}) \
that follows this structural template's beat sequence, emotional arc, and \
central secret/stakes mechanism exactly, but with entirely new specific \
content:

TEMPLATE:
{template_json}

Use these specifics -- do not deviate from them:
- Setting: {setting}
- Relationship between the two main people: {relationship_dynamic}
- The hidden truth, scheme, or high-stakes element driving the story \
should center on: {incident_seed}
- Character names: {name_a} and {name_b}
- Opening technique for the very first sentence: {hook_style}
- If it fits naturally, one character's occupation can be mentioned as \
minor supporting detail: {occupation} -- but don't build the plot around \
it; the family/relationship dynamic and the hidden secret are what drive \
the story, not the job.

Rules:
- Do not reuse any names, incidents, dialogue, or specific phrasing from \
any existing published story.
- Target length: 260-300 words.
- End on a natural hook for a comment/reaction, matching the emotional \
arc's final beat.
- Return ONLY the story text. No title, no preamble, no markdown.
"""

# Rough trope -> subreddit mapping so the tone matches. Falls back to
# AmItheAsshole, the most general-purpose one, if nothing matches.
SUBREDDIT_HINTS = {
    "paternity": "AmItheAsshole",
    "custody": "relationship_advice",
    "affair": "survivinginfidelity",
    "favoritism": "AmItheAsshole",
    "inheritance": "EntitledPeople",
    "financial": "MaliciousCompliance",
    "family betrayal": "AmItheAsshole",
    "relationship": "relationship_advice",
}


def build_extraction_prompt(story_text: str) -> str:
    return EXTRACTION_PROMPT.format(story_text=story_text)


def build_prompt(template: dict, combo: dict) -> str:
    trope = template.get("genre_trope", "").lower()
    subreddit_hint = next(
        (v for k, v in SUBREDDIT_HINTS.items() if k in trope),
        "AmItheAsshole",
    )
    # Prefer the SOURCE story's own extracted hook technique -- that's the
    # one that's actually proven to work for this specific template. Only
    # fall back to a randomly sampled one if extraction didn't capture it
    # (e.g. older templates run before this field existed, or a messy
    # extraction result). Independently randomizing this on every variant
    # would fight against the template's own demonstrated hook style,
    # which defeats the point of aligning with proven stories.
    hook_style = template.get("hook_style") or combo["hook_style"]
    return GENERATION_PROMPT.format(
        subreddit_hint=subreddit_hint,
        template_json=json.dumps(template),
        setting=combo["setting"],
        occupation=combo["occupation"],
        relationship_dynamic=combo["relationship_dynamic"],
        incident_seed=combo["incident_seed"],
        hook_style=hook_style,
        name_a=combo["character_names"][0],
        name_b=combo["character_names"][1],
    )
