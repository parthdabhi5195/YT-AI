"""
Shared prompt templates and prompt-building logic, used by BOTH the
real-time pilot script and the batch request builder -- so the two paths
always generate from an identical prompt and never drift apart.

Deliberately has no dependency on google-genai so it can be imported and
unit-tested without that package installed.

DESIGN NOTES (why things are the way they are):

* No subreddit-style hints. Style comes from the source stories
  themselves, via the extracted "narrative_voice" field, not from a
  hardcoded lookup table of subreddit names.

* "beats" is a flat object with fixed keys, not a list of objects. Models
  follow a fixed-key schema far more reliably than "a list of 5-7 objects
  each with these two fields".

* Names are OPTIONAL and driven by the source. Many proven stories use
  only relational descriptors ("my sister", "my boss") and never name
  anyone. Extraction records which style the source used, and generation
  matches it instead of always forcing names in.

* The sampled relationship_dynamic describes the TWO CENTRAL characters
  only. Templates whose character_roles list has more than two entries
  get the extra roles filled as supporting characters -- the pool does
  not need multi-party combinations for this to work.
"""
import json

# Every field extraction must return. Used by extract_templates.py to
# reject partial responses instead of writing them to disk.
REQUIRED_TEMPLATE_FIELDS = [
    "genre_trope",
    "character_roles",
    "uses_character_names",
    "central_secret_or_stakes",
    "beats",
    "emotional_arc",
    "hook_style",
    "narrative_voice",
    "pov_and_tense",
]

# Fixed beat keys. Extraction must return all of them; empty string is
# allowed for beats the source story genuinely doesn't have.
BEAT_KEYS = [
    "hook",
    "inciting_incident",
    "escalation",
    "complication",
    "twist",
    "resolution",
]

EXTRACTION_PROMPT = """You are analyzing the STRUCTURE of a short story, \
not its content. Return a single JSON object with exactly the fields \
below and nothing else.

"genre_trope"
  A short label, e.g. "family favoritism", "hidden paternity",
  "custody scheme", "financial betrayal".

"character_roles"
  A list of the significant people, described ONLY by their role and
  relationship, never by name. Example:
  ["narrator (adult daughter)", "antagonist (mother)", "ally (grandmother)"]

"uses_character_names"
  true if the source story gives any character an actual first name.
  false if it refers to everyone only by relationship ("my sister",
  "my boss", "the driver").

"central_secret_or_stakes"
  ONE abstract sentence describing the hidden truth, scheme, or
  high-stakes element that drives the story. No proper nouns, no
  specific incidents.

"beats"
  A JSON object with EXACTLY these six keys:
    "hook", "inciting_incident", "escalation", "complication",
    "twist", "resolution"
  Each value is ONE plain sentence describing what structurally happens
  at that point, abstractly. No names, no places, no specific objects,
  no wording copied from the source. If the story genuinely has no such
  beat, use an empty string "" for that key. Do not add or rename keys.

"emotional_arc"
  The feeling progression a listener experiences, e.g.
  "sympathy -> frustration -> tension -> vindication".

"hook_style"
  How the FIRST sentence works, described abstractly, e.g.
  "opens with a direct quote from another character" or
  "states the twist as a flat fact before any context".

"narrative_voice"
  One short phrase describing tone and register, e.g.
  "flat, matter-of-fact, understated" or
  "urgent and panicked, short sentences" or
  "wry and controlled, quietly bitter".

"pov_and_tense"
  e.g. "first person, past tense".

Do not include any character names, place names, or specific incidents \
from the source story anywhere in your output. Return ONLY the JSON \
object -- no markdown fences, no preamble.

STORY:
{story_text}
"""

GENERATION_PROMPT = """Write an original short story for a narrated \
short-form video. Follow the structural template below exactly in its \
beat sequence, emotional arc, narrative voice, and central secret/stakes \
mechanism -- but with entirely new specific content.

TEMPLATE:
{template_json}

Use these specifics -- do not deviate from them:
- Setting: {setting}
- Relationship between the TWO CENTRAL people: {relationship_dynamic}
- The hidden truth, scheme, or high-stakes element driving the story \
should center on: {incident_seed}
- Opening technique for the very first sentence: {hook_style}
- Narrative voice: {narrative_voice}
{roles_instruction}
{names_instruction}
{occupation_instruction}

Rules:
- Do not reuse any names, incidents, dialogue, or specific phrasing from \
any existing published story.
- Target length: 260-300 words. This is a hard requirement.
- Match the template's beat order exactly. Skip any beat whose template \
value is an empty string.
- End on a natural hook for a comment or reaction, matching the final \
beat of the emotional arc.
- Return ONLY the story text. No title, no preamble, no markdown, no \
headings, no quotation marks around the whole thing.
"""


def _roles_instruction(template: dict) -> str:
    """
    The sampled relationship_dynamic covers two people. If the template
    needs more, tell the model to invent appropriate supporting roles
    rather than leaving it to guess or dropping them.
    """
    roles = template.get("character_roles") or []
    if not isinstance(roles, list) or len(roles) <= 2:
        return ""
    extra = len(roles) - 2
    return (
        f"- The template has {len(roles)} significant roles. Beyond the two "
        f"central people above, include {extra} additional supporting "
        f"character(s) filling the remaining roles listed in the template, "
        f"inventing appropriate relationships for them that fit the setting."
    )


def _names_instruction(template: dict, combo: dict) -> str:
    """
    Match the source's naming convention. Forcing names into a story
    whose source used none (and vice versa) is one of the clearest ways
    generated output stops sounding like the proven material.
    """
    uses_names = template.get("uses_character_names")
    # Default to True only if the field is genuinely absent (older
    # templates). An explicit False must be respected.
    if uses_names is False:
        return (
            "- Do NOT use any proper names for any character. Refer to "
            "everyone only by their relationship to the narrator (\"my "
            "sister\", \"my boss\", \"the driver\"), the way the template's "
            "source did."
        )
    names = combo.get("character_names") or []
    if len(names) >= 2:
        name_str = f"{names[0]} and {names[1]}"
    elif names:
        name_str = names[0]
    else:
        return ""
    return (
        f"- Give characters ordinary American first names. Use {name_str} "
        f"for the two central people; pick other common American first "
        f"names for any supporting characters."
    )


def _occupation_instruction(combo: dict) -> str:
    occ = combo.get("occupation")
    if not occ:
        return ""
    return (
        f"- If it fits naturally, one character's occupation can be "
        f"mentioned as minor supporting detail: {occ}. Do not build the "
        f"plot around the job -- the relationship dynamic and the hidden "
        f"secret drive the story."
    )


def build_extraction_prompt(story_text: str) -> str:
    return EXTRACTION_PROMPT.format(story_text=story_text)


def build_prompt(template: dict, combo: dict) -> str:
    # Prefer the SOURCE story's own extracted hook technique and voice --
    # those are what's actually proven to work for this template. The
    # sampled hook_style is only a fallback for templates extracted
    # before that field existed, or where extraction returned it empty.
    hook_style = template.get("hook_style") or combo.get("hook_style", "")
    narrative_voice = template.get("narrative_voice") or "conversational, first person"

    return GENERATION_PROMPT.format(
        template_json=json.dumps(template),
        setting=combo["setting"],
        relationship_dynamic=combo["relationship_dynamic"],
        incident_seed=combo["incident_seed"],
        hook_style=hook_style,
        narrative_voice=narrative_voice,
        roles_instruction=_roles_instruction(template),
        names_instruction=_names_instruction(template, combo),
        occupation_instruction=_occupation_instruction(combo),
    )
