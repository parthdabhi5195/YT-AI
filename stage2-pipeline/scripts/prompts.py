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

# The same contract as the two lists above, in the form Gemini enforces
# at generation time (response_json_schema). The required-field lists are
# derived, not restated, so the prompt, the validator, and this schema
# cannot drift apart.
#
# Shape only. What each field MEANS stays in EXTRACTION_PROMPT, and the
# checks a JSON schema can't express -- notably a beats object that parses
# fine but is mostly empty -- stay in validate_template.
#
# Deliberately limited to core keywords. This dict is sent to the API
# verbatim (the SDK does not validate or rewrite it), so an unsupported
# keyword would fail every request rather than one row. That's why there's
# no additionalProperties: extra beat keys are caught by validate_template
# instead.
TEMPLATE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "genre_trope": {"type": "string"},
        "character_roles": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "uses_character_names": {"type": "boolean"},
        "central_secret_or_stakes": {"type": "string"},
        "beats": {
            "type": "object",
            # Empty string is a legal value here -- see the prompt. Only
            # the keys are required, not content in every one of them.
            "properties": {key: {"type": "string"} for key in BEAT_KEYS},
            "required": list(BEAT_KEYS),
        },
        "emotional_arc": {"type": "string"},
        "hook_style": {"type": "string"},
        "narrative_voice": {"type": "string"},
        "pov_and_tense": {"type": "string"},
    },
    "required": list(REQUIRED_TEMPLATE_FIELDS),
}

EXTRACTION_PROMPT = """You are analyzing the STRUCTURE of a short story, 
not its content. Return a single JSON object with exactly the fields 
below and nothing else.

"genre_trope"
  A short label, e.g. "family favoritism", "hidden paternity",
  "custody scheme", "financial betrayal".

"character_roles"
  A list of the significant people, described ONLY by their role in the
  conflict and their relationship to the narrator -- never by name, and
  never by a job title that exists only because of this story's setting.
  Example:
  ["narrator (adult daughter)", "antagonist (mother)", "ally (grandmother)"]

  Relationship words are exactly right: daughter, stepfather, in-law,
  ex-partner, neighbor, best friend, someone in a position of trust.

  BAD   ["narrator (student)", "antagonist (bus driver)",
         "authority (vice principal)", "authority (security guard)"]
  GOOD  ["narrator (child)", "antagonist (adult in a position of trust)",
         "authority figure", "bystander who sides against the narrator"]

  Keep an occupation ONLY when the conflict is impossible without it --
  when the person's professional duty is itself the betrayal.

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
  at that point. If the story genuinely has no such beat, use an empty
  string "" for that key. Do not add or rename keys.

  THE TRANSPLANT TEST -- apply this to every beat before you write it.
  This template will be reused to generate a NEW story in a completely
  different setting: a hospital, a cruise ship, a farm, a courtroom. A
  beat that names the original's setting, institution, vehicle, job
  title or objects becomes a contradiction the moment it is reused.
  Write each beat so it stays true no matter where the story is moved.

  Name the STRUCTURAL FUNCTION, not the furniture:

  BAD   "The protagonist is brought to the principal's office and
         confronted with evidence of their wrongdoing."
  GOOD  "The protagonist is summoned by an authority figure and
         confronted with evidence they cannot immediately explain."

  BAD   "A seemingly ordinary bus ride begins with an ominous statement
         from the driver."
  GOOD  "A routine journey turns threatening when the person in control
         refuses to follow the expected course."

  BAD   "The narrator discovers the partner's detailed diary revealing
         a fabricated history."
  GOOD  "The narrator finds a private record proving the other person
         invented their shared history."

  Never allowed in a beat: proper names, place names, institution names
  (school, hospital, courthouse), vehicles, job titles, or any object
  specific enough to belong only to this story. Reach instead for the
  abstract equivalent -- "an authority figure", "a private record",
  "written proof", "a shared obligation", "someone in a position of
  trust".

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

BEFORE YOU RETURN: re-read every beat and every character role. For each
one, ask whether it would have to change to move this story to a
completely different setting. If it would, rewrite it more abstractly.

Do not include any character names, place names, or specific incidents
from the source story anywhere in your output.

STORY:
{story_text}
"""

GENERATION_PROMPT = """Write an original short story for a narrated 
short-form video. Preserve the template's abstract conflict pattern, 
stakes, beat sequence, emotional arc, narrative voice, and POV and tense, 
but create entirely new concrete content. The result must feel like a 
fresh story built from the same underlying structure, not a disguised 
retelling. 


TEMPLATE:
{template_json}

Use these specifics -- do not deviate from them: 
- Setting: {setting} 
- Relationship between the TWO CENTRAL people: {relationship_dynamic} 
- Abstract conflict pattern and stakes to preserve: 
{central_secret_or_stakes} 
- New concrete situation that must express that same conflict pattern: 
{incident_seed} 
- Opening technique for the very first sentence: {hook_style} 
- Narrative voice: {narrative_voice} 
{roles_instruction} 
{names_instruction} 
{occupation_instruction} 

Rules:
- Do not reuse any names, incidents, dialogue, or specific phrasing from 
any existing published story.
- The new concrete situation must be compatible with the abstract conflict 
pattern above: preserve the underlying relationship tension, deception or 
problem, and type of stakes.
- Reveal information in a deliberate order so
the audience understands the situation before the final payoff.
- Match the template's beat order exactly. Skip any beat whose template
value is an empty string.
- End with a satisfying resolution that matches the final beat of the
emotional arc.
- Return ONLY the story text. No title, no preamble, no markdown, no
headings, no quotation marks around the whole thing.

PACING AND LENGTH CONSTRAINTS (CRITICAL):
- Target length: 230-260 words. Strict limit: NEVER exceed 280 words.
- Sentence budget: write 12 to 15 sentences total, broken into 3 to 4
short paragraphs.
- Beat budget: execute each template beat in 1 to 2 economical sentences.
- Style: keep sentences economical and cut filler, but stay within the
template's narrative voice. Tighten the voice; do not replace it.
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
    if not names:
        return ""
    # Every named character gets a name from the sampled pool, and the model
    # is told those are the only names allowed. Asked to "pick other common
    # names" for supporting characters, it reused the same handful across
    # the whole dataset. Deliberately no example surnames here -- naming one
    # in the prompt would prime the model to use it.
    central, supporting = names[:2], names[2:]
    text = f"- Use {' and '.join(central)} for the two central people."
    if supporting:
        text += (
            f" Use {', '.join(supporting)} for supporting characters, in any "
            f"order, only as many as the story needs."
        )
    text += (
        " These are the ONLY names allowed in the story. Do not invent any "
        "other first names or surnames, and do not attach a title (Mr., "
        "Mrs., Ms., Dr., Officer) to an invented surname. Refer to anyone "
        "without a name from this list by their role instead -- \"the "
        "officer\", \"my teacher\", \"her lawyer\"."
    )
    return text


MAX_NAMES_PER_STORY = 6


def names_needed(template: dict) -> int:
    """
    How many character names to sample for a template: one per role, so no
    character is left for the model to name itself. Clamped to 2..6 -- two
    covers the central pair, and six covers the largest casts extraction
    returns without handing the model a list it will mostly ignore.
    """
    roles = template.get("character_roles")
    count = len(roles) if isinstance(roles, list) else 2
    return max(2, min(MAX_NAMES_PER_STORY, count))


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
    central_secret_or_stakes = template.get("central_secret_or_stakes")

    return GENERATION_PROMPT.format(
        template_json=json.dumps(template),
        setting=combo["setting"],
        relationship_dynamic=combo["relationship_dynamic"],
        central_secret_or_stakes=central_secret_or_stakes,
        incident_seed=combo["incident_seed"],
        hook_style=hook_style,
        narrative_voice=narrative_voice,
        roles_instruction=_roles_instruction(template),
        names_instruction=_names_instruction(template, combo),
        occupation_instruction=_occupation_instruction(combo),
    )
