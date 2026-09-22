"""
Shared validation used by extract_templates.py, pilot_generate.py, and
clean_dataset.py.

The prompts ASK the model for a schema, a word count, and no markdown.
Those are requests, not guarantees. This module turns them into checks
that actually run, so bad rows get caught at the point they're produced
instead of silently ending up in training data.

No third-party dependencies -- importable anywhere in the pipeline.
"""
import re # regular expressions 
import sys
from pathlib import Path

# Self-contained path setup, so this module (and prompts.py through it)
# can be imported and unit-tested directly, not only as a side effect of
# a stage script having already appended scripts/ to sys.path.
sys.path.append(str(Path(__file__).parent))
from prompts import REQUIRED_TEMPLATE_FIELDS, BEAT_KEYS  # noqa: E402


def validate_template(template) -> list:
    """
    Check an extracted template against the schema the extraction prompt
    asked for. Returns a list of problem strings -- empty list means OK.
    """
    problems = []

    if not isinstance(template, dict):
        return ["template is not a JSON object"]

    for field in REQUIRED_TEMPLATE_FIELDS:
        if field not in template:
            problems.append(f"missing field: {field}")

    roles = template.get("character_roles")
    if roles is not None and not isinstance(roles, list): # character_roles is optional, but if present it must be a list
        problems.append("character_roles is not a list")
    elif isinstance(roles, list) and len(roles) == 0: # if character_roles is present and is a list, it must not be empty
        problems.append("character_roles is empty")

    uses_names = template.get("uses_character_names")
    if uses_names is not None and not isinstance(uses_names, bool):
        problems.append("uses_character_names is not a boolean")

    beats = template.get("beats")
    if beats is not None:
        if not isinstance(beats, dict):
            problems.append("beats is not an object")
        else:
            for key in BEAT_KEYS:
                if key not in beats:
                    problems.append(f"beats missing key: {key}") # There are missing keys
            extra = set(beats.keys()) - set(BEAT_KEYS)
            if extra:
                problems.append(f"beats has unexpected keys: {sorted(extra)}") # There are extra keys
            filled = [k for k in BEAT_KEYS if str(beats.get(k, "")).strip()]
            if len(filled) < 3: # If less than 3 beats have content, then the extraction is likely too weak to be useful
                problems.append(
                    f"only {len(filled)} beats have content -- extraction "
                    f"likely failed"
                )

    for field in ("genre_trope", "central_secret_or_stakes", "emotional_arc",
                  "hook_style", "narrative_voice", "pov_and_tense"):
        val = template.get(field)
        if val is not None and (not isinstance(val, str) or not val.strip()):
            problems.append(f"{field} is empty or not a string")

    return problems


# Regex list of common signs the model returned something other than clean story prose.
_MARKDOWN_PATTERNS = [
    (re.compile(r"^\s*#{1,6}\s"), "starts with a markdown heading"),
    (re.compile(r"^\s*```"), "contains a code fence"),
    (re.compile(r"\*\*[^*]+\*\*"), "contains bold markdown"),
    # Single-asterisk italics around a word or phrase: *real*, *never said*.
    # The lookarounds require both asterisks to sit OUTSIDE a word, so
    # censored profanity (f***ing, sh**) is not mistaken for markdown, and
    # the doubled asterisks of **bold** are left to the pattern above.
    (re.compile(r"(?<![*\w])\*(?=[^\s*])[^*\n]{1,60}?(?<=[^\s*])\*(?![*\w])"),
     "contains italic markdown"),
    (re.compile(r"^\s*(Title|TITLE)\s*:", re.MULTILINE), "contains a Title: line"),
    (re.compile(r"^\s*(Story|STORY)\s*:", re.MULTILINE), "contains a Story: label"),
]


def validate_story(
    text,
    min_words: int = 200,
    max_words: int = 340,
    forbidden_names=None,
) -> list:
    """
    Check a generated story. Returns a list of problem strings -- empty
    list means OK.

    forbidden_names: optional iterable of names that must NOT appear
    (e.g. names from the source story, to catch direct copying).
    """
    problems = []

    if text is None:
        return ["no text returned"]
    if not isinstance(text, str):
        return ["text is not a string"]

    stripped = text.strip() # strip whitespace
    if not stripped:
        return ["text is empty"]

    wc = word_count(stripped) # word count to see if within range.
    if wc < min_words:
        problems.append(f"too short: {wc} words (min {min_words})")
    elif wc > max_words:
        problems.append(f"too long: {wc} words (max {max_words})")

    for pattern, description in _MARKDOWN_PATTERNS:
        if pattern.search(stripped):
            problems.append(description)

    # A story wrapped entirely in quotes usually means the model treated the whole thing as a quoted block.
    if stripped.startswith('"') and stripped.endswith('"') and stripped.count('"') == 2:
        problems.append("entire story is wrapped in quotation marks")

    if forbidden_names: # Check if names are being copied from the source story.
        lowered = stripped.lower()
        hits = [n for n in forbidden_names
                if n and re.search(rf"\b{re.escape(n.lower())}\b", lowered)]
        if hits:
            problems.append(f"reuses source name(s): {sorted(set(hits))}")

    return problems


def word_count(text) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.strip().split())
