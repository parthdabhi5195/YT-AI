"""
Stage 1 -- Structural extraction.

Reads your proven source scripts and asks Gemini to output ONLY an abstract structural template for each one: 
            trope, 
            character roles, 
            plot
            beats, 
            emotional arc, 
            hook style, 
            narrative voice 
-- with no character names, settings, or verbatim phrases from the source.

Usage:
    python scripts/extract_templates.py \
        --input data/source_scripts/source_scripts.jsonl \
        --output data/templates/templates.jsonl \
        --project YOUR_PROJECT_ID \
        --limit 50    # start small; omit --limit to run all of them

Structure:
    source_scripts.jsonl
            |
            v
    extract_templates.py
            |
            v
    templates.jsonl

Each line of the input JSONL must have "id" and "text". Extra fields
(title, channel, views) are carried through to the output.

RESUME: safe to stop and re-run. IDs already present in the output file
are skipped.

BAD TEMPLATES: output is append-only, so an ID that was already written
is considered done even if you later decide its template is poor. To
redo one, delete its line from templates.jsonl and re-run. Schema
validation (below) means malformed templates are never written in the
first place, so this should be rare.

--limit counts SUCCESSFUL NEW WRITES, not input lines read. It skips
past already-done IDs and failed extractions until it has written N new
templates, which is what you want for a test run.
"""
import argparse # handles command-line arguments
import json # parses JSON data
import sys # importing local directory
from pathlib import Path # handles output paths

sys.path.append(str(Path(__file__).parent))

# importing Gemini-related helpers.
from gemini_io import (  # noqa: E402
    FatalConfigError, # detects fatal errors in API calls
    call_with_retry, # handles retry logic
    generate_text, # handles Gemini API calls and response parsing
)

from pipeline_io import (  # noqa: E402
    append_row, # write to file and flush to disk immediately
    carry_metadata, # copy over source metadata fields to the output row
    ensure_trailing_newline,  # edge case logic for resuming runs
    iter_jsonl, # iterates through JSONL files line by line
    load_ids, # loads IDs from output file to skip already processed rows (resume logic)
    print_stats, # summary blocks
)
from prompts import TEMPLATE_JSON_SCHEMA, build_extraction_prompt  # noqa: E402
from validation import validate_template  # noqa: E402

from google import genai

# Makes the model return raw JSON in the template's shape, instead of
# asking for it in the prompt and cleaning up afterward -- so there are no
# markdown fences to strip and no malformed JSON to parse around.
#
# This covers STRUCTURE only. validate_template still runs on the parsed
# result for the things the schema can't state: that enough beats actually
# have content, and that the string fields aren't blank.
EXTRACTION_CONFIG = {
    "response_mime_type": "application/json",
    "response_json_schema": TEMPLATE_JSON_SCHEMA,
}


def extract_one(client, model, story_text, retries=3, failures=None):
    """
    Returns a validated template dict, or raises exception.

    failures: optional list. Every attempt that parsed as JSON but failed
    validation is appended to it as {"template": ..., "problems": [...]},
    so the caller can keep the rejected output for diagnosis instead of
    only learning that three attempts failed.
    """
    prompt = build_extraction_prompt(story_text)

    def attempt():
        template = json.loads(
            generate_text(client, model, prompt, EXTRACTION_CONFIG)
        )
        problems = validate_template(template)
        if problems: # if list is not empty
            if failures is not None:
                failures.append({"template": template, "problems": problems})
            raise ValueError(f"schema validation failed: {problems}")
        return template

    return call_with_retry(attempt, retries=retries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", required=True, help="Your GCP project ID")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N successful NEW templates")
    parser.add_argument("--keep-invalid", action="store_true",
                        help="Also write templates that failed validation, with "
                             "a 'problems' field, to <output>.invalid.jsonl. They "
                             "go to a SEPARATE file on purpose -- a bad template "
                             "would otherwise seed hundreds of bad stories "
                             "downstream. Useful for diagnosing extraction.")
    args = parser.parse_args()

    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path = out_path.with_suffix(out_path.suffix + ".invalid.jsonl")

    ensure_trailing_newline(out_path)
    done_ids = load_ids(out_path)       # from previous runs; never mutated
    written_ids = set()                 # written by THIS run
    if done_ids:
        print(f"Resuming -- {len(done_ids)} templates already extracted.")

    stats = {"read": 0, "skipped_done": 0, "skipped_dupe_input": 0,
             "bad_input_line": 0, "failed": 0, "invalid_kept": 0, "written": 0}

    def bad_line(line_no, line, exc):
        print(f"SKIPPED input line {line_no}: malformed ({exc})")
        stats["bad_input_line"] += 1

    f_invalid = open(invalid_path, "a", encoding="utf-8") if args.keep_invalid else None

    with open(out_path, "a", encoding="utf-8") as f_out: # append mode to preserve previous outputs
        for line_no, row in iter_jsonl(args.input, on_bad=bad_line):
            if args.limit and stats["written"] >= args.limit:
                break
            stats["read"] += 1

            try:
                row_id = row["id"]
                story_text = row["text"]
            except (KeyError, TypeError) as e:
                print(f"SKIPPED input line {line_no}: malformed ({e})")
                stats["bad_input_line"] += 1
                continue

            # A repeat within this run means the source data has a problem
            # worth knowing about; an ID from a previous run is just done.
            if row_id in written_ids:
                stats["skipped_dupe_input"] += 1
                print(f"  warning: input line {line_no} repeats id "
                      f"{row_id!r} -- skipping the duplicate")
                continue
            if row_id in done_ids:
                stats["skipped_done"] += 1
                continue

            failures = [] if args.keep_invalid else None
            try:
                template = extract_one(client, args.model, story_text,
                                       failures=failures)
            except FatalConfigError as e:
                print(f"\nFATAL: {e}")
                print("Fix the configuration and re-run -- progress so far is saved.")
                break
            except RuntimeError as e:
                print(f"SKIPPED {row_id}: {e}")
                stats["failed"] += 1
                # The row is NOT added to written_ids, so a later re-run
                # retries it. This file is diagnostic only.
                if f_invalid is not None and failures:
                    last = failures[-1]
                    append_row(f_invalid, carry_metadata(row, {
                        "id": row_id,
                        "template": last["template"],
                        "problems": last["problems"],
                        "attempts": len(failures),
                    }))
                    stats["invalid_kept"] += 1
                continue

            append_row(f_out, carry_metadata(row, {"id": row_id, "template": template}))
            written_ids.add(row_id)
            stats["written"] += 1

            if stats["written"] % 25 == 0:
                print(f"Extracted {stats['written']} templates...")

    if f_invalid is not None:
        f_invalid.close()

    print_stats("Extraction summary", stats)
    print(f"\nOutput: {out_path}")
    if args.keep_invalid and stats["invalid_kept"]:
        print(f"Rejected templates: {invalid_path}")


if __name__ == "__main__":
    main()
