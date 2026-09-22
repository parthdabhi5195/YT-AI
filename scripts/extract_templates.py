"""
Stage 1 -- Structural extraction.

Reads your proven source scripts and asks Gemini to output ONLY an
abstract structural template for each one (trope, character roles, plot
beats, emotional arc, hook style, narrative voice) -- with no character
names, settings, or verbatim phrases from the source.

Usage:
    python scripts/extract_templates.py \
        --input data/source_scripts/source_scripts.jsonl \
        --output data/templates/templates.jsonl \
        --project YOUR_PROJECT_ID \
        --limit 50    # start small; omit --limit to run all of them

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
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from prompts import build_extraction_prompt  # noqa: E402
from validation import validate_template  # noqa: E402

from google import genai


# Substrings that indicate a problem no amount of retrying will fix --
# bad credentials, wrong project, model name that doesn't exist. Without
# this, a misconfigured run burns 3 retries x 6,500 rows before you find
# out anything is wrong.
FATAL_ERROR_MARKERS = (
    "permission_denied", "permission denied", "unauthenticated",
    "invalid_argument", "not_found", "was not found",
    "api key not valid", "could not automatically determine credentials",
    "billing", "has not been used in project", "is disabled",
)


class FatalExtractionError(RuntimeError):
    """Configuration-level failure -- stop the whole run, don't retry."""


def _is_fatal(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in msg for marker in FATAL_ERROR_MARKERS)


def _strip_code_fences(text: str) -> str:
    """Models sometimes wrap JSON in ``` fences despite being told not to."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_one(client, model: str, story_text: str, retries: int = 3) -> dict:
    """
    Returns a validated template dict, or raises.

    Catches broadly on purpose. Beyond google.genai.errors.APIError, real
    runs hit: AttributeError/TypeError when resp.text is missing or None
    (blocked/empty response), json.JSONDecodeError (a ValueError
    subclass) on malformed JSON, google.auth errors when credentials
    expire mid-run, and assorted httpx/socket errors on network blips.
    Enumerating those exactly across SDK versions is fragile, so the loop
    retries anything transient and fails fast on anything fatal.
    """
    prompt = build_extraction_prompt(story_text)
    last_err = None

    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)

            # resp.text can be absent or None when a response is blocked
            # or empty. Don't let that raise an unhandled AttributeError.
            text = getattr(resp, "text", None)
            if not text or not text.strip():
                raise ValueError("empty or missing response text")

            template = json.loads(_strip_code_fences(text))

            problems = validate_template(template)
            if problems:
                raise ValueError(f"schema validation failed: {problems}")

            return template

        except Exception as e:  # noqa: BLE001 -- see docstring
            if _is_fatal(e):
                raise FatalExtractionError(
                    f"Configuration error, stopping run: {type(e).__name__}: {e}"
                ) from e
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s

    raise RuntimeError(
        f"failed after {retries} attempts: {type(last_err).__name__}: {last_err}"
    )


def ensure_trailing_newline(path: Path) -> None:
    """
    If a previous run was killed mid-write, the file can end without a
    newline. Appending then concatenates the new row onto the partial one,
    corrupting BOTH -- the partial line AND the row being written, which
    silently disappears. Repair the boundary before appending.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            with open(path, "a", encoding="utf-8") as fa:
                fa.write("\n")
            print("  note: repaired a missing newline at the end of the "
                  "output file (previous run was interrupted mid-write)")


def load_done_ids(out_path: Path) -> set:
    """
    Read IDs already extracted. Tolerates a truncated final line, which
    happens if a previous run was killed mid-write -- that shouldn't
    block resuming.
    """
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  warning: output line {line_no} is not valid JSON "
                      f"(likely a truncated final line) -- ignoring it")
                continue
            if isinstance(row, dict) and "id" in row:
                done.add(row["id"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", required=True, help="Your GCP project ID")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N successful NEW templates")
    args = parser.parse_args()

    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ensure_trailing_newline(out_path)
    resumed_ids = load_done_ids(out_path)
    seen_ids = set(resumed_ids)  # grows as this run writes new rows
    if resumed_ids:
        print(f"Resuming -- {len(resumed_ids)} templates already extracted.")

    stats = {"read": 0, "skipped_done": 0, "skipped_dupe_input": 0,
             "bad_input_line": 0, "failed": 0, "written": 0}

    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "a", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, 1):
            if args.limit and stats["written"] >= args.limit:
                break

            line = line.strip()
            if not line:
                continue
            stats["read"] += 1

            try:
                row = json.loads(line)
                row_id = row["id"]
                story_text = row["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"SKIPPED input line {line_no}: malformed ({e})")
                stats["bad_input_line"] += 1
                continue

            if row_id in seen_ids:
                # Distinguish "already done in a previous run" from "this
                # ID appears twice in the input file" -- both are skipped,
                # but a within-run duplicate means your source data has a
                # problem worth knowing about.
                if row_id in resumed_ids:
                    stats["skipped_done"] += 1
                else:
                    stats["skipped_dupe_input"] += 1
                    print(f"  warning: input line {line_no} repeats id "
                          f"{row_id!r} -- skipping the duplicate")
                continue

            try:
                template = extract_one(client, args.model, story_text)
            except FatalExtractionError as e:
                print(f"\nFATAL: {e}")
                print("Fix the configuration and re-run -- progress so far is saved.")
                break
            except RuntimeError as e:
                print(f"SKIPPED {row_id}: {e}")
                stats["failed"] += 1
                continue

            out_row = {"id": row_id, "template": template}
            for extra_key in ("title", "channel", "views"):
                if extra_key in row:
                    out_row[extra_key] = row[extra_key]

            f_out.write(json.dumps(out_row) + "\n")
            f_out.flush()
            seen_ids.add(row_id)  # prevents reprocessing a duplicate input ID
            stats["written"] += 1

            if stats["written"] % 25 == 0:
                print(f"Extracted {stats['written']} templates...")

    print("\n--- Extraction summary ---")
    for k, v in stats.items():
        print(f"{k:>19}: {v}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
