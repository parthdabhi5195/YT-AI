"""
Stage 2 -- Pilot generation.

Takes the templates from extract_templates.py and generates a SMALL batch
of new stories from them (hundreds to a couple thousand), each with a
different diversity combo. Read the output before you go anywhere near
build_batch_requests.py / the full production run.

Usage:
    python scripts/pilot_generate.py \
        --templates data/templates/templates.jsonl \
        --output data/pilot_output/pilot.jsonl \
        --project YOUR_PROJECT_ID \
        --variants-per-template 15 \
        --max-templates 100

RESUME: safe to re-run. Story IDs already in the output are skipped, so
you won't get duplicate rows.

DETERMINISM: --seed makes the SAMPLER reproducible (same combos in the
same order). It does not make Gemini's text output reproducible --
--temperature controls that, and even at 0 the model is not guaranteed
byte-identical across runs.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from diversity_sampler import DiversitySampler  # noqa: E402
from gemini_io import FatalConfigError, call_with_retry, generate_text  # noqa: E402
from pipeline_io import (  # noqa: E402
    append_row, # write to file and flush to disk immediately
    carry_metadata, # copy over source metadata fields to the output row
    ensure_trailing_newline, # edge case logic for resuming runs
    iter_jsonl, # iterates through JSONL files line by line
    load_ids, # loads IDs from output file to skip already processed rows (resume logic)
    print_stats, # summary blocks
)
from prompts import build_prompt, names_needed  # noqa: E402
from validation import validate_story, word_count  # noqa: E402

from google import genai

# Generating 1 story per prrompt with retry logic.
def generate_one(client, model, prompt, temperature=1.0, max_output_tokens=500, retries=3):
    return call_with_retry(
        lambda: generate_text(client, model, prompt, {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens
            }),
        retries=retries,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True) # template file path
    parser.add_argument("--output", required=True) # pilot output file path
    parser.add_argument("--project", required=True) # GCP project ID
    parser.add_argument("--location", default="us-central1") # GCP location for Gemini API to ensure batch request pricing rates
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--variants-per-template", type=int, default=15)
    parser.add_argument("--max-templates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42) # deterministic sampling of diversity combos. Useful for checking diversity
    parser.add_argument("--temperature", type=float, default=1.0) # generation temperature for Gemini API. 0 is deterministic, 1 is default, >1 is more random
    parser.add_argument("--min-words", type=int, default=200)
    parser.add_argument("--max-words", type=int, default=340)
    parser.add_argument("--keep-invalid", action="store_true",
                        help="Write stories that fail validation too, flagged "
                             "with a 'problems' field, instead of dropping "
                             "them. Useful for diagnosing a bad prompt.")
    parser.add_argument("--max-output-tokens", type=int, default=500,
                        help="Must match build_batch_requests.py's default so "
                         "the pilot can reproduce the batch path's truncation.")
    args = parser.parse_args()

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    sampler = DiversitySampler(seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_trailing_newline(out_path)
    done_ids = load_ids(out_path)
    if done_ids:
        print(f"Resuming -- {len(done_ids)} stories already generated.")

    stats = {
        "bad_template_line": 0, 
        "skipped_done": 0, 
        "api_failed": 0,
        "invalid_dropped": 0, 
        "invalid_kept": 0, 
        "written": 0
    }

    def bad_line(line_no, line, exc):
        print(f"SKIPPED template line {line_no}: malformed ({exc})")
        stats["bad_template_line"] += 1

    with open(out_path, "a", encoding="utf-8") as f_out: # append mode to resume runs and not overwrite existing output
        t_idx = 0
        for line_no, row in iter_jsonl(args.templates, on_bad=bad_line):
            if t_idx >= args.max_templates:
                break

            try:
                # extract template and template_id from the source template row
                template = row["template"]
                template_id = row["id"]
            except (KeyError, TypeError) as e:
                print(f"SKIPPED template line {line_no}: malformed ({e})")
                stats["bad_template_line"] += 1
                continue

            t_idx += 1

            for v in range(args.variants_per_template):
                story_id = f"{template_id}_v{v}" # for each variant, it builds a unique story_id by appending the variant number to the template_id
                if story_id in done_ids: # Preventing a duplicate output from a rerun
                    stats["skipped_done"] += 1
                    continue

                # generate a new diversity combo for the current template variant    
                combo = sampler.sample_combo(names_needed(template))
                prompt = build_prompt(template, combo)

                try:
                    story_text = generate_one(
                        client, 
                        args.model, 
                        prompt, 
                        temperature=args.temperature, 
                        max_output_tokens=args.max_output_tokens,
                    )
                except FatalConfigError as e: # stops whole run
                    print(f"\nFATAL: {e}")
                    print("Progress so far is saved.")
                    print_stats("Pilot summary", stats)
                    return
                except RuntimeError as e: # continues running
                    print(f"SKIPPED {story_id}: {e}")
                    stats["api_failed"] += 1
                    continue

                # validate after recieving story from Gemini API    
                problems = validate_story(
                    story_text,
                    min_words=args.min_words,
                    max_words=args.max_words,
                )
                if problems and not args.keep_invalid: # drop story if --keep-invalid is not passed
                    stats["invalid_dropped"] += 1
                    print(f"DROPPED {story_id}: {problems}")
                    continue

                out_row = { # building the output row
                    "id": story_id,
                    "source_template_id": template_id,
                    **combo,
                    "story_text": story_text,
                    "word_count": word_count(story_text),
                    "model_used": args.model,
                    "temperature": args.temperature,
                }
                # Carry source-performance metadata through so you can
                # later weight or filter by how the source video did.
                carry_metadata(row, out_row, prefix="source_")
                if problems: # if --keep-invalid is passed, it will write the story to the output file with a "problems" field
                    out_row["problems"] = problems
                    stats["invalid_kept"] += 1

                append_row(f_out, out_row)
                done_ids.add(story_id)
                stats["written"] += 1

                if stats["written"] % 25 == 0:
                    print(f"Generated {stats['written']} stories...")

    print_stats("Pilot summary", stats)
    total_attempted = stats["written"] + stats["invalid_dropped"] + stats["api_failed"]
    if total_attempted:
        print(f"\nClean yield: {stats['written'] / total_attempted * 100:.1f}%")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
