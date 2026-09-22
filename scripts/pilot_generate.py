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
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from diversity_sampler import DiversitySampler  # noqa: E402
from prompts import build_prompt  # noqa: E402
from validation import validate_story, word_count  # noqa: E402
from extract_templates import (  # noqa: E402
    FatalExtractionError, _is_fatal, ensure_trailing_newline,
)

from google import genai


def generate_one(client, model, prompt, temperature=1.0, retries=3):
    """
    Same broad-catch rationale as extract_one: beyond APIError, real runs
    hit missing/None resp.text on blocked responses, auth expiry, and
    network blips. Fatal config errors fail fast instead of burning
    retries on every row.
    """
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": temperature},
            )
            text = getattr(resp, "text", None)
            if not text or not text.strip():
                raise ValueError("empty or missing response text")
            return text.strip()
        except Exception as e:  # noqa: BLE001
            if _is_fatal(e):
                raise FatalExtractionError(
                    f"Configuration error, stopping run: {type(e).__name__}: {e}"
                ) from e
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"failed after {retries} attempts: {type(last_err).__name__}: {last_err}"
    )


def load_done_story_ids(out_path: Path) -> set:
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "id" in row:
                done.add(row["id"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--variants-per-template", type=int, default=15)
    parser.add_argument("--max-templates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-words", type=int, default=200)
    parser.add_argument("--max-words", type=int, default=350)
    parser.add_argument("--keep-invalid", action="store_true",
                        help="Write stories that fail validation too, flagged "
                             "with a 'problems' field, instead of dropping "
                             "them. Useful for diagnosing a bad prompt.")
    args = parser.parse_args()

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    sampler = DiversitySampler(seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_trailing_newline(out_path)
    done_ids = load_done_story_ids(out_path)
    if done_ids:
        print(f"Resuming -- {len(done_ids)} stories already generated.")

    stats = {"bad_template_line": 0, "skipped_done": 0, "api_failed": 0,
             "invalid_dropped": 0, "invalid_kept": 0, "written": 0}

    with open(args.templates, "r", encoding="utf-8") as f_templates, \
         open(out_path, "a", encoding="utf-8") as f_out:

        t_idx = 0
        for line_no, line in enumerate(f_templates, 1):
            if t_idx >= args.max_templates:
                break
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                template = row["template"]
                template_id = row["id"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"SKIPPED template line {line_no}: malformed ({e})")
                stats["bad_template_line"] += 1
                continue

            t_idx += 1

            for v in range(args.variants_per_template):
                story_id = f"{template_id}_v{v}"
                if story_id in done_ids:
                    stats["skipped_done"] += 1
                    continue

                combo = sampler.sample_combo()
                prompt = build_prompt(template, combo)

                try:
                    story_text = generate_one(
                        client, args.model, prompt, temperature=args.temperature
                    )
                except FatalExtractionError as e:
                    print(f"\nFATAL: {e}")
                    print("Progress so far is saved.")
                    return
                except RuntimeError as e:
                    print(f"SKIPPED {story_id}: {e}")
                    stats["api_failed"] += 1
                    continue

                problems = validate_story(
                    story_text,
                    min_words=args.min_words,
                    max_words=args.max_words,
                )
                if problems and not args.keep_invalid:
                    stats["invalid_dropped"] += 1
                    print(f"DROPPED {story_id}: {problems}")
                    continue

                out_row = {
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
                for extra_key in ("title", "channel", "views"):
                    if extra_key in row:
                        out_row[f"source_{extra_key}"] = row[extra_key]
                if problems:
                    out_row["problems"] = problems
                    stats["invalid_kept"] += 1

                f_out.write(json.dumps(out_row) + "\n")
                f_out.flush()
                done_ids.add(story_id)
                stats["written"] += 1

                if stats["written"] % 25 == 0:
                    print(f"Generated {stats['written']} stories...")

    print("\n--- Pilot summary ---")
    for k, v in stats.items():
        print(f"{k:>19}: {v}")
    total_attempted = stats["written"] + stats["invalid_dropped"] + stats["api_failed"]
    if total_attempted:
        print(f"\nClean yield: {stats['written'] / total_attempted * 100:.1f}%")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
