"""
Stage 3 -- Build batch request files for the full production run.

Turns your templates + diversity injector into JSONL request files, split
into chunks so you can run (and check the cost of) one chunk at a time.

=== SCHEMA STATUS: PARTIALLY VERIFIED ===

Verified from Google's docs:
  * For native Gemini models on Vertex, each input line is a JSON object
    whose "request" field follows the GenerateContentRequest structure.
    That is the format built below (see batch_schema.py).
  * A DIFFERENT format exists -- OpenAI-shaped, with top-level
    "custom_id"/"method"/"url"/"body" -- but that one is for
    Model-as-a-Service partner models (Claude, Llama on Vertex), which
    your $300 free-trial credit does NOT cover. Do not mix them up.
  * Max 150,000 requests per batch job, so --chunk-size must stay under
    that. Default 50,000 is comfortably inside it.

NOT verified: whether native Gemini batch output preserves a top-level
"custom_id" field you send in. Because of that, this script writes TWO
join keys for every request:
  1. custom_id      -- used if the output echoes it back
  2. prompt_sha256  -- hash of the exact prompt text, which the output
                       DOES echo back inside the request object
clean_dataset.py tries custom_id first and falls back to prompt_sha256,
so the pipeline joins correctly either way. Do not remove the hash.

One 1-row batch job would settle which key survives and let one of the
two be deleted; until someone runs it, both stay.

Usage:
    python scripts/build_batch_requests.py \
        --templates data/templates/templates.jsonl \
        --output-dir data/batch_requests \
        --target-total 2000000 \
        --chunk-size 50000 \
        --seed 42
"""
import argparse # CLI parsing   
import json # JSON parsing
import sys # file/path creation
from itertools import islice # chunking
from pathlib import Path

# Create a path and add it to system path
sys.path.append(str(Path(__file__).parent))

from batch_schema import (  # noqa: E402
    MAX_REQUESTS_PER_JOB, # 150,000 max requests per batch job
    build_request_line, # request line builder per the GenerateContentRequest schema
    prompt_hash, # compute a stable hash for a prompt to use as id when custom_id is missing
)
from diversity_sampler import DiversitySampler  # noqa: E402
from pipeline_io import carry_metadata, iter_jsonl  # noqa: E402
from prompts import build_prompt, names_needed  # noqa: E402

# Takes entire collection and splits it into lists of size "size", yielding each list. The last list may be smaller than "size" but will never be empty.
def chunked(iterable, size):
    """Batches of `size`, never yielding an empty final batch."""
    it = iter(iterable)
    while batch := list(islice(it, size)): # if list is empty, break the loop
        yield batch # get current batch one by one

# Iterate through each template and generate specified amount of variants, yieliding a tuple
def iter_planned(templates, sampler, variants_per_template, target_total):
    """Yield (template_row, variant_index, combo) up to target_total."""
    planned = 0
    for t_row in templates:
        for v in range(variants_per_template):
            if planned >= target_total: # stop once target has been reached
                return
            yield t_row, v, sampler.sample_combo(names_needed(t_row["template"]))
            planned += 1


# scans templates line by line, and tolerate malformed lines and reject rows that don't have both id and template fields.
def load_templates(path):
    templates = []
    skipped = 0

    def bad_line(line_no, line, exc):
        nonlocal skipped
        print(f"  SKIPPED template line {line_no}: {exc}")
        skipped += 1

    for line_no, row in iter_jsonl(path, on_bad=bad_line):
        if "template" not in row or "id" not in row:
            print(f"  SKIPPED template line {line_no}: missing id/template")
            skipped += 1
            continue
        templates.append(row)
    return templates, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-total", type=int, default=2_000_000)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=500) # based on Gemini API and keep between 280-300 words
    parser.add_argument("--run-name", default="",
                        help="Prefix for custom_ids and chunk names, e.g. 'extra'. "
                             "Required for any run after the first: ids are "
                             "template_id + variant index, so a second run from "
                             "the same templates would otherwise reuse the first "
                             "run's ids and chunk names.")
    args = parser.parse_args()
    prefix = f"{args.run_name}_" if args.run_name else ""

    # Enforce the Gemini Enterprise Agent Platform batch ceiling.
    if args.chunk_size > MAX_REQUESTS_PER_JOB:
        raise SystemExit(
            f"--chunk-size {args.chunk_size:,} exceeds the {MAX_REQUESTS_PER_JOB:,} "
            f"requests-per-job limit. Lower it."
        )

    templates, skipped = load_templates(args.templates)
    if not templates:
        raise SystemExit("No usable templates -- run extract_templates.py first.")
    if skipped:
        print(f"  ({skipped} malformed template line(s) skipped)\n")

    variants_per_template = max(1, args.target_total // len(templates)) # In our case, around 267 variants per template. So, we need a large diversity_pools.json
    print(f"{len(templates):,} templates x {variants_per_template:,} variants "
          f"= {len(templates) * variants_per_template:,} requests planned "
          f"(capped at --target-total {args.target_total:,})")

    sampler = DiversitySampler(seed=args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    planned = iter_planned(
        templates, sampler, variants_per_template, args.target_total
    )
    total_written = 0
    chunk_count = 0

    with open(out_dir / "request_metadata.jsonl", "w", encoding="utf-8") as f_map:
        for chunk_idx, batch in enumerate(chunked(planned, args.chunk_size)): # iterate through each chunk
            chunk_name = f"{prefix}requests_chunk_{chunk_idx:04d}"
            with open(out_dir / f"{chunk_name}.jsonl", "w", encoding="utf-8") as f_out: # create a file for current chunk
                for t_row, v, combo in batch:
                    custom_id = f"{prefix}{t_row['id']}_v{v}"
                    prompt = build_prompt(t_row["template"], combo)

                    f_out.write(json.dumps(build_request_line(
                        custom_id, prompt, args.temperature, args.max_output_tokens
                    )) + "\n")

                    meta = {
                        "custom_id": custom_id,
                        "prompt_sha256": prompt_hash(prompt),
                        "source_template_id": t_row["id"],
                        "chunk": chunk_name,
                        **combo,
                    }
                    carry_metadata(t_row, meta, prefix="source_")
                    f_map.write(json.dumps(meta) + "\n")

            chunk_count += 1
            total_written += len(batch)

    print(f"\nWrote {total_written:,} requests across {chunk_count} chunk file(s) "
          f"in {out_dir}")
    print("Metadata (with both join keys) written to request_metadata.jsonl")
    print("\nBefore submitting a real chunk, compare one line against Google's "
          "current sample file -- see the schema note at the top of this script.")


if __name__ == "__main__":
    main()
