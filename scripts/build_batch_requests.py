"""
Stage 3 -- Build batch request files for the full production run.

Turns your templates + diversity injector into JSONL request files, split
into chunks so you can run (and check the cost of) one chunk at a time.

=== SCHEMA STATUS: PARTIALLY VERIFIED ===

Verified from Google's docs:
  * For native Gemini models on Vertex, each input line is a JSON object
    whose "request" field follows the GenerateContentRequest structure.
    That is the format built below.
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

Usage:
    python scripts/build_batch_requests.py \
        --templates data/templates/templates.jsonl \
        --output-dir data/batch_requests \
        --target-total 2000000 \
        --chunk-size 50000 \
        --seed 42
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from diversity_sampler import DiversitySampler  # noqa: E402
from prompts import build_prompt  # noqa: E402

MAX_REQUESTS_PER_JOB = 150_000


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def batch_request(custom_id: str, prompt: str, temperature: float,
                  max_output_tokens: int) -> dict:
    return {
        "custom_id": custom_id,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-total", type=int, default=2_000_000)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=500)
    args = parser.parse_args()

    if args.chunk_size > MAX_REQUESTS_PER_JOB:
        raise SystemExit(
            f"--chunk-size {args.chunk_size:,} exceeds the {MAX_REQUESTS_PER_JOB:,} "
            f"requests-per-job limit. Lower it."
        )

    templates = []
    skipped = 0
    with open(args.templates, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "template" not in row or "id" not in row:
                    raise KeyError("missing id/template")
                templates.append(row)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  SKIPPED template line {line_no}: {e}")
                skipped += 1

    if not templates:
        raise SystemExit("No usable templates -- run extract_templates.py first.")
    if skipped:
        print(f"  ({skipped} malformed template line(s) skipped)\n")

    variants_per_template = max(1, args.target_total // len(templates))
    print(f"{len(templates):,} templates x {variants_per_template:,} variants "
          f"= {len(templates) * variants_per_template:,} requests planned "
          f"(capped at --target-total {args.target_total:,})")

    sampler = DiversitySampler(seed=args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_idx = 0
    lines_in_chunk = 0
    total_written = 0

    def open_chunk(idx):
        return open(out_dir / f"requests_chunk_{idx:04d}.jsonl", "w", encoding="utf-8")

    f_out = open_chunk(chunk_idx)
    f_map = open(out_dir / "request_metadata.jsonl", "w", encoding="utf-8")

    try:
        for t_row in templates:
            if total_written >= args.target_total:
                break
            template = t_row["template"]
            for v in range(variants_per_template):
                if total_written >= args.target_total:
                    break

                combo = sampler.sample_combo()
                custom_id = f"{t_row['id']}_v{v}"
                prompt = build_prompt(template, combo)
                p_hash = prompt_hash(prompt)

                f_out.write(json.dumps(batch_request(
                    custom_id, prompt, args.temperature, args.max_output_tokens
                )) + "\n")

                meta = {
                    "custom_id": custom_id,
                    "prompt_sha256": p_hash,
                    "source_template_id": t_row["id"],
                    "chunk": f"requests_chunk_{chunk_idx:04d}",
                    **combo,
                }
                for extra_key in ("title", "channel", "views"):
                    if extra_key in t_row:
                        meta[f"source_{extra_key}"] = t_row[extra_key]
                f_map.write(json.dumps(meta) + "\n")

                lines_in_chunk += 1
                total_written += 1

                if lines_in_chunk >= args.chunk_size:
                    f_out.close()
                    chunk_idx += 1
                    lines_in_chunk = 0
                    f_out = open_chunk(chunk_idx)
    finally:
        f_out.close()
        f_map.close()

    last_chunk_path = out_dir / f"requests_chunk_{chunk_idx:04d}.jsonl"
    if last_chunk_path.exists() and last_chunk_path.stat().st_size == 0:
        last_chunk_path.unlink()
        chunk_idx -= 1

    print(f"\nWrote {total_written:,} requests across {chunk_idx + 1} chunk file(s) "
          f"in {out_dir}")
    print("Metadata (with both join keys) written to request_metadata.jsonl")
    print("\nBefore submitting a real chunk, compare one line against Google's "
          "current sample file -- see the schema note at the top of this script.")


if __name__ == "__main__":
    main()
