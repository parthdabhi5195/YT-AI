"""
Stage 3 -- Build batch request files for the full production run.

Turns your templates + diversity injector into JSONL request files, split
into chunks so you can run (and check the cost of) one chunk at a time
instead of committing your whole budget in one submission.

IMPORTANT: Before running a real chunk, download Google's own example file
(gs://cloud-samples-data/batch/prompt_for_batch_gemini_predict.jsonl) and
compare its structure to BATCH_REQUEST below. Vertex's batch schema for
native Gemini models has shifted across model generations and I have not
been able to verify this byte-for-byte -- treat this as a correct starting
shape, not a guarantee, and confirm on your first small chunk before
trusting it at scale.

Usage:
    python scripts/build_batch_requests.py \
        --templates data/templates/templates.jsonl \
        --output-dir data/batch_requests \
        --target-total 2000000 \
        --chunk-size 50000 \
        --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from diversity_sampler import DiversitySampler  # noqa: E402
from prompts import build_prompt  # noqa: E402


def batch_request(custom_id: str, prompt: str) -> dict:
    """
    Structurally matches a standard Gemini generateContent request body.
    VERIFY against Google's current sample file before a real run -- see
    module docstring.
    """
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generation_config": {"temperature": 1.0, "max_output_tokens": 500},
        },
        "custom_id": custom_id,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-total", type=int, default=2_000_000)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    templates = []
    with open(args.templates, "r", encoding="utf-8") as f:
        for line in f:
            templates.append(json.loads(line))

    if not templates:
        raise SystemExit("No templates found -- run extract_templates.py first.")

    variants_per_template = max(1, args.target_total // len(templates))
    total_planned = len(templates) * variants_per_template
    print(f"{len(templates)} templates x {variants_per_template} variants "
          f"= {total_planned:,} requests planned")

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
            template = t_row["template"]
            for v in range(variants_per_template):
                if total_written >= args.target_total:
                    break
                combo = sampler.sample_combo()
                custom_id = f"{t_row['id']}_v{v}"
                prompt = build_prompt(template, combo)

                f_out.write(json.dumps(batch_request(custom_id, prompt)) + "\n")
                f_map.write(json.dumps({
                    "custom_id": custom_id,
                    "source_template_id": t_row["id"],
                    **combo,
                }) + "\n")

                lines_in_chunk += 1
                total_written += 1

                if lines_in_chunk >= args.chunk_size:
                    f_out.close()
                    chunk_idx += 1
                    lines_in_chunk = 0
                    f_out = open_chunk(chunk_idx)
            if total_written >= args.target_total:
                break
    finally:
        f_out.close()
        f_map.close()

    # Clean up a trailing empty chunk file if total_written landed exactly
    # on a chunk boundary.
    last_chunk_path = out_dir / f"requests_chunk_{chunk_idx:04d}.jsonl"
    if last_chunk_path.exists() and last_chunk_path.stat().st_size == 0:
        last_chunk_path.unlink()
        chunk_idx -= 1

    print(f"Wrote {total_written:,} requests across {chunk_idx + 1} chunk file(s) in {out_dir}")
    print("Submit ONE chunk with submit_batch_job.py, check the output and your "
          "billing report, then continue to the next chunk -- don't submit all "
          "chunks back to back on the first run.")


if __name__ == "__main__":
    main()
