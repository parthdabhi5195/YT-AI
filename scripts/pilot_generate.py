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
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from diversity_sampler import DiversitySampler  # noqa: E402
from prompts import build_prompt  # noqa: E402

from google import genai
from google.genai import errors as genai_errors


def generate_one(client, model, prompt, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return resp.text.strip()
        except genai_errors.APIError as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Generation failed after {retries} attempts: {last_err}")


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
    args = parser.parse_args()

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    sampler = DiversitySampler(seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    story_count = 0
    with open(args.templates, "r", encoding="utf-8") as f_templates, \
         open(out_path, "a", encoding="utf-8") as f_out:
        for t_idx, line in enumerate(f_templates):
            if t_idx >= args.max_templates:
                break
            row = json.loads(line)
            template = row["template"]
            for v in range(args.variants_per_template):
                combo = sampler.sample_combo()
                prompt = build_prompt(template, combo)
                try:
                    story_text = generate_one(client, args.model, prompt)
                except RuntimeError as e:
                    print(f"SKIPPED {row['id']} variant {v}: {e}")
                    continue
                out_row = {
                    "id": f"{row['id']}_v{v}",
                    "source_template_id": row["id"],
                    **combo,
                    "story_text": story_text,
                    "word_count": len(story_text.split()),
                    "model_used": args.model,
                }
                f_out.write(json.dumps(out_row) + "\n")
                f_out.flush()
                story_count += 1
                if story_count % 25 == 0:
                    print(f"Generated {story_count} stories...")

    print(f"Done. {story_count} pilot stories written to {out_path}")


if __name__ == "__main__":
    main()
