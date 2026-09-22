"""
Stage 1 -- Structural extraction.

Reads your proven source scripts and asks Gemini to output ONLY an
abstract structural template for each one (trope, character roles, plot
beats, emotional arc, hook style) -- with no character names, settings, or
verbatim phrases from the source.

Usage:
    python scripts/extract_templates.py \
        --input data/source_scripts/source_scripts.jsonl \
        --output data/templates/templates.jsonl \
        --project YOUR_PROJECT_ID \
        --limit 50    # start small; omit --limit to run all of them

Each line of the input JSONL must look like:
    {"id": "src_0001", "text": "the full story text..."}

Re-running with the same --output file skips IDs already extracted, so
it's safe to stop and resume.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from prompts import build_extraction_prompt  # noqa: E402

from google import genai
from google.genai import errors as genai_errors


def extract_one(client: "genai.Client", model: str, story_text: str, retries: int = 3) -> dict:
    prompt = build_extraction_prompt(story_text)
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            text = resp.text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0]
            return json.loads(text)
        except (genai_errors.APIError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise RuntimeError(f"Extraction failed after {retries} attempts: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", required=True, help="Your GCP project ID")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N stories -- use this for a first test")
    args = parser.parse_args()

    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already_done = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                already_done.add(json.loads(line)["id"])
        print(f"Resuming -- {len(already_done)} templates already extracted.")

    count = 0
    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "a", encoding="utf-8") as f_out:
        for line in f_in:
            if args.limit and count >= args.limit:
                break
            row = json.loads(line)
            if row["id"] in already_done:
                continue
            try:
                template = extract_one(client, args.model, row["text"])
                out_row = {"id": row["id"], "template": template}
                # Carry through source metadata if present (title/channel/views
                # from prepare_source_data.py) -- harmless no-op if you used
                # the simpler {"id","text"}-only format instead.
                for extra_key in ("title", "channel", "views"):
                    if extra_key in row:
                        out_row[extra_key] = row[extra_key]
                f_out.write(json.dumps(out_row) + "\n")
                f_out.flush()
                count += 1
                if count % 25 == 0:
                    print(f"Extracted {count} templates...")
            except RuntimeError as e:
                print(f"SKIPPED {row['id']}: {e}")

    print(f"Done. {count} new templates written to {out_path}")


if __name__ == "__main__":
    main()
