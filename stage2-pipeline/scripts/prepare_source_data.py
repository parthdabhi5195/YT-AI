"""
Converts your export files -- fields VideoID, Channel, Title, Duration,
Views, Transcript -- into the internal format extract_templates.py
expects, preserving title/channel/views as metadata that flows through
the whole pipeline.

Files are concatenated in EXACTLY the order you list them.

Usage:
    python scripts/prepare_source_data.py \
        --inputs data/source_scripts/@Broken.jsonl \
                 data/source_scripts/@Requestedreads.jsonl \
                 data/source_scripts/@TaleReader_0.jsonl \
        --output data/source_scripts/source_scripts.jsonl

NOTE ON Duration: required of every input row as an exporter-contract
check, but deliberately not carried into the output -- no later stage
uses it. Add it to the written row below if you ever want to filter or
weight by video length.

Duplicate VideoIDs are not checked for as YouTube guarentees global uniqueness.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from pipeline_io import append_row, iter_jsonl, print_stats  # noqa: E402

REQUIRED_FIELDS = {"VideoID", "Channel", "Title", "Duration", "Views", "Transcript"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input JSONL files, concatenated in this exact order")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Reading {len(args.inputs)} file(s) in order:")
    for f in args.inputs:
        print(f"  - {f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
            "total_seen": 0, 
             "missing_fields": 0, 
             "bad_json": 0, 
             "kept": 0
            }
    with open(out_path, "w", encoding="utf-8") as f_out:
        for file_path in args.inputs:

            def bad_line(line_no, line, exc, _path=file_path):
                print(f"  SKIPPED {_path}:{line_no} -- bad JSON ({exc})")
                stats["bad_json"] += 1

            for line_no, row in iter_jsonl(file_path, on_bad=bad_line): # Iterate through each (#, JSON) in input file.
                stats["total_seen"] += 1

                if not REQUIRED_FIELDS.issubset(row.keys()): # Print warning and skip if required fields are missing in JSON
                    missing = sorted(REQUIRED_FIELDS - set(row.keys()))
                    print(f"  SKIPPED {file_path}:{line_no} -- missing {missing}")
                    stats["missing_fields"] += 1
                    continue

                append_row(f_out, { # What the final output row looks like for source_scripts.jsonl
                    "id": row["VideoID"],
                    "text": row["Transcript"],
                    "title": row["Title"],
                    "channel": row["Channel"],
                    "views": row["Views"],
                }) # writing to hard disk immendiately instead of storing buffer data in RAM.
                stats["kept"] += 1

            

    print_stats("Conversion summary", stats)

    print(f"\nWrote {stats['kept']} stories to {out_path}")


if __name__ == "__main__":
    main()
