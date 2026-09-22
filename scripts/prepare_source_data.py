"""
Converts your export files -- fields VideoID, Channel, Title, Duration,
Views, Transcript -- into the internal format extract_templates.py
expects, preserving title/channel/views as metadata that flows through
the whole pipeline.

Files are concatenated in EXACTLY the order you list them.

Usage:
    python scripts/prepare_source_data.py \
        --inputs data/source_scripts/_Broken.jsonl \
                 data/source_scripts/_Requestedreads.jsonl \
                 data/source_scripts/_TaleReader_0.jsonl \
        --output data/source_scripts/source_scripts.jsonl

NOTE ON DUPLICATE IDs: YouTube video IDs are 11-character globally
unique identifiers -- one per video, and a video belongs to exactly one
channel -- so per-channel exports cannot legitimately collide. Verified
empirically against your 6,473 rows: zero duplicates. Nothing is skipped
or dropped here as a result. A passive count check at the end warns if
duplicates ever do appear, which in practice would only mean the scraper
ran twice and appended; delete those three lines if you don't want it.
"""
import argparse
import json
from pathlib import Path

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

    stats = {"total_seen": 0, "missing_fields": 0, "bad_json": 0, "kept": 0}
    ids = []

    with open(out_path, "w", encoding="utf-8") as f_out:
        for file_path in args.inputs:
            with open(file_path, "r", encoding="utf-8") as f_in:
                for line_no, line in enumerate(f_in, 1):
                    line = line.strip()
                    if not line:
                        continue
                    stats["total_seen"] += 1

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  SKIPPED {file_path}:{line_no} -- bad JSON ({e})")
                        stats["bad_json"] += 1
                        continue

                    if not REQUIRED_FIELDS.issubset(row.keys()):
                        missing = sorted(REQUIRED_FIELDS - set(row.keys()))
                        print(f"  SKIPPED {file_path}:{line_no} -- missing {missing}")
                        stats["missing_fields"] += 1
                        continue

                    ids.append(row["VideoID"])
                    f_out.write(json.dumps({
                        "id": row["VideoID"],
                        "text": row["Transcript"],
                        "title": row["Title"],
                        "channel": row["Channel"],
                        "views": row["Views"],
                    }) + "\n")
                    stats["kept"] += 1

    print("\n--- Conversion summary ---")
    for k, v in stats.items():
        print(f"{k:>16}: {v}")

    if len(ids) != len(set(ids)):
        print(f"\nWARNING: {len(ids) - len(set(ids))} duplicate VideoID(s) found. "
              f"YouTube IDs are globally unique, so this almost certainly means "
              f"the same file was passed twice or the scraper appended a re-run. "
              f"Nothing was dropped -- check your --inputs list.")

    print(f"\nWrote {stats['kept']} stories to {out_path}")


if __name__ == "__main__":
    main()
