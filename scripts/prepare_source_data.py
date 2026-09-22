"""
Converts your real export files -- the ones with fields VideoID, Channel,
Title, Duration, Views, Transcript -- into the internal format
extract_templates.py expects, while PRESERVING title/channel/views as
extra metadata that flows through the whole pipeline (useful later if you
want to filter templates by how well the source video actually performed).

This replaces concat_jsonl.py for your real dataset -- concat_jsonl.py
still exists for the simpler hypothetical {"id","text"} case, but your
actual files need this instead.

Handles multiple input files, checks for duplicate VideoIDs across files,
and can optionally drop anything under a views threshold if you want to
train only on your best-performing subset.

Usage (merge everything, no extra filtering -- your dataset is already
all >=50k views):
    python scripts/prepare_source_data.py \
        --inputs /mnt/user-data/uploads/_Broken.jsonl \
                 /mnt/user-data/uploads/_Requestedreads.jsonl \
                 /mnt/user-data/uploads/_TaleReader_0.jsonl \
        --output data/source_scripts/source_scripts.jsonl

Usage (keep only your strongest performers, e.g. 200k+ views):
    python scripts/prepare_source_data.py \
        --input-glob "/mnt/user-data/uploads/*.jsonl" \
        --output data/source_scripts/source_scripts.jsonl \
        --min-views 200000
"""
import argparse
import glob
import json
from pathlib import Path

REQUIRED_FIELDS = {"VideoID", "Channel", "Title", "Duration", "Views", "Transcript"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=None,
                         help="Explicit list of input JSONL files")
    parser.add_argument("--input-glob", default=None,
                         help='Glob pattern instead of listing files, e.g. "data/raw/*.jsonl"')
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-views", type=int, default=0,
                         help="Drop stories below this view count. Your dataset is "
                              "already all >=50,000, so this only matters if you "
                              "want an even stricter 'best of the best' subset.")
    args = parser.parse_args()

    if args.input_glob:
        files = sorted(glob.glob(args.input_glob))
    elif args.inputs:
        files = args.inputs
    else:
        raise SystemExit("Provide either --inputs (list of files) or --input-glob (pattern)")

    if not files:
        raise SystemExit("No input files matched.")

    print(f"Reading {len(files)} file(s):")
    for f in files:
        print(f"  - {f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_video_ids = set()
    stats = {"total_seen": 0, "missing_fields": 0, "duplicate_id": 0,
             "below_min_views": 0, "kept": 0}

    with open(out_path, "w", encoding="utf-8") as f_out:
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    stats["total_seen"] += 1
                    row = json.loads(line)

                    if not REQUIRED_FIELDS.issubset(row.keys()):
                        stats["missing_fields"] += 1
                        continue

                    video_id = row["VideoID"]
                    if video_id in seen_video_ids:
                        stats["duplicate_id"] += 1
                        continue
                    seen_video_ids.add(video_id)

                    if row["Views"] < args.min_views:
                        stats["below_min_views"] += 1
                        continue

                    out_row = {
                        "id": video_id,
                        "text": row["Transcript"],
                        "title": row["Title"],
                        "channel": row["Channel"],
                        "views": row["Views"],
                    }
                    f_out.write(json.dumps(out_row) + "\n")
                    stats["kept"] += 1

    print("\n--- Conversion summary ---")
    for k, v in stats.items():
        print(f"{k:>17}: {v}")
    print(f"\nWrote {stats['kept']} stories to {out_path}")


if __name__ == "__main__":
    main()
