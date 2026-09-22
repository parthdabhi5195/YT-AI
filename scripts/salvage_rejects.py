"""
Stage 5b -- Recover usable stories from clean_dataset.py's rejects file.

Stories rejected for validation aren't all bad. Measured on the 1,000-row
calibration batch, most rejects were fine prose with a fixable flaw:

  * single-word italics (*was*, *now*) -- strip the asterisks
  * complete stories just outside the word bounds

What this will NOT recover:

  * stories cut off by max_output_tokens -- detected by a missing final
    sentence punctuation mark, since the rejects file doesn't carry the
    finishReason
  * api errors and exact/near duplicates (reason != "validation")

Salvaged stories skipped the dedup pass (validation runs before it in
clean_dataset.py), so every recovered story is deduplicated here against
the clean file AND against the other salvaged stories.

Writes a SEPARATE file. Nothing is appended to your clean dataset -- review
it, then concatenate:

    python3 scripts/salvage_rejects.py \
        --rejects data/clean/calib/rejects.jsonl \
        --clean   data/clean/calib/stories_clean.jsonl \
        --output  data/clean/calib/salvaged.jsonl \
        --min-words 180 --max-words 400

    cat data/clean/calib/salvaged.jsonl >> data/clean/calib/stories_clean.jsonl
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from clean_dataset import Deduper  # noqa: E402
from pipeline_io import iter_jsonl, print_stats  # noqa: E402
from validation import validate_story, word_count  # noqa: E402

# Same shape validation.py flags: *word* or *short phrase*, not ** bold **.
_ITALIC = re.compile(r"(?<![*\w])\*(?=[^\s*])([^*\n]{1,60}?)(?<=[^\s*])\*(?![*\w])")

# A story that ends mid-sentence was cut off. Allow a trailing closing quote.
_COMPLETE_ENDING = re.compile(r"""[.!?…]["'”’)\]]*\s*$""")


def strip_italics(text):
    return _ITALIC.sub(r"\1", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rejects", required=True)
    parser.add_argument("--clean", required=True,
                        help="The clean file salvaged stories are deduped against.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-words", type=int, default=180,
                        help="Relaxed floor for recovery (clean_dataset's is 200).")
    parser.add_argument("--max-words", type=int, default=400,
                        help="Relaxed ceiling for recovery (clean_dataset's is 340).")
    parser.add_argument("--near-dup-threshold", type=float, default=0.85)
    parser.add_argument("--dedup-against", action="append", default=[],
                        metavar="CLEAN_JSONL",
                        help="Additional clean file(s) salvaged stories must not "
                             "duplicate -- e.g. the first run's dataset when "
                             "salvaging a second run. Repeatable.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite a non-empty --output.")
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
        raise SystemExit(f"{out_path} already exists and is not empty. "
                         f"Re-run with --force to replace it.")

    # Seed the deduper with everything already kept.
    deduper = Deduper(args.near_dup_threshold)
    seeded = 0
    for path in [args.clean, *args.dedup_against]:
        for _, row in iter_jsonl(path):
            text = row.get("story_text")
            if text:
                deduper.check(text, row.get("id"))
                seeded += 1
    print(f"Seeded dedup index with {seeded:,} clean stories.")

    stats = Counter()
    still_failing = Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f_out:
        for _, row in iter_jsonl(args.rejects):
            stats["read"] += 1
            if row.get("reason") != "validation":
                stats["skipped_not_validation"] += 1
                continue

            text = strip_italics(row.get("story_text") or "").strip()
            if not text:
                stats["skipped_empty"] += 1
                continue

            if not _COMPLETE_ENDING.search(text):
                stats["dropped_truncated"] += 1
                continue

            problems = validate_story(
                text, min_words=args.min_words, max_words=args.max_words
            )
            if problems:
                stats["dropped_still_invalid"] += 1
                for p in problems:
                    still_failing[p.split(":")[0]] += 1
                continue

            if deduper.check(text, row.get("id")):
                stats["dropped_duplicate"] += 1
                continue

            recovered_from = [p.split(":")[0] for p in row.get("problems") or []]
            out = {k: v for k, v in row.items() if k not in ("reason", "problems")}
            out["story_text"] = text
            out["word_count"] = word_count(text)
            out["salvaged_from"] = recovered_from
            f_out.write(json.dumps(out) + "\n")
            stats["salvaged"] += 1

    print_stats("Salvage summary", stats)
    if still_failing:
        print("\nStill failing after repair:")
        for p, n in still_failing.most_common():
            print(f"  {n:5}  {p}")
    print(f"\nOutput: {out_path}")
    print("Review it, then append with:")
    print(f"  cat {out_path} >> {args.clean}")


if __name__ == "__main__":
    main()
