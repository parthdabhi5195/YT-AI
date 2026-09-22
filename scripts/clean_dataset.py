"""
Stage 5 -- Merge batch outputs with their metadata, deduplicate, and filter
by length/format to produce the final training-ready dataset.

Usage:
    pip install datasketch
    python scripts/clean_dataset.py \
        --batch-results-dir data/batch_results \
        --metadata data/batch_requests/request_metadata.jsonl \
        --output data/clean/stories_clean.jsonl \
        --min-words 200 --max-words 350

NOTE ON custom_id: this assumes the batch output preserves the "custom_id"
field you sent in. If your first real chunk's output doesn't include it,
run:
    head -n 1 data/batch_results/<chunk_name>/*.jsonl | python -m json.tool
to see the actual shape, and adjust the `custom_id = ...` line below to
match (e.g. falling back to matching by line order against your input
file instead).
"""
import argparse
import hashlib
import json
from pathlib import Path

from datasketch import MinHash, MinHashLSH


def _find_text_recursive(obj):
    """
    Best-effort search for a generated text string in an unknown response
    shape. Batch output schemas can nest the string differently across API
    versions, so this walks the structure looking for a "text" key rather
    than assuming one exact path.
    """
    if isinstance(obj, dict):
        if "text" in obj and isinstance(obj["text"], str):
            return obj["text"]
        for v in obj.values():
            found = _find_text_recursive(v)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_text_recursive(item)
            if found:
                return found
    return None


def load_metadata(path: str) -> dict:
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            meta[row["custom_id"]] = row
    return meta


def minhash_for(text: str, num_perm: int = 64) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for word in set(text.lower().split()):
        mh.update(word.encode("utf-8"))
    return mh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-results-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-words", type=int, default=200)
    parser.add_argument("--max-words", type=int, default=350)
    parser.add_argument("--near-dup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    print(f"Loaded metadata for {len(metadata)} requests.")

    seen_hashes = set()
    lsh = MinHashLSH(threshold=args.near_dup_threshold, num_perm=64)

    stats = {"total": 0, "no_text": 0, "bad_length": 0,
             "exact_dup": 0, "near_dup": 0, "no_metadata_match": 0, "kept": 0}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_files = sorted(Path(args.batch_results_dir).rglob("*.jsonl"))
    print(f"Found {len(result_files)} result file(s).")

    with open(out_path, "w", encoding="utf-8") as f_out:
        for rf in result_files:
            with open(rf, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    stats["total"] += 1
                    row = json.loads(line)
                    custom_id = row.get("custom_id") or row.get("id")
                    text = _find_text_recursive(row)

                    if not text:
                        stats["no_text"] += 1
                        continue

                    text = text.strip()
                    word_count = len(text.split())
                    if not (args.min_words <= word_count <= args.max_words):
                        stats["bad_length"] += 1
                        continue

                    exact_hash = hashlib.md5(text.lower().encode("utf-8")).hexdigest()
                    if exact_hash in seen_hashes:
                        stats["exact_dup"] += 1
                        continue

                    mh = minhash_for(text)
                    if lsh.query(mh):
                        stats["near_dup"] += 1
                        continue

                    seen_hashes.add(exact_hash)
                    lsh.insert(f"{custom_id}_{stats['kept']}", mh)

                    meta_row = metadata.get(custom_id)
                    if meta_row is None:
                        stats["no_metadata_match"] += 1
                        meta_row = {}

                    out_row = {
                        "id": custom_id,
                        "story_text": text,
                        "word_count": word_count,
                        **meta_row,
                    }
                    f_out.write(json.dumps(out_row) + "\n")
                    stats["kept"] += 1

    print("\n--- Cleaning summary ---")
    for k, v in stats.items():
        print(f"{k:>17}: {v}")
    yield_rate = stats["kept"] / stats["total"] * 100 if stats["total"] else 0
    print(f"\nYield rate: {yield_rate:.1f}% clean stories kept, written to {out_path}")


if __name__ == "__main__":
    main()
