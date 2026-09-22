"""
Stage 5 -- Merge batch outputs with their metadata, validate, deduplicate,
and produce the final training-ready dataset.

WHAT THIS IS: a practical cleaning pass. It removes malformed rows,
out-of-range lengths, markdown/format artifacts, exact duplicates, and
near-duplicates, then reattaches the metadata for each surviving story.

WHAT THIS IS NOT: a semantic relevance check, a safety filter, a
plagiarism detector, or a guarantee the model followed the beat order.
It catches low-entropy junk and obvious duplicates. Reading a random
sample by hand is still the only way to judge whether the stories are
actually good.

JOIN STRATEGY: tries custom_id first; if the batch output doesn't echo
that field back, falls back to hashing the prompt text out of the echoed
request and matching on prompt_sha256. build_batch_requests.py writes
both keys for exactly this reason.

Usage:
    pip install datasketch
    python scripts/clean_dataset.py \
        --batch-results-dir data/batch_results \
        --metadata data/batch_requests/request_metadata.jsonl \
        --output data/clean/stories_clean.jsonl \
        --min-words 200 --max-words 350
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from validation import validate_story, word_count  # noqa: E402

from datasketch import MinHash, MinHashLSH


def _find_text_recursive(obj, _skip_keys=("parts",)):
    """
    Best-effort search for generated text in an unknown response shape.
    Batch output nesting differs across API versions, so walk the
    structure rather than hard-coding one path.
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


def _find_prompt_in_request(row):
    """
    Pull the original prompt text back out of the echoed request object,
    so it can be hashed and matched against prompt_sha256.
    """
    req = row.get("request")
    if not isinstance(req, dict):
        return None
    try:
        return req["contents"][0]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return _find_text_recursive(req)


def _find_response_obj(row):
    """
    Prefer the response half of the row so we don't accidentally extract
    the PROMPT text as if it were the generated story.
    """
    for key in ("response", "predictions", "prediction", "candidates"):
        if key in row:
            return row[key]
    # No recognizable response key: use the whole row minus the request.
    return {k: v for k, v in row.items() if k != "request"}


def load_metadata(path: str):
    by_custom_id, by_prompt_hash = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "custom_id" in row:
                by_custom_id[row["custom_id"]] = row
            if "prompt_sha256" in row:
                by_prompt_hash[row["prompt_sha256"]] = row
    return by_custom_id, by_prompt_hash


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
    parser.add_argument("--rejects-output", default=None,
                        help="Optional path to write rejected rows with reasons, "
                             "for diagnosing a bad prompt")
    args = parser.parse_args()

    by_custom_id, by_prompt_hash = load_metadata(args.metadata)
    print(f"Loaded metadata: {len(by_custom_id)} by custom_id, "
          f"{len(by_prompt_hash)} by prompt hash.")

    seen_hashes = set()
    lsh = MinHashLSH(threshold=args.near_dup_threshold, num_perm=64)

    stats = {"total": 0, "bad_json": 0, "no_text": 0, "failed_validation": 0,
             "exact_dup": 0, "near_dup": 0, "joined_by_custom_id": 0,
             "joined_by_prompt_hash": 0, "no_metadata_match": 0, "kept": 0}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f_rejects = open(args.rejects_output, "w", encoding="utf-8") if args.rejects_output else None

    result_files = sorted(Path(args.batch_results_dir).rglob("*.jsonl"))
    print(f"Found {len(result_files)} result file(s).")

    def reject(row_id, reason, text=None):
        if f_rejects:
            f_rejects.write(json.dumps({
                "id": row_id, "reason": reason,
                "text_preview": (text or "")[:200],
            }) + "\n")

    with open(out_path, "w", encoding="utf-8") as f_out:
        for rf in result_files:
            with open(rf, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    stats["total"] += 1

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        stats["bad_json"] += 1
                        reject(None, "bad json")
                        continue

                    custom_id = row.get("custom_id") or row.get("id")
                    text = _find_text_recursive(_find_response_obj(row))

                    if not text:
                        stats["no_text"] += 1
                        reject(custom_id, "no text found")
                        continue
                    text = text.strip()

                    problems = validate_story(
                        text, min_words=args.min_words, max_words=args.max_words
                    )
                    if problems:
                        stats["failed_validation"] += 1
                        reject(custom_id, f"validation: {problems}", text)
                        continue

                    exact_hash = hashlib.md5(text.lower().encode("utf-8")).hexdigest()
                    if exact_hash in seen_hashes:
                        stats["exact_dup"] += 1
                        reject(custom_id, "exact duplicate", text)
                        continue

                    mh = minhash_for(text)
                    if lsh.query(mh):
                        stats["near_dup"] += 1
                        reject(custom_id, "near duplicate", text)
                        continue

                    seen_hashes.add(exact_hash)
                    lsh.insert(f"{custom_id}_{stats['kept']}", mh)

                    # Join: custom_id first, prompt hash as fallback.
                    meta_row = None
                    if custom_id and custom_id in by_custom_id:
                        meta_row = by_custom_id[custom_id]
                        stats["joined_by_custom_id"] += 1
                    else:
                        prompt = _find_prompt_in_request(row)
                        if prompt:
                            p_hash = hashlib.sha256(
                                prompt.encode("utf-8")
                            ).hexdigest()
                            meta_row = by_prompt_hash.get(p_hash)
                            if meta_row:
                                stats["joined_by_prompt_hash"] += 1
                    if meta_row is None:
                        meta_row = {}
                        stats["no_metadata_match"] += 1

                    out_row = {
                        "id": custom_id or meta_row.get("custom_id"),
                        "story_text": text,
                        "word_count": word_count(text),
                        **{k: v for k, v in meta_row.items() if k != "custom_id"},
                    }
                    f_out.write(json.dumps(out_row) + "\n")
                    stats["kept"] += 1

    if f_rejects:
        f_rejects.close()

    print("\n--- Cleaning summary ---")
    for k, v in stats.items():
        print(f"{k:>22}: {v}")
    if stats["total"]:
        print(f"\nYield rate: {stats['kept'] / stats['total'] * 100:.1f}%")
    if stats["no_metadata_match"]:
        print(f"WARNING: {stats['no_metadata_match']} rows had no metadata match. "
              f"If this is most of them, check that --metadata points at the "
              f"request_metadata.jsonl from the SAME build_batch_requests.py run.")
    print(f"Output: {out_path}")
    if args.rejects_output:
        print(f"Rejects: {args.rejects_output}")


if __name__ == "__main__":
    main()
