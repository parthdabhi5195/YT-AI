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

JOIN STRATEGY: tries custom_id first, then falls back to hashing the
prompt text out of the echoed request and matching on prompt_sha256.
build_batch_requests.py writes both keys for exactly this reason.

In practice the fallback is the path that runs: Vertex batch output rows
carry only status, processed_time, request and response -- custom_id is
accepted on the way in and not echoed back. So joined_by_custom_id: 0 in
the summary is expected, not a bug. The hash is taken over the extracted
prompt TEXT, not the request JSON, which is what makes it survive the
reshaping Vertex does to the echoed request (dropping generationConfig,
adding explicit nulls to parts).

MEMORY: metadata and the near-duplicate index are both held in RAM, and
both scale with the number of requests -- roughly 4.5GB at 2M rows
(1.7GB metadata + 2.6GB LSH). If that is tight, run one chunk at a time:
request_metadata.jsonl stamps a "chunk" field on every row.

Usage:
    pip install datasketch
    python scripts/clean_dataset.py \
        --batch-results-dir data/batch_results \
        --metadata data/batch_requests/request_metadata.jsonl \
        --output data/clean/stories_clean.jsonl \
        --min-words 200 --max-words 340
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from batch_schema import extract_prompt_text, prompt_hash  # noqa: E402
from pipeline_io import iter_jsonl, print_stats  # noqa: E402
from validation import validate_story, word_count  # noqa: E402

from datasketch import MinHash, MinHashLSH

MINHASH_PERMUTATIONS = 64

# walks nested JSON and finds the first string field names "text"
def _find_text_recursive(obj):
    """
    Best-effort search for generated text in an unknown response shape.
    Batch output nesting differs across API versions, so walk the
    structure rather than hard-coding one path.

    Note this will happily return prompt text if handed a request object;
    _find_response_obj is what keeps that from happening.
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
    return extract_prompt_text(req) or _find_text_recursive(req)


def _find_response_obj(row):
    """
    Prefer the response half of the row so we don't accidentally extract
    the PROMPT text as if it were the generated story.
    """
    for key in ("response", "predictions", "prediction", "candidates"): # defensive move, but usually it is listed as "response"
        if key in row:
            return row[key]
    # No recognizable response key: use the whole row minus the request.
    return {k: v for k, v in row.items() if k != "request"}


def load_metadata(path):
    """
    Returns (by_custom_id, by_prompt_hash, expected_per_chunk).

    expected_per_chunk counts how many requests each chunk was built with.
    Comparing it against what actually turns up in the results directory is
    the only way to notice a chunk that was never submitted or never
    downloaded -- otherwise those stories just quietly aren't there.
    """
    by_custom_id, by_prompt_hash = {}, {}
    expected_per_chunk = Counter()

    def warn(line_no, line, exc):
        print(f"  warning: metadata line {line_no} is not valid JSON -- ignoring it")

    for _, row in iter_jsonl(path, on_bad=warn):
        if "custom_id" in row:
            by_custom_id[row["custom_id"]] = row
        if "prompt_sha256" in row:
            by_prompt_hash[row["prompt_sha256"]] = row
        if row.get("chunk"):
            expected_per_chunk[row["chunk"]] += 1
    return by_custom_id, by_prompt_hash, expected_per_chunk

# create a MinHash signnature of the story text
# lower-case the text, tokenize by whitespace, de-duplicate tokens with a set, and hash them
# used by the deduper to estimate near-duplicate stories.
def minhash_for(text, num_perm=MINHASH_PERMUTATIONS):
    mh = MinHash(num_perm=num_perm)
    # update_batch does one vectorized pass instead of a Python call per
    # token -- ~6x faster, and this line dominates the whole script.
    mh.update_batch([w.encode("utf-8") for w in set(text.lower().split())])
    return mh


class Deduper:
    """Exact + near duplicate rejection. Returns a stats key, or None to keep."""

    def __init__(self, threshold, num_perm=MINHASH_PERMUTATIONS):
        self.seen_hashes = set()
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.kept = 0

    def check(self, text, key):
        exact = hashlib.md5(text.lower().encode("utf-8")).hexdigest()
        if exact in self.seen_hashes:
            return "exact_dup" # catches exact duplicates
        mh = minhash_for(text)
        if self.lsh.query(mh):
            return "near_dup"
        self.seen_hashes.add(exact)
        self.lsh.insert(f"{key}_{self.kept}", mh)
        self.kept += 1
        return None # New story is accepted and recorded


def join_metadata(row, custom_id, by_custom_id, by_prompt_hash):
    """Returns (metadata_row, stats_key). custom_id first, prompt hash as fallback."""
    if custom_id and custom_id in by_custom_id:
        return by_custom_id[custom_id], "joined_by_custom_id"
    prompt = _find_prompt_in_request(row)
    if prompt:
        meta_row = by_prompt_hash.get(prompt_hash(prompt))
        if meta_row:
            return meta_row, "joined_by_prompt_hash"
    return {}, "no_metadata_match"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-results-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-words", type=int, default=200)
    parser.add_argument("--max-words", type=int, default=340)
    parser.add_argument("--near-dup-threshold", type=float, default=0.85)
    parser.add_argument("--rejects-output", default=None,
                        help="Path to write rejected rows to, with full story "
                             "text, reason and combo metadata. Strongly "
                             "recommended: without it, every rejected story is "
                             "discarded with no record.")
    parser.add_argument("--dedup-against", action="append", default=[],
                        metavar="CLEAN_JSONL",
                        help="An existing clean file to deduplicate against. "
                             "Its stories are loaded into the duplicate index "
                             "first, so this run rejects anything that repeats "
                             "them. Repeatable. Use when cleaning a second run "
                             "whose stories must not duplicate the first.")
    parser.add_argument("--append", action="store_true",
                        help="Append to the output files instead of replacing "
                             "them. Use when cleaning one chunk at a time into "
                             "a single dataset.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite non-empty output files. Without this (or "
                             "--append) the run aborts rather than destroying a "
                             "previous run's results.")
    args = parser.parse_args()

    if args.append and args.force:
        raise SystemExit("--append and --force are mutually exclusive.")

    # A previous run's output represents real API spend. Replacing it is a
    # decision, not a default.
    mode = "a" if args.append else "w"
    if not args.append and not args.force:
        for path in (args.output, args.rejects_output):
            if path and Path(path).exists() and Path(path).stat().st_size > 0:
                raise SystemExit(
                    f"{path} already exists and is not empty.\n"
                    f"Re-run with --append to add to it, or --force to replace "
                    f"it. Replacing discards whatever a previous run produced."
                )

    by_custom_id, by_prompt_hash, expected_per_chunk = load_metadata(args.metadata)
    print(f"Loaded metadata: {len(by_custom_id)} by custom_id, "
          f"{len(by_prompt_hash)} by prompt hash, "
          f"{len(expected_per_chunk)} chunk(s) expected.")

    deduper = Deduper(args.near_dup_threshold)
    for path in args.dedup_against:
        seeded = 0
        for _, prior in iter_jsonl(path):
            if prior.get("story_text"):
                deduper.check(prior["story_text"], prior.get("id"))
                seeded += 1
        print(f"Dedup index seeded with {seeded:,} stories from {path}")
    stats = {"total": 0, "bad_json": 0, "no_text": 0, "failed_validation": 0,
             "exact_dup": 0, "near_dup": 0, "joined_by_custom_id": 0,
             "joined_by_prompt_hash": 0, "no_metadata_match": 0, "kept": 0}
    dup_reasons = {"exact_dup": "exact duplicate", "near_dup": "near duplicate"}
    seen_per_chunk = Counter()   # rows actually found, per chunk, for the audit below

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_files = sorted(Path(args.batch_results_dir).rglob("*.jsonl")) # script scans the entire directory tree recursively. It doesn't assume one result file.
    print(f"Found {len(result_files)} result file(s).")

    with ExitStack() as stack:
        f_out = stack.enter_context(open(out_path, mode, encoding="utf-8"))
        f_rejects = (
            stack.enter_context(open(args.rejects_output, mode, encoding="utf-8"))
            if args.rejects_output else None
        )

        def reject(row_id, reason, text=None, meta_row=None, problems=None):
            """
            Write a rejected row in the SAME shape as a kept one, plus why.

            Full story_text, not a preview: a story dropped for a markdown
            heading or a stray wrapping quote is good prose you already paid
            for, and recovering it later is a text edit -- but only if the
            text and its combo metadata survived.
            """
            if not f_rejects:
                return
            out = {
                "id": row_id or (meta_row or {}).get("custom_id"),
                "reason": reason,
                "story_text": text or "",
                "word_count": word_count(text) if text else 0,
            }
            if problems:
                out["problems"] = problems
            if meta_row:
                out.update({k: v for k, v in meta_row.items() if k != "custom_id"})
            f_rejects.write(json.dumps(out) + "\n")

        def bad_json(line_no, line, exc):
            stats["bad_json"] += 1
            reject(None, "bad json")

        for rf in result_files:
            for _, row in iter_jsonl(rf, on_bad=bad_json):
                stats["total"] += 1

                custom_id = row.get("custom_id") or row.get("id")

                # Join BEFORE every reject gate, not after. Two reasons: a
                # rejected row is only recoverable if you can still tell which
                # template and combo produced it, and the chunk audit below
                # needs to count rows that ARRIVED, however they turned out.
                # Vertex echoes the request even on a failed row, so this works
                # even when no text came back.
                meta_row, join_key = join_metadata(
                    row, custom_id, by_custom_id, by_prompt_hash
                )
                if meta_row.get("chunk"):
                    seen_per_chunk[meta_row["chunk"]] += 1

                text = _find_text_recursive(_find_response_obj(row))
                if not text:
                    stats["no_text"] += 1
                    # Vertex leaves "status" empty on success and puts the API
                    # error there on a failed row. Carrying it through is the
                    # difference between "8% vanished" and knowing why.
                    status = row.get("status")
                    reject(custom_id,
                           f"api error: {status}" if status else "no text found",
                           None, meta_row)
                    continue
                text = text.strip()

                problems = validate_story(
                    text, min_words=args.min_words, max_words=args.max_words
                )
                if problems:
                    stats["failed_validation"] += 1
                    reject(custom_id, "validation", text, meta_row, problems)
                    continue

                dup = deduper.check(text, custom_id)
                if dup:
                    stats[dup] += 1
                    reject(custom_id, dup_reasons[dup], text, meta_row)
                    continue

                stats[join_key] += 1

                f_out.write(json.dumps({
                    "id": custom_id or meta_row.get("custom_id"),
                    "story_text": text,
                    "word_count": word_count(text),
                    **{k: v for k, v in meta_row.items() if k != "custom_id"},
                }) + "\n")
                stats["kept"] += 1

    print_stats("Cleaning summary", stats)
    if stats["total"]:
        print(f"\nYield rate: {stats['kept'] / stats['total'] * 100:.1f}%")
    if stats["no_metadata_match"]:
        print(f"WARNING: {stats['no_metadata_match']} rows had no metadata match. "
              f"If this is most of them, check that --metadata points at the "
              f"request_metadata.jsonl from the SAME build_batch_requests.py run.")

    # Chunk audit. A chunk that was never submitted, or whose results were
    # never downloaded, produces no error anywhere else -- the run just
    # silently processes fewer stories than you paid for.
    if expected_per_chunk:
        missing = sorted(c for c in expected_per_chunk if not seen_per_chunk[c])
        short = sorted(
            (c, expected_per_chunk[c], seen_per_chunk[c])
            for c in expected_per_chunk
            if seen_per_chunk[c] and seen_per_chunk[c] < expected_per_chunk[c]
        )
        if missing:
            lost = sum(expected_per_chunk[c] for c in missing)
            print(f"\nWARNING: {len(missing)} of {len(expected_per_chunk)} chunk(s) "
                  f"produced NO rows -- {lost:,} requests unaccounted for.")
            print("  Not submitted, not downloaded, or missing from "
                  "--batch-results-dir:")
            for c in missing[:10]:
                print(f"    {c}  ({expected_per_chunk[c]:,} requests)")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
        if short:
            print(f"\nWARNING: {len(short)} chunk(s) returned fewer rows than built:")
            for c, exp, got in short[:10]:
                print(f"    {c}  expected {exp:,}, found {got:,} "
                      f"(short {exp - got:,})")
        if not missing and not short:
            print(f"\nAll {len(expected_per_chunk)} chunk(s) accounted for.")

    print(f"\nOutput: {out_path}" + ("  (appended)" if args.append else ""))
    if args.rejects_output:
        print(f"Rejects: {args.rejects_output}")


if __name__ == "__main__":
    main()
