"""
Step 1 -- turn story JSONL into train / validation corpora.

Reads either shape of row and works out which it is on its own:

  CLEAN  -- the output of clean_dataset.py / salvage_rejects.py, i.e. rows
            with a top-level "story_text".
  RAW    -- Vertex batch output, i.e. rows with "request" / "response" and
            the story buried in response.candidates[0].content.parts[0].text.

Raw mode exists so you can build the whole training stack and smoke-test it
on one 50k-row batch file TODAY, before the cleaning pass has run. It does
no deduplication and no validation beyond dropping empties, so a model
trained on raw data is a plumbing test, not a result.

THE SPLIT IS BY TEMPLATE, NOT BY STORY. Each of the ~7,270 templates backs
roughly 357 stories that share a beat sequence, an emotional arc and a
narrative voice. Split at random and a "held-out" story is a near-paraphrase
of 356 stories the model trained on -- validation loss then measures recall,
not generalisation, and it will look wonderful while the model has learned
nothing transferable. Holding out whole templates asks the honest question:
can it write a story shaped in a way it has never seen?

Three outputs:

  train.jsonl              templates the model learns from
  val_new_templates.jsonl  templates it has never seen  <- the real metric
  val_seen_templates.jsonl stories held out from TRAIN templates

The gap between the two validation losses is the diagnostic. Small gap means
the model learned English narrative prose. Large gap means it learned 7,270
beat skeletons and is reciting them.

Assignment is a hash of the template id, so it is deterministic, needs no
memory, and stays stable if you rebuild after adding more batch files -- a
story never migrates across the split and quietly contaminates validation.

Usage:
    python training/build_corpus.py \
        --inputs 'data/clean/final/stories.jsonl' \
        --out-dir training/corpus

    # smoke test on raw, uncleaned batch output:
    python training/build_corpus.py \
        --inputs 'data/batch_results/production/**/*.jsonl' \
        --out-dir training/corpus_raw \
        --strip-italics --min-words 150 --max-words 450 --max-stories 200000
"""
import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from glob import glob
from pathlib import Path

# Same shape validation.py flags: *word* or *short phrase*, not **bold**.
_ITALIC = re.compile(r"(?<![*\w])\*(?=[^\s*])([^*\n]{1,60}?)(?<=[^\s*])\*(?![*\w])")
# id looks like "KI7U8h4BmEE_v73" or "extra_LdHjeqlaljI_v3".
_ID = re.compile(r"^(?:[a-z]+_)?(.*)_v\d+$")

SPLITS = ("train", "val_new_templates", "val_seen_templates")


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def find_text(obj):
    """First string-valued "text" field anywhere in a nested structure."""
    if isinstance(obj, dict):
        t = obj.get("text")
        if isinstance(t, str):
            return t
        for v in obj.values():
            found = find_text(v)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_text(item)
            if found:
                return found
    return None


def extract(row):
    """(story_id, story_text) from either row shape, or (id, None)."""
    row_id = row.get("id") or row.get("custom_id")
    if isinstance(row.get("story_text"), str):
        return row_id, row["story_text"]
    # Raw: search the RESPONSE half only. Searching the whole row would happily
    # return the prompt, since the echoed request also has a "text" field --
    # you would train the model on its own instructions.
    for key in ("response", "predictions", "prediction", "candidates"):
        if key in row:
            return row_id, find_text(row[key])
    return row_id, find_text({k: v for k, v in row.items() if k != "request"})


def template_of(row, row_id):
    """
    Prefer the metadata field; fall back to parsing the id.

    The run-name prefix is stripped first, so production's "LdHje_v3" and the
    extra run's "extra_LdHje_v9" resolve to the SAME template -- otherwise a
    template could straddle the split and leak across it.
    """
    tid = row.get("source_template_id")
    if tid:
        return str(tid)
    if row_id:
        m = _ID.match(row_id)
        if m:
            return m.group(1)
        return row_id
    return ""


def bucket(key, mod):
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", action="append", required=True,
                    help="Glob of story JSONL (repeatable). Quote it so the "
                         "shell does not expand it. .gz is fine.")
    ap.add_argument("--out-dir", default="training/corpus")
    ap.add_argument("--val-template-permille", type=int, default=20,
                    help="Per-mille of TEMPLATES held out entirely. 20 = 2%%, "
                         "~145 templates, ~50k stories.")
    ap.add_argument("--seen-val-every", type=int, default=400,
                    help="1 in N train stories diverted to val_seen_templates.")
    ap.add_argument("--min-words", type=int, default=0, help="0 disables.")
    ap.add_argument("--max-words", type=int, default=0, help="0 disables.")
    ap.add_argument("--strip-italics", action="store_true",
                    help="Remove *single-word italics*. Recommended for RAW "
                         "input; clean input has already had this done.")
    ap.add_argument("--drop-exact-dups", action="store_true",
                    help="Drop byte-identical stories. Costs ~250MB of RAM at "
                         "2.4M rows. Unnecessary on clean input.")
    ap.add_argument("--max-stories", type=int, default=0,
                    help="Stop after N kept stories. For building a dev subset.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite a non-empty out-dir.")
    args = ap.parse_args()

    files = sorted({p for pat in args.inputs for p in glob(pat, recursive=True)})
    if not files:
        raise SystemExit(f"No files matched {args.inputs}. Quote the glob.")
    print(f"{len(files)} input file(s).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        p = out_dir / f"{s}.jsonl"
        if p.exists() and p.stat().st_size > 0 and not args.force:
            raise SystemExit(f"{p} exists and is not empty. Re-run with --force.")

    stats = Counter()
    words_hist = Counter()
    templates = {s: set() for s in SPLITS}
    seen_hashes = set() if args.drop_exact_dups else None
    handles = {s: open(out_dir / f"{s}.jsonl", "w", encoding="utf-8") for s in SPLITS}

    try:
        for path in files:
            with open_maybe_gz(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    stats["rows_read"] += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        stats["bad_json"] += 1
                        continue

                    row_id, text = extract(row)
                    if not text or not text.strip():
                        # On raw input this is usually an API error row; the
                        # reason, if any, is in row["status"].
                        stats["no_text"] += 1
                        continue

                    text = text.strip()
                    if args.strip_italics:
                        stripped = _ITALIC.sub(r"\1", text)
                        if stripped != text:
                            stats["italics_stripped"] += 1
                            text = stripped

                    n_words = len(text.split())
                    if args.min_words and n_words < args.min_words:
                        stats["too_short"] += 1
                        continue
                    if args.max_words and n_words > args.max_words:
                        stats["too_long"] += 1
                        continue

                    if seen_hashes is not None:
                        h = hashlib.md5(text.lower().encode("utf-8")).digest()
                        if h in seen_hashes:
                            stats["exact_dup"] += 1
                            continue
                        seen_hashes.add(h)

                    tpl = template_of(row, row_id)
                    if bucket(tpl, 1000) < args.val_template_permille:
                        split = "val_new_templates"
                    elif row_id and bucket(row_id, args.seen_val_every) == 0:
                        split = "val_seen_templates"
                    else:
                        split = "train"

                    handles[split].write(json.dumps(
                        {"id": row_id, "template": tpl, "text": text}) + "\n")
                    stats[split] += 1
                    stats["kept"] += 1
                    templates[split].add(tpl)
                    words_hist[min(n_words // 20 * 20, 600)] += 1

                    if args.max_stories and stats["kept"] >= args.max_stories:
                        raise StopIteration
            print(f"  {path}: {stats['rows_read']:,} rows read, "
                  f"{stats['kept']:,} kept so far")
    except StopIteration:
        print(f"Stopped at --max-stories {args.max_stories:,}.")
    finally:
        for h in handles.values():
            h.close()

    print("\n--- corpus ---")
    for k in ("rows_read", "bad_json", "no_text", "italics_stripped", "too_short",
              "too_long", "exact_dup", "kept"):
        if stats[k]:
            print(f"  {k:20s} {stats[k]:>12,}")
    print()
    for s in SPLITS:
        print(f"  {s:24s} {stats[s]:>10,} stories  "
              f"{len(templates[s]):>6,} templates")

    overlap = templates["train"] & templates["val_new_templates"]
    if overlap:  # would mean the hash split is broken; loud, not silent
        print(f"\n  ERROR: {len(overlap)} template(s) in BOTH train and "
              f"val_new_templates. Validation is contaminated.")

    total_words = sum(b * n for b, n in words_hist.items())
    mean_words = total_words / max(stats["kept"], 1)
    print(f"\n  mean length ~{mean_words:.0f} words. Distribution (20-word bins):")
    peak = max(words_hist.values()) if words_hist else 1
    for b in sorted(words_hist):
        if words_hist[b] * 60 // peak or words_hist[b] > stats["kept"] // 200:
            bar = "#" * max(1, words_hist[b] * 50 // peak)
            print(f"    {b:4d}-{b+19:<4d} {words_hist[b]:>9,}  {bar}")

    (out_dir / "stats.json").write_text(json.dumps({
        "counts": {s: stats[s] for s in SPLITS},
        "templates": {s: len(templates[s]) for s in SPLITS},
        "mean_words": mean_words,
        "total_words": total_words,
    }, indent=2))
    print(f"\nWrote {out_dir}/  (train.jsonl, val_*.jsonl, stats.json)")
    print("Next: python training/train_tokenizer.py --corpus", out_dir)


if __name__ == "__main__":
    main()
