"""
Step 3 -- encode each split to a flat uint16 token stream on disk.

PACKING. Stories are concatenated, each followed by <|endoftext|>, into one
long array. Training then samples random block_size windows out of it. Most
windows straddle a boundary, which is the point: the model learns what an
ending looks like, what a beginning looks like, and that the token after
<|endoftext|> has nothing to do with what came before.

The alternative -- one story per padded sequence -- wastes roughly 30% of
every batch on padding at these lengths, and teaches the model less about
starting and stopping. Packing is what nanoGPT and TinyStories both do.

uint16 holds 0..65535, so any vocabulary up to 65,536 fits. The file is a
plain array with no header, which is what lets train.py np.memmap it and
read random windows without loading gigabytes into RAM.

Memory here is O(batch), not O(corpus): stories stream in, tokens stream out.

Usage:
    python training/encode.py --corpus training/corpus
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

EOT = "<|endoftext|>"
SPLITS = ("train", "val_new_templates", "val_seen_templates")


def iter_batches(path, size):
    batch = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line).get("text")
            except json.JSONDecodeError:
                continue
            if t:
                batch.append(t)
                if len(batch) >= size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="training/corpus")
    ap.add_argument("--tokenizer", default=None, help="Defaults to <corpus>/tokenizer.json")
    ap.add_argument("--batch-size", type=int, default=2000,
                    help="Stories per encode_batch call. The Rust tokenizer "
                         "parallelises inside this call across all cores.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    tok_path = Path(args.tokenizer) if args.tokenizer else corpus / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit(f"{tok_path} not found. Run train_tokenizer.py first.")

    tok = Tokenizer.from_file(str(tok_path))
    vocab = tok.get_vocab_size()
    eot_id = tok.token_to_id(EOT)
    if eot_id is None:
        raise SystemExit(f"{tok_path} has no {EOT} token.")
    if vocab > 65536:
        raise SystemExit(f"vocab {vocab} exceeds uint16. Switch the dtype to uint32.")
    print(f"tokenizer: vocab={vocab:,}  {EOT}={eot_id}")

    meta = {"vocab_size": vocab, "eot_id": eot_id,
            "tokenizer": str(tok_path.name), "splits": {}}

    for split in SPLITS:
        src = corpus / f"{split}.jsonl"
        if not src.exists():
            print(f"  {split}: missing, skipped")
            continue
        dst = corpus / f"{split}.bin"
        if dst.exists() and dst.stat().st_size > 0 and not args.force:
            raise SystemExit(f"{dst} exists. Re-run with --force.")

        n_tokens = n_stories = n_words = 0
        with open(dst, "wb") as out:
            for batch in iter_batches(src, args.batch_size):
                ids = []
                for enc in tok.encode_batch(batch):
                    ids.extend(enc.ids)
                    ids.append(eot_id)
                arr = np.asarray(ids, dtype=np.uint16)
                # Guard against a silent uint16 wrap if someone hand-edits the
                # tokenizer: a wrapped id trains the model on the wrong token.
                if arr.size and int(arr.max()) >= vocab:
                    raise SystemExit("token id out of range -- vocab mismatch")
                arr.tofile(out)
                n_tokens += arr.size
                n_stories += len(batch)
                n_words += sum(len(t.split()) for t in batch)
                if n_stories % 200_000 < args.batch_size:
                    print(f"    {split}: {n_stories:,} stories, "
                          f"{n_tokens/1e6:.1f}M tokens")

        meta["splits"][split] = {"tokens": n_tokens, "stories": n_stories}
        print(f"  {split:24s} {n_stories:>10,} stories  {n_tokens:>13,} tokens  "
              f"{n_tokens/1e6:7.1f}M  {dst.stat().st_size/1e9:5.2f} GB  "
              f"({n_tokens/max(n_words,1):.3f} tok/word)")

    (corpus / "meta.json").write_text(json.dumps(meta, indent=2))
    tr = meta["splits"].get("train", {}).get("tokens", 0)
    print(f"\nWrote {corpus}/*.bin and meta.json")
    if tr:
        # The number that sets your schedule. Everything in train.py is
        # measured in tokens, not epochs, because packed windows are sampled
        # at random and "epoch" is only ever an average here.
        print(f"One pass over train = {tr:,} tokens.")
        for bs, ga, bl in ((32, 4, 512),):
            per = bs * ga * bl
            print(f"At batch_size={bs} x grad_accum={ga} x block={bl} "
                  f"= {per:,} tokens/step, that is {tr//per:,} steps per pass.")
    print(f"\nNext: python training/train.py --corpus {corpus} --preset slm_15m")


if __name__ == "__main__":
    main()
