"""
Step 2 -- train a byte-level BPE tokenizer on the training split.

WHY NOT JUST USE GPT-2's TOKENIZER. GPT-2's vocabulary is 50,257. At
n_embd=384 that is 50257 * 384 = 19.3M parameters in the embedding table
alone -- more than the entire 15M model budget, before a single transformer
block exists. You would be training a lookup table with a small transformer
attached. An 8,192-token vocabulary costs 3.15M, which is 23% of the budget:
still the single largest tensor in the model, but affordable.

WHY 8,192 AND NOT SMALLER. Shrinking the vocabulary does not make the model
cheaper for free -- it makes every story LONGER in tokens, so each epoch
costs more compute and the model must carry information across more steps.
8k is roughly where the curve flattens for single-domain English. Use
--vocab-size to try 4,096 or 16,384; the script reports tokens-per-word for
whatever you pick, which is the number that decides the trade.

WHY BYTE-LEVEL. The corpus is full of curly quotes, em dashes and ellipses
("...", "’", "“ ”") from the generator. A byte-level BPE has no unknown
token by construction: worst case it spends a few tokens, it never drops a
character or emits <unk>.

Trained on TRAIN ONLY. A tokenizer fitted on validation text is a small but
real leak -- merges tuned to held-out strings make them cheaper to predict.

Usage:
    python training/train_tokenizer.py --corpus training/corpus
    python training/train_tokenizer.py --corpus training/corpus --vocab-size 16384
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

# Marks a story boundary. Stories are packed back to back into fixed-length
# training blocks, so this token is the ONLY signal the model gets that one
# story ended and an unrelated one began. It is also the prompt you hand the
# model at sampling time to say "begin a story".
EOT = "<|endoftext|>"


def iter_texts(path, stride, limit=None):
    """Every `stride`-th story. Striding rather than head-N avoids fitting the
    tokenizer to whichever batch chunk happens to sort first."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride:
                continue
            try:
                t = json.loads(line).get("text")
            except json.JSONDecodeError:
                continue
            if t:
                n += 1
                yield t
                if limit and n >= limit:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="training/corpus")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--sample-stories", type=int, default=300_000,
                    help="Stories used to FIT the tokenizer. 300k is far more "
                         "than an 8k vocabulary needs; raising it costs time "
                         "and RAM and changes the merges barely at all.")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default=None, help="Defaults to <corpus>/tokenizer.json")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    train_path = corpus / "train.jsonl"
    if not train_path.exists():
        raise SystemExit(f"{train_path} not found. Run build_corpus.py first.")
    out_path = Path(args.out) if args.out else corpus / "tokenizer.json"

    total = 0
    stats_path = corpus / "stats.json"
    if stats_path.exists():
        total = json.loads(stats_path.read_text())["counts"].get("train", 0)
    if not total:
        with open(train_path, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
    stride = max(1, total // max(args.sample_stories, 1))
    scope = "all of them" if stride == 1 else f"every {stride}th"
    print(f"{total:,} training stories; fitting on {scope} "
          f"(~{min(total, args.sample_stories):,} stories).")

    tok = Tokenizer(models.BPE(unk_token=None))
    # add_prefix_space=False: stories start at a real word, not a space.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        # EOT is added first so it lands at id 0 -- a stable, memorable id
        # that train.py and sample.py both look up by name anyway.
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(iter_texts(train_path, stride, args.sample_stories),
                            trainer=trainer, length=min(total, args.sample_stories))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    eot_id = tok.token_to_id(EOT)
    print(f"\nSaved {out_path}  vocab={tok.get_vocab_size():,}  {EOT}={eot_id}")

    # ---- the measurement that decides your compute budget ----
    sample_words = sample_tokens = n = 0
    for text in iter_texts(train_path, max(1, total // 5000), 5000):
        sample_words += len(text.split())
        sample_tokens += len(tok.encode(text).ids) + 1  # +1 for the EOT
        n += 1
    tpw = sample_tokens / max(sample_words, 1)
    per_story = sample_tokens / max(n, 1)

    print(f"\nMeasured on {n:,} stories:")
    print(f"  {tpw:.3f} tokens/word, {per_story:.0f} tokens/story "
          f"(incl. {EOT})")
    if total:
        proj = per_story * total
        print(f"  projected TRAIN corpus: {proj/1e6:,.1f}M tokens "
              f"({proj*2/1e9:.2f} GB as uint16)")
        # A 14M-parameter model is Chinchilla-compute-optimal at ~280M tokens.
        # Small models keep improving well past that point, so more is good --
        # but a subset far below it is a plumbing test, not a training run.
        ratio = proj / 280e6
        if ratio >= 1.0:
            print(f"  = {ratio:.1f}x the ~280M tokens that are Chinchilla-optimal "
                  f"for 14M params.")
            print(f"  Over-training a small model well past compute-optimal is "
                  f"the right call:\n  it is how TinyStories-class models get "
                  f"fluent. Plan 1-2 passes.")
        else:
            print(f"  = only {ratio:.2f}x the ~280M tokens a 14M-param model "
                  f"wants. Fine for a\n  smoke test; expect incoherent output. "
                  f"Point --inputs at the full corpus\n  for a real run.")

    print("\nSample round-trip:")
    demo = next(iter_texts(train_path, max(1, total // 3), 1))[:180]
    ids = tok.encode(demo).ids
    print(f"  in  : {demo!r}")
    print(f"  ids : {ids[:24]}{' ...' if len(ids) > 24 else ''}")
    rt = tok.decode(ids)
    print(f"  out : {rt!r}")
    print(f"  lossless: {rt == demo}")
    print(f"\nNext: python training/encode.py --corpus {corpus}")


if __name__ == "__main__":
    main()
