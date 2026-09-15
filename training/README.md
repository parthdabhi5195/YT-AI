# Training a ~15M-parameter story model

Five scripts, run in order. Each one prints the command for the next.

```bash
pip install -r training/requirements.txt

python training/build_corpus.py   --inputs 'data/clean/final/stories.jsonl' --out-dir training/corpus
python training/train_tokenizer.py --corpus training/corpus
python training/encode.py          --corpus training/corpus
python training/train.py           --corpus training/corpus --preset slm_15m \
                                   --out-dir training/out/slm_15m --epochs 2
python training/sample.py          --ckpt training/out/slm_15m/ckpt_best.pt \
                                   --corpus training/corpus --num 10
```

`python training/model.py` prints the parameter budget of every preset without
training anything.

---

## The numbers this is built around

| | |
|---|---|
| Stories | ~2.45M (after cleaning + salvage) |
| Mean length | ~265 words |
| Tokens at an 8k vocab | **~850M** (measure it — `train_tokenizer.py` reports the real figure) |
| Model | 13.8M params (6 layers, 6 heads, d=384, block 512) |
| Chinchilla-optimal for 13.8M | ~280M tokens |
| So you have | **~3x more data than compute-optimal** |

That last row is the single most important fact about this project, and it is
good news. Compute-optimal is the point where, on a *fixed compute budget*,
you should stop training and build a bigger model. You do not have a fixed
compute budget — you have a fixed model size and free data. Small models keep
getting better well past compute-optimal; that is exactly how TinyStories-class
models get fluent. **Plan one to two full passes, not a fraction of one.**

It also means dropout should stay at 0. With 60x more tokens than parameters
the model cannot memorise the corpus, so regularisation only slows it down.

---

## Where to train

**Kaggle, not free Colab.** 12-hour sessions against Colab's ~4, roughly 30
GPU-hours a week, and Kaggle Datasets gives you persistent storage for the
token files. Colab free will disconnect mid-run and you will spend the project
babysitting it.

Rough single-GPU throughput for this model, and what one pass over 850M tokens
costs:

| GPU | tok/s (est.) | one pass |
|---|---|---|
| T4 (Kaggle / Colab free) | 40–80k | 3–6 h |
| P100 (Kaggle) | 40–70k | 3–6 h |
| A100 (Colab Pro) | 250–500k | 0.5–1 h |
| M-series MPS (your Mac) | 10–20k | 12–24 h |

Don't trust the table — `train.py` prints real tok/s within the first minute.
Multiply it out and set `--max-hours` accordingly.

**Move the `.bin` files, not the JSONL.** Run `build_corpus.py`,
`train_tokenizer.py` and `encode.py` locally on the Mac (all CPU, ~20–40 min
for the full corpus, well within 24 GB). That turns ~4 GB of JSONL into
~1.7 GB of `train.bin` plus a 400 KB tokenizer. Upload *that* as a private
Kaggle Dataset. Re-encoding on Kaggle every session wastes half your GPU hours.

**Python.** Your local 3.14 has no PyTorch wheels yet. Make a 3.11 or 3.12 venv
for training; Kaggle and Colab are already on 3.11.

Kaggle's 2×T4 option only gets used by one GPU here — `train.py` is
single-device. Multi-GPU DDP roughly halves wall-clock but adds a launcher and
a failure mode; at 3–6 hours a pass it is not worth it yet.

---

## Start before the data is clean

`build_corpus.py` reads **raw Vertex batch output** as well as cleaned rows.
So you can validate this entire stack today against one 50k chunk, while the
cleaning pass is still pending in the other chat:

```bash
python training/build_corpus.py \
    --inputs 'data/batch_results/production/chunk_0000/*.jsonl' \
    --out-dir training/corpus_dev \
    --strip-italics --min-words 150 --max-words 450
python training/train_tokenizer.py --corpus training/corpus_dev --vocab-size 8192
python training/encode.py --corpus training/corpus_dev
python training/train.py --corpus training/corpus_dev --preset slm_15m \
    --out-dir training/out/dev --max-steps 2000
```

Raw mode skips deduplication and validation entirely, so the output will be
incoherent — that is fine. You are checking that the loss falls, the
checkpoints resume, and sampling produces words. Swap in the clean corpus for
the real run.

---

## Decisions, and why

**8,192-token vocabulary, trained from scratch.** GPT-2's 50,257-token vocab
would cost `50257 x 384 = 19.3M` parameters in the embedding table alone —
more than the entire model budget. At 8k it costs 3.15M, still the largest
tensor in the model. Shrinking further is not free: fewer tokens means *longer*
sequences, so more compute per story. 8k is about where that trade flattens
for single-domain English.

**Split train/validation by template, never by story.** Each of the ~7,270
templates backs ~357 stories that share a beat sequence, an emotional arc, and
a narrative voice. Hold out random *stories* and every held-out story has 356
near-siblings in training — validation loss then measures recall and will look
excellent while the model has learned nothing transferable. `build_corpus.py`
holds out whole templates by hashing the template id, and additionally tracks
a second validation set of stories held out from *training* templates.

**Watch the gap between them.** `val_seen_templates` minus `val_new_templates`
is your memorisation signal. Near zero means the model learned English
narrative prose. Growing means it is learning 7,270 beat skeletons and
reciting them — which is the specific risk this dataset carries, given 45% of
source stories are school-related and each template repeats ~357 times.

**Stories are packed, not padded.** Concatenated with `<|endoftext|>` between
them, then sliced into random 512-token windows. Padding each story to 512
would waste ~30% of every batch. Packing also teaches the model what an ending
looks like and that the token after `<|endoftext|>` is unrelated to what came
before — which is what makes `sample.py` able to prompt with `<|endoftext|>`
and get a fresh story.

**block_size 512.** A story is ~350 tokens, so a 512-token window usually holds
one complete story plus the tail of the last. Dropping to 384 would save maybe
15% compute; 512 lets the model see whole-story structure. Keep it.

**No metadata conditioning in v1.** Training on raw `story_text` only. Adding
`<|hook:X|>` control tokens is a real option, but it spends capacity and adds a
sampling-time contract, and you should find out what the plain model does
first. `build_corpus.py` already carries `template` through, so adding it later
is a small change.

**Four departures from nanoGPT**, all documented inline in `model.py`: RoPE
instead of learned position embeddings (saves 197k params, ~1.4% of budget),
RMSNorm instead of LayerNorm, SwiGLU instead of the 4x GELU MLP (identical
parameter count, reliably lower loss), and no biases.

**You can have the plain GPT-2 architecture instead.** `--legacy-gpt2 --bias`
gives you LayerNorm, the 4x GELU MLP, learned position embeddings and biases --
i.e. exactly what nanoGPT and the reference notebook use:

```bash
python training/train.py --corpus training/corpus --preset slm_15m \
    --legacy-gpt2 --bias --out-dir training/out/legacy --epochs 2
```

At the same vocabulary and block size the two are within 1.6% on parameter
count (13.77M vs 13.99M) and have an identical 10.6M-parameter block stack.
The architecture is a small effect. What is *not* small, and what the flags
above deliberately do not change, is the 8k vocabulary, block_size 512, the
`<|endoftext|>` separator, the by-template split and the token budget. Those
are the decisions that matter; pick either architecture under them.

---

## Running it

```bash
python training/train.py --corpus training/corpus --preset slm_15m \
    --out-dir training/out/slm_15m --epochs 2 --max-hours 11.5
```

Re-running the **identical command** after a disconnect resumes from the last
checkpoint. There are no extra flags to remember — that is the whole design.
`--max-hours` checkpoints and exits *before* the platform kills the session,
which is the difference between resuming and starting over. Set it below your
limit: 11.5 on Kaggle, 3.5 on Colab free. Ctrl-C also checkpoints on the way out.

Defaults: batch 32 x grad-accum 4 x block 512 = **65,536 tokens per step**, so
~13,000 steps per pass over 850M tokens. Peak LR 1e-3 with 500 warmup steps and
cosine decay to 1e-4. If loss spikes or goes NaN, drop `--lr` toward 6e-4 first.
If you run out of GPU memory, halve `--batch-size` and double `--grad-accum` —
the tokens-per-step figure, and therefore the schedule, stays identical.

### What to watch

1. **`val_new` falling.** Steeply at first, then flattening. The absolute value
   is not comparable to anyone else's model because it depends on your
   vocabulary. To compare against published numbers, convert to bits per word:
   `bpw = loss x tokens_per_word / ln(2)`, using the tokens/word that
   `train_tokenizer.py` printed.
2. **The `gap` column staying near zero.** See above.
3. **The samples.** `--sample-interval 2000` prints stories mid-run. Loss will
   keep improving long after the prose stops improving, and prose is the
   product. Read them.
4. **`sample.py`'s "ended on their own" count.** A model that never emits
   `<|endoftext|>` has not learned how long a story is.

---

## If the output is not good enough

In the order I would try them:

1. **Train longer.** A second full pass is cheap and usually the biggest win.
2. **Go to `--preset slm_30m`** (29.9M params, 8 layers, d=512). With 850M
   tokens you have the data to support it; it costs roughly 2x the compute and
   nothing else in the pipeline changes. Parameter count is the main quality
   lever you have left, and at 30M you are still well past compute-optimal.
3. **Check the data before blaming the model.** If samples are fluent but
   formulaic, that is the template skew, not the architecture — no amount of
   training fixes it, and it is a data-pipeline problem.
4. Then tune LR / vocab size. These are the small knobs; do them last.

---

## Files

| | |
|---|---|
| `model.py` | architecture; run it directly to print parameter budgets |
| `build_corpus.py` | JSONL (clean **or** raw batch output) → train/val splits |
| `train_tokenizer.py` | 8k byte-level BPE, reports tokens/word and corpus size |
| `encode.py` | splits → packed `uint16` token streams |
| `train.py` | training loop, resumable, time-limited |
| `sample.py` | generate stories from a checkpoint |

`training/corpus/` and `training/out/` are gitignored — they are gigabytes.

## Not yet read

The reference notebook (`colab.research.google.com/drive/1k4G3G5MxYLxawmPfAknUN7dbbmyqldQv`)
is behind a Google login and could not be fetched. To compare its choices
against these: in Colab, File → Download → Download .ipynb, and save it here as
`training/reference_tinystories.ipynb`.
