"""
Step 4 -- train the model.

Built around one assumption: THE SESSION WILL DIE BEFORE TRAINING FINISHES.
Colab free disconnects after a few hours, Kaggle caps a session at 12. So:

  * every --ckpt-interval steps the full state (model, optimizer, AMP scaler,
    step, tokens seen, RNG) is written to out/ckpt.pt
  * --resume auto picks it up and carries on, with no flags to remember
  * --max-hours exits cleanly and checkpoints BEFORE the platform kills you,
    which is the difference between resuming and starting over
  * SIGINT (the stop button) also checkpoints on the way out

Progress is measured in TOKENS, not epochs. Training samples random windows
out of a packed token stream, so "epoch" is only ever an average -- but
tokens seen is exact, and it is what learning-rate schedules and scaling
comparisons are actually a function of.

Two validation losses are tracked:
  val_new_templates   stories built from beat structures never seen  <- watch this
  val_seen_templates  held-out stories from templates it trained on
The GAP between them is the memorisation signal. See build_corpus.py.

Usage:
    python training/train.py --corpus training/corpus --preset slm_15m \
        --out-dir training/out/slm_15m --epochs 2

    # resume after a disconnect -- identical command, nothing to remember
    python training/train.py --corpus training/corpus --preset slm_15m \
        --out-dir training/out/slm_15m --epochs 2

    # Kaggle: stop and checkpoint at 11h30 so the 12h kill never lands
    ... --max-hours 11.5
"""
import argparse
import json
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent))
from model import GPT, GPTConfig, PRESETS  # noqa: E402

SPLITS = ("train", "val_new_templates", "val_seen_templates")
_stop = {"now": False}


def _handle_sigint(signum, frame):
    if _stop["now"]:
        raise KeyboardInterrupt
    print("\n[signal] finishing this step, then checkpointing. "
          "Press Ctrl-C again to abort immediately.")
    _stop["now"] = True


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested, device):
    if requested != "auto":
        return requested
    if device == "cuda":
        # T4 and P100 -- the free-tier GPUs -- are pre-Ampere and have NO
        # bf16 support. Asking for it there fails or silently falls back, so
        # check rather than assume.
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    # MPS autocast is still uneven; fp32 is slower but trustworthy, and the
    # Mac is for smoke tests anyway.
    return "float32"


class Data:
    """Random block_size windows out of a packed uint16 token stream."""

    def __init__(self, corpus, block_size, device, device_type):
        self.block_size = block_size
        self.device = device
        self.device_type = device_type
        self.paths, self.lengths = {}, {}
        for split in SPLITS:
            p = Path(corpus) / f"{split}.bin"
            if p.exists() and p.stat().st_size > 0:
                self.paths[split] = p
                self.lengths[split] = p.stat().st_size // 2  # uint16
        if "train" not in self.paths:
            raise SystemExit(f"{corpus}/train.bin not found. Run encode.py first.")

    def batch(self, split, batch_size):
        # Re-open the memmap every call. Holding one open across thousands of
        # iterations leaks page-cache references and the process grows without
        # bound -- a known nanoGPT footgun.
        data = np.memmap(self.paths[split], dtype=np.uint16, mode="r")
        hi = len(data) - self.block_size - 1
        if hi <= 0:
            raise SystemExit(f"{split}.bin is shorter than block_size.")
        ix = np.random.randint(0, hi, size=batch_size)
        x = torch.from_numpy(np.stack([data[i:i + self.block_size] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + self.block_size] for i in ix]).astype(np.int64))
        if self.device_type == "cuda":
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device)


def lr_at(step, args):
    """Linear warmup, then cosine decay to --min-lr, then flat."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(args.warmup_steps, 1)
    if step >= args.lr_decay_steps:
        return args.min_lr
    progress = (step - args.warmup_steps) / max(args.lr_decay_steps - args.warmup_steps, 1)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)


@torch.no_grad()
def evaluate(model, data, args, ctx):
    model.eval()
    out = {}
    for split in data.paths:
        if split == "train" and args.eval_iters_train == 0:
            continue
        iters = args.eval_iters_train if split == "train" else args.eval_iters
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = data.batch(split, args.batch_size)
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def show_samples(raw_model, corpus, device, n=2, max_new_tokens=400):
    """Print a story. Validation loss cannot tell you whether the prose reads."""
    tok_path = Path(corpus) / "tokenizer.json"
    if not tok_path.exists():
        return
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tok_path))
    except Exception as e:  # noqa: BLE001
        print(f"  (sampling skipped: {e})")
        return
    eot = tok.token_to_id("<|endoftext|>")
    raw_model.eval()
    idx = torch.full((n, 1), eot, dtype=torch.long, device=device)
    with torch.no_grad():
        out = raw_model.generate(idx, max_new_tokens, temperature=0.8,
                                 top_k=200, eos_token_id=eot)
    raw_model.train()
    for row in out.tolist():
        ids = row[1:]
        if eot in ids:
            ids = ids[:ids.index(eot)]
        text = tok.decode(ids)
        print(f"  --- sample ({len(text.split())} words) ---")
        print("  " + text.replace("\n", "\n  ")[:1200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="training/corpus")
    ap.add_argument("--out-dir", default="training/out/slm_15m")
    ap.add_argument("--preset", default="slm_15m", choices=sorted(PRESETS))
    # architecture overrides (all optional; preset supplies the defaults)
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--n-head", type=int, default=None)
    ap.add_argument("--n-embd", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--legacy-gpt2", action="store_true",
                    help="LayerNorm + GELU MLP + learned positions, i.e. the "
                         "nanoGPT/GPT-2 shapes, for A/B comparison.")
    ap.add_argument("--bias", action="store_true",
                    help="Biases on every Linear and LayerNorm. GPT-2 has them; "
                         "they cost a few thousand parameters and measurably do "
                         "nothing. Use with --legacy-gpt2 to match the "
                         "reference notebook exactly.")
    # optimisation
    ap.add_argument("--batch-size", type=int, default=32, help="Micro-batch per step.")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="Micro-batches per optimizer step. batch_size x "
                         "grad_accum x block_size is the real batch in tokens.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Peak LR. 1e-3 suits models this small; drop toward "
                         "6e-4 if loss spikes or goes NaN.")
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # schedule
    ap.add_argument("--epochs", type=float, default=2.0,
                    help="Passes over the train token stream. Ignored if "
                         "--max-steps is set.")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="Checkpoint and exit after this long. Set it BELOW "
                         "your platform's session limit (Kaggle 12h, Colab ~4h).")
    # bookkeeping
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--eval-iters-train", type=int, default=20)
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--sample-interval", type=int, default=2000,
                    help="0 disables mid-training sampling.")
    ap.add_argument("--ckpt-interval", type=int, default=500)
    ap.add_argument("--resume", default="auto", choices=("auto", "never"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto",
                    choices=("auto", "bfloat16", "float16", "float32"))
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = pick_device(args.device)
    device_type = "cuda" if device.startswith("cuda") else device
    dtype = pick_dtype(args.dtype, device_type)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.corpus) / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found. Run encode.py first.")
    meta = json.loads(meta_path.read_text())

    kw = dict(PRESETS[args.preset])
    for k in ("n_layer", "n_head", "n_embd", "block_size"):
        v = getattr(args, k)
        if v is not None:
            kw[k] = v
    cfg = GPTConfig(vocab_size=meta["vocab_size"], dropout=args.dropout,
                    legacy_gpt2=args.legacy_gpt2, bias=args.bias, **kw)

    data = Data(args.corpus, cfg.block_size, device, device_type)
    if "val_new_templates" not in data.paths:
        print("WARNING: no val_new_templates.bin -- the held-out-template loss, "
              "which is\n  the only number here that measures generalisation, "
              "will not be computed.\n  Best-checkpoint selection falls back to "
              "train loss, which always improves.\n  Rebuild the corpus with "
              "--val-template-permille > 0.\n")
    tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size
    train_tokens = meta["splits"]["train"]["tokens"]
    steps_per_epoch = max(1, train_tokens // tokens_per_step)
    max_steps = args.max_steps or int(args.epochs * steps_per_epoch)
    args.lr_decay_steps = max_steps  # decay across the whole run

    print(f"device={device} dtype={dtype} preset={args.preset}")
    print(f"corpus: {train_tokens:,} train tokens, "
          f"{meta['splits']['train']['stories']:,} stories, vocab {cfg.vocab_size:,}")
    print(f"schedule: {tokens_per_step:,} tokens/step "
          f"({args.batch_size} x {args.grad_accum} x {cfg.block_size}), "
          f"{steps_per_epoch:,} steps/epoch, {max_steps:,} steps total "
          f"({max_steps * tokens_per_step / 1e6:,.0f}M tokens, "
          f"{max_steps * tokens_per_step / max(train_tokens,1):.2f} epochs)")

    model = GPT(cfg).to(device)
    print(f"model: {model.num_parameters()/1e6:.2f}M parameters "
          f"({model.num_parameters(non_embedding=True)/1e6:.2f}M non-embedding)")
    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (args.beta1, args.beta2), device_type)
    # GradScaler only matters for fp16: fp16 gradients underflow to zero
    # without loss scaling. bf16 has fp32's exponent range and needs none.
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))

    step, tokens_seen, best_val = 0, 0, float("inf")
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume == "auto" and ckpt_path.exists():
        print(f"resuming from {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ck["config"]
        current = {k: getattr(cfg, k) for k in saved}
        if saved != current:
            raise SystemExit(
                f"Checkpoint architecture differs from the requested one.\n"
                f"  checkpoint: {saved}\n  requested : {current}\n"
                f"Use a different --out-dir, or match the flags.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        step, tokens_seen, best_val = ck["step"], ck["tokens_seen"], ck["best_val"]
        if ck.get("torch_rng") is not None:
            torch.set_rng_state(ck["torch_rng"].cpu().to(torch.uint8))
        print(f"  at step {step:,}, {tokens_seen/1e6:.0f}M tokens, "
              f"best val {best_val:.4f}")

    raw_model = model
    if args.compile and device_type == "cuda":
        try:
            model = torch.compile(model)
            print("torch.compile: on (first step will be slow while it warms up)")
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile unavailable ({e}); continuing eager.")

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[dtype]
    ctx = (nullcontext() if device_type != "cuda" or dtype == "float32"
           else torch.amp.autocast(device_type=device_type, dtype=amp_dtype))

    def save(path, extra=None):
        payload = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "step": step, "tokens_seen": tokens_seen, "best_val": best_val,
            "config": {k: getattr(cfg, k) for k in
                       ("vocab_size", "block_size", "n_layer", "n_head",
                        "n_embd", "bias", "rope_theta", "legacy_gpt2",
                        "tie_embeddings")},
            "args": vars(args),
            "torch_rng": torch.get_rng_state(),
        }
        if extra:
            payload.update(extra)
        tmp = Path(str(path) + ".tmp")
        torch.save(payload, tmp)
        # Atomic replace: a session killed mid-write leaves the PREVIOUS
        # checkpoint intact instead of a truncated file that loads as garbage.
        tmp.replace(path)

    log_path = out_dir / "log.csv"
    if not log_path.exists():
        log_path.write_text("step,tokens,lr,train_loss,val_new,val_seen,tok_per_s\n")

    print(f"\ntraining -> {out_dir}\n")
    model.train()
    t_start = time.time()
    t_last = t_start
    tokens_at_last_log = tokens_seen
    running = None

    while step < max_steps:
        lr = lr_at(step, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(args.grad_accum):
            x, y = data.batch("train", args.batch_size)
            with ctx:
                _, loss = model(x, y)
                # Gradients accumulate by summing, so each micro-batch must
                # contribute 1/grad_accum of the loss or the effective LR
                # scales with grad_accum.
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            total_loss += loss.item()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)  # clip real gradients, not scaled ones
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_seen += tokens_per_step
        running = total_loss if running is None else 0.9 * running + 0.1 * total_loss

        if step % args.log_interval == 0:
            now = time.time()
            tps = (tokens_seen - tokens_at_last_log) / max(now - t_last, 1e-6)
            t_last, tokens_at_last_log = now, tokens_seen
            eta = (max_steps - step) * tokens_per_step / max(tps, 1)
            print(f"step {step:>7,}/{max_steps:,}  loss {total_loss:.4f} "
                  f"(avg {running:.4f})  lr {lr:.2e}  "
                  f"{tps/1e3:,.0f}k tok/s  "
                  f"{tokens_seen/1e6:,.0f}M seen  eta {eta/3600:.1f}h")

        if step % args.eval_interval == 0 or step == max_steps:
            losses = evaluate(model, data, args, ctx)
            vnew = losses.get("val_new_templates", float("nan"))
            vseen = losses.get("val_seen_templates", float("nan"))
            gap = vseen - vnew if vnew == vnew and vseen == vseen else float("nan")
            print(f"  eval @ {step:,}: train {losses.get('train', float('nan')):.4f}  "
                  f"val_new {vnew:.4f}  val_seen {vseen:.4f}  "
                  f"gap {gap:+.4f}  (ppl_new {math.exp(min(vnew, 20)):.1f})")
            with open(log_path, "a") as f:
                f.write(f"{step},{tokens_seen},{lr:.6e},"
                        f"{losses.get('train', '')},{vnew},{vseen},"
                        f"{(tokens_seen - 0) / max(time.time() - t_start, 1):.0f}\n")
            using_val = vnew == vnew  # NaN only when the split is absent
            score = vnew if using_val else losses.get("train", float("inf"))
            if score < best_val:
                best_val = score
                save(out_dir / "ckpt_best.pt",
                     {"val_loss": score, "val_metric": "val_new_templates"
                      if using_val else "train (no val split)"})
                label = "val_new" if using_val else "train loss (NO VAL SPLIT)"
                print(f"  new best {label} {best_val:.4f} -> ckpt_best.pt")

        if args.sample_interval and step % args.sample_interval == 0:
            show_samples(raw_model, args.corpus, device)

        hit_time_limit = args.max_hours and (time.time() - t_start) > args.max_hours * 3600
        if step % args.ckpt_interval == 0 or hit_time_limit or _stop["now"]:
            save(ckpt_path)
        if hit_time_limit:
            print(f"\n--max-hours {args.max_hours} reached at step {step:,}. "
                  f"Checkpointed. Re-run the same command to continue.")
            break
        if _stop["now"]:
            print(f"\nStopped at step {step:,}. Checkpointed. "
                  f"Re-run the same command to continue.")
            break

    save(ckpt_path)
    elapsed = (time.time() - t_start) / 3600
    print(f"\ndone: step {step:,}, {tokens_seen/1e6:,.0f}M tokens, "
          f"{elapsed:.2f}h this session, best val_new {best_val:.4f}")
    print(f"  {out_dir}/ckpt.pt       latest")
    print(f"  {out_dir}/ckpt_best.pt  lowest val_new_templates loss")
    print(f"\nSample from it:\n  python training/sample.py "
          f"--ckpt {out_dir}/ckpt_best.pt --corpus {args.corpus} --num 5")


if __name__ == "__main__":
    main()
