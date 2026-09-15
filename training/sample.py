"""
Step 5 -- generate stories from a checkpoint.

Priming with <|endoftext|> is what tells the model "a new story starts here",
because that is the only thing that ever preceded a story start in training.
Generation stops at the next <|endoftext|>, i.e. when the model decides the
story is over. A model that never emits one has not learned story length --
that is a real signal, so --num-tokens ending the story is reported.

Usage:
    python training/sample.py --ckpt training/out/slm_15m/ckpt_best.pt \
        --corpus training/corpus --num 5

    # continue a story you started
    python training/sample.py --ckpt ... --prompt '"You think you can just'

    # write them to a file for review
    python training/sample.py --ckpt ... --num 200 --out samples.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
from model import GPT, GPTConfig  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

EOT = "<|endoftext|>"


def load(ckpt_path, corpus, device, tokenizer_path=None):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    tok_path = Path(tokenizer_path) if tokenizer_path else Path(corpus) / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit(f"{tok_path} not found. Pass --tokenizer.")
    return model, Tokenizer.from_file(str(tok_path)), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", default="training/corpus")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="Below ~0.7 the prose gets repetitive and safe; above "
                         "~1.0 it loses the thread. 0.8 is a good default.")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--prompt", default=None,
                    help="Continue this text instead of starting fresh.")
    ap.add_argument("--out", default=None, help="Write JSONL here instead of stdout.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tok, ck = load(args.ckpt, args.corpus, device, args.tokenizer)
    eot = tok.token_to_id(EOT)
    print(f"{args.ckpt}: step {ck.get('step', '?'):,}, "
          f"{ck.get('tokens_seen', 0)/1e6:,.0f}M tokens, "
          f"val {ck.get('val_loss', ck.get('best_val', float('nan'))):.4f}, "
          f"{model.num_parameters()/1e6:.2f}M params, device {device}\n")

    prefix = [eot] + (tok.encode(args.prompt).ids if args.prompt else [])
    f_out = open(args.out, "w", encoding="utf-8") if args.out else None
    n_done = n_finished = 0
    total_words = 0

    while n_done < args.num:
        b = min(args.batch_size, args.num - n_done)
        idx = torch.tensor([prefix] * b, dtype=torch.long, device=device)
        out = model.generate(idx, args.max_new_tokens, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p, eos_token_id=eot)
        for row in out.tolist():
            ids = row[1:]                      # drop the priming EOT
            finished = eot in ids
            if finished:
                ids = ids[:ids.index(eot)]     # drop everything from the EOT on
                n_finished += 1
            text = tok.decode(ids).strip()
            words = len(text.split())
            total_words += words
            n_done += 1
            rec = {"n": n_done, "words": words, "finished": finished, "text": text}
            if f_out:
                f_out.write(json.dumps(rec) + "\n")
            else:
                print(f"=== story {n_done}  ({words} words, "
                      f"{'complete' if finished else 'TRUNCATED at max_new_tokens'}) ===")
                print(text)
                print()
    if f_out:
        f_out.close()
        print(f"wrote {n_done} stories to {args.out}")

    print(f"{n_finished}/{n_done} ended on their own; "
          f"mean {total_words/max(n_done,1):.0f} words "
          f"(training target was 230-260).")
    if n_finished < n_done:
        print("  Stories that never emit <|endoftext|> mean the model has not "
              "learned to end. Expected early in training; a persistent problem "
              "late on usually means too few tokens seen.")


if __name__ == "__main__":
    main()
