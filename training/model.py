"""
The model. A decoder-only transformer in the nanoGPT mould, with four
substitutions that are strictly cheaper or strictly better per parameter.
Every deviation from Karpathy's nanoGPT is called out where it happens,
because the point of a 15M-parameter budget is that you get to spend every
parameter on prose rather than on scaffolding.

  nanoGPT / GPT-2            here                why
  -----------------------    -----------------   -------------------------
  learned position embed     RoPE                saves block_size*n_embd
                                                 params (197k at 512x384,
                                                 ~1.4% of the budget) and
                                                 degrades more gracefully
                                                 past the trained length
  LayerNorm                  RMSNorm             one fewer stat, no bias,
                                                 same loss in practice
  GELU MLP, 4x wide          SwiGLU, 8/3x wide   same parameter count
                                                 (3*d*h with h=8d/3 equals
                                                 8d^2), consistently lower
                                                 loss
  biases on every Linear     no biases           a few thousand parameters
                                                 that measurably do nothing

Set `legacy_gpt2=True` in the config to get the GPT-2 shapes back if you
want to A/B against the reference notebooks.

Parameter arithmetic at the default config (6 layers, 6 heads, d=384,
vocab 8192, embeddings tied to the output head):

    per layer : 4*d^2 (attention) + 3*d*(8d/3) (SwiGLU) = 12*d^2 = 1.77M
    6 layers  : 10.62M
    embeddings: 8192 * 384         =  3.15M   (tied, counted once)
    norms     : negligible
    total     : ~13.8M

Scaling knob: n_layer=7 lands at ~15.6M, n_embd=512/n_head=8/n_layer=8 at
~30M. Nothing else in the codebase needs to change.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    # 0.0 is the right default here and it is not laziness. Dropout fights
    # memorisation, and with ~850M training tokens against ~14M parameters
    # there is roughly 60x more data than model. The model cannot memorise
    # the corpus; handicapping it just slows learning.
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    legacy_gpt2: bool = False
    tie_embeddings: bool = True


class RMSNorm(nn.Module):
    """Written out rather than using nn.RMSNorm so this file runs on torch 2.1."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class LayerNorm(nn.Module):
    """GPT-2's norm, with an optional bias, for legacy_gpt2 mode."""

    def __init__(self, dim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


def build_rope_cache(block_size, head_dim, theta, device=None):
    """
    Precompute cos/sin for rotary embeddings. Shape (block_size, head_dim//2).

    Rotary encodes position by rotating each (i, i + head_dim/2) pair of
    query/key channels by an angle proportional to the token's position.
    Attention then depends on the *difference* of two positions, which is
    what you actually want a relative-position scheme to do.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(block_size, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """x is (B, n_head, T, head_dim). cos/sin are (T, head_dim//2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide evenly by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.legacy = cfg.legacy_gpt2
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # In legacy mode there is no RoPE here: GPT-2 gets its positions from a
        # learned embedding added once at the bottom of the stack (GPT.wpe), so
        # attention itself is position-blind.

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if not self.legacy:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # PyTorch dispatches this to a fused (FlashAttention / memory-efficient)
        # kernel when it can. It is both faster and far lighter on memory than
        # materialising the T x T attention matrix, which matters at block 512.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class SwiGLU(nn.Module):
    """
    Gated feed-forward. hidden = 8/3 * n_embd keeps the parameter count
    identical to GPT-2's 4x GELU MLP: 3 * d * (8d/3) == 8 * d^2.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)  # keep matmul shapes tensor-core friendly
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class GELUMLP(nn.Module):
    """GPT-2's MLP, for legacy_gpt2 mode."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        norm = (lambda d: LayerNorm(d, cfg.bias)) if cfg.legacy_gpt2 else RMSNorm
        self.ln_1 = norm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = norm(cfg.n_embd)
        self.mlp = GELUMLP(cfg) if cfg.legacy_gpt2 else SwiGLU(cfg)

    def forward(self, x, cos, sin):
        # Pre-norm residual stream: the identity path from input to output is
        # never normalised, which is what makes deep stacks trainable.
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.legacy_gpt2 else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, cfg.bias) if cfg.legacy_gpt2 else RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Input and output embeddings share one matrix. At an 8k vocab this
            # is 3.15M parameters -- 23% of the model -- that you would
            # otherwise pay for twice. It also slightly improves loss.
            self.lm_head.weight = self.wte.weight

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(cfg.block_size, head_dim, cfg.rope_theta)
        # persistent=False: derived from config, no reason to bloat checkpoints.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Residual-projection init. Every layer adds into the residual stream,
        # so without this the stream's variance grows with depth and early
        # training is unstable. Scaling the projections by 1/sqrt(2*n_layer)
        # keeps it roughly constant. (GPT-2 paper, section 2.3.)
        for name, p in self.named_parameters():
            if name.endswith(("c_proj.weight", "w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Tied weights are stored once, so a plain sum already counts them once."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            if self.wpe is not None:
                n -= self.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence of {T} exceeds block_size {self.cfg.block_size}"

        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(x)

        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss

        # Inference: only the last position can produce the next token, so
        # project just that one. At an 8k vocab this is a small saving; at
        # 50k it is the difference between fast and sluggish sampling.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Decay 2D parameters (all the matmul weights), never 1D ones (norm
        gains, biases). Decaying a norm's gain toward zero is just a slow
        way of breaking the layer.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        extra = {}
        if device_type == "cuda":
            try:  # fused AdamW is a solid few-percent win, but not everywhere
                torch.optim.AdamW(decay[:1], lr=learning_rate, fused=True)
                extra["fused"] = True
            except (RuntimeError, TypeError):
                pass
        opt = torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)
        print(f"  optimizer: {sum(p.numel() for p in decay):,} decayed / "
              f"{sum(p.numel() for p in no_decay):,} undecayed params"
              f"{' (fused)' if extra.get('fused') else ''}")
        return opt

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, eos_token_id=None):
        """
        Autoregressive sampling. Stops early once every sequence in the batch
        has emitted eos_token_id, which for this corpus means "the story
        finished on its own" -- the signal you actually want to watch.
        """
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits)

            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if eos_token_id is not None:
                # Once a sequence is done, keep feeding it EOS so the tensor
                # stays rectangular; the caller truncates at the first EOS.
                nxt = torch.where(finished.unsqueeze(1),
                                  torch.full_like(nxt, eos_token_id), nxt)
                finished |= nxt.squeeze(1) == eos_token_id
            idx = torch.cat([idx, nxt], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        return idx


PRESETS = {
    # ~13.8M parameters. The stated target for this project.
    "slm_15m": dict(n_layer=6, n_head=6, n_embd=384, block_size=512),
    # ~15.6M. Same width, one more layer, if you want the number on the tin.
    "slm_15m_deep": dict(n_layer=7, n_head=6, n_embd=384, block_size=512),
    # ~29M. Noticeably better prose; still trains overnight on one T4.
    "slm_30m": dict(n_layer=8, n_head=8, n_embd=512, block_size=512),
    # Tiny, for smoke-testing the plumbing on CPU in seconds.
    "debug": dict(n_layer=2, n_head=2, n_embd=128, block_size=128),
}


if __name__ == "__main__":
    # `python training/model.py` prints the parameter budget for each preset,
    # so you can pick a size before committing to a training run.
    for name, kw in PRESETS.items():
        m = GPT(GPTConfig(**kw))
        total = m.num_parameters()
        print(f"{name:14s} {total/1e6:6.2f}M total   "
              f"{m.num_parameters(non_embedding=True)/1e6:6.2f}M non-embedding   "
              f"{kw}")
