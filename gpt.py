"""
A tiny GPT-style language model, built from scratch in PyTorch.

Everything an LLM needs is in this one file:
  1. Tokenizer   - turns text into integers and back (character level here)
  2. Model       - a decoder-only transformer (the same shape as GPT)
  3. Training    - learn to predict the next token
  4. Sampling    - generate new text one token at a time

Usage:
  python gpt.py train                 # train on input.txt, save model.pt
  python gpt.py sample --tokens 500   # load model.pt and generate text
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hyperparameters. Small enough to train on a laptop CPU in a few minutes.
# Scale these up (and use a GPU) for better results.
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128     # context length: how many tokens the model sees at once
N_EMBD = 192         # size of each token's vector
N_HEAD = 6           # attention heads per layer (N_EMBD must divide evenly)
N_LAYER = 4          # number of transformer blocks stacked
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_ITERS = 3000
EVAL_EVERY = 250

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)


# ---------------------------------------------------------------------------
# 1. Tokenizer
# Real LLMs use subword tokenizers (BPE) with ~50k-200k tokens. A character
# tokenizer is the simplest possible version of the same idea.
# ---------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Each token looks back at earlier tokens and pulls in what's relevant.

    "Causal" means a token can only attend to positions before it, so the
    model can't cheat by peeking at the token it's supposed to predict.
    """

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)  # query, key, value in one go
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
        # Lower-triangular mask: position t may see positions 0..t.
        mask = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        self.register_buffer("mask", mask.view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMBD, dim=2)
        # Split channels into heads: (B, T, C) -> (B, heads, T, head_size)
        hs = C // N_HEAD
        q = q.view(B, T, N_HEAD, hs).transpose(1, 2)
        k = k.view(B, T, N_HEAD, hs).transpose(1, 2)
        v = v.view(B, T, N_HEAD, hs).transpose(1, 2)

        # Attention scores: how much each token should look at each other one.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v                                       # weighted sum of values
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    """Per-token feed-forward network: where most of the 'thinking' happens."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention (communicate) then MLP (compute).

    The `x + ...` residual connections and LayerNorms are what let deep
    stacks of these train stably.
    """

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)   # what the token is
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)   # where it sits
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)         # scores for next token

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy: how surprised the model is by the real next token.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling: predict a token, append it, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]          # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # only the last position matters
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------
def get_batch(data):
    """Random chunks of text. Targets are the inputs shifted by one token."""
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, iters=50):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.tensor([model(*get_batch(data))[1].item() for _ in range(iters)])
        out[name] = losses.mean().item()
    model.train()
    return out


def train(args):
    text = open(args.data, encoding="utf-8").read()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))  # hold out 10% to check we're not memorizing
    train_data, val_data = data[:split], data[split:]

    model = GPT(tok.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{len(text):,} characters, vocab {tok.vocab_size}, model {n_params/1e6:.2f}M params, device {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Cosine learning-rate decay: big steps early, small careful steps late.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=LEARNING_RATE / 10)

    start = time.time()
    for step in range(args.iters + 1):
        if step % EVAL_EVERY == 0 or step == args.iters:
            l = estimate_loss(model, train_data, val_data)
            print(f"step {step:5d} | train loss {l['train']:.3f} | val loss {l['val']:.3f} | {time.time()-start:.0f}s", flush=True)
            # Checkpoint as we go, so an interrupted run still leaves a usable model.
            torch.save({"model": model.state_dict(), "chars": tok.chars}, args.out)

        if step == args.iters:
            break
        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    print(f"saved {args.out}")


# ---------------------------------------------------------------------------
# 4. Sampling
# ---------------------------------------------------------------------------
def sample(args):
    ckpt = torch.load(args.out, map_location=device)
    tok = CharTokenizer("".join(ckpt["chars"]))
    model = GPT(tok.vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = args.prompt or "\n"
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "sample"])
    p.add_argument("--data", default="input.txt")
    p.add_argument("--out", default="model.pt")
    p.add_argument("--iters", type=int, default=MAX_ITERS)
    p.add_argument("--prompt", default="")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    args = p.parse_args()
    train(args) if args.mode == "train" else sample(args)
