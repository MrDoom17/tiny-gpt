# A tiny GPT, from scratch

`gpt.py` is a complete, working language model in about 250 lines of PyTorch. It has the same architecture as GPT-2/3 (a decoder-only transformer), just very small, so you can read every line and train it on a laptop CPU.

## Run it

```bash
pip install torch
python gpt.py train                                  # trains on input.txt, writes model.pt
python gpt.py sample --tokens 500                    # generate text
python gpt.py sample --prompt "ROMEO:" --tokens 300  # continue a prompt
```

`input.txt` is Tiny Shakespeare (about 1 MB of plays). Swap in any plain-text file to train on something else.

## How it works

| Piece | What it does | Where in `gpt.py` |
|---|---|---|
| Tokenizer | Maps each character to an integer (65 of them for Shakespeare). Real LLMs use BPE subword tokens instead. | `CharTokenizer` |
| Embeddings | Turn each token id and its position into a vector. | `GPT.tok_emb`, `GPT.pos_emb` |
| Self-attention | Each token looks at earlier tokens and mixes in what is relevant. The causal mask stops it seeing the future. | `CausalSelfAttention` |
| MLP | A small per-token neural network applied after attention. | `MLP` |
| Transformer block | Attention plus MLP, with residual connections and LayerNorm. Stacked 4 times. | `Block` |
| Output head | Scores every possible next token. | `GPT.head` |
| Training | Show the model random 128-character chunks and minimise cross-entropy on predicting each next character. | `train()` |
| Sampling | Predict a distribution, sample a character, append it, repeat. `--temperature` controls randomness. | `GPT.generate()` |

The model has about 1.8M parameters. For scale, GPT-2 small has 124M and modern frontier models have hundreds of billions; the code would look almost the same.

## Making it better

- Train longer: `python gpt.py train --iters 10000`.
- Make it bigger: raise `N_LAYER`, `N_EMBD`, `BLOCK_SIZE` at the top of the file. Do this on a GPU, the script picks one up automatically.
- Use a real tokenizer: swap `CharTokenizer` for a BPE tokenizer such as `tiktoken`'s GPT-2 encoding, which lets the model work in words rather than letters.
- More data: a few hundred MB of text (e.g. OpenWebText) is where it starts producing coherent sentences.

The natural next step after this is Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), which this file is modelled on.
