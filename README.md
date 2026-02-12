# celebi-nanogpt

**Nano byte‑level GPT from scratch** — a lightweight, hackable training pipeline for quick prototyping.

- ✅ **No tokenizer needed**: byte‑level tokens (0..255) via UTF‑8 support all languages.
- ✅ **Pretraining + sampling**: a self‑contained training loop in a single repo.
- ✅ **Apple Silicon friendly**: automatically picks **mps → cuda → cpu** as available.

> This repository covers the **Nano** stage: the core model with pretraining and sampling only.  
> Later stages will live in separate repos: **celebi‑tinygpt** (Chat SFT) → **RAG**.

---

## Installation

```bash
pip install -r requirements.txt
```

## Generate sample data (OSCAR)

```bash
python scripts/download_oscar_sample.py --langs tr en --docs_per_lang 3000 --out_dir data
```

> **Note**: OSCAR might be gated or temporarily suspended on Hugging Face. If you have access, the above works. Otherwise, we recommend using Wikipedia:

### Recommended (non-gated): Wikipedia

```bash
python scripts/download_wikipedia_sample.py --langs tr en --docs_per_lang 3000 --out_dir data
```

This command produces:
- `data/train.bin`
- `data/val.bin`

## Pretraining

```bash
python -m src.train --config configs/nano.yaml
```

Checkpoints are saved to:
- `out/nano_pretrain/ckpt_step*.pt`

## Generate text (Sampling)

> If your shell shows `dquote>`, it means your quotes are unbalanced. Use **single quotes** in zsh.

Works on Apple Silicon (MPS):

```bash
python -m src.sample --config configs/nano.yaml --ckpt out/nano_pretrain/ckpt_step20000.pt --prompt 'Hello!'
```

Or with more parameters:

```bash
python -m src.sample --config configs/nano.yaml --ckpt out/nano_pretrain/ckpt_step20000.pt --prompt 'Hello!' --temperature 0.6 --max_new 200
```

For more consistent and guiding responses, try using a prompt like this:

```bash
python -m src.sample --config configs/nano.yaml --ckpt out/nano_pretrain/ckpt_step20000.pt --prompt 'Hello! This article is about' --temperature 0.6 --max_new 200
```

---

## Repository structure

```
configs/   # configuration files
scripts/   # data helper scripts
src/       # model, training, and sampling
out/       # checkpoints and outputs (ignored by git)
data/      # train/val bin files (ignored by git)
```

---

## Roadmap

- **celebi‑nanogpt** (this repo): pretraining + sampling ✅
- **celebi‑tinygpt**: chat SFT + interactive conversation
- **celebi‑tinygpt (v1)**: RAG (document‑grounded answers)

---

## License

MIT (see `LICENSE`).
