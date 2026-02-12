import os
import argparse
import torch
from tqdm import tqdm

from .utils import load_yaml, set_seed, pick_device, ensure_dir, cosine_lr, estimate_loss, Timer
from .data import BinDataset
from .model import GPT


def save_ckpt(out_dir: str, step: int, model, opt, cfg: dict):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"ckpt_step{step}.pt")
    torch.save(
        {"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg},
        path,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nano.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    device = pick_device(cfg["device"])
    print("device:", device)

    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]
    t_cfg = cfg["train"]

    train_ds = BinDataset(ds_cfg["train_bin"], ds_cfg["block_size"], ds_cfg["batch_size"], device)
    val_ds = BinDataset(ds_cfg["val_bin"], ds_cfg["block_size"], ds_cfg["batch_size"], device)

    model = GPT(
        vocab_size=m_cfg["vocab_size"],
        block_size=ds_cfg["block_size"],
        n_layer=m_cfg["n_layer"],
        n_head=m_cfg["n_head"],
        n_embd=m_cfg["n_embd"],
        dropout=m_cfg["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(t_cfg["lr"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )

    out_dir = t_cfg["out_dir"]
    timer = Timer()

    max_steps = int(t_cfg["max_steps"])
    pbar = tqdm(range(1, max_steps + 1), desc="pretrain")

    for step in pbar:
        lr = cosine_lr(step, max_steps, float(t_cfg["lr"]))
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = train_ds.get_batch()
        _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(t_cfg["grad_clip"]))
        opt.step()

        if step % 20 == 0:
            pbar.set_postfix(loss=float(loss.item()), lr=lr, sec=round(timer.s(), 1))

        if step % int(t_cfg["eval_every"]) == 0:
            tr = estimate_loss(model, train_ds.get_batch, int(t_cfg["eval_batches"]))
            va = estimate_loss(model, val_ds.get_batch, int(t_cfg["eval_batches"]))
            print(f"\n[eval] step {step} | train {tr:.4f} | val {va:.4f}")

        if step % int(t_cfg["save_every"]) == 0:
            save_ckpt(out_dir, step, model, opt, cfg)

    save_ckpt(out_dir, max_steps, model, opt, cfg)
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
