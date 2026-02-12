import argparse
import torch

from .utils import load_yaml, pick_device
from .model import GPT


def main():
    """
    Loads a checkpoint and generates text based on a prompt.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nano.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Merhaba!")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_new", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = pick_device(cfg["device"])
    print("device:", device)

    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]
    s_cfg = cfg.get("sample", {})

    temperature = args.temperature if args.temperature is not None else float(s_cfg.get("temperature", 0.9))
    max_new = args.max_new if args.max_new is not None else int(s_cfg.get("max_new_tokens", 300))

    model = GPT(
        vocab_size=m_cfg["vocab_size"],
        block_size=ds_cfg["block_size"],
        n_layer=m_cfg["n_layer"],
        n_head=m_cfg["n_head"],
        n_embd=m_cfg["n_embd"],
        dropout=0.0,
    ).to(device)

    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()

    out = model.generate(
        args.prompt.encode("utf-8"),
        max_new=max_new,
        temperature=temperature,
        device=device,
    )
    print(out.decode("utf-8", errors="ignore"))


if __name__ == "__main__":
    main()
