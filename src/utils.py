import os
import time
import math
import random
import yaml
import torch


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg
    # Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    # CUDA
    if torch.cuda.is_available():
        return "cuda"
    # CPU
    return "cpu"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def cosine_lr(step: int, max_steps: int, base_lr: float, min_ratio: float = 0.1) -> float:
    min_lr = base_lr * min_ratio
    t = step / max(1, max_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def s(self) -> float:
        return time.time() - self.t0


@torch.no_grad()
def estimate_loss(model, get_batch_fn, eval_batches: int) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch_fn()
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))
