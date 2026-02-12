import os
import numpy as np
import torch


def read_bin(path: str):
    """
    Reads a binary file using numpy memmap (uint8 read-only).
    """
    assert os.path.exists(path), f"Not found: {path}"
    return np.memmap(path, dtype=np.uint8, mode="r")


class BinDataset:
    """
    Byte-level LM dataset:
      x: (B,T) current bytes
      y: (B,T) next bytes
    """

    def __init__(self, bin_path: str, block_size: int, batch_size: int, device: str):
        self.data = read_bin(bin_path)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self):
        """
        Samples a random batch of blocks from the dataset.
        Returns:
            x (B, T): inputs
            y (B, T): targets (shifted by 1)
        """
        n = len(self.data)
        T = self.block_size
        B = self.batch_size

        ix = np.random.randint(0, n - (T + 1), size=(B,))
        x = np.stack([self.data[i : i + T] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + T + 1] for i in ix]).astype(np.int64)

        x = torch.from_numpy(x).to(self.device)
        y = torch.from_numpy(y).to(self.device)
        return x, y
