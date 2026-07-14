import random
import numpy as np
import torch

from torch.utils.data import Dataset


class TokenDataset(Dataset):

    def __init__(self, filename, context_length):

        self.data = np.memmap(
            filename,
            dtype=np.uint16,
            mode="r"
        )

        self.context_length = context_length

        self.length = len(self.data) - context_length - 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # Ignore the sequential index from the DataLoader and
        # sample a random position in the token stream.
        idx = random.randint(
            0,
            self.length - 1
        )

        x = torch.from_numpy(
            self.data[
                idx:
                idx + self.context_length
            ].astype(np.int64)
        )

        y = torch.from_numpy(
            self.data[
                idx + 1:
                idx + self.context_length + 1
            ].astype(np.int64)
        )

        return x, y
