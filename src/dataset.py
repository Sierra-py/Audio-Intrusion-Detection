import numpy as np
import torch
from torch.utils.data import Dataset


class AcousticDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, 128, 130), y: (N,)
        # Add channel dim and convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 128, 130)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    dataset = AcousticDataset(X, y)
    print(f"Dataset size: {len(dataset)}")

    sample_x, sample_y = dataset[0]
    print(f"Sample X shape: {sample_x.shape}")  # expect (1, 128, 130)
    print(f"Sample y: {sample_y}")
    print(f"X dtype: {sample_x.dtype}")
    print(f"y dtype: {sample_y.dtype}")