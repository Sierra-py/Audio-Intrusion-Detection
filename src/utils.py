import numpy as np
import torch


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Computes inverse-frequency class weights for CrossEntropyLoss.
    Minority classes get higher weight so the loss penalizes missing them more.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize so weights sum to num_classes
    return torch.tensor(weights, dtype=torch.float32)


def normalize(X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Standardizes X to zero mean and unit variance across the entire dataset.
    Returns normalized X, mean, and std so you can apply the same transform at inference.
    """
    mean = X.mean()
    std = X.std()
    return (X - mean) / (std + 1e-8), mean, std


if __name__ == "__main__":
    import numpy as np
    from config.config import LABELS

    y = np.load("data/processed/y.npy")
    X = np.load("data/processed/X.npy")

    weights = compute_class_weights(y, num_classes=len(LABELS))
    print("Class weights:")
    from config.config import LABELS_INVERTED
    for i, w in enumerate(weights):
        print(f"  {LABELS_INVERTED[i]:<15} {w:.4f}")

    X_norm, mean, std = normalize(X)
    print(f"\nBefore: mean={X.mean():.4f}, std={X.std():.4f}")
    print(f"After:  mean={X_norm.mean():.4f}, std={X_norm.std():.4f}")
    print(f"Saved mean={mean:.4f}, std={std:.4f} (needed at inference)")