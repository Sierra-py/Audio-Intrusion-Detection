import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LABELS,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, EARLY_STOPPING_PATIENCE,
    VAL_SPLIT, TEST_SPLIT, RANDOM_SEED, NUM_CLASSES, DROPOUT
)
from src.model import AcousticCNN
from src.dataset import AcousticDataset
from src.utils import compute_class_weights, normalize


def train():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load & normalize ---
    X = np.load(PROCESSED_DATA_DIR / "X.npy")
    y = np.load(PROCESSED_DATA_DIR / "y.npy")

    # Split indices before normalizing to avoid data leakage
    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        stratify=y[idx_trainval], random_state=RANDOM_SEED
    )

    # Normalize using training set stats only
    X_train_raw = X[idx_train]
    mean = X_train_raw.mean()
    std = X_train_raw.std()
    X_norm = (X - mean) / (std + 1e-8)

    # Save normalization stats alongside checkpoint
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / "norm_stats.npy", np.array([mean, std]))
    print(f"Norm stats saved — mean: {mean:.4f}, std: {std:.4f}")

    # --- Datasets & loaders ---
    train_dataset = AcousticDataset(X_norm[idx_train], y[idx_train])
    val_dataset   = AcousticDataset(X_norm[idx_val],   y[idx_val])
    test_dataset  = AcousticDataset(X_norm[idx_test],  y[idx_test])

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, loss, optimizer ---
    model = AcousticCNN(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)

    class_weights = compute_class_weights(y[idx_train], num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training loop ---
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        train_loss, train_correct = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()

        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)

        # Validate
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()

        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)

        print(f"Epoch {epoch:>3}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        # Early stopping & checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pt")
            print(f"  ✓ Saved checkpoint (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} — no improvement for {EARLY_STOPPING_PATIENCE} epochs.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to {MODELS_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    train()