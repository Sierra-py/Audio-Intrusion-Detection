import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score
)
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, VISUALIZATIONS_DIR,
    LABELS, LABELS_INVERTED, BATCH_SIZE, TEST_SPLIT, VAL_SPLIT,
    RANDOM_SEED, NUM_CLASSES, DROPOUT
)
from src.model import AcousticCNN
from src.dataset import AcousticDataset


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load data & reproduce same split ---
    X = np.load(PROCESSED_DATA_DIR / "X.npy")
    y = np.load(PROCESSED_DATA_DIR / "y.npy")

    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    idx_train, _ = train_test_split(
        idx_trainval, test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        stratify=y[idx_trainval], random_state=RANDOM_SEED
    )

    # Normalize using training stats (same as train.py)
    norm_stats = np.load(MODELS_DIR / "norm_stats.npy")
    mean, std = norm_stats[0], norm_stats[1]
    X_norm = (X - mean) / (std + 1e-8)

    test_dataset = AcousticDataset(X_norm[idx_test], y[idx_test])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Load best checkpoint ---
    model = AcousticCNN(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pt", map_location=device))
    model.eval()

    # --- Run inference on test set ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Metrics ---
    class_names = [LABELS_INVERTED[i] for i in range(NUM_CLASSES)]

    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    overall_acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")
    print()
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # --- Confusion matrix ---
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_ylabel("True label")
    axes[0].set_xlabel("Predicted label")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalized)")
    axes[1].set_ylabel("True label")
    axes[1].set_xlabel("Predicted label")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save_path = VISUALIZATIONS_DIR / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


if __name__ == "__main__":
    evaluate()