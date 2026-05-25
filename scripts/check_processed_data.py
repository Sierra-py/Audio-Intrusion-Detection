""" Checks sample distributiobn of the processed data"""
import numpy as np
from config.config import LABELS

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

print(f"X shape: {X.shape}")
print(f"X dtype: {X.dtype}")
print(f"X value range: {X.min():.2f} to {X.max():.2f}")

print("\nSamples per category:")
for name, label in LABELS.items():
    count = np.sum(y == label)
    print(f"  {name}: {count}")