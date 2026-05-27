import numpy as np
import torch
import librosa

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    MODELS_DIR, LABELS_INVERTED, NUM_CLASSES, DROPOUT,
    SR, N_MELS, N_FFT, HOP_LENGTH, WINDOW_SAMPLES, TOP_DB
)
from src.model import AcousticCNN


ABNORMAL_CLASSES = {"gunshots", "glass_breaking", "chainsaw"}


def load_model(device):
    model = AcousticCNN(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pt", map_location=device))
    model.eval()
    return model


def load_norm_stats():
    stats = np.load(MODELS_DIR / "norm_stats.npy")
    return stats[0], stats[1]  # mean, std


def audio_to_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel, ref=np.max)


def extract_window(y: np.ndarray) -> np.ndarray:
    """
    Extracts a 3-second window from raw audio.
    Mirrors preprocessing: peak-energy centering for impulsive sounds,
    but at inference we don't know the class — so we always use peak-energy.
    If the audio is shorter than 3s, pad it.
    """
    if len(y) <= WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
        return y

    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)

    if len(y_trimmed) <= WINDOW_SAMPLES:
        y_trimmed = np.pad(y_trimmed, (0, WINDOW_SAMPLES - len(y_trimmed)))
        return y_trimmed

    # Center on peak energy
    energy = np.array([
        np.sum(y_trimmed[i:i+1024]**2)
        for i in range(0, len(y_trimmed) - 1024, 512)
    ])
    peak_sample = np.argmax(energy) * 512
    start = max(0, peak_sample - WINDOW_SAMPLES // 2)
    end = start + WINDOW_SAMPLES

    if end > len(y_trimmed):
        end = len(y_trimmed)
        start = max(0, end - WINDOW_SAMPLES)

    window = y_trimmed[start:end]
    if len(window) < WINDOW_SAMPLES:
        window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))

    return window


def predict(audio_path: str, model, mean: float, std: float, device) -> dict:
    """
    Loads an audio file, preprocesses it, and returns prediction.
    """
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    window = extract_window(y)
    mel = audio_to_mel(window)

    # Normalize with training stats
    mel_norm = (mel - mean) / (std + 1e-8)

    # Shape: (1, 1, 128, 130) — batch=1, channel=1
    tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted_idx = int(np.argmax(probs))
    predicted_class = LABELS_INVERTED[predicted_idx]
    confidence = float(probs[predicted_idx])
    is_breach = predicted_class in ABNORMAL_CLASSES

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "is_breach": is_breach,
        "probabilities": {LABELS_INVERTED[i]: float(p) for i, p in enumerate(probs)}
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scripts/inference.py <path_to_audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device)
    mean, std = load_norm_stats()

    result = predict(audio_path, model, mean, std, device)

    print(f"\nFile: {audio_path}")
    print(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%} confidence)")
    print(f"Breach detected: {result['is_breach']}")
    print("\nAll probabilities:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<15} {prob:.4f}  {bar}")