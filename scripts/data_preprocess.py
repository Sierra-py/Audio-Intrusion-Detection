"""Preprocess the raw data
Converts the raw audio signals into numpy arrays.
Make an array for each category and also for the whole data as X, y
"""

"""
This scripts converts the raw data in to the processed data
"""
import os
import numpy as np
import librosa
from config.config import SR, WINDOW_SIZE, HOP_LENGTH, N_FFT, N_MELS, OVERLAP, CONTINUOUS, IMPULSIVE, LABELS, WINDOW_SAMPLES



def to_melspectrogram(window):
    """ 
    Returns a 2-D mel spectrogram converted to db
    """
    mel = librosa.feature.melspectrogram(
        y=window,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def process_continuous(y):
    """Slide windows with 50% overlap, no trimming."""
    samples = []
    step = int(WINDOW_SAMPLES * (1 - OVERLAP))
    
    for start in range(0, len(y) - WINDOW_SAMPLES + 1, step):
        window = y[start:start + WINDOW_SAMPLES]
        mel = to_melspectrogram(window)
        samples.append(mel)
    
    return samples

def augment_window(window):
    """Adds augmented audio sample by adding background noise to original"""
    augmented = [window]  # keeps original
    
    # Add background noise
    noise = np.random.normal(0, 0.005, len(window)).astype(np.float32)
    augmented.append(window + noise)
    return augmented

def process_impulsive(y):
    """Trim silence, center window around peak energy."""
    # Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    if len(y_trimmed) < WINDOW_SAMPLES:
        # Pad if trimmed audio is shorter than window
        y_trimmed = np.pad(y_trimmed, (0, WINDOW_SAMPLES - len(y_trimmed)))
        windows = augment_window(y_trimmed[:WINDOW_SAMPLES])
        return [to_melspectrogram(w) for w in windows]
    
    # Find peak energy frame
    energy = np.array([
        np.sum(y_trimmed[i:i+1024]**2)
        for i in range(0, len(y_trimmed) - 1024, 512)
    ])
    peak_frame = np.argmax(energy)
    peak_sample = peak_frame * 512
    
    # Center window around peak
    start = max(0, peak_sample - WINDOW_SAMPLES // 2)
    end = start + WINDOW_SAMPLES
    
    # Adjust if window goes out of bounds
    if end > len(y_trimmed):
        end = len(y_trimmed)
        start = max(0, end - WINDOW_SAMPLES)
    
    window = y_trimmed[start:end]
    
    if len(window) < WINDOW_SAMPLES:
        window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))
    
    windows = augment_window(window)
    return [to_melspectrogram(w) for w in windows]

def process_folder(folder_path, category):
    """Processes all samples of one folder and return list of mel spectrograms of each folder"""
    all_mels = []
    files = [f for f in os.listdir(folder_path) if f.endswith(('.mp3', '.wav', '.ogg'))]
    is_continuous = category in CONTINUOUS
    
    for i, fname in enumerate(files):
        print(f"  {i+1}/{len(files)}: {fname}", end='\r')
        try:
            y, _ = librosa.load(os.path.join(folder_path, fname), sr=SR, mono=True)
            
            if is_continuous:
                mels = process_continuous(y)
            else:
                mels = process_impulsive(y)
            
            all_mels.extend(mels)
        
        except Exception as e:
            print(f"\n  Error on {fname}: {e}")
    
    print(f"\n  {category}: {len(all_mels)} windows extracted")
    return all_mels

def preprocess_all(raw_dir="data/raw", save_dir="data/processed"):
    """Saves mels into disk as numpy arrays"""
    os.makedirs(save_dir, exist_ok=True)
    
    X, y = [], []
    
    for category, label in LABELS.items():
        folder = os.path.join(raw_dir, category)
        if not os.path.exists(folder):
            print(f"Skipping {category} - folder not found")
            continue
        
        print(f"Processing: {category}")
        mels = process_folder(folder, category)
        
        category_dir = os.path.join(save_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        np.save(os.path.join(category_dir, "mels.npy"), np.array(mels))

        X.extend(mels)
        y.extend([label] * len(mels))
    
    X = np.array(X)
    y = np.array(y)
    
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)
    print(f"\nSaved to {save_dir}/X.npy and {save_dir}/y.npy")
    
    return X, y

if __name__ == "__main__":
    X, y = preprocess_all()