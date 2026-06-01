
import time
import random
import librosa
import numpy as np
from pathlib import Path
from kafka import KafkaProducer

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import SR, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC

def get_random_audio_file(raw_dir="data/raw"):
    all_files = list(Path(raw_dir).rglob("*.mp3"))
    return random.choice(all_files)

def stream_audio(filepath):
    y, _ = librosa.load(filepath, sr=SR, mono=True)
    window_samples = SR  # 1 second

    producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)

    print(f"Streaming: {filepath}")
    chunks = 0

    for start in range(0, y.shape[0] - window_samples + 1, window_samples):
        chunk = y[start:start + window_samples]
        producer.send(KAFKA_TOPIC, chunk.tobytes())
        producer.flush()
        chunks += 1
        print(f"  Sent chunk {chunks} ({start/SR:.1f}s - {(start+window_samples)/SR:.1f}s)")
        time.sleep(1)

    print(f"Done. {chunks} chunks sent.")


if __name__ == "__main__":
    while True:
        filepath = get_random_audio_file()
        stream_audio(filepath)