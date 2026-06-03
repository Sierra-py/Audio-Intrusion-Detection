
import time
import random
import librosa
import numpy as np
from pathlib import Path
from kafka import KafkaProducer

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import SR, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, KAFKA_SASL_MECHANISM, KAFKA_SECURITY_PROTOCOL, KAFKA_USERNAME, KAFKA_PASSWORD

def get_random_audio_file(raw_dir="data/raw"):
    all_files = list(Path(raw_dir).rglob("*.mp3"))
    return random.choice(all_files)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    security_protocol=KAFKA_SECURITY_PROTOCOL,
    sasl_mechanism=KAFKA_SASL_MECHANISM,
    sasl_plain_username=KAFKA_USERNAME,
    sasl_plain_password=KAFKA_PASSWORD,
)

def stream_audio(filepath):
    y, _ = librosa.load(filepath, sr=SR, mono=True)
    window_samples = SR  # 1 second

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
        try:
            filepath = get_random_audio_file()
            stream_audio(filepath)
        except Exception as e:
            print(f"Error: {e} — retrying in 5s")
            time.sleep(5)