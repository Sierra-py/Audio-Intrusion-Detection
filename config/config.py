import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

# Load .env
load_dotenv()

# Paths from .env
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models/checkpoints"))
VISUALIZATIONS_DIR = Path(os.getenv("VISUALIZATIONS_DIR", "visualizations"))
FREESOUND_API_KEY = os.getenv("FREESOUND_API_KEY")

# Load yaml
_config_path = Path(__file__).parent / "config.yaml"
with open(_config_path, "r") as f:
    _cfg = yaml.safe_load(f)

# Audio
SR = _cfg["audio"]["sample_rate"]
WINDOW_SIZE = _cfg["audio"]["window_size"]
WINDOW_SAMPLES = SR * WINDOW_SIZE
OVERLAP = _cfg["audio"]["overlap"]
N_MELS = _cfg["audio"]["n_mels"]
N_FFT = _cfg["audio"]["n_fft"]
HOP_LENGTH = _cfg["audio"]["hop_length"]
TOP_DB = _cfg["audio"]["top_db"]
NOISE_FACTOR = _cfg["augmentation"]["noise_factor"]

# Labels
LABELS = _cfg["labels"]
LABELS_INVERTED = {v: k for k, v in LABELS.items()}
CONTINUOUS = _cfg["categories"]["continuous"]
IMPULSIVE = _cfg["categories"]["impulsive"]

# Training
BATCH_SIZE = _cfg["training"]["batch_size"]
LEARNING_RATE = _cfg["training"]["learning_rate"]
EPOCHS = _cfg["training"]["epochs"]
EARLY_STOPPING_PATIENCE = _cfg["training"]["early_stopping_patience"]
VAL_SPLIT = _cfg["training"]["val_split"]
TEST_SPLIT = _cfg["training"]["test_split"]
RANDOM_SEED = _cfg["training"]["random_seed"]

# Model
FILTERS = _cfg["model"]["filters"]
DROPOUT = _cfg["model"]["dropout"]
NUM_CLASSES = _cfg["model"]["num_classes"]

#Kafka
KAFKA_BOOTSTRAP_SERVERS = _cfg["kafka"]["bootstrap_servers"]
KAFKA_TOPIC = _cfg["kafka"]["topic"]