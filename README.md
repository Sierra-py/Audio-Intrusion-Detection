# Acoustic Perimeter Intrusion Detection System

A real-time, end-to-end machine learning system that detects security breaches by classifying environmental audio. Trained on 8 sound categories, deployed as a streaming pipeline with a live dashboard.

---

## Live Demo

🔗 **Dashboard:** [Live Frontend](https://audio-intrusion-detection.vercel.app/)  
🎥 **Demo Video:** [Watch on YouTube/Loom](https://your-video-link)  
⚙️ **Backend Health:** [API Status](https://audio-intrusion-detection-production.up.railway.app/health)

---

## How It Works

![Dashboard GIF]("C:\Users\sys_a\Videos\audio-intrusion-dashboard-demo-live-gif-hd.gif")

The system runs as a continuous pipeline across three separate infrastructure components:

**1. Edge Device (`virtual_edge.py`)**  
Simulates a physical microphone at the perimeter. Picks random audio files, slices them into 1-second chunks, and streams them to a cloud Kafka broker. In a real deployment this would run on embedded hardware (Raspberry Pi or similar) at the site — decoupled from the backend so the stream continues regardless of backend state.

**2. Backend (Railway)**  
FastAPI server running a background Kafka consumer thread. Every incoming chunk is converted to a mel spectrogram, normalized, and passed through the trained CNN. The result — predicted class, confidence, breach flag — is broadcast in real time to all connected WebSocket clients.

**3. Dashboard (Vercel)**  
React frontend connected via WebSocket. Updates live with every prediction. Green = perimeter secure. Red = breach detected. Keeps a scrollable log of the last 20 events.

```
Edge Device (local/embedded)
    → Redpanda Cloud (Kafka broker)
        → Railway (FastAPI + CNN inference)
            → WebSocket
                → Vercel (React dashboard)
```

### Breach vs Normal

| Class | Type | Action |
|---|---|---|
| Wind | Normal | ✅ Secure |
| Rain | Normal | ✅ Secure |
| Birds | Normal | ✅ Secure |
| Crickets | Normal | ✅ Secure |
| Thunder | Normal | ✅ Secure |
| Gunshots | **Breach** | 🚨 Alert |
| Glass Breaking | **Breach** | 🚨 Alert |
| Chainsaw | **Breach** | 🚨 Alert |

### Dashboard Preview

![Secure State](https://your-secure-screenshot-link)
*Normal state — perimeter secure*

![Breach State](https://your-breach-screenshot-link)
*Breach detected — red alert with event log*

---

## The Problem

Perimeter security systems typically rely on cameras or motion sensors — both have blind spots and require line of sight. Audio-based detection works in the dark, through obstacles, and can distinguish *what* triggered the alert, not just *that* something did.

This system classifies environmental audio in real time, separating normal farmhouse sounds (wind, rain, birds, crickets, thunder) from breach events (gunshots, chainsaw, glass breaking), and streams predictions to a live dashboard.

---

## Architecture

```
virtual_edge.py → Kafka (Redpanda Cloud) → FastAPI backend → WebSocket → React dashboard
```

A simulated edge device streams 1-second audio chunks to a cloud Kafka broker. The FastAPI backend consumes the stream, runs CNN inference on each chunk, and broadcasts results to connected WebSocket clients. The React frontend displays predictions in real time.

---

## Stack

| Layer | Technology |
|---|---|
| ML Model | Custom CNN — PyTorch |
| Audio Processing | Librosa |
| Streaming | Apache Kafka (Redpanda Cloud) |
| Backend | FastAPI, WebSockets |
| Frontend | React, Tailwind CSS |
| Deployment | Docker, Railway (backend), Vercel (frontend) |

---

## Model

**Architecture:** 4-block CNN — each block is Conv2d → BatchNorm → ReLU → MaxPool, followed by AdaptiveAvgPool and a single Linear classifier head.

Output: 8-class softmax

~1.3M parameters


**Input:** Mel spectrograms (128 × 130) converted from 3-second audio windows at 22050 Hz.

**Training:**
- 3,007 samples across 8 classes
- Adam optimizer, CrossEntropyLoss with inverse-frequency class weights
- Early stopping (patience=10), stopped at epoch 35, best checkpoint at epoch 25

**Results:**

```
Overall Accuracy:  87.83%
Macro F1:          0.88

Notable per-class results:
  glass_breaking   98% recall
  chainsaw         96% recall
  thunder          92% recall  ← zero confusion with gunshots ✓
  gunshots         87% recall  ← misses classified as glass_breaking (still a breach) ✓
```

---

## Design Decisions

**Why these 8 classes?**
The class set was designed deliberately. Thunder was included specifically to prevent gunshot false positives — they share impulsive acoustic characteristics. Car sounds and voices were excluded because their threat level depends on context (a car at 3am vs midday), which audio alone cannot determine.

**Two preprocessing paths**
Continuous sounds (wind, rain, birds) use overlapping sliding windows. Impulsive sounds (gunshots, glass breaking, chainsaw) use peak-energy centering after silence trimming. Applying silence trimming to wind would destroy the signal — the two-path logic prevents this.

**Why one linear layer?**
The CNN extracts rich spatial features from the spectrogram. A single linear layer is sufficient for the classification step — additional layers would add parameters without improving the feature extraction that matters.

**Why Kafka?**
A direct HTTP POST from the edge device to the backend creates tight coupling — if the backend is down, data is lost. Kafka decouples the producer and consumer. The edge device streams regardless of backend state, and the backend processes when it's ready. This mirrors real production IoT architectures.

**Config architecture**
Three-layer config: `.env` for secrets, `config.yaml` for hyperparameters, `config.py` as the loader. Keeps secrets out of version control, hyperparameters readable and editable without touching code.

---

## Project Structure

```
acoustic-perimeter/
├── config/              # config.yaml + config.py loader
├── data/                # raw + processed audio (gitignored)
├── scripts/
│   ├── data_collection.py      # Freesound API downloader
│   ├── data_preprocess.py      # mel spectrogram extraction
│   ├── train.py                # model training
│   ├── evaluate.py             # test set evaluation + confusion matrix
│   ├── inference.py            # single-file prediction
│   └── virtual_edge.py         # simulated edge microphone device
├── src/
│   ├── model.py                # CNN architecture
│   ├── dataset.py              # PyTorch Dataset
│   └── utils.py                # class weights, normalization
├── backend/
│   ├── main.py                 # FastAPI server
│   └── Dockerfile
├── frontend/                   # React dashboard
└── models/checkpoints/         # best_model.pt + norm_stats.npy
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- A Freesound API key (for data collection)
- A Kafka/Redpanda cluster
- Docker

### Environment Variables

Create a `.env` file in the project root:

```env
FREESOUND_API_KEY=your_key_here
REDPANDA_USERNAME=your_username
REDPANDA_PASSWORD=your_password

# Optional overrides (defaults shown)
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
MODELS_DIR=models/checkpoints
VISUALIZATIONS_DIR=visualizations
```

## Running Locally

**Backend:**
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

**Edge device (streams audio to Kafka):**
```bash
python scripts/virtual_edge.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` — dashboard updates in real time as audio streams through the pipeline.

---

## Data

Audio collected from the Freesound API using similarity-based search (`similar_to` parameter) rather than keyword filtering, which produced more acoustically consistent results. 

Preprocessing produces 3,007 mel spectrogram windows across 8 classes. Raw audio and processed data are excluded from the repository due to size.

---

## Pipeline

### 1. Collect Data

Downloads ~150 audio samples per class from Freesound using similarity search:

```bash
python scripts/data_collection.py
```

### 2. Preprocess

Converts raw `.mp3` files into normalized mel spectrograms saved as numpy arrays:

```bash
python scripts/data_preprocess.py
```

Outputs `data/processed/X.npy` and `data/processed/y.npy`.

### 3. Train

```bash
python scripts/train.py
```

Trains with stratified train/val/test splits, class-weighted cross-entropy loss, Adam optimizer, and early stopping. Saves the best checkpoint to `models/checkpoints/best_model.pt` along with normalization stats.

### 4. Evaluate

```bash
python scripts/evaluate.py
```

Runs inference on the held-out test set and prints accuracy, macro F1, and a per-class classification report. Saves a confusion matrix to `visualizations/`.

### 5. Single File Inference

```bash
python scripts/inference.py path/to/audio.mp3
```

---

