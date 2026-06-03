"""
FastAPI inference server for the Acoustic Perimeter Breach Detection system.

Exposes two endpoints:
    GET  /health  — confirms the server is running
    POST /predict — accepts an audio file, runs CNN inference, returns prediction

The model and normalization stats are loaded once at startup and reused across
all requests. Incoming audio is written to a temporary file, preprocessed into
a mel spectrogram, normalized, and passed through the trained CNN. The response
includes the predicted class, confidence score, breach flag, and full probability
distribution across all 8 classes.
"""
import torch
import tempfile
import os
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import threading
from contextlib import asynccontextmanager
from kafka import KafkaConsumer
import numpy as np
import asyncio
from scripts.inference import predict as run_predict, load_model, load_norm_stats
from scripts.inference import audio_to_mel
from config.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, LABELS_INVERTED


def kafka_consumer_loop(loop):

    ABNORMAL_CLASSES = {"gunshots", "chainsaw", "glass_breaking"}
    consumer = KafkaConsumer(KAFKA_TOPIC, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)

    for msg in consumer:
        chunk = np.frombuffer(msg.value, dtype=np.float32)
        mel = audio_to_mel(chunk)
        mel_norm = (mel - mean) / (std + 1e-8)
        tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        predicted_class = LABELS_INVERTED[predicted_idx]
        confidence = round(float(probs[predicted_idx]), 2) * 100
        is_breach = predicted_class in ABNORMAL_CLASSES

        result = {'confidence': confidence, "is_breach": is_breach, "predicted_class": predicted_class}
        if is_breach:
            print(f"breach Detected, type:{predicted_class}")
        else:
            print("All Normal")
        for client in connected_clients:
            asyncio.run_coroutine_threadsafe(
                client.send_json(result),
                loop  # the event loop captured at startup
            )
        

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the kafka consumer thread in bg
    loop = asyncio.get_running_loop()
    thread = threading.Thread(target=kafka_consumer_loop, args=(loop,),  daemon=True)
    thread.start()
    yield
    print("Shutdown Event!")
    # Stop the kafka consumer

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)
mean, std = load_norm_stats()

connected_clients = set() # tracks the connected clients through web sockets


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Save upload to a temp file, run inference, delete it
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = run_predict(tmp_path, model, mean, std, device)
    finally:
        os.remove(tmp_path)

    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keeps connection open
    except:
        connected_clients.remove(websocket)

