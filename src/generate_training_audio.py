"""
generate_training_audio.py
==========================
XTTS-v2 синтез аудио для тренировочных данных (Audio SFT).
Берёт первые N_SAMPLES примеров из data/training_data_text.json
и синтезирует .wav для каждого.

Запуск (из корня проекта, в conda activate xtts_env):
    python src/generate_training_audio.py

Выход:
    data/wavs_training/sample_XXXX.wav   (~1500 файлов)
    data/training_audio_manifest.json
"""

import json
import os
import torch
from TTS.api import TTS
from tqdm import tqdm

# === НАСТРОЙКИ ===
INPUT_JSON      = "data/training_data_text.json"
OUTPUT_JSON     = "data/training_audio_manifest.json"
OUTPUT_FOLDER   = "data/wavs_training"
SPEAKER_WAV     = "data/Cameron_Russell-chunk-38.wav"
N_SAMPLES       = 1500          # сколько примеров синтезировать (subset из 3000)
RANDOM_SEED     = 123           # для воспроизводимости выбора subset
TTS_SPEED       = 1.15
# =================

import random
rng = random.Random(RANDOM_SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading XTTS v2 model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    all_data = json.load(f)

# Берём stratified subset: соблюдаем пропорцию positive/negative
positives = [x for x in all_data if x["label"] is not None]
negatives = [x for x in all_data if x["label"] is None]
rng.shuffle(positives)
rng.shuffle(negatives)

n_pos = int(N_SAMPLES * 0.8)
n_neg = N_SAMPLES - n_pos
data = positives[:n_pos] + negatives[:n_neg]
rng.shuffle(data)

print(f"Synthesizing {len(data)} examples ({n_pos} positive, {n_neg} negative)...")

processed = []

for i, item in tqdm(enumerate(data), total=len(data)):
    text = item["text"]
    filename = f"sample_{i:04d}.wav"
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=SPEAKER_WAV,
            language="en",
            file_path=filepath,
            speed=TTS_SPEED,
            split_sentences=False,
        )
        processed.append({
            "text":       text,
            "label":      item["label"],
            "audio_path": filepath,
        })
    except Exception as e:
        print(f"\n[SKIP] Error on sample {i}: {e}")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(processed, f, indent=2, ensure_ascii=False)

print(f"\nDone! Saved {len(processed)} items to {OUTPUT_JSON}")
