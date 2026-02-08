import json
import os
import torch
from TTS.api import TTS
from tqdm import tqdm

# === НАСТРОЙКИ ===
INPUT_JSON = "generated_dataset.json"   # Файл от ChatGPT со сгенерированными примерами в формате json
OUTPUT_JSON = "dataset_audio_Cameron_Russell_115.json"      # Результат
OUTPUT_FOLDER = "wavs_Cameron_Russell_115"  # Куда кладем аудио
SPEAKER_WAV = "Cameron_Russell-chunk-38.wav"  # Референс голоса
# =================

# 1. Проверяем наличие GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 2. Создаем папку для аудио
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 3. Загружаем модель XTTS v2
print("Loading XTTS v2 model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 4. Читаем датасет
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Found {len(data)} examples. Starting generation...")

# 5. Генерируем аудио
processed_data = []

for i, item in tqdm(enumerate(data), total=len(data)):
    text = item['text']
    filename = f"sample_{i:04d}.wav"
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    # Генерация
    # language="en" - т.к. мы решили делать на английском
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=SPEAKER_WAV,
            language="en",
            file_path=filepath,
            speed=1.15,
            split_sentences=False
        )

        # Обновляем запись (добавляем путь к аудио)
        # Важно сохранить абсолютный путь или относительный, как удобно для обучения
        # Для локальных тестов лучше относительный
        item['audio_path'] = filepath
        processed_data.append(item)

    except Exception as e:
        print(f"\nError generating sample {i}: {e}")

# 6. Сохраняем итоговый манифест
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)

print(f"\nDone! Saved {len(processed_data)} items to {OUTPUT_JSON}")
