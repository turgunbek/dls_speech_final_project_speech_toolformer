import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import torch
import json

from tqdm import tqdm
import re

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor
)

from qwen_omni_utils import process_mm_info


# === НАСТРОЙКИ ===
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
INPUT_FILE = "dataset_audio_Cameron_Russell_115.json"
OUTPUT_FILE = "results_omni_7b_v2.json"
DEVICE = torch.device('cuda:0')
# =================

print(f"Loading {MODEL_ID}...")

# 1. Загрузка модели
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto"
)

processor = Qwen2_5OmniProcessor.from_pretrained(
    MODEL_ID
)


# Системный промпт
SYSTEM_PROMPT = """You are a banking assistant.
You must NOT generate audio or speech output. Generate TEXT ONLY.

Available tools:
1. transfer_money(recipient: str, amount: float)
2. get_exchange_rate(currency_from: str, currency_to: str)

Output ONLY valid JSON format: {"tool_name": "...", "arguments": {...}}.
If unrelated, return null."""

ASR_PROMPT = "Please transcribe the audio accurately."

def generate_omni(conversation, generate_audio=False):
    # 1. Текст (промпт)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    # 2. Аудио
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    
    # 3. Токенизация
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False
    )
    inputs = inputs.to(model.device).to(model.dtype)
    
    # Запоминаем длину входа, чтобы потом отрезать
    input_len = inputs.input_ids.shape[1]
    
    # 4. Генерация
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            use_audio_in_video=False, 
            return_audio=False, 
            max_new_tokens=128
        )
        
        if isinstance(out, tuple):
            text_ids = out[0]
        else:
            text_ids = out

    # 5. Декодирование (ОТРЕЗАЕМ ВХОД)
    # Берем токены только начиная с input_len
    new_tokens = text_ids[0][input_len:]
    response_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response_text.strip()

def extract_json(text):
    try:
        # 1. Пытаемся найти блок кода ```json ... ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
            
        # 2. Если нет, ищем просто фигурные скобки
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
            
        if "null" in text.lower(): return None
        return "PARSE_ERROR"
    except:
        return "PARSE_ERROR"

# -------------------
conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Hello!"}]
    }
]

print(generate_omni(conversation))
# --------------------

# === ЗАПУСК ===

with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

results = []
print(f"Processing {len(data)} items using Qwen2.5-Omni...")

for item in tqdm(data):
    audio_path = item['audio_path']
    if not os.path.exists(audio_path):
        print(f"Skipping missing file: {audio_path}")
        continue
        
    ground_truth_text = item['text']
    
    # --- PIPELINE A: TEXT ONLY (Oracle) ---
    # Подаем только текст
    conv_a = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": ground_truth_text}]}
    ]
    res_a_raw = generate_omni(conv_a)
    json_a = extract_json(res_a_raw)
    
    # --- PIPELINE B: ASR (Audio -> Transcript) ---
    conv_b = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": ASR_PROMPT},
        ]},
    ]
    transcript_b = generate_omni(conv_b)
    
    # --- PIPELINE C: DIRECT (Audio -> JSON) ---
    conv_c = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": "Extract the intent."},
        ]},
    ]
    res_c_raw = generate_omni(conv_c)
    json_c = extract_json(res_c_raw)
    
    # --- PIPELINE D: CASCADED (Generated Transcript -> JSON) ---
    conv_d = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": transcript_b}]} # Используем выход шага B
    ]
    res_d_raw = generate_omni(conv_d)
    json_d = extract_json(res_d_raw)
    
    results.append({
        "ground_truth": item.get('label'),
        "input_text_gt": ground_truth_text,
        "pipeline_a_json": json_a,
        "pipeline_b_transcript": transcript_b,
        "pipeline_c_json": json_c,
        "pipeline_d_json": json_d
    })

# Сохранение
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Done. Saved to {OUTPUT_FILE}")
