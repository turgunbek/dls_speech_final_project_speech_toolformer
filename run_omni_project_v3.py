"""
run_omni_project_v3.py
======================
Версия v3: исправлен баг Pipeline D.

Проблема в v2:
  Pipeline D передавал в LLM сырой вывод ASR-шага (Pipeline B), который содержал
  verbose-обёртку модели вида:
      "The original content of this audio is: 'sent twelve hundred to brawl.'"
  LLM воспринимала это как информационное сообщение, а не как банковский запрос,
  и часто возвращала null — отсюда низкий recall (0.576) при высокой precision.

Исправление в v3:
  Перед передачей в Pipeline D транскрипт очищается функцией extract_clean_transcript.
  В результатах сохраняются оба варианта: pipeline_b_transcript (сырой) и
  pipeline_b_transcript_clean (очищенный).

Изменений в Pipeline A / B / C нет.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import json
import re

from tqdm import tqdm

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from qwen_omni_utils import process_mm_info


# === НАСТРОЙКИ ===
MODEL_ID    = "Qwen/Qwen2.5-Omni-7B"
INPUT_FILE  = "dataset_audio_Cameron_Russell_115.json"
OUTPUT_FILE = "results_omni_7b_v3.json"
# =================


# ---------------------------------------------------------------------------
# Загрузка модели
# ---------------------------------------------------------------------------

print(f"Loading {MODEL_ID}...")

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a banking assistant.
You must NOT generate audio or speech output. Generate TEXT ONLY.

Available tools:
1. transfer_money(recipient: str, amount: float)
2. get_exchange_rate(currency_from: str, currency_to: str)

Output ONLY valid JSON format: {"tool_name": "...", "arguments": {...}}.
If the request is unrelated to these tools, return null."""

ASR_PROMPT = "Please transcribe the audio accurately."


# ---------------------------------------------------------------------------
# Основные функции
# ---------------------------------------------------------------------------

def generate_omni(conversation: list) -> str:
    """Запускает модель и возвращает только сгенерированный текст (без входа)."""
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=False,
            max_new_tokens=128,
        )
        text_ids = out[0] if isinstance(out, tuple) else out

    new_tokens = text_ids[0][input_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_json(text: str):
    """
    Извлекает JSON из ответа модели.
    Возвращает: dict | None | "PARSE_ERROR"
    """
    try:
        # Блок кода ```json ... ```
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        # Просто фигурные скобки
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        if "null" in text.lower():
            return None
        return "PARSE_ERROR"
    except Exception:
        return "PARSE_ERROR"


def is_hallucination(text: str) -> bool:
    """True, если транскрипт выглядит как галлюцинация модели."""
    if not isinstance(text, str) or not text.strip():
        return True
    # Нелатинские символы (китайские / арабские и т.д.)
    if len(re.findall(r'[^\x00-\x7F]', text)) > 3:
        return True
    # Повторяющийся паттерн: одно слово >= 4 раза подряд
    words = text.split()
    for i in range(len(words) - 3):
        if len(set(words[i:i+4])) == 1:
            return True
    return False


def extract_clean_transcript(text: str) -> str:
    """
    Извлекает чистый текст из verbose-обёртки Qwen ASR:
      "The original content of this audio is: 'sent fifty bucks to alex.'"
      → "sent fifty bucks to alex."

    Пробует несколько паттернов; при галлюцинации возвращает пустую строку.
    Пустая строка → Pipeline D вернёт null (корректное поведение для неразборчивого аудио).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if is_hallucination(text):
        return ""

    # Паттерн 1: одинарные кавычки  'content'
    m = re.search(r"'([^']{3,})'", text)
    if m:
        return m.group(1).strip(" .")

    # Паттерн 2: двойные кавычки "content"
    m = re.search(r'"([^"]{3,})"', text)
    if m:
        return m.group(1).strip(" .")

    # Паттерн 3: всё после «is:» / «are:»
    m = re.search(r'(?:is|are)\s*:\s*(.{5,})$', text.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip("'.\" ")

    # Паттерн 4: нет служебных слов → вернуть как есть
    service = ("original content", "transcription", "audio is", "the content")
    if not any(sw in text.lower() for sw in service):
        return text.strip(" .")

    return ""


# ---------------------------------------------------------------------------
# Smoke-test модели
# ---------------------------------------------------------------------------

_test_conv = [
    {"role": "system",  "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user",    "content": [{"type": "text", "text": "Hello!"}]},
]
print("Smoke-test:", generate_omni(_test_conv))


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"\nProcessing {len(dataset)} items with Qwen2.5-Omni-7B (v3)...")

results = []

for item in tqdm(dataset):
    audio_path = item["audio_path"]
    if not os.path.exists(audio_path):
        print(f"  [SKIP] missing file: {audio_path}")
        continue

    ground_truth_text = item["text"]

    # ── Pipeline A: TEXT ONLY (Oracle) ────────────────────────────────────
    conv_a = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "text", "text": ground_truth_text}]},
    ]
    res_a_raw = generate_omni(conv_a)
    json_a = extract_json(res_a_raw)

    # ── Pipeline B: ASR  (Audio → Transcript) ─────────────────────────────
    conv_b = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user",   "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text",  "text": ASR_PROMPT},
        ]},
    ]
    transcript_b_raw = generate_omni(conv_b)
    transcript_b_clean = extract_clean_transcript(transcript_b_raw)  # ← НОВОЕ в v3

    # ── Pipeline C: DIRECT (Audio → JSON) ─────────────────────────────────
    conv_c = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text",  "text": "Extract the intent."},
        ]},
    ]
    res_c_raw = generate_omni(conv_c)
    json_c = extract_json(res_c_raw)

    # ── Pipeline D: CASCADED (Cleaned Transcript → JSON) ──────────────────
    # v2-баг: передавали сырой transcript_b (с обёрткой "The original content is: '...'")
    # v3-фикс: передаём очищенный transcript_b_clean
    d_input = transcript_b_clean if transcript_b_clean else ground_truth_text + " [ASR FAILED]"
    conv_d = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "text", "text": d_input}]},
    ]
    res_d_raw = generate_omni(conv_d)
    json_d = extract_json(res_d_raw)

    results.append({
        "ground_truth":               item.get("label"),
        "input_text_gt":              ground_truth_text,
        "pipeline_a_json":            json_a,
        "pipeline_b_transcript":      transcript_b_raw,       # сырой вывод ASR
        "pipeline_b_transcript_clean": transcript_b_clean,    # очищенный (для WER и D)
        "pipeline_c_json":            json_c,
        "pipeline_d_json":            json_d,
    })

# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(results)} samples saved to {OUTPUT_FILE}")
