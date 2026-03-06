# Plan: Улучшение результатов — Audio SFT v2 + word2num WER

## Статус входной точки

| | Baseline | Text SFT | Audio SFT v1 |
|--|--|--|--|
| A accuracy | 98.83% | **99.61%** | 98.83% (= baseline → merge failed) |
| C accuracy | 68.30% | 67.51% | 68.30% |
| D accuracy | 67.51% | **70.06%** | 67.51% |
| B WER | 66.74% | 67.16% | 66.74% |

**Что делаем:**
1. Audio SFT v2 — исправить merge, 5 эпох, LR 5e-5, новый скрипт + новые результаты
2. word2num — нормализация чисел при подсчёте WER, снизить 66.74% → ~40–50%

---

## Часть 1: Audio SFT v2

### Гипотеза о причине провала v1

После `model.thinker = model.thinker.merge_and_unload()` Pipeline A должен давать **99.61%** (text SFT) до начала audio-обучения. Но eval показал **98.83%** (baseline) — значит, merge не применился к весам, которые потом использовались при generate().

**Диагностика (до переобучения):** добавить в начало скрипта eval на 5 примерах до старта epoch 1. Если A ≈ 99.6% → merge работает. Если A ≈ 98.8% → merge не работает.

### 1.1 Изменения в скрипте

Создать **`src/finetune_audio_v2.py`** на базе `finetune_audio.py` со следующими изменениями:

#### Изменение 1: LR и эпохи

```python
NUM_EPOCHS    = 5      # было 2
LEARNING_RATE = 5e-5   # было 1e-4
OUTPUT_DIR    = "checkpoints/qwen_omni_audio_sft_v2"
```

#### Изменение 2: Диагностика merge — добавить функцию quick_eval и вызов после merge

Добавить после строки `model.thinker = model.thinker.merge_and_unload()` в `main()`:

```python
# === MERGE DIAGNOSTICS ===
logger.info(f"[DIAG] type(model.thinker) after merge = {type(model.thinker).__name__}")
# Если PeftModel — merge не сработал!

# Quick eval на 3 примерах из eval set (pipeline A — text input)
import random
from pathlib import Path
eval_manifest = "data/dataset_audio_Cameron_Russell_115.json"
if Path(eval_manifest).exists():
    with open(eval_manifest, "r") as f:
        eval_data_all = json.load(f)
    eval_sample = random.sample(eval_data_all, min(5, len(eval_data_all)))
    model.eval()
    correct = 0
    with torch.no_grad():
        for ex in eval_sample:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",   "content": [{"type": "text", "text": ex["text"]}]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, return_tensors="pt").to(next(model.parameters()).device)
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            resp = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            logger.info(f"  [DIAG] pred={resp[:80]!r}  gt={json.dumps(ex['label'])[:60]}")
    model.train()
logger.info("[DIAG] Quick eval done — check output above to verify merge.")
# === END DIAGNOSTICS ===
```

#### Изменение 3: Исправить _save_lora — прописать base_model_name_or_path

```python
def _save_lora(thinker, lora_cfg, output_dir, processor=None, base_model_id=None):
    """Сохраняет LoRA-веса тинкера в формате PeftModel.from_pretrained."""
    os.makedirs(output_dir, exist_ok=True)
    # Явно прописываем правильный base model id, чтобы loader мог найти модель
    if base_model_id is not None:
        lora_cfg.base_model_name_or_path = base_model_id
    lora_cfg.save_pretrained(output_dir)   # adapter_config.json
    lora_sd = {k: v.cpu() for k, v in thinker.state_dict().items() if "lora_" in k}
    torch.save(lora_sd, os.path.join(output_dir, "adapter_model.bin"))
    if processor is not None:
        processor.save_pretrained(output_dir)
```

И вызывать с `base_model_id`:

```python
# Вместо:
_save_lora(model.thinker, lora_cfg, OUTPUT_DIR, processor)
# Использовать:
_save_lora(model.thinker, lora_cfg, OUTPUT_DIR, processor, base_model_id=MODEL_ID)
# То же для checkpoint saves:
_save_lora(model.thinker, lora_cfg, ckpt_dir, base_model_id=MODEL_ID)
```

#### Изменение 4 (если merge не работает): альтернативный вариант

Если диагностика показала, что `type(model.thinker)` после merge всё ещё `PeftModel`,
пробуем явный merge через state_dict:

```python
# Вместо merge_and_unload(), сделать вручную:
if os.path.isdir(TEXT_SFT_ADAPTER):
    logger.info("Loading text SFT adapter...")
    peft_thinker = PeftModel.from_pretrained(model.thinker, TEXT_SFT_ADAPTER)
    logger.info("Merging manually via merge_adapter()...")
    peft_thinker.merge_adapter()  # in-place merge без создания нового объекта
    # Теперь peft_thinker.base_model содержит merged weights
    # Присваиваем thinker напрямую через базовую модель:
    model.thinker = peft_thinker.base_model.model
    logger.info(f"[DIAG] type after manual merge: {type(model.thinker).__name__}")
```

### 1.2 Команды на сервере (L40s, omni_env)

```bash
conda activate omni_env
cd ~/speech_toolformer   # путь к проекту на сервере

# Скопировать finetune_audio.py → finetune_audio_v2.py и внести правки (см. 1.1)
cp src/finetune_audio.py src/finetune_audio_v2.py
# ... внести изменения из раздела 1.1 ...

# Запустить обучение (5 эпох, ~3-5 часов на L40s)
python src/finetune_audio_v2.py 2>&1 | tee logs/audio_sft_v2.log
# Проверить: DIAG output в начале лога.

# После обучения — eval
python src/run_omni_project_v3.py \
    --model_path checkpoints/qwen_omni_audio_sft_v2 \
    --output results/results_audio_sft_v2.json \
    2>&1 | tee logs/eval_audio_sft_v2.log

# Метрики
python src/calculate_metrics.py results/results_audio_sft_v2.json
# → results/results_audio_sft_v2_metrics.txt
```

### 1.3 Ожидаемый результат

- Если merge сработал + 5 эпох: Pipeline C **70–75%** (оптимистично)
- Если merge всё ещё не работает: результаты ≈ v1 (но с лог-диагностикой будет ясна причина)
- В любом случае: добавляем v2 в отчёт как попытку с анализом

---

## Часть 2: word2num — нормализация WER

### Проблема

WER = 66.74% потому что:
- Referece: `"send 1500 dollars to Alice"` (цифры)
- ASR hyp: `"send fifteen hundred dollars to Alice"` (слова)
- WER считает это как 2 ошибки из 5 слов

### Решение

Добавить в `calculate_metrics.py` конвертацию числовых слов → цифры.

#### Установка библиотеки

```bash
conda activate omni_env
pip install word2number
```

#### Правка calculate_metrics.py

Добавить функцию `normalize_numbers_in_text` **после** существующей `normalize_text_for_wer`:

```python
def normalize_numbers_in_text(text: str) -> str:
    """
    Конвертирует числовые слова в цифры для честного WER.
    Примеры: "fifteen hundred" → "1500", "fifty" → "50",
             "one thousand two hundred" → "1200".
    Использует word2number; фразы находятся жадным поиском.
    """
    try:
        from word2number import w2n
    except ImportError:
        return text  # если нет библиотеки — не меняем

    # Список «числовых» слов для определения границ фраз
    NUM_WORDS = {
        "zero","one","two","three","four","five","six","seven","eight","nine","ten",
        "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
        "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
        "eighty","ninety","hundred","thousand","million","billion","a",
    }

    words = text.split()
    result = []
    i = 0
    while i < len(words):
        # Пробуем найти максимально длинную числовую фразу начиная с i
        best_end = i  # по умолчанию — одно слово
        best_val = None
        if words[i].lower() in NUM_WORDS:
            for j in range(len(words), i, -1):
                phrase = " ".join(words[i:j])
                try:
                    val = w2n.word_to_num(phrase)
                    best_val = str(val)
                    best_end = j
                    break
                except ValueError:
                    pass
        if best_val is not None:
            result.append(best_val)
            i = best_end
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)
```

Затем в `normalize_text_for_wer` добавить вызов:

```python
def normalize_text_for_wer(text: str, normalize_numbers: bool = False) -> str:
    """Нормализует текст для подсчёта WER: lowercase, без пунктуации."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\w\s]", "", text).lower()
    text = " ".join(text.split())
    if normalize_numbers:
        text = normalize_numbers_in_text(text)
    return text
```

И добавить в скрипт второй проход WER с нормализацией (без изменения оригинального):

```python
# === WER с нормализацией чисел ===
wer_refs_norm = [normalize_text_for_wer(r, normalize_numbers=True) for r in wer_refs]
wer_hyps_norm = [normalize_text_for_wer(h, normalize_numbers=True) for h in wer_hyps]
final_wer_norm = compute_wer(wer_refs_norm, wer_hyps_norm) if wer_refs_norm else float("nan")
```

И добавить в вывод:

```python
f"  WER (raw):          {final_wer:.2%}",
f"  WER (word2num):     {final_wer_norm:.2%}  ← digit↔word normalized",
```

> **Важно:** оба WER считаются по тем же подмножествам (без галлюцинаций).
> Оригинальные файлы результатов не меняются — только метрики-скрипт.

### Команды на сервере

```bash
pip install word2number
# Правки в calculate_metrics.py (см. выше)

# Пересчитать метрики для всех файлов:
python src/calculate_metrics.py results/results_omni_7b_v3.json
python src/calculate_metrics.py results/results_text_sft.json
python src/calculate_metrics.py results/results_audio_sft.json
# (и v2 если сделан): python src/calculate_metrics.py results/results_audio_sft_v2.json
```

---

## Часть 3: Обновление документации после экспериментов

После получения результатов:

### Что обновить в `report.md`

- **Section 6 (WER)**: добавить строку "WER (word2num normalized): XX%" с объяснением
- **Section 10.4 (Audio SFT)**: добавить результаты v2:
  - Если улучшение: обновить таблицу 10.2 с новым столбцом Audio SFT v2
  - Если снова нет: дополнить секцию "Root causes" с диагностикой merge

### Что обновить в `README.md`

- Таблица "После LoRA Fine-tuning": добавить строку WER (word2num)
- Если Audio SFT v2 дал улучшение: обновить таблицу

### Новые файлы результатов (сохранить старые!)

```
results/results_audio_sft_v2.json          ← новый
results/results_audio_sft_v2_metrics.txt   ← новый
```

Старые `results_audio_sft.json` и `results_audio_sft_metrics.txt` **не трогать** — это
историческая точка v1.

---

## Порядок выполнения

```
[1] word2num (30 мин, CPU, не требует GPU)
    pip install word2number
    правки в calculate_metrics.py
    запустить пересчёт метрик
    проверить WER_norm в выводе

[2] finetune_audio_v2.py (создать скрипт — 1 час, обучение — 3–5 часов на L40s)
    скопировать и внести правки из раздела 1.1
    запустить обучение
    проверить DIAG в начале лога (merge diagnostics)
    после завершения — eval + метрики

[3] Обновить report.md + README.md + commit
```

---

## Что НЕ делать

- Не трогать `results_omni_7b_v3.json` и `results_text_sft.json` — baseline зафиксирован
- Не изменять `finetune_audio.py` — оставить как есть для истории
- Не менять LORA_RANK > 16 — данных мало, overfitting
- Не увеличивать training set сверх 1500 — новые аудио не генерировались
