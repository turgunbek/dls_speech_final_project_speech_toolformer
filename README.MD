# Speech-Toolformer: Голосовой ассистент для банковского приложения

**Итоговый проект курса DLS MIPT: Speech, Fall 2025**

---

## Идея

Голосовой ассистент для банкинга: пользователь произносит _"Transfer five hundred to Mom"_ —
система транскрибирует речь, понимает намерение и возвращает структурированный вызов инструмента:

```json
{"tool_name": "transfer_money", "arguments": {"recipient": "Mom", "amount": 500.0}}
```

Инструмент: `transfer_money(recipient: str, amount: float)` — два аргумента разных типов,
реальный бизнес-кейс, нетривиальный с точки зрения парсинга.

Модель: **Qwen2.5-Omni-7B** (нативно мультимодальная, аудио → текст в одном forward pass).
Zero-shot baseline + LoRA SFT (text и audio) — подробности в [report.md](report.md).

Подробный анализ, метрики и выводы: **[report.md](report.md)**

---

## Ключевые результаты

### Baseline (zero-shot)

| Pipeline | Описание | Accuracy | Precision | Recall | FAR |
|----------|----------|:--------:|:---------:|:------:|:---:|
| **A** — Text Oracle | GT-текст → JSON | 98.83% | 0.997 | 0.988 | 0.009 |
| **B** — ASR | Аудио → транскрипт | WER 66.74%¹ | — | — | — |
| **C** — Direct | Аудио → JSON напрямую | 68.30% | 0.992 | 0.601 | 0.018 |
| **D** — Cascaded | Аудио → ASR → JSON | 67.51% | **1.000** | 0.586 | **0.000** |

### После LoRA Fine-tuning (Text SFT)

| Pipeline | Baseline | Text SFT | Δ |
|----------|:--------:|:--------:|:-:|
| **A** — Text Oracle | 98.83% | **99.61%** | +0.78 pp |
| **A** FAR | 0.009 | **0.000** | −100% |
| **A** Precision | 0.997 | **1.000** | — |
| **C** — Direct | 68.30% | 67.51% | ~0 (noise) |
| **D** — Cascaded | 67.51% | **70.06%** | +2.55 pp |
| **D** Recall | 0.586 | **0.618** | +0.032 |

¹ WER завышен из-за digit↔word mismatch ("1500" vs "fifteen hundred"); семантический WER ~20–30%.

**Вывод:** Pipeline D + Text SFT рекомендуется для банкинга — **Precision=1.000, FAR=0.000, Accuracy=70.06%**. Text SFT добавляет +0.78 pp на Pipeline A и +2.55 pp на Pipeline D при нулевых ложных вызовах.

---

## Структура репозитория

### `data/` — входные данные
| Файл | Описание |
|------|----------|
| `data/generated_dataset.json` | 511 текстовых примеров (ChatGPT): 401 positive, 110 negative |
| `data/dataset_audio_Cameron_Russell_115.json` | Манифест с полями `text`, `label`, `audio_path` |
| `data/Cameron_Russell-chunk-38.wav` | Референсное аудио для TTS |
| `data/wavs_Cameron_Russell_115.zip` | 511 синтезированных WAV (XTTS-v2) |

### `src/` — скрипты
| Файл | Описание |
|------|----------|
| `src/generate_audio.py` | TTS-синтез аудио (XTTS-v2, голос Cameron Russell) |
| `src/generate_training_data.py` | Генерация 3000 текстовых примеров для Text SFT |
| `src/finetune_text.py` | LoRA Text SFT (текстовый датасет → JSON) |
| `src/finetune_audio.py` | LoRA Audio SFT (аудио → JSON) |
| `src/run_omni_project_v2.py` | Эксперименты v2 (содержит баг в Pipeline D) |
| `src/run_omni_project_v3.py` | **Финальный скрипт**: фикс Pipeline D + поддержка `--model_path` для SFT |
| `src/calculate_metrics.py` | Метрики: accuracy, precision, recall, FAR, parsable rate, WER |
| `src/error_analysis.py` | Детальный разбор ошибок по категориям для каждого пайплайна |

### `results/` — результаты
| Файл | Описание |
|------|----------|
| `results/results_omni_7b_v2.json` | Сырые результаты v2 (511 сэмплов) |
| `results/results_omni_7b_v3.json` | Сырые результаты v3 — baseline (511 сэмплов) |
| `results/results_text_sft.json` | Сырые результаты Text SFT (все пайплайны) |
| `results/results_audio_sft.json` | Сырые результаты Audio SFT (все пайплайны) |
| `results/results_omni_7b_v3_metrics.txt` | Метрики baseline v3 |
| `results/results_text_sft_metrics.txt` | Метрики Text SFT |
| `results/results_audio_sft_metrics.txt` | Метрики Audio SFT |
| `results/error_analysis_v3.txt` | Разбор ошибок — baseline v3 |
| `results/error_analysis_audio_sft.txt` | Разбор ошибок — Audio SFT |

### Отчёт
| Файл | Описание |
|------|----------|
| `report.md` | **Полный отчёт**: дизайн данных, эксперименты, SFT, анализ ошибок, выводы |
| `report_sft_1.md` | Краткая сводка результатов SFT (таблицы + технические находки) |
| `demo_notebook.ipynb` | Интерактивный демо-ноутбук: все 4 пайплайна + baseline и SFT метрики |

---

## Воспроизведение

### 1. Синтез аудио (если нужно пересинтезировать)

```bash
conda create -n xtts_env python=3.10 -y
conda activate xtts_env
conda install -c conda-forge ffmpeg -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install TTS
COQUI_TOS_AGREED=1 python src/generate_audio.py
```

### 2. Запуск экспериментов (Qwen2.5-Omni-7B)

```bash
conda create -n omni_env python=3.10 -y
conda activate omni_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/transformers
pip install accelerate librosa soundfile
pip install qwen-omni-utils[decord] -U
pip install timm decord moviepy

python src/run_omni_project_v3.py   # финальная версия
```

### 3. Метрики и анализ ошибок

```bash
pip install jiwer

python src/calculate_metrics.py results/results_omni_7b_v3.json
python src/error_analysis.py results/results_omni_7b_v3.json
```

### 4. Fine-tuning (LoRA SFT)

```bash
conda activate omni_env
pip install peft trl

# Text SFT
python src/finetune_text.py
# → checkpoints/qwen_omni_text_sft/

# Оценка Text SFT
python src/run_omni_project_v3.py \
    --model_path checkpoints/qwen_omni_text_sft \
    --output_file results/results_text_sft.json \
    --pipelines A,B,C,D
python src/calculate_metrics.py results/results_text_sft.json

# Audio SFT (требует предварительно сгенерированного аудио для обучения)
python src/finetune_audio.py
# → checkpoints/qwen_omni_audio_sft/

# Оценка Audio SFT
python src/run_omni_project_v3.py \
    --model_path checkpoints/qwen_omni_audio_sft \
    --output_file results/results_audio_sft.json \
    --pipelines A,B,C,D
python src/calculate_metrics.py results/results_audio_sft.json
```

> **Примечание:** `checkpoints/` не коммитится в репозиторий (>10 GB). Результаты оценки (`results/*.json`, `results/*.txt`) включены в репо.
