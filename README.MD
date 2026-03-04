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
Никакого файнтюна — только zero-shot с системным промптом.

Подробный анализ, метрики и выводы: **[report.md](report.md)**

---

## Ключевые результаты

| Pipeline | Описание | Accuracy | Precision | Recall | FAR |
|----------|----------|:--------:|:---------:|:------:|:---:|
| **A** — Text Oracle | GT-текст → JSON | **98.83%** | 0.997 | 0.988 | 0.009 |
| **B** — ASR | Аудио → транскрипт | WER **66.74%**¹ | — | — | — |
| **C** — Direct | Аудио → JSON напрямую | **68.30%** | 0.992 | 0.601 | 0.018 |
| **D** — Cascaded | Аудио → ASR → JSON | **67.51%** | **1.000** | 0.586 | **0.000** |

¹ WER завышен из-за digit↔word mismatch ("1500" vs "fifteen hundred"); семантический WER ~20–30%.

**Вывод:** Pipeline D (каскадный) рекомендуется для банкинга — precision=1.000, FAR=0.000 (ни одного ложного срабатывания). Pipeline C чуть лучше по recall, но даёт 2 ложных вызова на 511 сэмплов.

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
| `src/run_omni_project_v2.py` | Эксперименты v2 (все пайплайны A–D; содержит баг в D) |
| `src/run_omni_project_v3.py` | Эксперименты v3 — **финальная версия**, фикс Pipeline D |
| `src/calculate_metrics.py` | Метрики: accuracy, precision, recall, FAR, parsable rate, WER |
| `src/error_analysis.py` | Детальный разбор ошибок по категориям для каждого пайплайна |

### `results/` — результаты
| Файл | Описание |
|------|----------|
| `results/results_omni_7b_v2.json` | Сырые результаты v2 (511 сэмплов) |
| `results/results_omni_7b_v3.json` | Сырые результаты v3 (511 сэмплов, фикс D) |
| `results/results_omni_7b_v2_metrics.txt` | Метрики v2 |
| `results/results_omni_7b_v3_metrics.txt` | Метрики v3 |
| `results/error_analysis_v3.txt` | Разбор ошибок по категориям (финальный) |

### Отчёт
| Файл | Описание |
|------|----------|
| `report.md` | **Полный отчёт**: дизайн данных, эксперименты, анализ ошибок, выводы |
| `demo_notebook.ipynb` | Интерактивный демо-ноутбук: все 4 пайплайна + метрики |

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
