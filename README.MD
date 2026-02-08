## Проект - "Speech-Toolformer" - в рамках DLS: Speech

---

## Тип проекта - "Applied Science (исследовательско-прикладной)"

---

## Идея - **Голосовой ассистент в банковском приложении**

### Описание

Есть запрос со стороны бизнеса (банк) на создание сервиса в мобильном приложении, где клиент говорит, например, "покажи историю транзакций", "какой там курс доллара?", "переведи 1000 рублей маме", и т.п.

Решил взять для MVP одну функцию - `"transfer_money"`, но тут будет 2 аргумента - `"recipient"` и `"amount"`.

Для этого прошел по такому пайплайну:
1. Сгенерировал с помощью LLM (ChatGPT) 511 текстовых JSON
2. Получил синтезированные аудио с помощью xtts-v2
3. Подготовил манифест-файл
4. Сгруппировал эксперименты A, B, C, D в один скрипт
5. Анализ результатов


---

### Files:
*  `generated_dataset.json` - синтетический текстовый датасет (с помощью ChatGPT)
*  `generate_audio.py` - скрипт для TTS-синтеза (с помощью xtts-v2)
*  `dataset_audio_Cameron_Russell_115.json` - файл манифеста
*  `Cameron_Russell-chunk-38.wav` - референсное аудио для TTS-генерации
*  `wavs_Cameron_Russell_115.zip` - архив со сгенерированными аудиофайлами
*  `run_omni_project.py` - основной скрипт запуска экспериментов с моделью "Qwen2.5-Omni-7B" - *пока дает не идеальные результаты, нужно доработать*
*  `results_omni_7b.json` - результат запуского основного скрипта, содержит выдачу для каждого пайплайна (a, b, c, d) - *пока не финальный результат*
*  `run_omni_project_v2.py` - основной скрипт запуска экспериментов с моделью "Qwen2.5-Omni-7B" - исправил ошибки PARSE_ERROR. *текущий вариант*
*  `results_omni_7b.json` - результат запуского основного скрипта (run_omni_project_v2.py), содержит выдачу для каждого пайплайна (a, b, c, d) - *текущий результат*
*  `calculate_metrics.py` - скрипт для подсчета метрик после получения результатов экспериментов
*  `metrics_report.txt` - вывод результатов с метриками.


---

## Ход работ

### Фаза 1: TTS: Синтез аудио

```
conda create -n xtts_env python=3.10 -y
conda activate xtts_env
conda install -c conda-forge ffmpeg -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install TTS
```


```
pip install "transformers==4.33.0" --force-reinstall
pip install "numpy<2.0" --force-reinstall
```

TTS-generate audios:
```
COQUI_TOS_AGREED=1 python generate_audio.py
```


### Фаза 2: Omni-model

```
conda create -n omni_env python=3.10 -y
conda activate omni_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/transformers 
pip install accelerate librosa soundfile

pip install qwen-omni-utils[decord] -U
pip install timm decord moviepy
```


Run experiments in one script:
```
python run_omni_project.py
```

Получил файл `results_omni_7b.json`, содержащий местами "PARSE_ERROR" в некоторых пайплайнах. Но это не критично, т.к. модель реально услышала аудио. Однако функция extract_json ищет чистый JSON, а видит кучу текста с промптами, поэтому выдает "PARSE_ERROR". Поэтому пофиксил функции generate_omni и extract_json.

Rerun experiments:
```
python run_omni_project_v2.py
```
Получил файл `results_omni_7b_v2.json`, где уже нет "PARSE_ERROR"

## Результат
```
========================================
FINAL METRICS REPORT
========================================
Pipeline A (Text Oracle):        98.63% Accuracy
Pipeline C (Direct Audio):       70.65% Accuracy
Pipeline D (Cascaded ASR+NLP):   55.58% Accuracy
----------------------------------------
Pipeline B (ASR Quality, raw):  103.37% WER
(Note: WER is inflated due to assistant-style transcription output)
========================================

ANALYSIS:
Direct audio-to-intent inference outperforms the cascaded ASR-based pipeline by 15.07% accuracy.
This suggests that intermediate transcription introduces compounding errors that degrade intent extraction performance.

Modality Gap (Direct vs Text Oracle): -27.98% accuracy.

📄 Metrics saved to metrics_report.txt
```

---


***Проект не оформлен до конца, находится в процессе завершения.***
