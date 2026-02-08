## Проект - "Speech-Toolformer" - в рамках DLS

---

## Тип проекта - "Applied Science (исследовательско-прикладной)"

---

### Files:
*  `generated_dataset.json` - синтетический текстовый датасет (с помощью ChatGPT)
*  `generate_audio.py` - скрипт для TTS-синтеза (с помощью xtts-v2)
*  `dataset_audio_Cameron_Russell_115.json` - файл манифеста
*  `Cameron_Russell-chunk-38.wav` - референсное аудио для TTS-генерации
*  `wavs_Cameron_Russell_115.zip` - архив со сгенерированными аудиофайлами
*  `run_omni_project.py` - основной скрипт запуска экспериментов с моделью "Qwen2.5-Omni-7B" - *пока дает не идеальные результаты, нужно доработать*
*  `results_omni_7b.json` - результат запуского основного скрипта, содержит выдачу для каждого пайплайна (a, b, c, d) - *пока не финальный результат*


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

---

***Проект не оформлен до конца, находится в процессе завершения.***
