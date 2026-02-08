import json
import re
from jiwer import wer
from num2words import num2words


# === НАСТРОЙКИ ===
INPUT_FILE = "results_omni_7b_v2.json"
# =================


def normalize_text_for_wer(text):
    """Чистит текст для честного подсчета WER"""
    if not isinstance(text, str): return ""
    # Убираем префиксы, если модель любит болтать (типа "The transcription is: ...")
    text = re.sub(r"The original content.*?:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"The transcription.*?:", "", text, flags=re.IGNORECASE)
    # Нижний регистр и удаление пунктуации
    text = re.sub(r"[^\w\s]", "", text).lower()
    return " ".join(text.split())

def compare_json(pred, truth):
    """
    Сравнивает два JSON объекта (Soft Match).
    Возвращает True, если они совпадают по смыслу.
    """
    # 1. Обработка Null (Negative samples)
    if pred is None and truth is None: return True
    if pred is None or truth is None: return False
    
    # 2. Проверка имени инструмента
    if pred.get("tool_name") != truth.get("tool_name"):
        return False
        
    # 3. Проверка аргументов
    pred_args = pred.get("arguments", {})
    truth_args = truth.get("arguments", {})
    
    for key, t_val in truth_args.items():
        p_val = pred_args.get(key)
        
        # Сравнение чисел (50 == 50.0)
        if isinstance(t_val, (int, float)) and isinstance(p_val, (int, float)):
            if abs(t_val - p_val) > 0.1: # Допускаем погрешность 0.1
                return False
        # Сравнение строк (имена)
        elif isinstance(t_val, str) and isinstance(p_val, str):
            if t_val.lower().strip() != p_val.lower().strip():
                # Можно добавить логику: если "Paul" внутри "Paul Elilvo" -> True
                # Но пока строгое сравнение (case-insensitive)
                return False
        else:
            if str(t_val) != str(p_val):
                return False
                
    return True


def extract_asr_text(text):
    if not isinstance(text, str):
        return ""
    m = re.search(r"'(.+?)'", text)
    if m:
        return m.group(1)
    return text


# === ЗАГРУЗКА ===
with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

print(f"Calculating metrics for {len(data)} samples...\n")

# Счетчики
metrics = {
    "A": {"correct": 0, "total": 0},
    "C": {"correct": 0, "total": 0},
    "D": {"correct": 0, "total": 0},
}

wer_refs = []
wer_hyps = []

# === ПРОГОН ===
for item in data:
    truth = item.get("ground_truth")
    
    # 1. Pipeline A (Text Oracle)
    if compare_json(item.get("pipeline_a_json"), truth):
        metrics["A"]["correct"] += 1
    metrics["A"]["total"] += 1
    
    # 2. Pipeline C (Direct Audio)
    if compare_json(item.get("pipeline_c_json"), truth):
        metrics["C"]["correct"] += 1
    metrics["C"]["total"] += 1
    
    # 3. Pipeline D (Cascaded)
    if compare_json(item.get("pipeline_d_json"), truth):
        metrics["D"]["correct"] += 1
    metrics["D"]["total"] += 1
    
    # 4. Pipeline B (ASR WER)
    ref_text = normalize_text_for_wer(item.get("input_text_gt"))
    # hyp_text = normalize_text_for_wer(item.get("pipeline_b_transcript"))
    hyp_text = normalize_text_for_wer(extract_asr_text(item.get("pipeline_b_transcript")))

    
    if ref_text.strip(): # Игнорируем пустые
        wer_refs.append(ref_text)
        wer_hyps.append(hyp_text)

# === ВЫВОД РЕЗУЛЬТАТОВ ===

print("="*40)
print("FINAL METRICS REPORT")
print("="*40)

# Accuracy
acc_a = metrics["A"]["correct"] / metrics["A"]["total"] * 100
acc_c = metrics["C"]["correct"] / metrics["C"]["total"] * 100
acc_d = metrics["D"]["correct"] / metrics["D"]["total"] * 100

print(f"Pipeline A (Text Oracle):  {acc_a:.2f}% Accuracy")
print(f"Pipeline C (Direct Audio): {acc_c:.2f}% Accuracy")
print(f"Pipeline D (Cascaded):     {acc_d:.2f}% Accuracy")

print("-" * 40)

# WER
final_wer = wer(wer_refs, wer_hyps)
print(f"Pipeline B (ASR Quality):  {final_wer:.2%} WER")

print("="*40)

# Выводы для отчета
print("\nANALYSIS SUGGESTIONS:")
gap_c = acc_a - acc_c
gap_d = acc_a - acc_d

if acc_c > acc_d:
    print(f"🚀 WINNER: Direct Pipeline (C) is better by {acc_c - acc_d:.2f}%!")
    print("Conclusion: Qwen2.5-Omni understands audio intent better directly.")
else:
    print(f"🐢 WINNER: Cascaded Pipeline (D) is better by {acc_d - acc_c:.2f}%!")
    print("Conclusion: Converting to text first is still safer.")

print(f"Modality Gap (Direct): -{gap_c:.2f}% compared to text.")


# ==============
REPORT_FILE = "metrics_report.txt"

with open(REPORT_FILE, "w") as f:
    f.write("="*40 + "\n")
    f.write("FINAL METRICS REPORT\n")
    f.write("="*40 + "\n\n")

    f.write(f"Pipeline A (Text Oracle):  {acc_a:.2f}% Accuracy\n")
    f.write(f"Pipeline C (Direct Audio): {acc_c:.2f}% Accuracy\n")
    f.write(f"Pipeline D (Cascaded):     {acc_d:.2f}% Accuracy\n\n")

    f.write(f"Pipeline B (ASR Quality):  {final_wer:.2%} WER\n\n")

    if acc_c > acc_d:
        f.write(
            f"Direct pipeline outperforms cascaded by "
            f"{acc_c - acc_d:.2f}% accuracy.\n"
        )
    else:
        f.write(
            f"Cascaded pipeline outperforms direct by "
            f"{acc_d - acc_c:.2f}% accuracy.\n"
        )

print(f"\n📄 Metrics saved to {REPORT_FILE}")
