import json
import re
import sys
from collections import Counter
from jiwer import wer as compute_wer


# === НАСТРОЙКИ ===
# Можно передать файл с результатами как аргумент: python calculate_metrics.py results_omni_7b_v3.json
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "results_omni_7b_v2.json"
REPORT_FILE = INPUT_FILE.replace(".json", "_metrics.txt")
# =================


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def is_hallucination(text: str) -> bool:
    """
    Определяет, является ли транскрипт галлюцинацией модели.

    Проверки (от очевидного к тонкому):
    1. Нелатинские символы > 3  (китайские, арабские и т.д.)
    2. Четыре одинаковых слова подряд  (простое повторение)
    3. Любой 3-граммный паттерн встречается > 2 раз  (периодическое зацикливание:
       "eight two one zero eight two one..." — слова разные, но последовательность повторяется)
    4. После очистки от служебного префикса длина > 40 слов  (защитный порог)
    """
    if not isinstance(text, str) or not text.strip():
        return True

    # 1. Нелатинские символы
    if len(re.findall(r'[^\x00-\x7F]', text)) > 3:
        return True

    words = text.split()

    # 2. Четыре одинаковых слова подряд
    for i in range(len(words) - 3):
        if len(set(words[i:i+4])) == 1:
            return True

    # 3. Периодическое зацикливание: 3-грамм встречается > 2 раз
    if len(words) >= 9:  # минимум для возможного повтора 3-граммы 3 раза
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if Counter(trigrams).most_common(1)[0][1] > 2:
            return True

    # 4. Аномально длинный транскрипт (нормальные фразы < 20 слов)
    if len(words) > 40:
        return True

    return False


def extract_asr_text(text: str) -> str:
    """
    Извлекает чистый текст транскрипта из verbose-вывода модели Qwen.
    Qwen оборачивает транскрипт в разные форматы — пробуем несколько паттернов.
    Возвращает пустую строку при галлюцинации.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if is_hallucination(text):
        return ""

    # Паттерн 1: одинарные кавычки 'content' (наиболее частый вариант Qwen)
    m = re.search(r"'([^']{3,})'", text)
    if m:
        return m.group(1).strip(" .")

    # Паттерн 2: двойные кавычки "content"
    m = re.search(r'"([^"]{3,})"', text)
    if m:
        return m.group(1).strip(" .")

    # Паттерн 3: всё после «is:» / «are:» / двоеточия в конце строки
    m = re.search(r'(?:is|are)\s*:\s*(.{5,})$', text.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip("'.\" ")

    # Паттерн 4: возможно, модель вернула чистый текст без обёртки
    # (только если нет служебных слов)
    service_words = ("original content", "transcription", "audio is", "the content")
    if not any(sw in text.lower() for sw in service_words):
        return text.strip(" .")

    return ""


def normalize_text_for_wer(text: str) -> str:
    """Нормализует текст для подсчёта WER: lowercase, без пунктуации."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\w\s]", "", text).lower()
    return " ".join(text.split())


def compare_json(pred, truth) -> bool:
    """
    Soft-match: проверяет совпадение предсказания и GT.
    - None == None → True (верно отклонил вызов инструмента)
    - pred=dict, truth=None → False (ложная тревога)
    - pred=None, truth=dict → False (пропуск)
    - оба dict → сравниваем tool_name и arguments (числа с допуском 0.1,
      строки case-insensitive)
    """
    if pred == "PARSE_ERROR":
        return False
    if pred is None and truth is None:
        return True
    if pred is None or truth is None:
        return False
    if not isinstance(pred, dict) or not isinstance(truth, dict):
        return False

    if pred.get("tool_name") != truth.get("tool_name"):
        return False

    pred_args = pred.get("arguments", {})
    truth_args = truth.get("arguments", {})

    for key, t_val in truth_args.items():
        p_val = pred_args.get(key)
        if isinstance(t_val, (int, float)) and isinstance(p_val, (int, float)):
            if abs(t_val - p_val) > 0.1:
                return False
        elif isinstance(t_val, str) and isinstance(p_val, str):
            if t_val.lower().strip() != p_val.lower().strip():
                return False
        else:
            if str(t_val) != str(p_val):
                return False

    return True


def compute_pipeline_metrics(data: list, pipeline_key: str) -> dict:
    """
    Вычисляет полный набор метрик для одного пайплайна:
    accuracy, precision, recall, FAR, parsable_rate, TP/FP/FN/TN.

    Определения:
      TP  — GT=positive, предсказание верное (tool + все аргументы)
      FN  — GT=positive, предсказание неверное (null, PARSE_ERROR или неверные args)
      FP  — GT=negative (null), предсказание — любой tool call
      TN  — GT=negative (null), предсказание — null или PARSE_ERROR
    """
    tp = fp = fn = tn = parse_errors = 0
    total = len(data)

    for item in data:
        truth = item.get("ground_truth")
        pred = item.get(pipeline_key)

        if pred == "PARSE_ERROR":
            parse_errors += 1

        is_pos_truth = truth is not None
        is_pos_pred = pred is not None and pred != "PARSE_ERROR"
        correct = compare_json(pred, truth)

        if is_pos_truth:
            if correct:
                tp += 1
            else:
                fn += 1
        else:
            if is_pos_pred:
                fp += 1
            else:
                tn += 1

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    parsable_rate = (total - parse_errors) / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "far": far,
        "parsable_rate": parsable_rate,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "parse_errors": parse_errors,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples from {INPUT_FILE}\n")

# ---------------------------------------------------------------------------
# Метрики пайплайнов A / C / D
# ---------------------------------------------------------------------------

m_a = compute_pipeline_metrics(data, "pipeline_a_json")
m_c = compute_pipeline_metrics(data, "pipeline_c_json")
m_d = compute_pipeline_metrics(data, "pipeline_d_json")

# ---------------------------------------------------------------------------
# WER для пайплайна B
# ---------------------------------------------------------------------------

wer_refs = []
wer_hyps = []
hallucination_count = 0
extraction_failed_count = 0

for item in data:
    ref_raw = item.get("input_text_gt", "")
    ref = normalize_text_for_wer(ref_raw)
    if not ref.strip():
        continue

    # v3-файл содержит уже извлечённый чистый транскрипт
    if "pipeline_b_transcript_clean" in item:
        # Проверяем сырой транскрипт (нелатинские символы и т.д.)
        if is_hallucination(item.get("pipeline_b_transcript", "")):
            hallucination_count += 1
            continue
        hyp_raw = item["pipeline_b_transcript_clean"]
    else:
        raw_transcript = item.get("pipeline_b_transcript", "")
        if is_hallucination(raw_transcript):
            hallucination_count += 1
            continue
        hyp_raw = extract_asr_text(raw_transcript)
        if not hyp_raw:
            extraction_failed_count += 1
            continue

    # Проверяем также сам очищенный текст — периодическое зацикливание
    # могло оказаться внутри кавычек и не было поймано по сырому тексту
    if is_hallucination(hyp_raw):
        hallucination_count += 1
        continue

    hyp = normalize_text_for_wer(hyp_raw)
    wer_refs.append(ref)
    wer_hyps.append(hyp)

final_wer = compute_wer(wer_refs, wer_hyps) if wer_refs else float("nan")
wer_samples_used = len(wer_refs)
wer_samples_skipped = hallucination_count + extraction_failed_count

# ---------------------------------------------------------------------------
# Вывод результатов
# ---------------------------------------------------------------------------

HEADER = "=" * 56
SEP    = "-" * 56

def fmt(m: dict, name: str) -> str:
    return (
        f"Pipeline {name}:\n"
        f"  Accuracy:      {m['accuracy']:.2%}\n"
        f"  Precision:     {m['precision']:.3f}\n"
        f"  Recall:        {m['recall']:.3f}\n"
        f"  FAR:           {m['far']:.3f}\n"
        f"  Parsable rate: {m['parsable_rate']:.2%}\n"
        f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}"
        + (f"  PARSE_ERR={m['parse_errors']}" if m['parse_errors'] else "")
    )

lines = [
    HEADER,
    f"METRICS REPORT  —  {INPUT_FILE}",
    f"Total samples: {len(data)}",
    HEADER,
    "",
    fmt(m_a, "A (Text Oracle)"),
    "",
    fmt(m_c, "C (Direct Audio → JSON)"),
    "",
    fmt(m_d, "D (Cascaded: ASR transcript → JSON)"),
    "",
    SEP,
    f"Pipeline B — ASR Quality (WER)",
    f"  WER:              {final_wer:.2%}",
    f"  Samples used:     {wer_samples_used} / {len(data)}",
    f"  Hallucinations:   {hallucination_count}  (skipped from WER)",
    f"  Extraction failed:{extraction_failed_count}  (skipped from WER)",
    "",
    SEP,
    "SUMMARY",
    SEP,
]

# Modality gap
gap_ca = m_a["accuracy"] - m_c["accuracy"]
gap_da = m_a["accuracy"] - m_d["accuracy"]
gap_cd = m_c["accuracy"] - m_d["accuracy"]

lines += [
    f"Modality gap (A vs C):  {gap_ca:+.2%}  (direct audio vs text oracle)",
    f"Modality gap (A vs D):  {gap_da:+.2%}  (cascaded vs text oracle)",
    f"C vs D gap:             {gap_cd:+.2%}",
    "",
]

if m_c["accuracy"] >= m_d["accuracy"]:
    lines.append(
        f"WINNER: Pipeline C (Direct) outperforms D by "
        f"{gap_cd:.2%} accuracy and recall "
        f"({m_c['recall']:.3f} vs {m_d['recall']:.3f})."
    )
else:
    lines.append(
        f"WINNER: Pipeline D (Cascaded) outperforms C by "
        f"{-gap_cd:.2%} accuracy."
    )

output_str = "\n".join(lines)
print(output_str)

# ---------------------------------------------------------------------------
# Запись в файл
# ---------------------------------------------------------------------------

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(output_str + "\n")

print(f"\nReport saved to: {REPORT_FILE}")
