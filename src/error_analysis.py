"""
error_analysis.py
=================
Разбор ошибок по каждому пайплайну.
Запуск из корня проекта:
    python src/error_analysis.py results/results_omni_7b_v2.json
    python src/error_analysis.py results/results_omni_7b_v3.json
"""

import json
import re
import sys
from collections import Counter


def is_hallucination(text: str) -> bool:
    """Полная проверка на галлюцинацию (синхронизирована с calculate_metrics.py)."""
    if not isinstance(text, str) or not text.strip():
        return True
    if len(re.findall(r'[^\x00-\x7F]', text)) > 3:
        return True
    words = text.split()
    for i in range(len(words) - 3):
        if len(set(words[i:i+4])) == 1:
            return True
    if len(words) >= 9:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if Counter(trigrams).most_common(1)[0][1] > 2:
            return True
    if len(words) > 40:
        return True
    return False


INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "results/results_omni_7b_v2.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples from {INPUT_FILE}\n")


# ---------------------------------------------------------------------------
# Утилиты (дублируем compare_json, чтобы не зависеть от calculate_metrics.py)
# ---------------------------------------------------------------------------

def compare_json(pred, truth) -> bool:
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


def categorize_error(pred, truth) -> str:
    """
    Категоризирует ошибку предсказания:
      false_positive  — GT=null, model вызвала инструмент
      false_negative  — GT=positive, model вернула null
      wrong_recipient — оба positive, recipient не совпадает
      wrong_amount    — оба positive, amount не совпадает
      both_wrong      — оба positive, оба поля не совпадают
      parse_error     — модель не смогла выдать валидный JSON
      correct         — совпадает (не ошибка)
    """
    if compare_json(pred, truth):
        return "correct"
    if pred == "PARSE_ERROR":
        return "parse_error"

    is_pos_truth = truth is not None
    is_pos_pred = pred is not None and pred != "PARSE_ERROR"

    if not is_pos_truth and is_pos_pred:
        return "false_positive"
    if is_pos_truth and not is_pos_pred:
        return "false_negative"

    # Оба positive, но что-то не так
    if not isinstance(pred, dict) or not isinstance(truth, dict):
        return "wrong_format"

    pred_args = pred.get("arguments", {})
    truth_args = truth.get("arguments", {})

    wrong_recipient = False
    wrong_amount = False

    t_recip = truth_args.get("recipient", "")
    p_recip = pred_args.get("recipient", "")
    if isinstance(t_recip, str) and isinstance(p_recip, str):
        if t_recip.lower().strip() != p_recip.lower().strip():
            wrong_recipient = True

    t_amt = truth_args.get("amount", None)
    p_amt = pred_args.get("amount", None)
    if isinstance(t_amt, (int, float)) and isinstance(p_amt, (int, float)):
        if abs(t_amt - p_amt) > 0.1:
            wrong_amount = True
    elif t_amt != p_amt:
        wrong_amount = True

    if wrong_recipient and wrong_amount:
        return "both_wrong"
    if wrong_recipient:
        return "wrong_recipient"
    if wrong_amount:
        return "wrong_amount"

    if pred.get("tool_name") != truth.get("tool_name"):
        return "wrong_tool"

    return "other_mismatch"


# ---------------------------------------------------------------------------
# Анализ по пайплайнам
# ---------------------------------------------------------------------------

pipeline_keys = {
    "A (Text Oracle)": "pipeline_a_json",
    "C (Direct Audio → JSON)": "pipeline_c_json",
    "D (Cascaded ASR → JSON)": "pipeline_d_json",
}

SEP = "=" * 64
SEP_M = "-" * 64

for pipeline_name, key in pipeline_keys.items():
    print(SEP)
    print(f"Pipeline {pipeline_name}  —  key: '{key}'")
    print(SEP)

    errors = []
    category_counts = Counter()

    for item in data:
        truth = item.get("ground_truth")
        pred = item.get(key)
        cat = categorize_error(pred, truth)
        category_counts[cat] += 1

        if cat != "correct":
            errors.append({
                "gt_text": item.get("input_text_gt", ""),
                "truth": truth,
                "pred": pred,
                "category": cat,
                "b_transcript": item.get("pipeline_b_transcript", ""),
                "b_clean": item.get("pipeline_b_transcript_clean", ""),
            })

    total = len(data)
    correct = category_counts["correct"]
    print(f"Correct: {correct}/{total}  ({correct/total:.1%})")
    print(f"Errors:  {total - correct}/{total}  ({(total-correct)/total:.1%})")
    print()

    print("Error breakdown:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        if cat == "correct":
            continue
        print(f"  {cat:<20} {cnt:>4}  ({cnt/total:.1%})")
    print()

    # Примеры по категориям
    shown_cats = set()
    for e in errors:
        cat = e["category"]
        if cat in shown_cats:
            continue
        shown_cats.add(cat)
        print(SEP_M)
        print(f"Example of [{cat}]:")
        print(f"  GT text : {e['gt_text']}")
        if key == "pipeline_d_json":
            clean = e["b_clean"] or e["b_transcript"][:80]
            print(f"  ASR     : {clean[:100]}")
        print(f"  Truth   : {e['truth']}")
        print(f"  Pred    : {e['pred']}")

    print()

# ---------------------------------------------------------------------------
# ASR-специфический анализ (Pipeline B)
# ---------------------------------------------------------------------------

print(SEP)
print("Pipeline B — ASR Transcript Quality")
print(SEP)

is_hallucination_count = 0
no_quotes_count = 0
clean_ok_count = 0

first_word_errors = Counter()
total_b = 0

for item in data:
    raw = item.get("pipeline_b_transcript", "")
    total_b += 1

    if is_hallucination(raw):
        is_hallucination_count += 1
        continue

    m = re.search(r"'([^']{3,})'", raw)
    if not m:
        m = re.search(r'"([^"]{3,})"', raw)
    if not m:
        no_quotes_count += 1
        continue

    clean_ok_count += 1
    extracted = m.group(1).strip()

    gt = item.get("input_text_gt", "")
    if gt and extracted:
        gt_first = gt.split()[0].lower().strip(".,!")
        hyp_first = extracted.split()[0].lower().strip(".,!")
        if gt_first != hyp_first:
            first_word_errors[f"'{gt_first}' -> '{hyp_first}'"] += 1

print(f"Total samples:               {total_b}")
print(f"  Clean extraction:          {clean_ok_count}")
print(f"  Hallucinations (skipped):  {is_hallucination_count}")
print(f"  No quotes / format issue:  {no_quotes_count}")
print()
print("Most common first-word errors (TTS clipping / ASR mismatch):")
for pair, cnt in first_word_errors.most_common(10):
    print(f"  {pair:<35} x {cnt}")

print()
print("Done.")
