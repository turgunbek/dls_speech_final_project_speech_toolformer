"""
finetune_audio_v2.py
====================
LoRA Audio SFT v2 для Qwen2.5-Omni-7B.

Исправления по сравнению с v1 (finetune_audio.py):
  1. Диагностика merge: проверяет тип тинкера и делает quick-eval ДО обучения.
  2. Альтернативный merge: если merge_and_unload() вернул PeftModel (merge не сработал),
     используем merge_adapter() + base_model.model.
  3. _save_lora сохраняет правильный base_model_name_or_path в adapter_config.json.
  4. Больше эпох (5 вместо 2) и ниже LR (5e-5 вместо 1e-4).

Запуск из корня проекта:
    python src/finetune_audio_v2.py

Output: checkpoints/qwen_omni_audio_sft_v2/
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
import logging
from pathlib import Path
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
from peft import LoraConfig, PeftModel, TaskType, inject_adapter_in_model
from qwen_omni_utils import process_mm_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID           = "Qwen/Qwen2.5-Omni-7B"
TEXT_SFT_ADAPTER   = "checkpoints/qwen_omni_text_sft"
EVAL_MANIFEST      = "data/dataset_audio_Cameron_Russell_115.json"
TRAIN_FILE         = "data/training_audio_manifest.json"
OUTPUT_DIR         = "checkpoints/qwen_omni_audio_sft_v2"
LORA_RANK          = 16
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.05
NUM_EPOCHS         = 5       # было 2
BATCH_SIZE         = 1
GRAD_ACCUM         = 16
LEARNING_RATE      = 5e-5    # было 1e-4
MAX_SEQ_LEN        = 1024
WARMUP_STEPS       = 30
LOG_INTERVAL       = 10
SAVE_INTERVAL      = 100
DIAG_SAMPLES       = 5       # сколько примеров брать для quick-eval
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a banking assistant.
You must NOT generate audio or speech output. Generate TEXT ONLY.

Available tools:
1. transfer_money(recipient: str, amount: float)
2. get_exchange_rate(currency_from: str, currency_to: str)

Output ONLY valid JSON format: {"tool_name": "...", "arguments": {...}}.
If the request is unrelated to these tools, return null."""

INTENT_PROMPT = "Extract the intent."


def label_to_str(label) -> str:
    if label is None:
        return "null"
    return json.dumps(label, ensure_ascii=False, separators=(",", ":"))


# ── Dataset ──────────────────────────────────────────────────────────────────

class AudioToolDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "audio_path":     item["audio_path"],
            "assistant_text": label_to_str(item["label"]),
        }


# ── Merge helpers ─────────────────────────────────────────────────────────────

def merge_text_adapter(model, adapter_dir: str) -> None:
    """
    Загружает text SFT adapter и вмёрживает его в model.thinker in-place.
    Пробует два метода:
      1. merge_and_unload() — стандартный PEFT-способ
      2. merge_adapter() + ручное переприсваивание — fallback если (1) вернул PeftModel
    После успеха model.thinker должен иметь тип Qwen2_5OmniThinkerForConditionalGeneration.
    """
    logger.info(f"Loading text SFT adapter from {adapter_dir} ...")
    peft_thinker = PeftModel.from_pretrained(model.thinker, adapter_dir)

    # Метод 1: стандартный merge_and_unload
    merged = peft_thinker.merge_and_unload()
    thinker_type = type(merged).__name__

    if "PeftModel" not in thinker_type:
        model.thinker = merged
        logger.info(f"[DIAG] merge_and_unload() succeeded. type(thinker)={thinker_type}")
        return

    # Метод 2: merge_adapter() + base_model.model
    logger.warning(
        f"[DIAG] merge_and_unload() returned {thinker_type} — PeftModel wrapper still present! "
        "Falling back to merge_adapter() + base_model.model."
    )
    peft_thinker.merge_adapter()
    model.thinker = peft_thinker.base_model.model
    thinker_type2 = type(model.thinker).__name__
    logger.info(f"[DIAG] After fallback merge: type(thinker)={thinker_type2}")


# ── Quick eval (Pipeline A — text input) ─────────────────────────────────────

def quick_eval_pipeline_a(model, processor, manifest_path: str, n: int = DIAG_SAMPLES):
    """
    Запускает Pipeline A (text → JSON) на n случайных примерах.
    Если text SFT merge сработал — accuracy должна быть ~99%, иначе ~98%.
    """
    if not Path(manifest_path).exists():
        logger.warning(f"[DIAG] eval manifest not found: {manifest_path}")
        return

    import random
    with open(manifest_path, "r", encoding="utf-8") as f:
        all_items = json.load(f)
    sample = random.sample(all_items, min(n, len(all_items)))

    device = next(model.parameters()).device
    model.eval()
    correct = 0

    with torch.no_grad():
        for ex in sample:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",   "content": [{"type": "text", "text": ex["text"]}]},
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=text, return_tensors="pt").to(device)
            out = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            resp = processor.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            gt_str = json.dumps(ex["label"], ensure_ascii=False) if ex["label"] else "null"
            is_ok = resp.startswith(gt_str[:20]) or gt_str[:20] in resp
            correct += int(is_ok)
            logger.info(
                f"  [DIAG A]  pred={resp[:70]!r}  gt={gt_str[:50]!r}  {'✓' if is_ok else '✗'}"
            )

    logger.info(f"[DIAG] Quick eval Pipeline A: {correct}/{n} correct")
    logger.info(
        "[DIAG] If ~5/5 → merge WORKED (text SFT active). "
        "If ~0-1/5 and answers look like baseline → merge FAILED."
    )
    model.train()


# ── Build inputs ─────────────────────────────────────────────────────────────

def build_inputs(processor, audio_path: str, assistant_text: str, max_len: int, device):
    conv_full = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text",  "text": INTENT_PROMPT},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]
    conv_prompt = conv_full[:-1]

    text_full = processor.apply_chat_template(conv_full, tokenize=False, add_generation_prompt=False)
    audios_full, _, _ = process_mm_info(conv_full, use_audio_in_video=False)
    enc_full = processor(
        text=text_full, audio=audios_full, return_tensors="pt",
        padding=True, use_audio_in_video=False,
    )

    text_prompt = processor.apply_chat_template(conv_prompt, tokenize=False, add_generation_prompt=True)
    audios_prompt, _, _ = process_mm_info(conv_prompt, use_audio_in_video=False)
    enc_prompt = processor(
        text=text_prompt, audio=audios_prompt, return_tensors="pt",
        padding=True, use_audio_in_video=False,
    )

    input_ids = enc_full["input_ids"][0]
    if input_ids.shape[0] > max_len:
        input_ids = input_ids[:max_len]

    attention_mask = enc_full["attention_mask"][0]
    if attention_mask.shape[0] > max_len:
        attention_mask = attention_mask[:max_len]

    prompt_len = enc_prompt["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:prompt_len] = -100

    if (labels != -100).sum() == 0:
        raise ValueError(
            f"All labels masked out (prompt_len={prompt_len}, seq_len={input_ids.shape[0]}). "
            "Increase MAX_SEQ_LEN."
        )

    result = {
        "input_ids":      input_ids.unsqueeze(0).to(device),
        "attention_mask": attention_mask.unsqueeze(0).to(device),
        "labels":         labels.unsqueeze(0).to(device),
    }
    if "input_features" in enc_full:
        result["input_features"] = enc_full["input_features"].to(device)
    if "feature_attention_mask" in enc_full:
        result["feature_attention_mask"] = enc_full["feature_attention_mask"].to(device)
    return result


# ── Save ─────────────────────────────────────────────────────────────────────

def _save_lora(thinker, lora_cfg, output_dir, processor=None, base_model_id=None):
    """Сохраняет LoRA-веса тинкера. base_model_id прописывается в adapter_config.json."""
    os.makedirs(output_dir, exist_ok=True)
    if base_model_id is not None:
        lora_cfg.base_model_name_or_path = base_model_id
    lora_cfg.save_pretrained(output_dir)
    lora_sd = {k: v.cpu() for k, v in thinker.state_dict().items() if "lora_" in k}
    torch.save(lora_sd, os.path.join(output_dir, "adapter_model.bin"))
    if processor is not None:
        processor.save_pretrained(output_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"Loading base model {MODEL_ID}...")
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    # ── Merge text SFT adapter ───────────────────────────────────────────────
    if os.path.isdir(TEXT_SFT_ADAPTER):
        merge_text_adapter(model, TEXT_SFT_ADAPTER)
    else:
        logger.warning(
            f"Text SFT adapter not found at {TEXT_SFT_ADAPTER}. "
            "Training audio SFT on top of BASE model (no text SFT)."
        )

    # ── DIAG: quick eval before training ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("DIAGNOSTICS: Pipeline A eval before audio training")
    logger.info("Expected ~5/5 if text merge worked, ~0-1/5 if not.")
    logger.info("=" * 60)
    quick_eval_pipeline_a(model, processor, EVAL_MANIFEST)
    logger.info("=" * 60)

    # ── Apply audio LoRA ─────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    inject_adapter_in_model(lora_cfg, model.thinker)

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.thinker.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info(f"Loading audio training data from {TRAIN_FILE}...")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    logger.info(f"  {len(train_data)} audio examples")
    dataset = AudioToolDataset(train_data)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=0.01,
    )
    total_opt_steps = (len(dataset) // GRAD_ACCUM) * NUM_EPOCHS

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        return max(0.1, 1.0 - (step - WARMUP_STEPS) / max(1, total_opt_steps - WARMUP_STEPS))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    opt_step = 0
    running_loss = 0.0
    device = next(model.thinker.parameters()).device

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        epoch_loss = 0.0

        for i, item in tqdm(enumerate(dataset), total=len(dataset),
                             desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            try:
                inputs = build_inputs(
                    processor=processor,
                    audio_path=item["audio_path"],
                    assistant_text=item["assistant_text"],
                    max_len=MAX_SEQ_LEN,
                    device=device,
                )
            except Exception as e:
                logger.warning(f"  [SKIP] item {i}: {e}")
                continue

            outputs = model.thinker(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                input_features=inputs.get("input_features"),
                feature_attention_mask=inputs.get("feature_attention_mask"),
            )
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            running_loss += loss.item() * GRAD_ACCUM
            epoch_loss   += loss.item() * GRAD_ACCUM

            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                if opt_step % LOG_INTERVAL == 0:
                    avg = running_loss / LOG_INTERVAL
                    logger.info(f"  step={opt_step}  loss={avg:.4f}  "
                                f"lr={scheduler.get_last_lr()[0]:.2e}")
                    running_loss = 0.0

                if opt_step % SAVE_INTERVAL == 0:
                    ckpt_dir = f"{OUTPUT_DIR}/checkpoint-{opt_step}"
                    _save_lora(model.thinker, lora_cfg, ckpt_dir, base_model_id=MODEL_ID)
                    logger.info(f"  Saved checkpoint → {ckpt_dir}")

        avg_epoch = epoch_loss / len(dataset)
        logger.info(f"Epoch {epoch+1} done — avg loss: {avg_epoch:.4f}")

    # ── Save final adapter ────────────────────────────────────────────────────
    _save_lora(model.thinker, lora_cfg, OUTPUT_DIR, processor, base_model_id=MODEL_ID)
    logger.info(f"\nFinal LoRA adapter saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
