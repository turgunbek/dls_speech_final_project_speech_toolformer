"""
finetune_audio.py
=================
LoRA Audio SFT for Qwen2.5-Omni-7B.
Дообучает языковую модель (thinker) на парах (audio → tool_call_JSON).
Audio encoder (audio_tower) и talker заморожены.

Это соответствует Pipeline C: audio → JSON напрямую.

Dependencies (omni_env):
    pip install peft trl

Run from project root:
    python src/finetune_audio.py

Output: checkpoints/qwen_omni_audio_sft/  (LoRA adapter ~200 MB)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
import logging
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
from peft import LoraConfig, PeftModel, TaskType, inject_adapter_in_model
from qwen_omni_utils import process_mm_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID           = "Qwen/Qwen2.5-Omni-7B"
TEXT_SFT_ADAPTER   = "checkpoints/qwen_omni_text_sft"   # стартуем от text SFT весов
TRAIN_FILE         = "data/training_audio_manifest.json"
OUTPUT_DIR         = "checkpoints/qwen_omni_audio_sft"
LORA_RANK          = 16
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.05
NUM_EPOCHS         = 2
BATCH_SIZE         = 1        # audio batching — держим 1 для простоты
GRAD_ACCUM         = 16       # effective batch size = 16
LEARNING_RATE      = 1e-4     # чуть ниже, чтобы не перебить text SFT
MAX_SEQ_LEN        = 1024
WARMUP_STEPS       = 30
LOG_INTERVAL       = 10
SAVE_INTERVAL      = 100
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
    """Returns raw conversation dicts; collation done in the training loop."""

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


# ── Training ─────────────────────────────────────────────────────────────────

def build_inputs(processor, audio_path: str, assistant_text: str, max_len: int, device):
    """
    Строит input tensors для одного аудио-примера.
    Возвращает словарь пригодный для model.forward().
    labels содержат -100 на позициях system+user tokens.
    """
    # Full conversation (for computing loss on assistant part)
    conv_full = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text",  "text": INTENT_PROMPT},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]
    # Prompt only (no assistant) — to find mask boundary
    conv_prompt = conv_full[:-1]

    # Tokenize full conversation
    text_full = processor.apply_chat_template(conv_full, tokenize=False, add_generation_prompt=False)
    audios_full, _, _ = process_mm_info(conv_full, use_audio_in_video=False)
    enc_full = processor(
        text=text_full,
        audio=audios_full,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )

    # Tokenize prompt only (no audio needed for length computation)
    text_prompt = processor.apply_chat_template(conv_prompt, tokenize=False, add_generation_prompt=True)
    audios_prompt, _, _ = process_mm_info(conv_prompt, use_audio_in_video=False)
    enc_prompt = processor(
        text=text_prompt,
        audio=audios_prompt,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
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

    # Если после усечения нет ни одного обучаемого токена → скип
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

    # Audio features — keep as-is (already on CPU from processor)
    if "input_features" in enc_full:
        result["input_features"] = enc_full["input_features"].to(device)
    if "feature_attention_mask" in enc_full:
        result["feature_attention_mask"] = enc_full["feature_attention_mask"].to(device)

    return result


def _save_lora(thinker, lora_cfg, output_dir, processor=None):
    """Сохраняет LoRA-веса тинкера в формате, совместимом с PeftModel.from_pretrained."""
    os.makedirs(output_dir, exist_ok=True)
    lora_cfg.save_pretrained(output_dir)          # adapter_config.json
    lora_sd = {k: v.cpu() for k, v in thinker.state_dict().items() if "lora_" in k}
    torch.save(lora_sd, os.path.join(output_dir, "adapter_model.bin"))
    if processor is not None:
        processor.save_pretrained(output_dir)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    logger.info(f"Loading base model {MODEL_ID}...")
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},   # один GPU — иначе accelerate-хуки ломают PEFT
    )

    # ── Load text SFT adapter if available, else apply fresh LoRA ───────────
    if os.path.isdir(TEXT_SFT_ADAPTER):
        logger.info(f"Loading text SFT adapter from {TEXT_SFT_ADAPTER}...")
        model.thinker = PeftModel.from_pretrained(model.thinker, TEXT_SFT_ADAPTER)
        # Merge text adapter into weights for clean re-application of audio LoRA
        logger.info("Merging text SFT adapter weights...")
        model.thinker = model.thinker.merge_and_unload()

    # Apply LoRA for audio SFT
    # inject_adapter_in_model модифицирует тинкер in-place без PEFT-обёртки,
    # что позволяет сохранить корректный forward внешней модели.
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

    # Freeze всё, потом разморозить только LoRA-параметры тинкера
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.thinker.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"trainable params: {trainable:,} || all params: {total:,} "
                f"|| trainable%: {100 * trainable / total:.4f}")

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

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    opt_step = 0
    running_loss = 0.0

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
                    device=next(model.thinker.parameters()).device,
                )
            except Exception as e:
                logger.warning(f"  [SKIP] item {i}: {e}")
                continue

            # Qwen2_5OmniForConditionalGeneration.forward не реализован —
            # обучение идёт через thinker напрямую (он обрабатывает audio_features)
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
                    _save_lora(model.thinker, lora_cfg, ckpt_dir)
                    logger.info(f"  Saved checkpoint → {ckpt_dir}")

        avg_epoch = epoch_loss / len(dataset)
        logger.info(f"Epoch {epoch+1} done — avg loss: {avg_epoch:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    _save_lora(model.thinker, lora_cfg, OUTPUT_DIR, processor)
    logger.info(f"\nFinal LoRA adapter saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
