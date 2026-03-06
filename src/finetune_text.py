"""
finetune_text.py
================
LoRA Text SFT for Qwen2.5-Omni-7B.
Fine-tunes the language model (thinker) on text→tool_call pairs.
Audio encoder and talker are NOT touched.

Dependencies (add to omni_env):
    pip install peft trl bitsandbytes

Run from project root:
    python src/finetune_text.py

Output: checkpoints/qwen_omni_text_sft/   (LoRA adapter ~200 MB)
"""

import os
import json
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID            = "Qwen/Qwen2.5-Omni-7B"
TRAIN_FILE          = "data/training_data_text.json"
OUTPUT_DIR          = "checkpoints/qwen_omni_text_sft"
LORA_RANK           = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
NUM_EPOCHS          = 2
BATCH_SIZE          = 4
GRAD_ACCUM          = 4       # effective batch size = BATCH_SIZE × GRAD_ACCUM = 16
LEARNING_RATE       = 2e-4
MAX_SEQ_LEN         = 256
WARMUP_STEPS        = 50
LOG_INTERVAL        = 20      # log loss every N optimizer steps
SAVE_INTERVAL       = 200     # save checkpoint every N optimizer steps
# ────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a banking assistant.
You must NOT generate audio or speech output. Generate TEXT ONLY.

Available tools:
1. transfer_money(recipient: str, amount: float)
2. get_exchange_rate(currency_from: str, currency_to: str)

Output ONLY valid JSON format: {"tool_name": "...", "arguments": {...}}.
If the request is unrelated to these tools, return null."""


def label_to_str(label) -> str:
    if label is None:
        return "null"
    return json.dumps(label, ensure_ascii=False, separators=(",", ":"))


# ── Dataset ──────────────────────────────────────────────────────────────────

class TextToolDataset(Dataset):
    def __init__(self, data: list, tokenizer, max_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        assistant_text = label_to_str(item["label"])

        messages_full = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": item["text"]},
            {"role": "assistant", "content": assistant_text},
        ]
        messages_prompt = messages_full[:-1]

        full_text   = self.tokenizer.apply_chat_template(
            messages_full,   tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )

        full_enc   = self.tokenizer(full_text,   truncation=True, max_length=self.max_len)
        prompt_enc = self.tokenizer(prompt_text, truncation=True, max_length=self.max_len)

        input_ids      = torch.tensor(full_enc["input_ids"],      dtype=torch.long)
        attention_mask = torch.tensor(full_enc["attention_mask"], dtype=torch.long)
        prompt_len     = len(prompt_enc["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_len] = -100          # mask system + user tokens

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def pad_collate(batch):
    """Right-pad all tensors to the longest sequence in the batch."""
    max_len = max(x["input_ids"].shape[0] for x in batch)
    ids, masks, lbls = [], [], []
    for item in batch:
        n = item["input_ids"].shape[0]
        p = max_len - n
        ids.append(torch.cat([item["input_ids"],
                               torch.zeros(p, dtype=torch.long)]))
        masks.append(torch.cat([item["attention_mask"],
                                 torch.zeros(p, dtype=torch.long)]))
        lbls.append(torch.cat([item["labels"],
                                torch.full((p,), -100, dtype=torch.long)]))
    return {
        "input_ids":      torch.stack(ids),
        "attention_mask": torch.stack(masks),
        "labels":         torch.stack(lbls),
    }


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load processor & model ──────────────────────────────────────────────
    logger.info(f"Loading {MODEL_ID}...")
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── Apply LoRA only to thinker (language model) ─────────────────────────
    # model.thinker is the Qwen2.5 LM; audio_tower and talker stay frozen.
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA to the thinker sub-model directly for clean separation
    thinker = model.thinker
    thinker = get_peft_model(thinker, lora_cfg)
    thinker.print_trainable_parameters()
    model.thinker = thinker

    # ── Dataset & Dataloader ────────────────────────────────────────────────
    logger.info(f"Loading training data from {TRAIN_FILE}...")
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    logger.info(f"  {len(train_data)} examples")

    dataset    = TextToolDataset(train_data, tokenizer, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=pad_collate, num_workers=0)

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=0.01
    )

    # Linear warmup scheduler
    total_optimizer_steps = (len(dataloader) // GRAD_ACCUM) * NUM_EPOCHS
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        return max(0.1, 1.0 - (step - WARMUP_STEPS) / max(1, total_optimizer_steps - WARMUP_STEPS))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    opt_step = 0
    running_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        epoch_loss = 0.0

        for step, batch in tqdm(enumerate(dataloader),
                                 total=len(dataloader),
                                 desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            input_ids      = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels         = batch["labels"].to(model.device)

            # Forward through thinker only (text path)
            outputs = model.thinker(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            running_loss += loss.item() * GRAD_ACCUM
            epoch_loss   += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
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
                    model.thinker.save_pretrained(ckpt_dir)
                    logger.info(f"  Saved checkpoint → {ckpt_dir}")

        avg_epoch = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} done — avg loss: {avg_epoch:.4f}")

    # ── Save final adapter ───────────────────────────────────────────────────
    model.thinker.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    logger.info(f"\nFinal LoRA adapter saved → {OUTPUT_DIR}")
    logger.info("Load for eval:  model.thinker = PeftModel.from_pretrained(model.thinker, OUTPUT_DIR)")


if __name__ == "__main__":
    main()
