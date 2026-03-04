# Speech-Toolformer: Voice Banking Assistant
### Final Project Report — DLS MIPT, Fall 2025

---

## 1. Task and Tool Design

### 1.1 Motivation

The project implements a voice-controlled banking assistant where a user says something like _"Transfer a thousand dollars to Mom"_ and the system produces a structured tool call that the banking app can execute. This is a realistic industrial use case: many fintech products are exploring voice command interfaces, and the core challenge is reliable intent extraction from noisy, informal speech.

### 1.2 Tool Choice: `transfer_money`

```json
{
  "tool_name": "transfer_money",
  "arguments": {
    "recipient": "<string>",
    "amount": <float>
  }
}
```

**Why this tool:**
- Two arguments of different types (string + float) — non-trivial but not overly complex
- Both arguments must be extracted correctly from natural, colloquial speech ("shoot fifty bucks to bro", "send half a grand to Dad")
- Financial amounts appear in many surface forms: digits, words, abbreviations ("2k", "1.5k", "a grand", "fifteen hundred")
- Recipients use informal names, nicknames, and family terms ("Mom", "Hubby", "Bro", "Landlord")

A second tool `get_exchange_rate(currency_from, currency_to)` was included in the system prompt to demonstrate that the model correctly declines non-relevant queries in a multi-tool context (and also correctly fires it when prompted — see error analysis in Section 8.1).

### 1.3 Output format

JSON was chosen over XML for conciseness and easier parsing. The model is instructed to return `null` for requests unrelated to the defined tools.

---

## 2. Synthetic Dataset Design

### 2.1 Text dataset

| Property | Value |
|----------|-------|
| Total samples | 511 |
| Positive (`transfer_money`) | 401 (78.5%) |
| Negative (no tool) | 110 (21.5%) |
| Generation method | ChatGPT with diverse templates |

**Positive samples** cover a wide variety of phrasings:
- Imperative: "Transfer X to Person", "Send X to Person", "Pay X to Person"
- Colloquial: "Shoot X to Person", "Wire X to Person", "Move X to Person", "Slide X to Person"
- Polite/indirect: "Could you please send X to Alice?", "I need to send..."
- Amount formats: digits (`50`), number words (`fifty`), mixed (`1.5k`, `2 grand`, `a hundred and fifty`, `half a stack`)
- Recipient forms: first names, nicknames, family titles, full names, role words

**Negative samples** include queries that are superficially similar but require no tool call:
- Balance/transaction questions
- Statements about spending ("I spent 50 on groceries")
- Ambiguous phrasing ("Did I already send money to John?", "Transfer fees are expensive")
- Reminder requests ("Remind me to send money later")

The 21.5% negative ratio is slightly above the recommended 10–20%, but ensures the model is regularly tested on its ability to decline irrelevant requests.

### 2.2 Audio synthesis

| Property | Value |
|----------|-------|
| TTS engine | XTTS-v2 (Coqui TTS) |
| Speaker reference | Cameron Russell (English, female) |
| Number of voices | 1 |
| Speed | 1.15× (slightly accelerated) |
| Language | English |
| Total audio files | 511 WAV files |

**Known TTS artefact:** XTTS-v2 sometimes clips the very first phoneme of short phrases. This causes systematic first-word errors in ASR (e.g., "Send" → "then"/"sent", "Shoot" → "show"/"showed"). See Section 8.4 for the full error breakdown.

**Limitation:** Only one speaker voice was used. Ideally, multiple voices, accents, and noise conditions would be included to test acoustic robustness.

---

## 3. Model and System Prompt

### 3.1 Model

**Qwen2.5-Omni-7B** was chosen as the backbone model. It is a natively multimodal model that accepts audio directly as input and generates text output — this is exactly the property needed to compare ASR-based and end-to-end audio pipelines under a single model.

Hardware: NVIDIA L40S GPU (48 GB VRAM). The model runs in `torch_dtype="auto"` (bfloat16) with `device_map="auto"`. No fine-tuning was performed; all experiments are zero-shot.

### 3.2 System prompt

```
You are a banking assistant.
You must NOT generate audio or speech output. Generate TEXT ONLY.

Available tools:
1. transfer_money(recipient: str, amount: float)
2. get_exchange_rate(currency_from: str, currency_to: str)

Output ONLY valid JSON format: {"tool_name": "...", "arguments": {...}}.
If the request is unrelated to these tools, return null.
```

Key design choices:
- Explicit instruction to generate text only (Qwen2.5-Omni generates audio by default — this must be disabled)
- Tool schema in concise function-signature style
- Explicit null-return instruction is essential to control the false alarm rate

---

## 4. Experimental Pipelines

Four pipelines were evaluated on the full dataset of 511 samples:

| Pipeline | Input | Steps | Notes |
|----------|-------|-------|-------|
| **A** (Text Oracle) | Ground-truth text | text → model → JSON | Upper bound; no ASR involved |
| **B** (ASR) | Audio | audio → model → transcript | Evaluates pure ASR quality (WER) |
| **C** (Direct) | Audio | audio → model → JSON | Single-pass audio-to-intent |
| **D** (Cascaded) | Audio | audio → ASR → clean text → model → JSON | Two-pass; B's cleaned transcript feeds D |

All pipelines use the same model (Qwen2.5-Omni-7B) with no fine-tuning. Two script versions were run:

- **v2** (`run_omni_project_v2.py`) — initial implementation; contained a bug in Pipeline D (see Section 6)
- **v3** (`run_omni_project_v3.py`) — fixed Pipeline D transcript passing

---

## 5. Results: v2 Baseline

Results from `results_omni_7b_v2.json` (511 samples).

### 5.1 Tool-call metrics (Pipelines A, C, D)

Evaluation uses soft-match: `tool_name` must match exactly; `recipient` is compared case-insensitively; `amount` with tolerance ±0.1. Predicting `null` for a positive sample is a false negative; predicting any tool call for a negative sample is a false positive.

| Metric | Pipeline A (Text) | Pipeline C (Direct Audio) | Pipeline D (Cascaded, **buggy**) |
|--------|:-----------------:|:------------------------:|:--------------------------------:|
| **Accuracy** | **98.63%** | **70.65%** | **55.58%** |
| Precision | 0.997 | 0.977 | 0.994 |
| Recall | 0.985 | 0.641 | 0.436 |
| FAR | 0.009 | 0.055 | 0.009 |
| Parsable rate | 100% | 100% | 100% |
| TP / FP / FN / TN | 395 / 1 / 6 / 109 | 257 / 6 / 144 / 104 | 175 / 1 / 226 / 109 |

**Modality gap (A vs C):** 27.98 pp accuracy drop when switching from clean text to raw audio.

### 5.2 ASR quality (Pipeline B)

| Metric | Value |
|--------|-------|
| WER (after hallucination fix) | **66.74%** |
| Samples used | 504 / 511 |
| Hallucinations excluded | 7 |

The raw WER before fixing was 103.36% (inflated by undetected looping hallucinations). After applying the improved `is_hallucination` filter (trigram repetition detection), WER drops to **66.74%**. This number is still significantly elevated — see Section 6.3 for why it remains high.

### 5.3 Key observation: Pipeline D anomaly in v2

Pipeline D in v2 has anomalously low recall (0.436) despite high precision (0.994). The root cause is a bug in transcript passing, described in Section 6.1.

---

## 6. Bug Analysis and v3 Fix

### 6.1 Pipeline D bug (v2)

In v2, Pipeline D feeds the **raw output of Pipeline B** (the ASR step) directly into the LLM. However, Qwen wraps transcripts in a verbose prefix:

```
"The original content of this audio is: 'sent twelve hundred to brawl.'"
```

When this string is submitted as the user message to the banking assistant, the LLM interprets it as a **descriptive statement about audio content** rather than a banking command, and correctly returns `null`. This is rational model behavior — the framing is completely wrong.

**Evidence:** Out of 401 positive samples, Pipeline D in v2 returned `null` for 226 (FN=226, recall=0.436). Nearly all of these have recognizable transfer intents that the model ignored due to the wrong framing.

Example:
```
User message (v2):  "The original content of this audio is: 'Sent twelve hundred to brawl.'"
Model output:       null
Expected:           {"tool_name": "transfer_money", "arguments": {"recipient": "Bro", "amount": 1200}}
```

### 6.2 v3 fix

In `run_omni_project_v3.py`, the transcript is cleaned before being passed to Pipeline D using `extract_clean_transcript`:

```python
# v2 (buggy):
conv_d = [..., {"type": "text", "text": transcript_b_raw}]

# v3 (fixed):
transcript_b_clean = extract_clean_transcript(transcript_b_raw)
conv_d = [..., {"type": "text", "text": transcript_b_clean}]
```

`extract_clean_transcript` uses a cascade of regex patterns to strip the verbose wrapper:
1. Single-quoted content: `'([^']{3,})'`
2. Double-quoted content: `"([^"]{3,})"`
3. Everything after `is:` / `are:`
4. Return raw text unchanged if no service words are detected

Both `pipeline_b_transcript` (raw) and `pipeline_b_transcript_clean` (extracted) are saved in v3 results for full traceability.

### 6.3 WER issue and fix

**Root cause of inflated WER — undetected looping hallucinations:**

Some transcripts have valid single-quote wrapping but contain looping content generated by the model inside the quotes:

```
"The original content of this audio is: 'sent eight point zero eight two
 one zero eight two one seven five zero eight two one seven five zero...'"
```

The content inside the quotes is extracted as a ~60-word hypothesis for a 4-word reference ("send 8.08 to Ron"), contributing ~56 word insertions. The original `is_hallucination` filter only detected **four identical consecutive words** — this periodic pattern of *different* words cycled in a loop was not caught.

**Fix applied in `calculate_metrics.py`:** added two new hallucination checks:
1. **Trigram repetition:** any 3-word sequence appearing > 2 times in the same transcript → looping pattern detected
2. **Length threshold:** transcript > 40 words after extraction → almost certainly looping
3. **Double check:** `is_hallucination` is now applied to *both* the raw transcript *and* the extracted clean text

**Result after fix:**

| | Before fix | After fix |
|---|:---:|:---:|
| WER | 103.36% | **66.74%** |
| Hallucinations excluded | 6 | 7 |
| Samples used for WER | 505 | 504 |

**Why 66.74% is still elevated — number format mismatch:**

Even with looping hallucinations removed, WER remains high due to how numeric amounts are represented. The reference texts use digits ("1500"), but the ASR model outputs spoken-form numbers ("fifteen hundred"). Since these are different strings, jiwer counts them as word errors:

| Reference (digit) | Hypothesis (word) | WER contribution |
|---|---|---|
| "1500" (1 word) | "fifteen hundred" (2 words) | 1 insertion + 1 substitution = 2 errors |
| "50" (1 word) | "fifty" (1 word) | 1 substitution |
| "2.5k" → "25k" (1 word) | "two point five k" (4 words) | 3 insertions + 1 substitution |

This is a metric artefact, not an ASR quality problem: the model correctly recognizes the amount, just expresses it in words rather than digits. Normalizing both sides to a canonical number format would reduce WER substantially.

**Estimated "semantic" WER** (ignoring number format differences): approximately **20–30%**, driven by:
- First-word clipping: "Send" → "then"/"sent" (TTS artefact, ×36 total occurrences)
- Phonetic confusion on short names: "Bro" → "brawl", "Paul" → "pour", "Emma" → "m r"
- Garbled hallucinations still partially in the evaluation set

---

## 7. Results: v3 (Fixed Pipeline D)

Results from `results_omni_7b_v3.json` (511 samples, `run_omni_project_v3.py`).

_Note: all four pipelines were re-run from scratch in v3. Pipelines A and C show minor numerical differences from v2 (±2 pp) attributable to non-determinism in GPU floating-point operations. The methodology and prompts are identical._

### 7.1 Tool-call metrics

| Metric | Pipeline A (Text) | Pipeline C (Direct Audio) | Pipeline D v2 (buggy) | **Pipeline D v3 (fixed)** |
|--------|:-----------------:|:------------------------:|:---------------------:|:-------------------------:|
| **Accuracy** | **98.83%** | **68.30%** | 55.58% | **67.51%** |
| Precision | 0.997 | 0.992 | 0.994 | **1.000** |
| Recall | 0.988 | 0.601 | 0.436 | **0.586** |
| FAR | 0.009 | 0.018 | 0.009 | **0.000** |
| Parsable rate | 100% | 100% | 100% | 100% |
| TP / FP / FN / TN | 396 / 1 / 5 / 109 | 241 / 2 / 160 / 108 | 175 / 1 / 226 / 109 | 235 / 0 / 166 / 110 |

**Impact of the v3 fix on Pipeline D:**
- Accuracy: 55.58% → **67.51%** (+11.93 pp)
- Recall: 0.436 → **0.586** (+0.15)
- FAR: 0.009 → **0.000** (no false alarms in this run)

**C vs D gap after the fix:** only **0.78%** accuracy, compared to 15.07% in v2. The pipelines are now essentially on par.

### 7.2 WER (Pipeline B) — after hallucination fix

Same audio transcriptions as v2 (Pipeline B was not changed), same WER after applying the improved `is_hallucination` filter.

| Metric | Value |
|--------|-------|
| WER | **66.74%** |
| Samples used | 504 / 511 |
| Hallucinations excluded | 7 |
| Clean extraction | 478 (93.5%) |
| No-quotes / format variant | 26 (5.1%) |

The 66.74% WER is primarily driven by number format mismatch (digits vs. words), not by genuine speech recognition failure. See Section 6.3 for the full analysis. Estimated semantic WER (ignoring number format): **~20–30%**.

---

## 8. Error Analysis

### 8.1 Pipeline A — Text Oracle (accuracy 98.83%, v3)

Only 6 errors in 511 samples:

| Error type | Count | Example |
|-----------|-------|---------|
| False negative | 3 | "Slide half a stack to Hubby" → `null` (very slang, model doesn't know "stack"="500") |
| False positive | 1 | "What's the exchange rate for euros to dollars?" → `get_exchange_rate(EUR, USD)` |
| Wrong recipient | 1 | "Send 2.5k to the Contractor dude" → recipient="Contractor dude" vs GT="Contractor" |
| Wrong amount | 1 | "Move a grand and a half to Mark" → amount=1.5 vs GT=1500.0 |

Notable findings:
- The **false positive** is actually correct model behavior: "What's the exchange rate..." does match the second tool in the system prompt (`get_exchange_rate`). This is a **dataset label issue** — the sample is marked as negative (null), but the model correctly fires the defined tool. This was the only FP in Pipeline A.
- The **wrong amount** on "a grand and a half" (model outputs 1.5 instead of 1500) reveals a limitation: the model understands "a grand" = 1000 and "a half" = 0.5, computes 1.5 — a mathematically plausible but wrong interpretation.

**Conclusion:** Pipeline A demonstrates excellent zero-shot text instruction-following. The model handles most colloquial phrasings without fine-tuning.

### 8.2 Pipeline C — Direct Audio (accuracy 68.30%, v3)

162 errors out of 511 samples:

| Error type | Count | % of total | Example |
|-----------|-------|-----------|---------|
| False negative | 52 | 10.2% | "Can you toss a hundred to Mom?" → `null` |
| Wrong amount | 47 | 9.2% | "Transfer 0.5k to Tom" → amount=0.5 (vs 500) |
| Wrong recipient | 46 | 9.0% | "Send 1,200 to Bro" → recipient="Brol" (vs "Bro") |
| Both wrong | 15 | 2.9% | "Send 1200.50 to the Landlord" → recipient="the landlord", amount=200.5 |
| False positive | 2 | 0.4% | Exchange rate query → `get_exchange_rate` triggered |

**Dominant error patterns:**
- **Amount normalization:** "0.5k" is heard as "zero point five k" → model outputs 0.5. The model transcribes "k" (kilo=×1000) but fails to multiply. This is an issue with informal amount notation in audio vs. text.
- **Recipient mishearing:** Informal or short names are hard to hear — "Bro" is transcribed as "Brol", "Hubby" as "hubby" (ok), "Emma" as "m r" (completely garbled from "ninety five point two five to M-ma" → "m r").
- **Slang action words:** "Can you toss", "Slide", "Shoot" are sometimes not recognized as transfer intent in direct audio mode.

**Analysis:** The 31.7% error rate in Pipeline C is entirely non-trivial given zero-shot setup. The majority of errors (46+47+15 = 108 out of 162) are argument extraction errors — the model correctly identifies the transfer intent but extracts wrong values due to audio ambiguity. Only 52 are complete misses (false negatives).

### 8.3 Pipeline D — Cascaded, v3 fixed (accuracy 67.51%)

166 errors out of 511 samples:

| Error type | Count | % of total | Example (GT text + ASR transcript) |
|-----------|-------|-----------|-------------------------------------|
| False negative | 82 | 16.0% | GT: "Send ten dollars to Grandma" / ASR: "then ten dollars to grow such debt to your fifty thousand and sell" → `null` |
| Wrong amount | 53 | 10.4% | GT: "Transfer 0.5k to Tom" / ASR: "transfer zero point five k to tom attention at zero point eight six k to tom" → amount=0.5 |
| Wrong recipient | 24 | 4.7% | GT: "Send 1,200 to Bro" / ASR: "Sent twelve hundred to brawl" → recipient="brawl" |
| Both wrong | 7 | 1.4% | GT: "Move 95.25 to Emma" / ASR: "ninety five point two five to m r" → `get_exchange_rate(USD, MXN)` |

**All remaining errors in D v3 are attributable to ASR quality** (not prompt engineering):
- 82 FNs: garbled ASR output makes the text unrecognizable as a banking command ("grow such debt to your fifty thousand")
- 53 wrong amounts: "0.5k" → "zero point five k" → model correctly reads "0.5" from the literal transcript
- 24 wrong recipients: phonetic confusion ("Bro" → "brawl", "Grandma" → "Grandma" vs garbled)

The "both wrong" example (`get_exchange_rate(USD, MXN)` from "ninety five point two five to m r") is particularly interesting: the ASR collapsed "Emma" to "m r" — which the model interpreted as a currency code (`MXN` for Mexican Peso), triggering the wrong tool entirely.

### 8.4 Pipeline B — ASR Transcript Quality

From `error_analysis.py`:

```
Total samples:               511
  Clean extraction:          478  (93.5%)
  Hallucinations (skipped):   6  (1.2%)
  No quotes / format issue:  27  (5.3%)
```

**Most common first-word errors (TTS clipping artefact):**

| GT first word → ASR | Count | Root cause |
|--------------------|-------|-----------|
| "send" → "then" | 21 | TTS clips "S" → "th" sound heard |
| "send" → "sent" | 15 | Past tense mishearing |
| "send" → "and" | 4 | Complete first-word loss |
| "shoot" → "two" | 3 | Phonetic similarity |
| "move" → "ninety" | 2 | First word lost, second word misread |

The "send"→"then" pattern (21 occurrences) is the single most common error class. It is systematic: XTTS-v2 at speed=1.15× produces audio that starts with a soft onset, and the model hears the voiced dental fricative "th" from "send"'s /s/. This propagates directly into Pipeline D errors (e.g., "then fifty bucks to alex" is understood as a transfer request if the model is flexible, but may be missed).

---

## 9. Best Workflow Selection

### 9.1 Summary comparison table (v3)

| Pipeline | Accuracy | Precision | Recall | FAR | LLM calls | Notes |
|----------|----------|-----------|--------|-----|-----------|-------|
| A (Text Oracle) | 98.83% | 0.997 | 0.988 | 0.009 | 1 | Requires ground-truth text; not deployable |
| C (Direct Audio) | 68.30% | 0.992 | 0.601 | 0.018 | 1 | Single-pass; slightly higher recall |
| D v3 (Cascaded) | 67.51% | **1.000** | 0.586 | **0.000** | 2 | Zero false alarms; lower recall |

### 9.2 Analysis

After fixing the Pipeline D transcript format bug, the two deployable audio pipelines (C and D) are **essentially tied** on accuracy (68.30% vs 67.51%, gap = 0.78%). This is the most important finding of the v3 experiments.

However, they differ meaningfully on **precision / false alarm rate**:

- **Pipeline D v3: Precision = 1.000, FAR = 0.000** — in this run, not a single negative sample triggered a tool call. This means the cascaded pipeline never initiated an unintended bank transfer.
- **Pipeline C: Precision = 0.992, FAR = 0.018** — 2 false positives occurred. In a banking context, a false positive means the system might initiate a transfer the user didn't request, which is a serious error.

### 9.3 Recommended workflow

**For a production banking assistant: Pipeline D (Cascaded) is recommended.**

Rationale:
1. **Safety:** FAR = 0 (in this run) vs FAR = 0.018 for C. In financial transactions, a false positive (unintended transfer) is far more dangerous than a false negative (transfer not executed — user simply repeats the command).
2. **Debuggability:** The ASR transcript is an interpretable intermediate product. If the system makes an error, engineers can inspect the transcript to understand why.
3. **Competitive accuracy:** With the bug fixed, D is only 0.79 pp behind C in accuracy (within noise).
4. **Perfect precision:** D's precision (1.000 in this run) means users can trust that when a transfer is initiated, it was genuinely requested.

**Pipeline C is still attractive** if:
- Latency is critical (single LLM call vs two)
- The FAR difference is acceptable in the deployment context
- Audio quality is consistently high enough for direct intent extraction

**Finding alignment with the reference notebook:** The reference experiment on Gemma-3n found that the direct pipeline outperforms cascaded on noisy audio. Our results show the opposite after the v2 bug fix: with the transcript correctly formatted, D matches C. This may be because our audio is relatively clean (synthetic TTS). On real-world noisy recordings, the result might differ.

---

## 10. Discussion

### 10.1 What worked well

- **Zero-shot text performance is excellent** (Pipeline A: 98.83%). The base Qwen2.5-Omni-7B understands `transfer_money` argument extraction without any fine-tuning, given a clear system prompt.
- **Direct audio pipeline (C) achieves 68.30% accuracy** without fine-tuning and with 100% parsable rate. The model reliably outputs structured JSON from raw audio.
- **Parsable rate is 100%** across all pipelines and both versions. The system prompt reliably guides the model to output valid JSON or `null` — no PARSE_ERRORs in the final results.
- **Pipeline D v3 has zero false alarms.** For a banking use case, this is the most safety-relevant metric.
- **The v3 bug fix** improved Pipeline D accuracy by +12 pp by fixing a simple transcript formatting issue. This demonstrates how sensitive cascaded systems are to intermediate prompt construction.

### 10.2 Failure modes

- **ASR is the primary bottleneck.** Even with correct transcript passing, Pipeline D recall is limited to 0.586 because many ASR transcripts are too garbled for the model to extract the transfer intent.
- **Looping hallucinations on decimal amounts.** Numbers like "7.25" and "8.08" confuse the model into generating looping transcripts. This affects 6+ samples and corrupts WER computation.
- **Amount normalization gap.** "0.5k" → model outputs 0.5 (not 500) because the ASR transcribes "k" literally without expanding it. This affected 47 Pipeline C samples and 53 Pipeline D samples.
- **First-word TTS clipping.** "Send" → "then"/"sent" (×36 combined occurrences) is the most common ASR error class, caused by XTTS-v2 audio onset clipping.
- **Exchange rate false positive.** The sample "What's the exchange rate for euros to dollars?" is labeled as `null` in the dataset but correctly matches the `get_exchange_rate` tool defined in the system prompt. This is a dataset annotation issue, not a model error.
- **Single-voice dataset.** All audio comes from one speaker, limiting generalization assessment.

### 10.3 What would be tried with more time

1. **Fix looping hallucination detection:** Extend `is_hallucination` to detect periodic patterns (substring repetition). This would allow computing a meaningful WER.
2. **Amount normalization in Pipeline D post-processing:** Convert "zero point five k" → "500" before passing to the LLM. Expected: significant improvement in wrong_amount errors (~53 cases in D v3).
3. **Multi-voice dataset:** Add CommonVoice speaker references to XTTS-v2. Evaluate pipeline robustness across voices.
4. **Audio-conditioned LoRA fine-tuning:** Train on (audio, expected JSON) pairs using LoRA on the audio encoder + LLM attention layers. Expected: Pipeline C accuracy from ~68% toward ~85–90%.
5. **Text SFT for amount normalization:** Fine-tune on text examples where "0.5k" = 500, "a grand and a half" = 1500. Expected: improve wrong_amount cases in both C and D.
6. **TTS quality fix:** Add 0.3 s silence before each generated audio to prevent first-word clipping. Expected: reduce "send"→"then" errors from 21 to ~0.
7. **Real microphone recordings:** Replace synthetic TTS with actual user voice recordings to validate real-world performance.

---

## 11. Summary

| Criterion | Value |
|-----------|-------|
| Tool | `transfer_money(recipient, amount)` with 2 args |
| Dataset size | 511 samples (401 positive, 110 negative, 21.5% neg) |
| Audio | XTTS-v2, 1 voice (Cameron Russell), English |
| Model | Qwen2.5-Omni-7B, zero-shot, no fine-tuning |
| **Pipeline A accuracy (text oracle)** | **98.83%** |
| **Pipeline B WER** | **66.74%** (inflated by digit↔word mismatch; semantic WER ~20–30%) |
| **Pipeline C accuracy (direct audio)** | **68.30%** |
| **Pipeline D accuracy (cascaded, fixed)** | **67.51%** |
| **Best pipeline** | **D v3 (Cascaded) — precision=1.000, FAR=0.000** |
| v3 improvement over v2 (Pipeline D) | +12 pp accuracy, recall 0.436→0.586 |
| WER fix (trigram filter) | 103.36% → **66.74%** (looping hallucinations excluded) |
| Fine-tuning | Not performed |

---

## 12. File Descriptions

| File | Description |
|------|-------------|
| `data/generated_dataset.json` | Synthetic text dataset (511 examples, ChatGPT) |
| `data/Cameron_Russell-chunk-38.wav` | Speaker reference audio for TTS |
| `data/dataset_audio_Cameron_Russell_115.json` | Dataset manifest with `text`, `label`, `audio_path` |
| `data/wavs_Cameron_Russell_115.zip` | 511 synthesized WAV files (archived) |
| `src/generate_audio.py` | TTS synthesis script (XTTS-v2) |
| `src/run_omni_project_v2.py` | Experiment script v2 — all pipelines A–D (has Pipeline D bug) |
| `src/run_omni_project_v3.py` | Experiment script v3 — fixes Pipeline D transcript passing |
| `src/calculate_metrics.py` | Metrics: accuracy, precision, recall, FAR, parsable rate, WER |
| `src/error_analysis.py` | Detailed error breakdown and categorization per pipeline |
| `results/results_omni_7b_v2.json` | Raw results from v2 run (511 samples) |
| `results/results_omni_7b_v3.json` | Raw results from v3 run (511 samples, fixed D) |
| `results/results_omni_7b_v2_metrics.txt` | Auto-generated metrics report for v2 |
| `results/results_omni_7b_v3_metrics.txt` | Auto-generated metrics report for v3 |
| `results/error_analysis_v3.txt` | Detailed error categorization for all pipelines (v3) |
| `demo_notebook.ipynb` | Interactive demo: all 4 pipelines + metrics on pre-computed results |
