#!/usr/bin/env python
"""
eval_base.py ─ Evaluate *vanilla* Qwen3‑32B (no LoRA) on the same test set.

• Single‑process, layer‑parallel across GPUs
• Generates translations, writes JSONL identical to eval_lora.py
• Prints SacreBLEU for side‑by‑side comparison
"""

# ──────────────── Disable platform‑injected distributed vars ────────────────
import os
for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
    os.environ.pop(v, None)
os.environ["ACCELERATE_DISABLE_DISTRIBUTED"] = "1"
os.environ["TRANSFORMERS_NO_TP"] = "1"

# ─────────────────────────── Imports ────────────────────────────────────────
import json, argparse, tqdm, sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM

# ───────────────────── CLI arguments ────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--base_dir",   required=True, help="Full Qwen3‑32B base weights")
p.add_argument("--test_file",  required=True, help="Alpaca‑format test.jsonl")
p.add_argument("--output_dir", required=True, help="Where to write metrics & preds")
args = p.parse_args()

PRED_FILE = os.path.join(args.output_dir, "test_predictions_base.jsonl")
os.makedirs(args.output_dir, exist_ok=True)

# ───────────────────── Load base model ──────────────────────────────────────
print("🔄 Loading tokenizer & base model …")
tokenizer = AutoTokenizer.from_pretrained(
    args.base_dir, trust_remote_code=True, local_files_only=True
)
model     = AutoModelForCausalLM.from_pretrained(
    args.base_dir,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)
model.eval()

# ───────────────────── Evaluation loop ──────────────────────────────────────
refs, hyps = [], []
with open(args.test_file, encoding="utf-8") as fin, \
     open(PRED_FILE, "w", encoding="utf-8") as fout:

    for line in tqdm.tqdm(fin, desc="✓ generating (base)"):
        ex = json.loads(line)
        prompt = f"{ex['instruction']}\n### 输入:\n{ex['input']}\n### 输出:\n"
        in_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        print(f"Current Source Text: {ex['input']}")

        with model.no_sync():              # no grad
            out_ids = model.generate(
                in_ids,
                max_new_tokens=1024,       # keep same as eval_lora
                temperature=0.7,
                top_p=0.9,
            )
        pred = tokenizer.decode(out_ids[0][in_ids.shape[1]:], skip_special_tokens=True)
        print(f"Current Translation: {pred}")

        refs.append(ex["output"].strip())
        hyps.append(pred.strip())

        json.dump(
            {
                "instruction": ex["instruction"],
                "input": ex["input"],
                "reference": ex["output"],
                "prediction": pred,
            },
            fout,
            ensure_ascii=False,
        )
        fout.write("\n")

# ───────────────────── Metrics ──────────────────────────────────────────────
bleu = sacrebleu.corpus_bleu(hyps, [refs]).score

print("\n=================  BASE MODEL RESULT  ================")
print(f"SacreBLEU   : {bleu:6.2f}")
print(f"Predictions saved to →  {PRED_FILE}")
print("======================================================\n")
