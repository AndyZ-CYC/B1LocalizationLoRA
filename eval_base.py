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
import torch
import re
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
tokenizer.use_default_system_prompt = False   # 关闭默认模板，手动构造

def build_prompt(src_zh: str) -> str:
    """使用官方 chat_template 生成 prompt"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个游戏《黑神话：悟空》的翻译专家。请将给定的中文文本翻译成英文。"
                "对于尖括号<>或大括号{{}}包围的占位符，请保持英文部分不变，仅翻译其中的中文部分。"
                "❗只输出英文译文，严禁输出任何解释、标注、<think> 思考、代码或多余文本。"
            ),
        },
        {"role": "user", "content": src_zh},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 结尾自动带 <|im_start|>assistant\n

model = AutoModelForCausalLM.from_pretrained(
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
        prompt = build_prompt(ex["input"])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        print(f"Current Source Text: {ex['input']}")

        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=1024,
                temperature=0.2,      # 更确定
                top_p=0.95,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(out_ids[0][input_ids.size(1):], skip_special_tokens=False)
        # print("RAW >>>", repr(raw))        # 调试

        raw  = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
        pred = raw.split("<|im_end|>")[0].strip()
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
