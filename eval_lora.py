#!/usr/bin/env python
"""
eval_lora.py  ─  Batch‑evaluate a LoRA‑tuned Qwen3‑32B on TI‑ONE

• 关闭平台注入的分布式 / 张量并行环境变量（单进程推理）
• 逐条生成测试集，保存预测 JSONL
• 打印 SacreBLEU 与 ROUGE‑L‑F1
"""

# ─────────────────── 环境变量屏蔽 —───────────────────
import os
for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
    os.environ.pop(v, None)                     # 擦掉可能注入的 DDP 变量
os.environ["ACCELERATE_DISABLE_DISTRIBUTED"] = "1"
os.environ["TRANSFORMERS_NO_TP"] = "1"         # 禁用 HF tensor parallel

# ───────────────────── imports ──────────────────────
import json, argparse, tqdm, numpy as np, sacrebleu, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─────────────── CLI / 路径参数 ────────────────
p = argparse.ArgumentParser()
p.add_argument("--base_dir",   required=True, help="Full Qwen3‑32B base weights")
p.add_argument("--lora_key",   required=True, help="Directory with adapter_model.bin")
p.add_argument("--test_file",  required=True, help="Alpaca‑format test.jsonl")
p.add_argument("--output_dir", required=True, help="Where to write metrics & preds")
args = p.parse_args()

PRED_FILE = os.path.join(args.output_dir, "test_predictions.jsonl")
os.makedirs(args.output_dir, exist_ok=True)

# ─────────────── Load model + LoRA ────────────────
print("🔄 Loading tokenizer / base / LoRA …")
tokenizer = AutoTokenizer.from_pretrained(args.lora_key, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    args.base_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(
    model, 
    args.lora_key, 
    is_trainable=False,
    local_files_only=True,
)
model.eval()

# ─────────────── Evaluate loop ────────────────
refs, hyps = [], []
with open(args.test_file, encoding="utf-8") as fin, \
     open(PRED_FILE, "w", encoding="utf-8") as fout:

    for line in tqdm.tqdm(fin, desc="✓ generating"):
        ex = json.loads(line)
        prompt = f"{ex['instruction']}\n### 输入:\n{ex['input']}\n### 输出:\n"
        in_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        print(f"Current Source Text: {ex['input']}")

        with torch.no_grad():
            out_ids = model.generate(
                in_ids,
                max_new_tokens=1024,
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

# ─────────────── Metrics ────────────────
bleu   = sacrebleu.corpus_bleu(hyps, [refs]).score
# rl_f   = np.mean([s["rouge-l"]["f"] for s in rouge.get_scores(hyps, refs)]) * 100

print("\n=================  EVAL RESULT  =================")
print(f"SacreBLEU   : {bleu:6.2f}")
print(f"Predictions saved to →  {PRED_FILE}")
print("=================================================\n")