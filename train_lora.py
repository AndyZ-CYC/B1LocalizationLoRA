#!/usr/bin/env python
"""
LoRA 微调脚本 – 适配 TI‑ONE & Accelerate (torch‑2.7.0)

功能:
1. 载入 Qwen3‑32B‑Chat 本地权重 (BF16 + device_map=auto)
2. 注入 LoRA (可切换 rsLoRA)
3. 使用 TRL 的 SFTTrainer 做指令微调 (Alpaca 格式)
"""

import os
os.environ["TRANSFORMERS_NO_TP"] = "1"
os.environ["ACCELERATE_DISABLE_DISTRIBUTED"] = "1"
for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
    os.environ.pop(v, None)
import argparse
import logging
import sys
import json
from typing import Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

import torch.nn.functional as F

class MPFriendlySFTTrainer(SFTTrainer):
    """SFTTrainer that moves `labels` to the logits device (last layer GPU)."""
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # 把 labels 挪到与 logits 相同的 GPU
        labels = labels.to(logits.device, non_blocking=True)

        # from transformers.loss.loss_utils import fixed_cross_entropy
        # —— GPT-style 右移一位 —— 
        shift_logits = logits[..., :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[..., 1:].contiguous()       # (B, T-1)

        # —— 展平后计算 CE —— 
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),   # (B·(T-1), V)
            shift_labels.view(-1),                          # (B·(T-1))
            ignore_index=-100,
        )

        # print("shift_logits.requires_grad =", shift_logits.requires_grad)

        return (loss, outputs) if return_outputs else loss

# ────────────────────── CLI ──────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # 路径
    p.add_argument("--model_dir",   type=str, required=True)
    p.add_argument("--train_file",  type=str, required=True)
    p.add_argument("--dev_file",    type=str, required=True)
    p.add_argument("--output_dir",  type=str, required=True)
    # LoRA
    p.add_argument("--lora_r",       type=int,   default=64)
    p.add_argument("--lora_alpha",   type=int,   default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_rslora",   type=lambda x: str(x).lower() == "true", default=False)
    # 训练
    p.add_argument("--per_device_batch_size", type=int,   default=1)
    p.add_argument("--grad_acc",              type=int,   default=32)
    p.add_argument("--num_epochs",            type=int,   default=3)
    p.add_argument("--learning_rate",         type=float, default=2e-4)
    p.add_argument("--seed",                  type=int,   default=42)
    return p.parse_args()


# ────────────────────── 数据映射 ──────────────────────
def alpaca_to_text(ex: Dict) -> Dict:
    return {
        "text": f"{ex['instruction']}\n### 输入:\n{ex['input']}\n### 输出:\n{ex['output']}"
    }


# ────────────────────── 主流程 ──────────────────────
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "train.log")),
        ],
    )
    logging.info("Effective args:\n%s", json.dumps(vars(args), indent=2))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        use_rslora=args.use_rslora,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset
    train_ds = load_dataset("json", data_files=args.train_file, split="train").map(alpaca_to_text)
    dev_ds   = load_dataset("json", data_files=args.dev_file,   split="train").map(alpaca_to_text)

    # Trainer config (SFTConfig 仍然适用)
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="no",
        eval_strategy="epoch",
        bf16=True,
        # fp16=False,  # 显式禁用混合精度
        seed=args.seed,
        dataset_text_field="text",
        optim="adamw_torch",
        gradient_checkpointing=False,
    )

    # SFTTrainer (trl 0.18.x)
    # trainer = SFTTrainer(
    #     model=model,
    #     args=sft_cfg,
    #     train_dataset=train_ds,
    #     eval_dataset=dev_ds,
    #     processing_class=tokenizer,
    # )
    trainer = MPFriendlySFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save adapter & tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info("✅ Training finished — LoRA adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    main()