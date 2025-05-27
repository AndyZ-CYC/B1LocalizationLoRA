#!/usr/bin/env python
"""
train_lora.py – LoRA fine‑tune Qwen‑32B‑Chat on Alpaca‑format jsonl

✓ 兼容 TI‑ONE 两种传参方式：
   • 启动命令中显式 --arg value
   • 「调优参数」JSON → /opt/ml/input/config/hyperparameters.json
   （同名字段被 JSON 覆盖；命令行优先）

✓ 针对 32B + 8 × H20：
   • torch_dtype = bfloat16
   • device_map="auto"  (Transformers 自动切分)
   • gradient_checkpointing=True 降显存
   • 默认 rank 64 / alpha 128 / rsLoRA 可开关
"""
import os
os.environ["TRANSFORMERS_NO_TP"] = "1"
import argparse, json, logging, sys, math, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ────────────────────────────────────────────────────────────────────────────────
# argparse – base defaults
# ────────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # data & model
    p.add_argument("--model_dir", type=str, default="/opt/ml/input/model",
                   help="本地 Qwen‑32B 权重目录; 若不存在则尝试 HF Hub")
    p.add_argument("--base_model", type=str,
                   default="Qwen/Qwen-32B-Chat",
                   help="HF Hub id; 离线时忽略")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--dev_file",   type=str, required=True)
    p.add_argument("--output_dir", type=str,
                   default=os.getenv("SM_MODEL_DIR", "./lora_out"))
    # LoRA
    p.add_argument("--lora_r",        type=int,   default=64)
    p.add_argument("--lora_alpha",    type=int,   default=128)
    p.add_argument("--lora_dropout",  type=float, default=0.05)
    p.add_argument("--bias",          type=str,   default="none",
                   choices=["none", "lora_only", "all"])
    p.add_argument("--use_rslora",    type=lambda x: str(x).lower()=="true",
                   default=False)
    # training
    p.add_argument("--batch_size",    type=int,   default=1,
                   help="per‑device batch size")
    p.add_argument("--grad_acc",      type=int,   default=32,
                   help="gradient accumulation steps")
    p.add_argument("--num_epochs",    type=int,   default=3)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--deepspeed",     type=str,   default="", help="json config")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────────
# override with hyperparameters.json if exists
# ────────────────────────────────────────────────────────────────────────────────
def merge_hyperparameters(args: argparse.Namespace) -> None:
    hp_json = "/opt/ml/input/config/hyperparameters.json"
    if not os.path.exists(hp_json):
        return
    with open(hp_json) as f:
        raw = json.load(f)
    for k, v in raw.items():
        if not hasattr(args, k):
            continue
        cur = getattr(args, k)
        if isinstance(cur, bool):
            casted = str(v).lower() == "true"
        else:
            casted = type(cur)(v)
        setattr(args, k, casted)


# ────────────────────────────────────────────────────────────────────────────────
# data format helper
# ────────────────────────────────────────────────────────────────────────────────
def alpaca_to_text(ex):
    return {
        "text":
        f"{ex['instruction']}\n### 输入:\n{ex['input']}\n### 输出:\n{ex['output']}"
    }


# ────────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    merge_hyperparameters(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Effective args:\n%s", json.dumps(vars(args), indent=2))

    # 1) select path
    model_src = args.model_dir if os.path.exists(os.path.join(args.model_dir, "config.json")) else args.base_model
    logging.info("Loading model from: %s", model_src)

    # 2) tokenizer
    try:
        # 先尝试 fast（若未来镜像升级，可自动生效）
        tok = AutoTokenizer.from_pretrained(
            model_src,
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception as e:
        logging.warning("Fast tokenizer failed (%s). Fallback to slow tokenizer.", e)
        tok = AutoTokenizer.from_pretrained(
            model_src,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False        # ← 强制 slow 版本
        )

    # 3) base model – BF16 + auto sharding
    model = AutoModelForCausalLM.from_pretrained(
        model_src,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    logging.info("Base model loaded")

    # 4) LoRA config
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        task_type="CAUSAL_LM",
        use_rslora=args.use_rslora,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(model, lora_cfg)
    logging.info("LoRA wrapped (rsLoRA=%s)", args.use_rslora)

    # datasets
    ds_train = load_dataset("json", data_files=args.train_file,
                            split="train").map(alpaca_to_text)
    ds_dev   = load_dataset("json", data_files=args.dev_file,
                            split="train").map(alpaca_to_text)

    # training arguments
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="no",                  # 只在结束时手动保存
        logging_steps=50,
        gradient_checkpointing=True,
        bf16=True,
        seed=args.seed,
        report_to="none",
        deepspeed=args.deepspeed if args.deepspeed else None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        dataset_text_field="text",
    )

    trainer.train()

    # 手动保存一次 – PEFT adapter + tokenizer
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    logging.info("Finished – adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
