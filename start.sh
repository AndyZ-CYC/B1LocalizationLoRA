#!/usr/bin/env bash
# TI‑ONE 启动脚本 – LoRA 微调 Qwen3‑32B
set -euo pipefail

###############################################################################
# 配置区：如需多风格循环，可把 STYLE 写成环境变量由平台注入
###############################################################################
STYLE=${STYLE:-style_2}            # 训练数据子目录名称

MODEL_DIR="/opt/ml/input/model"                # 预训练模型
CODE_DIR="/opt/ml/input/code"                  # 本脚本与 python 源码所在
DATA_DIR="/opt/ml/input/data/${STYLE}"         # train.jsonl / dev.jsonl
OUTPUT_DIR="/opt/ml/model/${STYLE}"            # 适配器保存目录
REQ_FILE="${CODE_DIR}/requirements.txt"

mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"

###############################################################################
# 安装依赖（镜像已含 torch2.7.0+cu12.6，因此 requirements 不再包含 torch）
###############################################################################
echo "📦 Installing Python dependencies from requirements.txt"
pip3 install --no-cache-dir -r "${REQ_FILE}"

###############################################################################
# 使用 Accelerate 启动 8 张 H20 卡的分布式训练
###############################################################################
echo "🚀 Launching LoRA training on 8 GPUs"
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  "${CODE_DIR}/train_lora.py" \
    --model_dir   "${MODEL_DIR}" \
    --train_file  "${DATA_DIR}/train.jsonl" \
    --dev_file    "${DATA_DIR}/dev.jsonl" \
    --output_dir  "${OUTPUT_DIR}" \
    --lora_r 96 \
    --lora_alpha 192 \
    --lora_dropout 0.1 \
    --use_rslora true \
    --per_device_batch_size 1 \
    --grad_acc 32 \
    --num_epochs 4 \
    --learning_rate 2e-4
