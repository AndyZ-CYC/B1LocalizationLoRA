#!/usr/bin/env bash
# TI‑ONE 启动脚本 – Qwen3‑32B 评测（支持 LoRA / Base 对比）
set -euo pipefail

###############################################################################
# ---------- 配置路径 & 任务参数 ---------- #
###############################################################################
STYLE=${STYLE:-style_4}            # 当前风格：style_1~4
BASE_ONLY=${BASE_ONLY:-false}      # true → 只跑原生模型

BASE_DIR="/opt/ml/input/model"           # 预训练模型
LORA_DIR="/opt/ml/input/lora/${STYLE}"   # LoRA 适配器目录
CODE_DIR="/opt/ml/input/code"            # 脚本目录
DATA_DIR="/opt/ml/input/data/${STYLE}"   # 测试集
OUTPUT_DIR="/opt/ml/model/${STYLE}"      # 评测结果输出
REQ_FILE="${CODE_DIR}/requirements_test.txt"

mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"

###############################################################################
# ---------- 安装推理 & 评测依赖 ---------- #
###############################################################################
echo "📦 Installing Python dependencies (requirements_test.txt)"
pip3 install --no-cache-dir -r "${REQ_FILE}"

###############################################################################
# ---------- 选择评测脚本 ---------- #
###############################################################################
if [[ "${BASE_ONLY,,}" == "true" ]]; then
  echo "🚀 Launching *BASE* model testing (no LoRA)"
  python -u "${CODE_DIR}/eval_base.py" \
    --base_dir   "${BASE_DIR}" \
    --test_file  "${DATA_DIR}/test.jsonl" \
    --output_dir "${OUTPUT_DIR}"
else
  echo "🚀 Launching *LoRA* testing"
  python -u "${CODE_DIR}/eval_lora.py" \
    --base_dir   "${BASE_DIR}" \
    --lora_key   "${LORA_DIR}" \
    --test_file  "${DATA_DIR}/test.jsonl" \
    --output_dir "${OUTPUT_DIR}"
fi
