# localizationLoRA 项目说明

> **目标**  
> 使用 **Qwen3‑32B** 基座模型，分别针对 4 种翻译场景（风格）训练 **LoRA 适配器**，并在独立的测试任务中输出预测与自动评测结果，方便与原生模型对比。

---

## 一、训练任务 `localizationLoRA_train`

### 1. 关键脚本  
* `code/start.sh` — 入口脚本  
* `code/train_lora.py` — LoRA 训练逻辑  
* `code/requirements.txt` — 依赖列表（已包含 `transformers 4.52` / `trl 0.18.1` / `peft 0.9+` 等）

### 2. 平台配置  
| 选项 | 值 |
|-----|----|
| 镜像 | `2.7.0-cuda12.6-cudnn9-runtime` |
| 训练模式 | **DDP**（节点 1，GPU 8）|
| 启动命令 | `bash /opt/ml/input/code/start.sh` |

### 3. 超参（可在 UI「调优参数」填写，或直接修改 `start.sh`）

| 名称 | 默认 | 说明 |
|------|------|------|
| `STYLE` | `style_1` | 选择训练语料：<br>`style_1` = 游戏文本<br>`style_2` = 游戏对话<br>`style_3` = 物品描述<br>`style_4` = 系统信息 |
| 其它 LoRA 超参 | `lora_r=64 lora_alpha=128 lora_dropout=0.05` | 可在 `start.sh` 末行修改 |

### 4. 产物  
训练结束后，适配器目录会写入  
`output/{任务id}/model/${STYLE}/`

```
├── adapter_config.json
├── adapter_model.bin
└── tokenizer_*  # 同步保存 tokenizer
```

日志在 `output/{任务id}/model/${STYLE}/train.log`；验证集指标将自动打印在控制台。

---

## 二、测试任务 `localizationLoRA_test`

### 1. 关键脚本  
* `code/start.sh`（同名、内容改成测试逻辑）  
* `code/eval_lora.py` — LoRA 评测  
* `code/eval_base.py` — 原生模型评测  
* `code/requirements_test.txt` — 推理依赖（`sacrebleu` 等）

### 2. 平台配置  
| 选项 | 值 |
|-----|----|
| 镜像 | **与训练相同** |
| 训练模式 | **DDP**（节点 1，GPU 8）|
| 启动命令 | `bash /opt/ml/input/code/start.sh` |

> `start.sh` 内会根据任务类型自动选择 `eval_lora.py` 或 `eval_base.py`。  
> 若要对比两者，只需分别指向不同脚本或创建两个任务。

### 3. 超参示例  

| 名称 | 示例 | 说明 |
|------|------|------|
| `STYLE` | `style_2` | 选择要测试的风格目录 |
| `BASE_ONLY` | `true` | 若设为 `true` 则调用 `eval_base.py` 输出原生模型结果；否则默认 `eval_lora.py` |

### 4. 输出结果  
测试完成后，控制台会显示
SacreBLEU : 35.67
Predictions saved to → /opt/ml/model/style_2/test_predictions.jsonl

JSONL 文件中每行格式：
```json
{
  "instruction": "...",
  "input": "...",
  "reference": "...",
  "prediction": "..."
}
```

便于后续人工比对或进一步评测（COMET、人工打分等）。