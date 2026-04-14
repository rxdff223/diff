# Student 4 接手指南

## 1. 项目分工回顾

| Student | 负责内容 |
|---------|---------|
| Student 1 | 环境搭建、专家演示采集、数据预处理 |
| Student 2 | 经典BC、DAgger实现、协变量偏移分析 |
| Student 3 | Diffusion Policy核心模型（本仓库） |
| **Student 4** | 完整实验流程、可视化、统计分析、最终报告+理论讨论 |

---

## 2. Diffusion Policy 核心结论（Student 3 已完成）

### 2.1 主模型性能

**配置**: History=4, Horizon=8, Diffusion Steps=100, Epochs=10

| 指标 | 值 | 说明 |
|-----|-----|-----|
| MSE_real | 0.1129 | 比基线(0.132)提升14% |
| MAE_real | 0.2469 | 平均动作误差 |
| Smoothness | 0.1397 | 动作平滑度 |

### 2.2 消融实验结论（来自 `ablations_fast/summary.csv`）

#### History (观测历史长度 H)
| H | MSE_real (mean±std) | 结论 |
|---|---------------------|------|
| 1 | 0.1064 ± 0.006 | 短历史可工作 |
| 2 | **0.0938 ± 0.004** | **最佳** |
| 4 | 0.1168 ± 0.007 | 默认配置 |
| 8 | 0.0973 ± 0.002 | 长历史也不错 |

**结论**: H=2 最优，但 H=4 是常用配置，性能差距不大。

#### Horizon (动作预测长度 K)
| K | MSE_real (mean±std) | Smoothness | 结论 |
|---|---------------------|------------|------|
| 1 | 0.0412 ± 0.002 | NaN | 单步预测最准，但无平滑度 |
| 4 | 0.0608 ± 0.000 | 0.145 | 短chunk |
| 8 | 0.0973 ± 0.006 | 0.143 | **默认平衡点** |
| 16 | 0.1209 ± 0.010 | 0.102 | 长chunk更平滑但误差大 |

**结论**: K=1 MSE最低但无法体现动作连续性；K=8 是误差与平滑度的平衡点。

#### Diffusion Steps (扩散步数 T)
| T | MSE_real (mean±std) | Smoothness | 结论 |
|---|---------------------|------------|------|
| 20 | 0.1454 ± 0.003 | 0.217 | 步数少，质量差 |
| 50 | 0.1268 ± 0.005 | 0.125 | |
| 100 | 0.0970 ± 0.006 | 0.142 | 默认 |
| 200 | **0.0696 ± 0.006** | 0.123 | **最佳质量** |

**结论**: 更多扩散步数显著提升预测质量，但推理时间增加。

#### Demonstration Count (训练数据量)
| Episodes | MSE_real (mean±std) | 结论 |
|----------|---------------------|------|
| 50 | 0.1182 ± 0.005 | 数据少，欠拟合 |
| 100 | 0.1152 ± 0.000 | |
| 200 | **0.1008 ± 0.009** | |
| 400 | 0.1072 ± 0.001 | 数据饱和，无明显提升 |

**结论**: 200 episodes后数据饱和，更多数据不一定更好。

### 2.3 核心发现总结

1. **扩散步数影响最大**: T=200 比 T=20 MSE降低52%
2. **Horizon选择权衡**: K=1 误差小但无平滑度，K=8 是平衡点
3. **History影响较小**: H=2最优，但H=4-8都可工作
4. **数据饱和**: 200 episodes 后增加数据无明显收益

---

## 3. 可用的数据和文件

### 3.1 已上传的关键文件（git已追踪）

| 文件路径 | 用途 | 格式 |
|---------|------|------|
| `ablations_fast/summary.csv` | 主消融实验汇总表 | CSV (32行, 16设置, 2种子) |
| `ablations_fast/summary.jsonl` | 同上，JSON格式 | JSONL |
| `ablations_fast/plots/*.png` | 消融实验图表 (8张) | PNG |
| `ablations_full/summary.csv` | 完整消融汇总 | CSV |
| `ablations_history/summary.csv` | History专项消融 | CSV |
| `ablations_horizon/summary.csv` | Horizon专项消融 | CSV |
| `outputs_full/train_log.jsonl` | 主模型训练曲线 | JSONL |
| `outputs_full/train_config.json` | 主模型配置 | JSON |
| `preprocessed/stats.json` | 动作归一化范围 | JSON |
| 各实验 `eval_metrics.json` | 详细评估指标 | JSON |
| 各实验 `train_config.json` | 实验配置 | JSON |

### 3.2 数据格式说明

**summary.csv 列名**:
```
checkpoint, mae_norm, mae_real, mse_norm, mse_real, 
num_batches, run_name, seed, smoothness_l2_step, study, value, variable
```

**train_log.jsonl 格式**:
```json
{"epoch": 1, "train_loss": 0.182, "val_loss": 0.088}
{"epoch": 2, "train_loss": 0.097, "val_loss": 0.069}
...
```

**eval_metrics.json 格式**:
```json
{
  "mse_norm": 0.22,
  "mae_norm": 0.35,
  "mse_real": 0.11,
  "mae_real": 0.25,
  "smoothness_l2_step": 0.14,
  "num_batches": 51
}
```

**stats.json 格式**:
```json
{
  "action_min": [16个关节角度最小值],
  "action_max": [16个关节角度最大值]
}
```

---

## 4. 环境依赖

### 4.1 代码依赖

```bash
# Python 3.9
# PyTorch 2.4.1 + CUDA 12.1

pip install numpy>=1.24 tqdm>=4.66 matplotlib>=3.8
```

### 4.2 代码结构

```
diffusion_policy/
├── model.py        # ActionDiffusionTransformer
├── diffusion.py    # DiffusionScheduler (DDPM)
├── policy.py       # DiffusionPolicy
├── data.py         # DualArmSequenceDataset
└── utils.py        # seed, device

train_diffusion_policy.py   # 训练入口
eval_diffusion_policy.py    # 评估入口
sample_policy.py            # 采样入口
run_ablations.py            # 消融实验自动化
plot_ablations.py           # 绘图脚本
```

---

## 5. Student 4 可以做的事情

### 5.1 静态分析（无需额外数据）

```python
import pandas as pd

# 加载消融实验汇总
df = pd.read_csv('ablations_fast/summary.csv')

# 按实验类型分组统计
for study in ['history', 'horizon', 'diffusion_steps', 'demo_count']:
    sub = df[df['study'] == study]
    grouped = sub.groupby('value').agg({
        'mse_real': ['mean', 'std'],
        'smoothness_l2_step': ['mean', 'std']
    })
    print(f"\n{study}:")
    print(grouped)
```

### 5.2 训练曲线绘制

```python
import json
import matplotlib.pyplot as plt

logs = []
with open('outputs_full/train_log.jsonl') as f:
    for line in f:
        logs.append(json.loads(line))

epochs = [l['epoch'] for l in logs]
train_loss = [l['train_loss'] for l in logs]
val_loss = [l['val_loss'] for l in logs]

plt.plot(epochs, train_loss, label='train')
plt.plot(epochs, val_loss, label='val')
plt.legend()
plt.savefig('training_curve.png')
```

### 5.3 使用已有图表

`ablations_fast/plots/` 已包含8张图：
- `history_mse_real.png` / `history_smoothness_l2_step.png`
- `horizon_mse_real.png` / `horizon_smoothness_l2_step.png`
- `diffusion_steps_mse_real.png` / `diffusion_steps_smoothness_l2_step.png`
- `demo_count_mse_real.png` / `demo_count_smoothness_l2_step.png`

可直接用于报告。

---

## 6. Student 4 无法做的事情（缺少大文件）

### 6.1 需要但未上传的文件

| 文件 | 大小 | 原因 |
|-----|------|------|
| `outputs_full/best.pt` | ~60MB | 模型checkpoint |
| `preprocessed/*.npy` | ~150MB | 原始数据 |

### 6.2 这些文件能做什么

- 运行 `sample_policy.py` 采样动作序列
- 对比真实动作 vs 预测动作
- 生成机器人轨迹动画/视频
- 重新评估模型在不同split上的表现

### 6.3 如果需要动画

**方案A**: 让 Student 1 或 Student 3 提供 checkpoint 和数据文件

**方案B**: 自己重新训练（需要原始数据）

```bash
# 如果有数据，重新训练
python train_diffusion_policy.py \
  --data-dir preprocessed \
  --history 4 --horizon 8 \
  --diffusion-steps 100 \
  --epochs 10 \
  --out-dir outputs_full \
  --device cuda

# 然后采样
python sample_policy.py \
  --checkpoint outputs_full/best.pt \
  --episode 0 --timestep 10
```

---

## 7. 最终报告建议

### 7.1 报告结构

```
1. 项目概述
2. 方法对比（BC vs DAgger vs Diffusion Policy）
   - 从 Student 2 获取 BC/DAgger 结果
   - 与本仓库的 Diffusion Policy 对比
3. 实验结果
   - 主模型性能
   - 消融实验分析（使用 summary.csv）
4. 理论讨论
   - 为什么 Diffusion Policy 更好？
   - Action Chunking 的优势
   - Diffusion 步数的影响机制
5. 结论与未来工作
```

### 7.2 合并其他 Student 的数据

需要从其他仓库获取：
- Student 2: BC/DAgger 的 MSE/MAE/success rate
- Student 1: 数据采集方式说明
- 确保所有方法使用**相同的 split/seed**

---

## 8. 快速开始命令

```bash
# 拉取最新代码和结果
git pull

# 查看消融实验汇总
python -c "import pandas as pd; print(pd.read_csv('ablations_fast/summary.csv'))"

# 查看训练曲线
python -c "
import json
with open('outputs_full/train_log.jsonl') as f:
    for line in f: print(json.loads(line))
"

# 查看动作范围
python -c "
import json
stats = json.load(open('preprocessed/stats.json'))
print('action_min:', stats['action_min'])
print('action_max:', stats['action_max'])
"
```

---

## 9. 联系方式

如有问题，联系：
- Student 1: 数据相关问题
- Student 2: BC/DAgger 对比数据
- Student 3: Diffusion Policy 代码和模型细节

---

*Generated for Student 4 handoff - 2026-04-14*