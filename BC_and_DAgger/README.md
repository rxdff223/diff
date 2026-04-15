# Behavioral Cloning (BC) and DAgger Implementation

本文件夹实现了行为克隆（Behavioral Cloning, BC）和 DAgger 算法，用于机器人行为学习。

## 项目概述

这是课程作业的一部分，实现了两种模仿学习方法：
- **Behavioral Cloning (BC)**: 直接从专家演示中学习策略
- **DAgger (Dataset Aggregation)**: 通过迭代收集专家修正数据来缓解协变量漂移问题

## 文件结构

| 文件 | 描述 |
|------|------|
| `main.py` | 主要实现代码，包含 BC 和 DAgger 的完整流程 |
| `bc_model_best.pth` | BC 训练过程中验证集损失最小的模型 |
| `bc_model_final.pth` | BC 训练结束时的最终模型 |
| `bc_results.json` | 包含训练损失、验证损失、每维度 MAE、时间步误差和 DAgger 结果 |
| `bc_training_curve.png` | BC 训练曲线（线性和对数尺度） |
| `bc_mae_per_dim.png` | 各动作维度的平均绝对误差柱状图 |
| `bc_prediction_scatter.png` | 预测值与真实值的散点图对比 |
| `covariate_shift_analysis.png` | 时间步误差累积分析图 |
| `dagger_progress.png` | DAgger 各轮训练的验证损失变化 |

## 数据格式

代码适配预处理过的特征向量输入：
- **images**: `(500, 75, 1024)` - 500 个演示，每个 75 步，1024 维特征向量
- **actions**: `(500, 75, 16)` - 500 个演示，每个 75 步，16 维动作

## 模型架构

使用 MLP 网络结构：
- 输入层: 1024 维特征向量
- 隐藏层: 512 → 256 → 128（每层带 ReLU 激活和 Dropout）
- 输出层: 16 维动作（带 Sigmoid 激活，输出范围 [0, 1]）

## 训练流程

1. **数据加载与预处理**: 归一化特征和动作到 [0, 1] 范围
2. **划分数据集**: 80% 训练集，20% 验证集
3. **BC 训练**: 使用 MSE 损失，Adam 优化器，学习率调度
4. **模型评估**: 计算 MSE、MAE 和各维度误差
5. **协变量漂移分析**: 分析不同时间步的误差累积
6. **DAgger 训练**: 5 轮迭代，每轮添加新数据并重新训练

## 运行方式

```bash
python main.py
```

需要确保数据目录中包含：
- `liftpot_actions.npy`
- `liftpot_images.npy`
- `stats.json`

## 实验结果

- BC 最终训练损失: ~0.003
- BC 最终验证损失: ~0.0014
- DAgger 5 轮后验证损失逐步下降
- 时间步误差分析显示协变量漂移问题存在

## 依赖

- Python 3.x
- PyTorch
- NumPy
- Matplotlib