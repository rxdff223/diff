# 项目 2：双臂操纵扩散策略模仿学习

本仓库是 **Student 3** 对 Project 2 的贡献，聚焦于双臂机器人任务中的扩散策略模仿学习。此项目实现了：
- 条件扩散策略模型
- 训练与 checkpoint 保存
- 多步动作块采样与推理
- 离线评估与消融实验自动化

## 1. 项目简介

本项目旨在让双臂机器人通过专家演示数据学习协调动作。与传统行为克隆不同，扩散策略直接预测一个动作块（action chunk），从而生成更平滑、更一致的控制信号。

Student 3 的工作范围包括：
- `diffusion_policy` 模块的核心实现
- 训练与验证流程
- 消融实验设计与结果分析
- 中文文档与报告整理

## 2. 数据格式

请将预处理数据放在 `preprocessed/` 目录下：
- `liftpot_images.npy`：形状 `[N, T, D_obs]`，当前示例为 `[500, 75, 1024]`
- `liftpot_actions.npy`：形状 `[N, T, D_act]`，当前示例为 `[500, 75, 16]`
- `stats.json`：动作归一化所需的 min/max 数据（可选）

代码使用的样本构造规则：
- 观察历史长度：`H`
- 动作块长度：`K`

对于时间步 `t`：
- 条件输入：`obs[t-H+1:t+1]`
- 目标输出：`act[t:t+K]`

## 3. 环境安装

推荐环境：
- Python 3.9
- CUDA 12.1
- RTX 2060 或兼容 GPU

安装命令：

```bash
python -m pip install --upgrade pip
conda install pytorch=2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

`requirements.txt` 包含：
- `numpy>=1.24`
- `tqdm>=4.66`
- `matplotlib>=3.8`

如果仅需 CPU 运行，可安装 CPU 版本 PyTorch，并省略 `pytorch-cuda`。

## 4. 训练流程

主训练脚本：`train_diffusion_policy.py`

示例命令：

```bash
python train_diffusion_policy.py \
  --data-dir preprocessed \
  --history 4 \
  --horizon 8 \
  --diffusion-steps 100 \
  --batch-size 64 \
  --epochs 10 \
  --out-dir outputs_full \
  --device cuda
```

训练输出：
- `outputs_full/best.pt`
- `outputs_full/latest.pt`
- `outputs_full/train_log.jsonl`
- `outputs_full/train_config.json`

主要参数说明：
- `--history`：观察历史长度 `H`
- `--horizon`：动作块长度 `K`
- `--diffusion-steps`：扩散步骤数
- `--batch-size`：训练批大小
- `--epochs`：训练轮数
- `--out-dir`：输出目录
- `--device`：运行设备（`cuda` 或 `cpu`）

## 5. 离线评估

评估脚本：`eval_diffusion_policy.py`

示例命令：

```bash
python eval_diffusion_policy.py \
  --checkpoint outputs_full/best.pt \
  --data-dir preprocessed \
  --split val \
  --batch-size 64 \
  --device cuda
```

主要指标：
- `mse_norm`：归一化动作空间均方误差
- `mae_norm`：归一化动作空间平均绝对误差
- `mse_real`：真实动作空间均方误差
- `mae_real`：真实动作空间平均绝对误差
- `smoothness_l2_step`：轨迹平滑性指标

当前主模型效果参考值：
- `mse_real = 0.1129`
- `mae_real = 0.2469`
- `smoothness_l2_step = 0.1397`

## 6. 推理采样

采样脚本：`sample_policy.py`

示例命令：

```bash
python sample_policy.py \
  --checkpoint outputs_full/best.pt \
  --data-dir preprocessed \
  --episode 0 \
  --timestep 10 \
  --device cuda
```

输出内容包括：
- 归一化动作块预测
- 反归一化后的真实动作块
- 第一帧动作（用于 receding-horizon 控制）

## 7. 代码结构说明

### `diffusion_policy/model.py`

- 条件 Transformer 模型
- observation 与 noisy action 分别投影到 `d_model`
- 使用位置编码处理时序信息
- 通过时间步嵌入融合扩散步骤信息
- 模型输出为动作噪声预测

### `diffusion_policy/diffusion.py`

- 线性 beta DDPM 调度器实现
- 支持噪声前向注入与反向采样

### `diffusion_policy/policy.py`

- 训练损失计算
- 动作块采样与推理接口

### `diffusion_policy/data.py`

- `DualArmSequenceDataset`：数据窗口化与批次生成
- 动作归一化 / 反归一化工具
- 训练/验证集拆分逻辑

### `diffusion_policy/utils.py`

- 设备选择与随机种子配置

## 8. 消融实验

消融实验脚本：`run_ablations.py`

### history 消融示例：

```bash
python run_ablations.py \
  --study history \
  --history-values 2,4 \
  --out-root ablations_history \
  --epochs 1 \
  --batch-size 64 \
  --eval-batch-size 64 \
  --device cuda \
  --seeds 42 \
  --max-train-batches 50 \
  --max-val-batches 10 \
  --max-eval-batches 10
```

### horizon 消融示例：

```bash
python run_ablations.py \
  --study horizon \
  --horizon-values 1,4,8,16 \
  --out-root ablations_horizon \
  --epochs 1 \
  --batch-size 64 \
  --eval-batch-size 64 \
  --device cuda \
  --seeds 42 \
  --max-train-batches 30 \
  --max-val-batches 10 \
  --max-eval-batches 10
```

消融输出：
- `summary.csv`
- `summary.jsonl`
- 每个实验子目录下的 `best.pt` 和 `eval_metrics.json`

当前已生成目录：
- `ablations_history/`
- `ablations_horizon/`

## 9. 绘图结果

绘图脚本：`plot_ablations.py`

示例命令：

```bash
python plot_ablations.py \
  --summary-csv ablations_history/summary.csv \
  --out-dir ablations_history/plots \
  --metric mse_real \
  --secondary-metric smoothness_l2_step
```

当前已生成图像：
- `ablations_history/plots/history_mse_real.png`
- `ablations_history/plots/history_smoothness_l2_step.png`
- `ablations_horizon/plots/horizon_mse_real.png`
- `ablations_horizon/plots/horizon_smoothness_l2_step.png`

## 10. 报告与模板

仓库内提供：
- `REPORT_STUDENT3_TEMPLATE.md`：实验报告模板
- `REPORT_STUDENT3.md`：当前报告草稿
- `AI_PROMPT_LOG_TEMPLATE.md`：AI 交互日志模板

## 11. 结果总结

### 主模型结果
- `mse_real = 0.1129`
- `mae_real = 0.2469`
- `smoothness_l2_step = 0.1397`

### 消融结果参考（`ablations_fast/summary.csv`，仅统计存在对应 `best.pt` 与 `eval_metrics.json` 的实验）
- history：`H=1/2/4/8` 对应 `mse_real=0.1064/0.0938/0.1168/0.0973`（2 seeds 均值）
- horizon：当前完整双 seed 结果仅覆盖 `K=1/16`，对应 `mse_real=0.0412/0.1209`（`K=1` 的 smoothness 为 NaN 属预期）；`K=4` 仅有 seed=123，`K=8` 结果缺失
- diffusion_steps：当前完整双 seed 结果仅覆盖 `T=20/50/200`，对应 `mse_real=0.1454/0.1268/0.0696`；`T=100` 结果缺失
- demo_count：`50/100/200/400` 对应 `mse_real=0.1182/0.1152/0.1008/0.1072`（2 seeds 均值）



## 12. 后续建议

- 使用更长训练时间和更大 batch 重新跑消融实验
- 添加行为克隆 / DAgger 基线对比
- 在仿真环境中测量任务成功率而非仅看回归误差
- 将本项目结果整理为最终报告附件

---

如果你希望，我可以继续帮助你：
1. 将此中文 README 保存并提交到仓库；
2. 精炼 `REPORT_STUDENT3.md` 为最终版本；
3. 生成更完整的实验结果和图表说明。


