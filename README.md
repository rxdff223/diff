# 项目 2：双臂操纵扩散策略模仿学习

本仓库是 **Student 3** 对 Project 2 的贡献，聚焦于双臂机器人任务中的扩散策略模仿学习。此项目实现了：
- 条件扩散策略模型
- 训练与 checkpoint 保存
- 多步动作块采样与推理
- 离线评估与消融实验自动化

## 0. 团队分工与交接

为避免最终报告口径混乱，项目按以下责任边界协作：
- Student 1: Environment setup, expert demonstration collection (scripted + noisy), and data preprocessing pipeline.
- Student 2: Classic BC and DAgger implementation + covariate shift analysis.
- Student 3: Diffusion Policy core model (network architecture, training loop, sampling).
- Student 4: Full experiment pipeline, visualization, statistical analysis, and final report + theoretical discussion.

给 Student 4 的交接要点：
- Student 3 代码责任集中在 `diffusion_policy/`、`train_diffusion_policy.py`、`eval_diffusion_policy.py`、`sample_policy.py`。
- 复现实验主入口是 `run_ablations.py`，结果汇总口径以 `ablations_fast/summary.csv` 为准。
- `ablations_fast/summary.csv` 当前为完整双 seed 结果（32 行，16 组配置）。
- 最终总报告若需 BC/DAgger 对比，请直接合并 Student 2 的表格并统一 split/seed 口径。

## 1. 项目简介

本项目旨在让双臂机器人通过专家演示数据学习协调动作。与传统行为克隆不同，扩散策略直接预测一个动作块（action chunk），从而生成更平滑、更一致的控制信号。

Student 3 的工作范围包括：
- `diffusion_policy` 模块核心实现
- 训练与验证流程
- 消融实验设计与结果分析
- 文档与报告整理（便于 Student 4 汇总）

## 2. 数据格式

请将预处理数据放在 `preprocessed/` 目录下：
- `liftpot_images.npy`：形状 `[N, T, D_obs]`，当前示例 `[500, 75, 1024]`
- `liftpot_actions.npy`：形状 `[N, T, D_act]`，当前示例 `[500, 75, 16]`
- `stats.json`：动作归一化所需的 `min/max` 数据（可选）

样本构造规则：
- 观察历史长度：`H`
- 动作块长度：`K`

对时间步 `t`：
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

主要输出：
- `outputs_full/best.pt`
- `outputs_full/latest.pt`
- `outputs_full/train_log.jsonl`
- `outputs_full/train_config.json`

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

指标：
- `mse_norm` / `mae_norm`
- `mse_real` / `mae_real`
- `smoothness_l2_step`

当前主模型参考值：
- `mse_real = 0.1129`
- `mae_real = 0.2469`
- `smoothness_l2_step = 0.1397`

## 6. 推理采样

采样脚本：`sample_policy.py`

```bash
python sample_policy.py \
  --checkpoint outputs_full/best.pt \
  --data-dir preprocessed \
  --episode 0 \
  --timestep 10 \
  --device cuda
```

输出包括：
- 归一化动作块预测
- 反归一化后的真实动作块
- 第一帧动作（用于 receding-horizon 控制）

## 7. 代码结构说明

- `diffusion_policy/model.py`：条件 Transformer 噪声预测网络
- `diffusion_policy/diffusion.py`：DDPM 调度与反向采样
- `diffusion_policy/policy.py`：训练损失与采样接口
- `diffusion_policy/data.py`：窗口化数据集与归一化
- `diffusion_policy/utils.py`：设备与随机种子工具

## 8. 消融实验

消融脚本：`run_ablations.py`

```bash
python run_ablations.py \
  --study full \
  --out-root ablations_fast \
  --epochs 30 \
  --batch-size 128 \
  --device cuda \
  --seeds 42,123
```

输出：
- `ablations_fast/summary.csv`
- `ablations_fast/summary.jsonl`
- 各实验子目录下的 `best.pt` 与 `eval_metrics.json`

## 9. 绘图结果

绘图脚本：`plot_ablations.py`

```bash
python plot_ablations.py \
  --summary-csv ablations_fast/summary.csv \
  --out-dir ablations_fast/plots \
  --metric mse_real \
  --secondary-metric smoothness_l2_step
```

## 10. 报告与模板

Student 4 集成建议：
- 固定团队分工口径（见“0. 团队分工与交接”）后再整合。
- 先接入 Student 2 的 BC/DAgger 表，再和 diffusion 结果并排比较。
- 统一统计协议：相同 split、相同 seeds、均值±标准差。

仓库内模板：
- `REPORT_STUDENT3_TEMPLATE.md`
- `REPORT_STUDENT3.md`
- `AI_PROMPT_LOG_TEMPLATE.md`

## 11. 结果总结

主模型结果：
- `mse_real = 0.1129`
- `mae_real = 0.2469`
- `smoothness_l2_step = 0.1397`

`ablations_fast/summary.csv`（2 seeds 均值）关键结果：
- history：`H=1/2/4/8` -> `mse_real=0.1064/0.0938/0.1168/0.0973`
- horizon：`K=1/4/8/16` -> `mse_real=0.0412/0.0608/0.0973/0.1209`
- diffusion_steps：`T=20/50/100/200` -> `mse_real=0.1454/0.1268/0.0970/0.0696`
- demo_count：`50/100/200/400` -> `mse_real=0.1182/0.1152/0.1008/0.1072`

## 12. 后续建议

- 接入 Student 2 BC/DAgger 最终表并做统一对比图
- 在 Student 4 总报告中补齐理论分析与统计显著性说明
- 增加任务成功率指标（不仅是回归误差）
