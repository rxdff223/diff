# Project 2: Diffusion Policy Imitation Learning for Dual-Arm Manipulation

This repository focuses on diffusion-policy imitation learning for dual-arm manipulation tasks. It includes:
- A conditional diffusion-policy model
- Training and checkpointing
- Multi-step action-chunk sampling and inference
- Offline evaluation and ablation automation

## 0. Work Packages and Handoff

Project ownership is partitioned by work package:
- Work package A: environment setup, expert demonstration collection (scripted + noisy), and data preprocessing pipeline
- Work package B: classic BC and DAgger implementation, plus covariate-shift analysis
- Work package C: diffusion-policy core model (architecture, training loop, sampling)
- Work package D: full experiment pipeline, visualization, statistical analysis, and final report with theoretical discussion

This repository currently covers work package C and provides reusable outputs for final integration:
- Code entry points: `diffusion_policy/`, `train_diffusion_policy.py`, `eval_diffusion_policy.py`, `sample_policy.py`
- Experiment entry point: `run_ablations.py`
- Aggregated results: `ablations_fast/summary.csv` (currently complete two-seed results, 32 rows / 16 settings)

## 1. Overview

The goal is to learn coordinated dual-arm behavior from expert demonstrations. Unlike one-step behavior cloning, the diffusion policy predicts an action chunk directly, producing smoother and more consistent control signals.

## 2. Data Format

Put preprocessed data under `preprocessed/`:
- `liftpot_images.npy`: shape `[N, T, D_obs]`, current example `[500, 75, 1024]`
- `liftpot_actions.npy`: shape `[N, T, D_act]`, current example `[500, 75, 16]`
- `stats.json`: optional action `min/max` statistics for normalization

Window construction:
- Observation history length: `H`
- Action chunk horizon: `K`

At timestep `t`:
- Condition input: `obs[t-H+1:t+1]`
- Target output: `act[t:t+K]`

## 3. Environment Setup

Recommended environment:
- Python 3.9
- CUDA 12.1
- RTX 2060 or compatible GPU

Install commands:

```bash
python -m pip install --upgrade pip
conda install pytorch=2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

`requirements.txt` includes:
- `numpy>=1.24`
- `tqdm>=4.66`
- `matplotlib>=3.8`

## 4. Training

Main training script: `train_diffusion_policy.py`

Example:

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

Main outputs:
- `outputs_full/best.pt`
- `outputs_full/latest.pt`
- `outputs_full/train_log.jsonl`
- `outputs_full/train_config.json`

## 5. Offline Evaluation

Evaluation script: `eval_diffusion_policy.py`

Example:

```bash
python eval_diffusion_policy.py \
  --checkpoint outputs_full/best.pt \
  --data-dir preprocessed \
  --split val \
  --batch-size 64 \
  --device cuda
```

Metrics:
- `mse_norm` / `mae_norm`
- `mse_real` / `mae_real`
- `smoothness_l2_step`

Current main-model reference values:
- `mse_real = 0.1129`
- `mae_real = 0.2469`
- `smoothness_l2_step = 0.1397`

## 6. Sampling

Sampling script: `sample_policy.py`

```bash
python sample_policy.py \
  --checkpoint outputs_full/best.pt \
  --data-dir preprocessed \
  --episode 0 \
  --timestep 10 \
  --device cuda
```

Outputs include:
- Predicted normalized action chunk
- Predicted denormalized action chunk
- First action for receding-horizon execution

## 7. Code Structure

- `diffusion_policy/model.py`: conditional Transformer noise predictor
- `diffusion_policy/diffusion.py`: DDPM scheduling and reverse sampling
- `diffusion_policy/policy.py`: training loss and sampling interface
- `diffusion_policy/data.py`: windowed dataset and normalization
- `diffusion_policy/utils.py`: device and random-seed helpers

## 8. Ablation Experiments

Ablation script: `run_ablations.py`

```bash
python run_ablations.py \
  --study full \
  --out-root ablations_fast \
  --epochs 30 \
  --batch-size 128 \
  --device cuda \
  --seeds 42,123
```

Outputs:
- `ablations_fast/summary.csv`
- `ablations_fast/summary.jsonl`
- Per-run directories containing `best.pt` and `eval_metrics.json`

## 9. Plots

Plot script: `plot_ablations.py`

```bash
python plot_ablations.py \
  --summary-csv ablations_fast/summary.csv \
  --out-dir ablations_fast/plots \
  --metric mse_real \
  --secondary-metric smoothness_l2_step
```

## 10. Integration Notes (for Final Merge)

- Lock work-package boundaries before merging report sections to avoid attribution drift
- Compare BC/DAgger tables and diffusion tables under the same split/seed protocol
- Use a single reporting convention (mean ± std) with explicit experiment configs and seeds

## 11. Result Summary

Main-model result:
- `mse_real = 0.1129`
- `mae_real = 0.2469`
- `smoothness_l2_step = 0.1397`

Key results from `ablations_fast/summary.csv` (two-seed means):
- history: `H=1/2/4/8` -> `mse_real=0.1064/0.0938/0.1168/0.0973`
- horizon: `K=1/4/8/16` -> `mse_real=0.0412/0.0608/0.0973/0.1209`
- diffusion_steps: `T=20/50/100/200` -> `mse_real=0.1454/0.1268/0.0970/0.0696`
- demo_count: `50/100/200/400` -> `mse_real=0.1182/0.1152/0.1008/0.1072`

## 12. Next Steps

- Merge final BC/DAgger tables and generate unified comparison plots
- Add theoretical discussion and significance notes in the final report
- Add task-success metrics (beyond regression errors)
