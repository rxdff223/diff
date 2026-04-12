# Diffusion Policy Technical Report

## 0. Scope and Handoff

Project ownership is partitioned by work packages:
- Work package A: environment setup, expert demonstrations (scripted + noisy), data preprocessing pipeline
- Work package B: classic BC and DAgger implementation, covariate-shift analysis
- Work package C: diffusion-policy core model (architecture, training loop, sampling)
- Work package D: full experiment pipeline, visualization, statistical analysis, final report and theoretical discussion

This report covers work package C only.
For final integration:
- Merge BC/DAgger result tables from work package B with the diffusion table in this report
- Keep the same split/seed protocol across all compared methods

## 1. Objective

This report documents implementation and evaluation of a conditional diffusion policy for dual-arm imitation learning. The objective is to learn policy `π(a_{t:t+K-1} | o_{t-H+1:t})` that predicts a multi-step action chunk conditioned on recent observation history.

## 2. Method

### 2.1 Problem Setup

- Observation history: `o_{t-H+1:t} ∈ ℝ^{H × D_obs}`
- Action chunk: `a_{t:t+K-1} ∈ ℝ^{K × D_act}`

The dataset is windowed by history length `H` and horizon `K`.

### 2.2 Data Normalization

Actions are min-max normalized to `[-1, 1]` using training statistics.

### 2.3 Network Architecture

The model is a conditional Transformer encoder:
- observation projection: `obs_dim -> d_model`
- noisy action projection: `action_dim -> d_model`
- positional embeddings for observation/action tokens
- sinusoidal timestep embedding injected into action tokens
- Transformer encoder + linear output head for noise prediction

### 2.4 Diffusion Training

For action chunk `x_0` and timestep `τ`:
1. sample `ε ~ N(0, I)`
2. build noisy sample `x_τ = √(ᾱ_τ) x_0 + √(1 - ᾱ_τ) ε`
3. predict `ε_θ(x_τ, o_hist, τ)`
4. optimize `L = ||ε - ε_θ||₂²`

### 2.5 Inference

At inference, start from Gaussian noise and iteratively denoise from `T-1` to `0`, then denormalize predicted action chunk.

## 3. Experimental Protocol

### 3.1 Data and Split

- `preprocessed/liftpot_images.npy`: 500 × 75 × 1024
- `preprocessed/liftpot_actions.npy`: 500 × 75 × 16
- `preprocessed/stats.json`

Split: 90% train / 10% validation, seed 42.

### 3.2 Main Training Configuration

| Parameter | Value |
|-----------|-------|
| History `H` | 4 |
| Horizon `K` | 8 |
| Diffusion steps `T` | 100 |
| Batch size | 64 |
| Epochs | 10 |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Transformer d_model | 256 |
| Transformer layers | 6 |
| Attention heads | 8 |
| Device | CUDA |
| Environment | Python 3.9, PyTorch 2.4.1 + CUDA 12.1, RTX 2060 |

### 3.3 Metrics

- `mse_norm`, `mae_norm`
- `mse_real`, `mae_real`
- `smoothness_l2_step`

Baseline (predict mean action per dimension): `MSE_real ≈ 0.132`.

## 4. Results

### 4.1 Main Training Outcome

Main run (`H=4, K=8, T=100`) completed 10 epochs.
Checkpoint: `outputs_full/best.pt`

Validation metrics:
- `mse_norm = 0.2200`
- `mae_norm = 0.3507`
- `mse_real = 0.1129`
- `mae_real = 0.2469`
- `smoothness_l2_step = 0.1397`

This is about 14% better than baseline MSE.

### 4.2 History Ablation (H)

| H | MSE_real (mean±std) | MAE_real | Smoothness |
|----|---------------------|----------|------------|
| 1 | 0.1064 ± 0.006 | 0.236 | 0.150 |
| 2 | 0.0938 ± 0.004 | 0.219 | 0.142 |
| 4 | 0.1168 ± 0.007 | 0.254 | 0.149 |
| 8 | 0.0973 ± 0.002 | 0.220 | 0.127 |

### 4.3 Horizon Ablation (K)

| K | MSE_real (mean±std) | MAE_real | Smoothness | Notes |
|----|---------------------|----------|------------|-------|
| 1 | 0.0412 ± 0.002 | 0.136 | NaN | single-step, smoothness undefined |
| 4 | 0.0608 ± 0.000 | 0.176 | 0.145 | |
| 8 | 0.0973 ± 0.006 | 0.224 | 0.143 | default |
| 16 | 0.1209 ± 0.010 | 0.247 | 0.102 | smoother but harder |

### 4.4 Diffusion-Step Ablation (T)

| T | MSE_real (mean±std) | MAE_real | Smoothness |
|----|---------------------|----------|------------|
| 20 | 0.1454 ± 0.003 | 0.293 | 0.217 |
| 50 | 0.1268 ± 0.005 | 0.263 | 0.125 |
| 100 | 0.0970 ± 0.006 | 0.224 | 0.142 |
| 200 | 0.0696 ± 0.006 | 0.182 | 0.123 |

### 4.5 Demonstration-Count Ablation

| # Episodes | MSE_real (mean±std) | MAE_real | Smoothness |
|------------|---------------------|----------|------------|
| 50 | 0.1182 ± 0.005 | 0.263 | 0.321 |
| 100 | 0.1152 ± 0.000 | 0.253 | 0.154 |
| 200 | 0.1008 ± 0.009 | 0.230 | 0.141 |
| 400 | 0.1072 ± 0.001 | 0.237 | 0.141 |

## 5. Discussion

- Diffusion steps significantly affect quality (`T=200` best MSE, `T=20` worst)
- `K=8` is a balanced operating point between error and stability
- Increasing data from 200 to 400 episodes does not improve MSE in this setup

## 6. Reproducibility

```bash
# main training
python train_diffusion_policy.py --data-dir preprocessed --history 4 --horizon 8 --diffusion-steps 100 --batch-size 64 --epochs 10 --out-dir outputs_full --device cuda

# evaluation
python eval_diffusion_policy.py --checkpoint outputs_full/best.pt --data-dir preprocessed --split val --batch-size 64 --device cuda

# ablations
python run_ablations.py --study full --out-root ablations_fast --epochs 30 --batch-size 128 --d-model 256 --num-layers 6 --nhead 8 --diffusion-steps 100 --seeds "42,123" --device cuda
```
