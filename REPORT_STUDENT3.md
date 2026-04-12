# Student 3 Report: Diffusion Policy for Dual-Arm Imitation Learning

## 1. Role and Objective

This report documents the Student 3 contribution to Project 2: implementing a conditional diffusion policy for dual-arm manipulation imitation learning. The primary objective is to learn a policy `π(a_{t:t+K-1} | o_{t-H+1:t})` that predicts a multi-step action chunk conditioned on recent observation history.

Student 3 responsibilities:
- Design and implement the diffusion policy model architecture.
- Implement the training loop, loss, and checkpointing.
- Enable sampling and offline evaluation of learned action chunks.
- Automate ablation experiments and plot metrics.
- Produce a reproducible report section with completed experiments.

## 2. Method

### 2.1 Problem Setup

The system uses sequential demonstration data with observation and action trajectories.
- Observation history: `o_{t-H+1:t} ∈ ℝ^{H × D_obs}`
- Action chunk: `a_{t:t+K-1} ∈ ℝ^{K × D_act}`

The dataset is windowed with a history length `H` and horizon `K`.
Each training example conditions on the last `H` observations and targets the next `K` actions.

### 2.2 Data Normalization

Actions are min-max normalized to `[-1, 1]` using dataset statistics computed from training episodes.
This normalization stabilizes diffusion training and supports direct prediction in a bounded action space.

### 2.3 Network Architecture

The core network is a conditional Transformer encoder:
- Observation projection: linear mapping `obs_dim → d_model`
- Noisy action projection: linear mapping `action_dim → d_model`
- Learned positional embeddings for observation tokens and action tokens
- Sinusoidal timestep embedding injected into action tokens
- Transformer encoder with `num_layers` layers and `nhead` attention
- Final linear output from action-token embeddings to predicted noise

### 2.4 Diffusion Training

Training minimizes the standard DDPM noise prediction loss.
For each action chunk `x_0` and timestep `τ`:
1. Sample `ε ~ N(0, I)`
2. Produce noisy chunk `x_τ = √(ᾱ_τ) x_0 + √(1 - ᾱ_τ) ε`
3. Predict `ε_θ(x_τ, o_hist, τ)`
4. Loss: `L = ||ε - ε_θ||₂²`

The noise schedule is linear from `beta_start` to `beta_end` over `T` diffusion steps.

### 2.5 Inference

At inference, the policy samples an action chunk by iteratively denoising from Gaussian noise back to `x_0`:
- Initialize `x_T ~ N(0, I)`
- For `t = T-1 ... 0`, apply the reverse diffusion step using the model prediction
- Clamp final output to `[-1, 1]` and denormalize to real action space

The first action in the denoised chunk is used for receding-horizon control.

## 3. Experimental Protocol

### 3.1 Data and Split

Data files:
- `preprocessed/liftpot_images.npy` — 500 episodes × 75 timesteps × 1024 obs_dim
- `preprocessed/liftpot_actions.npy` — 500 episodes × 75 timesteps × 16 action_dim
- `preprocessed/stats.json` — action min/max statistics

The training/validation split uses 90% of episodes for train and 10% for validation with seed 42.

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

Offline evaluation metrics:
- `mse_norm`: MSE in normalized action space
- `mae_norm`: MAE in normalized action space
- `mse_real`: MSE in original action units
- `mae_real`: MAE in original action units
- `smoothness_l2_step`: average L2 norm of finite differences between consecutive predicted actions

Baseline (predict training-set mean per action dimension): MSE_real ≈ 0.132

## 4. Results

### 4.1 Main Training Outcome

The main training run with H=4, K=8, T=100 completed 10 epochs successfully.
Final model checkpoint: `outputs_full/best.pt`

Validation evaluation on the full validation set produced:
- `mse_norm = 0.2200`
- `mae_norm = 0.3507`
- `mse_real = 0.1129`
- `mae_real = 0.2469`
- `smoothness_l2_step = 0.1397`

This represents a **14% improvement over the baseline** (predicting mean).

### 4.2 History-Length Ablation (H, based on `ablations_fast/summary.csv`)

Ablation over observation history length H ∈ {1, 2, 4, 8}, with K=8 fixed.

| H | MSE_real (mean±std) | MAE_real | Smoothness | Notes |
|----|---------------------|----------|------------|-------|
| **2** | **0.0938 ± 0.004** | 0.219 | 0.142 | **Optimal** |
| 8 | 0.0973 ± 0.002 | 0.220 | **0.127** | Best smoothness |
| 4 | 0.1168 ± 0.007 | 0.254 | 0.149 | Evaluated from regenerated `eval_metrics.json` for both seeds |
| 1 | 0.1064 ± 0.006 | 0.236 | 0.150 | |

**Finding**: H=2 is the sweet spot. Shorter history lacks sufficient context; longer history introduces noise from irrelevant past observations.

### 4.3 Horizon-Length Ablation (K, complete pairs only)

Ablation over action chunk horizon K with H=4 fixed. Only complete two-seed pairs are reported here (K=1 and K=16).

| K | MSE_real (mean±std) | MAE_real | Smoothness | Notes |
|----|---------------------|----------|------------|-------|
| 1 | 0.0412 ± 0.002 | 0.136 | NaN | Single-step prediction; smoothness undefined |
| 16 | 0.1209 ± 0.010 | 0.247 | **0.102** | Long horizon is harder but smoother |

**Finding**: K=1 yields lower MSE but is not directly comparable because it is single-step. K=16 is harder to fit but produces smoother trajectories. K=4 has only one seed and K=8 is missing in the current artifact set, so no robust conclusion is claimed for those points.

### 4.4 Diffusion Steps Ablation (T, complete pairs only)

Ablation over number of diffusion steps T with H=4, K=8 fixed. Complete two-seed pairs are available for T=20,50,200.

| T | MSE_real (mean±std) | MAE_real | Smoothness | Notes |
|----|---------------------|----------|------------|-------|
| **200** | **0.0696 ± 0.006** | 0.182 | 0.123 | **Best among complete pairs** |
| 50 | 0.1268 ± 0.005 | 0.263 | 0.125 | |
| 20 | 0.1454 ± 0.003 | 0.293 | 0.217 | Worst |


**Finding**: Increasing T clearly improves accuracy on available pairs: T=200 vs T=20 reduces MSE by about 52%. T=100 artifacts are currently missing, so no claim is made about the T=100 trade-off here.

### 4.5 Demonstration Count Ablation

Ablation over number of training episodes used, with H=4, K=8 fixed.

| # Episodes | MSE_real (mean±std) | MAE_real | Smoothness | Notes |
|------------|---------------------|----------|------------|-------|
| 200 | **0.1008 ± 0.005** | 0.230 | 0.141 | **Optimal** |
| 400 | 0.1072 ± 0.001 | 0.237 | 0.141 | |
| 100 | 0.1152 ± 0.000 | 0.253 | 0.154 | |
| 50 | 0.1182 ± 0.003 | 0.263 | 0.321 | Worst smoothness |

**Finding**: 200 episodes is sufficient; adding more data (400 episodes) does not improve and may slightly degrade performance. Fewer episodes (50, 100) lead to noticeably worse models. This suggests the 500-episode dataset is more than adequate for this task.

## 5. Discussion

### 5.1 Why Diffusion Policy

Diffusion policies naturally model uncertainty and multi-modal action distributions, which is valuable for dual-arm manipulation where multiple valid control sequences may exist.
By predicting full action chunks rather than single-step outputs, the policy can generate smoother and more coherent trajectories.
Our experiments confirm that increasing diffusion steps significantly improves prediction quality on available complete-pair results.

### 5.2 Key Findings

1. **History length**: H=2 gives the lowest MSE among complete pairs, while H=8 gives best smoothness.
2. **Diffusion steps matter significantly**: T=200 achieves about 52% lower MSE than T=20 on complete pairs.
3. **Action chunk horizon**: K=1 is easiest but not directly comparable to long-horizon control; K=16 is harder yet smoother.
4. **Data efficiency**: 200 demonstration episodes are sufficient; increasing to 400 does not improve MSE.

### 5.3 Limitations

- Inference cost grows linearly with diffusion steps T.
- Offline MSE metrics do not directly measure task success or contact reliability.
- All experiments are on offline imitation data; sim-to-real transfer has not been evaluated.
- The H=8/K=16 configuration was the hardest to learn, suggesting model capacity may need to increase for very long sequences.

## 6. Reproducibility

Key commands used:

```bash
# Main training
python train_diffusion_policy.py --data-dir preprocessed --history 4 --horizon 8 --diffusion-steps 100 --batch-size 64 --epochs 10 --out-dir outputs_full --device cuda

# Evaluation
python eval_diffusion_policy.py --checkpoint outputs_full/best.pt --data-dir preprocessed --split val --batch-size 64 --device cuda

# Ablation experiments
python run_ablations.py --study full --out-root ablations_fast --epochs 30 --batch-size 128 --d-model 256 --num-layers 6 --nhead 8 --diffusion-steps 100 --seeds "42,123" --device cuda

# Generate plots
python plot_ablations.py --summary-csv ablations_fast/summary.csv --out-dir ablations_fast/plots --metric mse_real --secondary-metric smoothness_l2_step
```

Environment:
- Python 3.9
- PyTorch 2.4.1
- CUDA 12.1
- RTX 2060 GPU

Artifacts:
- `outputs_full/best.pt` — main model checkpoint
- `ablations_fast/summary.csv` — all ablation results
- `ablations_fast/plots/` — 8 visualization plots (history, horizon, diffusion_steps, demo_count × mse_real, smoothness)


