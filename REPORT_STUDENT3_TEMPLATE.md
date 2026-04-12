# Diffusion Policy Report Template

## 0. Scope and Interface

Use this ownership block in the merged report:
- Work package A: environment setup, expert demonstrations (scripted + noisy), data preprocessing pipeline
- Work package B: classic BC and DAgger implementation, covariate-shift analysis
- Work package C: diffusion-policy core model (architecture, training loop, sampling)
- Work package D: full experiment pipeline, visualization, statistical analysis, final report and theoretical discussion

Integration notes:
- Keep this section strictly focused on work package C
- Pull BC/DAgger numbers from work package B report
- Ensure all compared methods share the same split/seed protocol

## 1. Objective

Describe the diffusion-policy objective:
- learn `π(a_{t:t+K-1} | o_{t-H+1:t})`
- predict multi-step action chunks conditioned on observation history

## 2. Method

### 2.1 Setup

- Observation history: `o_{t-H+1:t} ∈ ℝ^{H × D_obs}`
- Action chunk: `a_{t:t+K-1} ∈ ℝ^{K × D_act}`
- Action normalization to `[-1, 1]`

### 2.2 Forward Diffusion

`q(x_τ | x_0) = N(√(ᾱ_τ) x_0, (1-ᾱ_τ)I)`

### 2.3 Training Objective

`L(θ) = E[|| ε - ε_θ(x_τ, o_hist, τ) ||_2^2]`

### 2.4 Network

- conditional Transformer encoder
- observation tokens + noisy action tokens + timestep embedding
- output predicted noise for each action token

### 2.5 Inference

- start from Gaussian noise
- iterative denoising `T -> 0`
- denormalize action chunk and execute first action

## 3. Protocol

### 3.1 Data and Split

- dataset source:
- split ratio:
- random seeds:

### 3.2 Hyperparameters

| Parameter | Value |
|---|---|
| History `H` | ... |
| Horizon `K` | ... |
| Diffusion steps `T` | ... |
| Batch size | ... |
| Epochs | ... |
| Learning rate | ... |
| d_model / layers / heads | ... |

### 3.3 Metrics

- `mse_real`
- `mae_real`
- `smoothness_l2_step`
- optional task success rate

## 4. Results

### 4.1 Main Run

- checkpoint:
- metrics:

### 4.2 History Ablation

| H | MSE_real (mean±std) | Smoothness |
|---|---|---|
| 1 | ... | ... |
| 2 | ... | ... |
| 4 | ... | ... |
| 8 | ... | ... |

### 4.3 Horizon Ablation

| K | MSE_real (mean±std) | Smoothness |
|---|---|---|
| 1 | ... | ... |
| 4 | ... | ... |
| 8 | ... | ... |
| 16 | ... | ... |

### 4.4 Diffusion-Step Ablation

| T | MSE_real (mean±std) | Inference Cost | Smoothness |
|---|---|---|---|
| 20 | ... | ... | ... |
| 50 | ... | ... | ... |
| 100 | ... | ... | ... |
| 200 | ... | ... | ... |

### 4.5 Demonstration-Count Ablation

| #Train Episodes | MSE_real (mean±std) | Smoothness |
|---|---|---|
| 50 | ... | ... |
| 100 | ... | ... |
| 200 | ... | ... |
| 400 | ... | ... |

## 5. Discussion

- key findings:
- limitations:
- implications for final integration:

## 6. Reproducibility

- exact commands
- commit hash
- random seeds
- hardware/runtime
