# Student 3 Report Section Template (Diffusion Policy)

## 0. Team Scope and Interface

Use this scope block unchanged in the final merged report to avoid ownership ambiguity:
- Student 1: Environment setup, expert demonstration collection (scripted + noisy), and data preprocessing pipeline.
- Student 2: Classic BC and DAgger implementation + covariate shift analysis.
- Student 3: Diffusion Policy core model (network architecture, training loop, sampling).
- Student 4: Full experiment pipeline, visualization, statistical analysis, and final report + theoretical discussion.

Integration notes for Student 4:
- Keep Student 3 section strictly diffusion-focused.
- Pull BC/DAgger numbers from Student 2 report; do not restate them here unless explicitly cited.
- Ensure all compared methods use the same split/seed protocol before drawing final conclusions.


## 1. Role and Objective

This section covers the Student 3 contribution: implementing a conditional diffusion policy for dual-arm imitation learning, including model architecture, training objective, and denoising inference.

Goal: learn a policy `pi(a_{t:t+K-1} | o_{t-H+1:t})` that predicts a multi-step action chunk conditioned on observation history.

## 2. Method

### 2.1 Problem Setup

- Observation history: `o_{t-H+1:t} in R^(H x D_obs)`
- Action chunk: `a_{t:t+K-1} in R^(K x D_act)`
- We normalize each action dimension to `[-1, 1]` with dataset min-max stats.

### 2.2 Forward Diffusion

For clean action chunk `x0` and timestep `tau`:

`q(x_tau | x0) = N(sqrt(alpha_bar_tau) * x0, (1 - alpha_bar_tau) * I)`

where:

- `beta_tau` follows a linear schedule from `beta_start` to `beta_end`
- `alpha_tau = 1 - beta_tau`
- `alpha_bar_tau = product_{s=1..tau} alpha_s`

### 2.3 Training Objective

Model `epsilon_theta(x_tau, o_{hist}, tau)` predicts Gaussian noise added to action chunk.

Loss:

`L(theta) = E_{x0, epsilon, tau}[ || epsilon - epsilon_theta(x_tau, o_{hist}, tau) ||_2^2 ]`

### 2.4 Network Architecture

- Backbone: conditional Transformer encoder
- Inputs:
  - observation tokens (length `H`)
  - noisy action tokens (length `K`)
  - sinusoidal timestep embedding injected into action tokens
- Output: predicted noise for each action token

### 2.5 Denoising Inference

At test time:

1. Start from Gaussian noise `x_T ~ N(0, I)`
2. Iteratively denoise from `T -> 0` using DDPM reverse step
3. Obtain normalized chunk `x_0_hat`, then denormalize to real action range
4. Execute first action (receding horizon), then replan at next control cycle

## 3. Experimental Protocol

### 3.1 Data and Split

- Dataset: dual-arm liftpot demonstrations
- Train/validation split: `val_ratio = ...`
- Seed(s): `...`

### 3.2 Main Hyperparameters

| Parameter | Value |
|---|---|
| History `H` | ... |
| Action chunk `K` | ... |
| Diffusion steps `T` | ... |
| Batch size | ... |
| Epochs | ... |
| Learning rate | ... |
| d_model / layers / heads | ... |

### 3.3 Metrics

- `MSE_real`: action-space mean squared error
- `MAE_real`: action-space mean absolute error
- `Smoothness_l2_step`: average L2 finite difference between consecutive predicted actions
- (Optional in simulator) task success rate

## 4. Ablation Results

### 4.1 History Length Ablation (`H`)

| H | MSE_real (mean+-std) | Smoothness |
|---|---|---|
| 1 | ... | ... |
| 2 | ... | ... |
| 4 | ... | ... |
| 8 | ... | ... |

Short analysis:
- ...

### 4.2 Action Chunk Ablation (`K`)

| K | MSE_real (mean+-std) | Smoothness |
|---|---|---|
| 1 | ... | ... |
| 4 | ... | ... |
| 8 | ... | ... |
| 16 | ... | ... |

Short analysis:
- ...

### 4.3 Diffusion Step Ablation (`T`)

| T | MSE_real (mean+-std) | Inference Cost | Smoothness |
|---|---|---|---|
| 20 | ... | ... | ... |
| 50 | ... | ... | ... |
| 100 | ... | ... | ... |
| 200 | ... | ... | ... |

Short analysis:
- ...

### 4.4 Demonstration Count Ablation

| #Train Episodes | MSE_real (mean+-std) | Smoothness |
|---|---|---|
| 50 | ... | ... |
| 100 | ... | ... |
| 200 | ... | ... |
| 400 | ... | ... |

Short analysis:
- ...

## 5. Discussion

### 5.1 Why Diffusion Helps in Dual-Arm Manipulation

- Captures multi-modal expert behaviors better than deterministic BC
- Produces smoother chunked trajectories via denoising prior
- Reduces compounding error versus one-step BC in long horizons

### 5.2 Limitations

- Inference cost grows with diffusion steps
- Offline metric does not fully reflect task completion quality
- Distribution shift still exists for unseen objects and contacts

### 5.3 Sim-to-Real Considerations

- Sensor noise and latency mismatch
- Dynamics gap and contact model mismatch
- Need domain randomization / calibration / residual adaptation

## 6. Reproducibility

Provide:
- Exact training/evaluation commands
- Commit hash
- Random seeds
- Hardware and runtime (Python 3.9, PyTorch 2.4.1+cu121)

