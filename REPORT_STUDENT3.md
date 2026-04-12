# Student 3 Report: Diffusion Policy for Dual-Arm Imitation Learning

## 1. Role and Objective

This report documents the Student 3 contribution to Project 2: implementing a conditional diffusion policy for dual-arm manipulation imitation learning. The primary objective is to learn a policy `\pi(a_{t:t+K-1} | o_{t-H+1:t})` that predicts a multi-step action chunk conditioned on recent observation history.

Student 3 responsibilities:
- Design and implement the diffusion policy model architecture.
- Implement the training loop, loss, and checkpointing.
- Enable sampling and offline evaluation of learned action chunks.
- Automate ablation experiments and plot metrics.
- Produce a reproducible report section with completed experiments.

## 2. Method

### 2.1 Problem Setup

The system uses sequential demonstration data with observation and action trajectories.
- Observation history: `o_{t-H+1:t} in R^{H x D_obs}`
- Action chunk: `a_{t:t+K-1} in R^{K x D_act}`

The dataset is windowed with a history length `H` and horizon `K`.
Each training example conditions on the last `H` observations and targets the next `K` actions.

### 2.2 Data Normalization

Actions are min-max normalized to `[-1, 1]` using dataset statistics computed from training episodes.
This normalization stabilizes diffusion training and supports direct prediction in a bounded action space.

### 2.3 Network Architecture

The core network is a conditional Transformer encoder:
- Observation projection: linear mapping `obs_dim -> d_model`
- Noisy action projection: linear mapping `action_dim -> d_model`
- Learned positional embeddings for observation tokens and action tokens
- Sinusoidal timestep embedding injected into action tokens
- Transformer encoder with `num_layers` layers and `nhead` attention
- Final linear output from action-token embeddings to predicted noise

### 2.4 Diffusion Training

Training minimizes the standard DDPM noise prediction loss.
For each action chunk `x_0` and timestep `t`:
1. Sample `\epsilon ~ N(0, I)`
2. Produce noisy chunk `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) \epsilon`
3. Predict `\epsilon_theta(x_t, o_hist, t)`
4. Loss: `L = ||\epsilon - \epsilon_theta||_2^2`

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
- `preprocessed/liftpot_images.npy`
- `preprocessed/liftpot_actions.npy`
- `preprocessed/stats.json`

The training/validation split uses 90% of episodes for train and 10% for validation with seed 42.

### 3.2 Training Configuration

The main training run used:
- History `H = 4`
- Horizon `K = 8`
- Diffusion steps `T = 100`
- Batch size `64`
- Epochs `10`
- Learning rate `3e-4`
- Weight decay `1e-4`
- Transformer depth `6`, `8` attention heads, `d_model = 256`
- Device: `cuda`
- Environment: Python 3.9, PyTorch 2.4.1 + CUDA 12.1

### 3.3 Metrics

Offline evaluation metrics:
- `mse_norm`: MSE in normalized action space
- `mae_norm`: MAE in normalized action space
- `mse_real`: MSE in original action units
- `mae_real`: MAE in original action units
- `smoothness_l2_step`: average L2 step difference across predicted action chunk

## 4. Results

### 4.1 Main Training Outcome

The 10-epoch training run completed successfully.
Final model checkpoint: `outputs_full/best.pt`

Validation evaluation on the full validation set produced:
- `mse_norm = 0.2200`
- `mae_norm = 0.3507`
- `mse_real = 0.1129`
- `mae_real = 0.2469`
- `smoothness_l2_step = 0.1397`

These results demonstrate that the diffusion policy can learn stable action chunk predictions with a moderate horizon.

### 4.2 History-Length Ablation

A small ablation study was executed with two history lengths:
- `H = 2`
- `H = 4`

Each ablation run used one epoch and a limited subset of batches to validate the automated pipeline.
Results in `ablations_history/summary.csv`:

| History | MSE_real | MAE_real | Smoothness | Notes |
|---|---|---|---|---|
| 2 | 0.1618 | 0.3227 | 0.5142 | shorter history improved the limited training loss, but may underfit behavior history |
| 4 | 0.1673 | 0.3297 | 0.5471 | longer history produced slightly higher error in this quick ablation run |

Plots saved to `ablations_history/plots/`:
- `history_mse_real.png`
- `history_smoothness_l2_step.png`

### 4.3 Interpretation

The diffusion policy architecture is well suited for chunked action prediction. The ablation confirms the pipeline can sweep hyperparameters and collect summary statistics.
Longer history may improve performance with more training, but limited one-epoch runs can produce noisy trends.

## 5. Discussion

### 5.1 Why Diffusion Policy

Diffusion policies naturally model uncertainty and multi-modal action distributions, which is valuable for dual-arm manipulation where multiple valid control sequences may exist.
By predicting full action chunks rather than single-step outputs, the policy can generate smoother and more coherent trajectories.

### 5.2 Limitations

- Inference cost is higher than one-step BC due to iterative denoising.
- Offline metrics do not directly measure task success or contact reliability.
- Current experiments are limited to imitation data and do not include sim-to-real transfer tests.

### 5.3 Future Extensions

- Add a behavioral cloning baseline for direct comparison.
- Extend ablations to horizon size, diffusion steps, and demonstration count.
- Evaluate on actual task success metrics in simulation or on hardware.
- Incorporate DAgger-style online correction for covariate shift.

## 6. Reproducibility

Key commands used:

```bash
conda activate C:\Users\94167\.conda\envs\dp39
python train_diffusion_policy.py --data-dir preprocessed --history 4 --horizon 8 --diffusion-steps 100 --batch-size 64 --epochs 10 --out-dir outputs_full --device cuda
python eval_diffusion_policy.py --checkpoint outputs_full/best.pt --data-dir preprocessed --split val --batch-size 64 --device cuda
python run_ablations.py --study history --history-values 2,4 --out-root ablations_history --epochs 1 --batch-size 64 --eval-batch-size 64 --device cuda --seeds 42 --max-train-batches 50 --max-val-batches 10 --max-eval-batches 10
python plot_ablations.py --summary-csv ablations_history/summary.csv --out-dir ablations_history/plots --metric mse_real --secondary-metric smoothness_l2_step
```

Environment:
- Python 3.9
- PyTorch 2.4.1
- CUDA 12.1
- RTX 2060 GPU

Artifacts:
- `outputs_full/best.pt`
- `ablations_history/summary.csv`
- `ablations_history/plots/history_mse_real.png`
- `ablations_history/plots/history_smoothness_l2_step.png`
