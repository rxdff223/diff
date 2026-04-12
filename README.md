# Project 2: Diffusion Policy Imitation Learning for Dual-Arm Manipulation

This repository contains the **Student 3** contribution for Project 2:
- Diffusion Policy core model
- Training loop and checkpointing
- Multi-step denoising sampling for action chunk inference
- Offline evaluation and ablation automation

The project is part of a larger dual-arm imitation learning pipeline. Student 3 focuses on the generative diffusion policy model, experimental evaluation, and documentation of results.

## 1. Data format (already prepared)

Place data under `preprocessed/`:
- `liftpot_images.npy`: shape `[N, T, D_obs]` (current file: `[500, 75, 1024]`)
- `liftpot_actions.npy`: shape `[N, T, D_act]` (current file: `[500, 75, 16]`)
- `stats.json`: optional action min/max for normalization

The code assumes each window uses:
- observation history length `H`
- action chunk length `K`

From each episode/time index `t`:
- condition = `obs[t-H+1:t+1]`
- target = `act[t:t+K]`

## 2. Install

This project works well with a Python 3.9 environment and a GPU like RTX 2060.

```bash
python -m pip install --upgrade pip
conda install pytorch=2.4.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

Recommended runtime:
- Python 3.9
- PyTorch 2.4.1 + CUDA 12.1 (`cu121`)
- NVIDIA driver compatible with CUDA 12.1

If you need CPU-only execution, install PyTorch without CUDA support and omit `pytorch-cuda`.

## 3. Train diffusion policy

```bash
python train_diffusion_policy.py \
  --data-dir preprocessed \
  --history 4 \
  --horizon 8 \
  --diffusion-steps 100 \
  --batch-size 64 \
  --epochs 60 \
  --out-dir outputs
```

Saved files:
- `outputs/best.pt`
- `outputs/latest.pt`
- `outputs/train_log.jsonl`
- `outputs/train_config.json`

## 4. Offline evaluation

```bash
python eval_diffusion_policy.py \
  --checkpoint outputs/best.pt \
  --data-dir preprocessed \
  --split val
```

Metrics:
- `mse_norm`, `mae_norm`: in normalized action space
- `mse_real`, `mae_real`: in original action scale
- `smoothness_l2_step`: finite-difference smoothness proxy

## 5. Sample action chunk from checkpoint

```bash
python sample_policy.py \
  --checkpoint outputs/best.pt \
  --data-dir preprocessed \
  --episode 0 \
  --timestep 10
```

This prints:
- predicted normalized action chunk
- predicted denormalized action chunk
- first action (for receding-horizon execution)

## 6. Implementation notes

- Backbone: conditional Transformer (`diffusion_policy/model.py`)
- Diffusion training target: epsilon/noise prediction
- Noise schedule: linear beta DDPM (`diffusion_policy/diffusion.py`)
- Inference: iterative denoising from Gaussian noise to action chunk
- Normalization: min-max to `[-1, 1]`, inverse at inference/evaluation

## 7. Recommended ablations (for report)

Use the same scripts and vary:
- number of demos: subsample episodes
- diffusion steps: 20 / 50 / 100 / 200
- history length: 1 / 2 / 4 / 8
- action chunk size: 1 / 4 / 8 / 16
- demo noise level: regenerate demo dataset with different perturbation

These directly support your required analysis section.

## 8. One-command ablation runner

Run all ablation studies:

```bash
python run_ablations.py \
  --study full \
  --out-root ablations \
  --epochs 30 \
  --batch-size 64 \
  --device cuda \
  --seeds 42,43,44
```

Quick smoke ablation (for debugging):

```bash
python run_ablations.py \
  --study history \
  --history-values 2,4 \
  --out-root ablations_smoke \
  --epochs 1 \
  --max-train-batches 1 \
  --max-val-batches 1 \
  --max-eval-batches 1 \
  --device cpu
```

Outputs:
- `ablations/summary.csv`
- `ablations/summary.jsonl`
- per-run folders with checkpoints and eval metrics

## 9. Plot ablation curves

```bash
python plot_ablations.py \
  --summary-csv ablations/summary.csv \
  --out-dir ablations/plots \
  --metric mse_real \
  --secondary-metric smoothness_l2_step
```

## 10. Report and AI log templates

- `REPORT_STUDENT3_TEMPLATE.md`: Student 3 method/experiment section skeleton
- `AI_PROMPT_LOG_TEMPLATE.md`: AI programming usage log skeleton
