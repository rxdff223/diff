# Behavioral Cloning (BC) and DAgger Implementation

This folder implements Behavioral Cloning (BC) and DAgger algorithms for robot behavior learning.

## Overview

This is part of a course assignment implementing two imitation learning methods:
- **Behavioral Cloning (BC)**: Learn policy directly from expert demonstrations
- **DAgger (Dataset Aggregation)**: Iteratively collect expert corrections to mitigate covariate shift

## File Structure

| File | Description |
|------|-------------|
| `main.py` | Main implementation code with complete BC and DAgger pipeline |
| `bc_model_best.pth` | Best BC model with lowest validation loss during training |
| `bc_model_final.pth` | Final BC model at the end of training |
| `bc_results.json` | Training/validation losses, per-dimension MAE, temporal errors, and DAgger results |
| `bc_training_curve.png` | BC training curves (linear and log scale) |
| `bc_mae_per_dim.png` | Bar chart of MAE for each action dimension |
| `bc_prediction_scatter.png` | Scatter plots comparing predicted vs true values |
| `covariate_shift_analysis.png` | Temporal error accumulation analysis |
| `dagger_progress.png` | Validation loss changes across DAgger rounds |

## Data Format

The code is adapted for preprocessed feature vector inputs:
- **images**: `(500, 75, 1024)` - 500 demonstrations, 75 steps each, 1024-dim feature vectors
- **actions**: `(500, 75, 16)` - 500 demonstrations, 75 steps each, 16-dim actions

## Model Architecture

MLP network structure:
- Input layer: 1024-dim feature vector
- Hidden layers: 512 → 256 → 128 (with ReLU activation and Dropout)
- Output layer: 16-dim action (with Sigmoid activation, output range [0, 1])

## Training Pipeline

1. **Data Loading & Preprocessing**: Normalize features and actions to [0, 1] range
2. **Dataset Split**: 80% training set, 20% validation set
3. **BC Training**: MSE loss, Adam optimizer, learning rate scheduling
4. **Model Evaluation**: Compute MSE, MAE, and per-dimension errors
5. **Covariate Shift Analysis**: Analyze error accumulation across time steps
6. **DAgger Training**: 5 rounds of iteration, adding new data and retraining each round

## Usage

```bash
python main.py
```

Ensure the data directory contains:
- `liftpot_actions.npy`
- `liftpot_images.npy`
- `stats.json`

## Results

- BC final training loss: ~0.003
- BC final validation loss: ~0.0014
- DAgger validation loss decreases over 5 rounds
- Temporal error analysis shows covariate shift issue exists

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib