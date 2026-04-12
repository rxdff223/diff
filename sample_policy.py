import argparse
from pathlib import Path

import numpy as np
import torch

from diffusion_policy import ActionDiffusionTransformer, DiffusionPolicy, DiffusionScheduler
from diffusion_policy.data import ActionNormStats
from diffusion_policy.utils import default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample action chunk from a trained diffusion policy")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--obs-file", type=str, default="liftpot_images.npy")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=10, help="Current env time index")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = default_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = ActionDiffusionTransformer(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    scheduler = DiffusionScheduler(**ckpt["scheduler_kwargs"], device=device)
    policy = DiffusionPolicy(
        model=model,
        scheduler=scheduler,
        history=ckpt["model_kwargs"]["history"],
        horizon=ckpt["model_kwargs"]["horizon"],
        action_dim=ckpt["model_kwargs"]["action_dim"],
        obs_dim=ckpt["model_kwargs"]["obs_dim"],
    ).to(device)
    policy.eval()

    obs = np.load(args.data_dir / args.obs_file, mmap_mode="r")
    history = ckpt["model_kwargs"]["history"]
    ep = args.episode
    t = args.timestep
    if t < history - 1:
        raise ValueError(f"timestep must be >= history-1 ({history - 1}), got {t}")
    if ep < 0 or ep >= obs.shape[0]:
        raise ValueError(f"episode out of range: {ep}")
    if t >= obs.shape[1]:
        raise ValueError(f"timestep out of range: {t}")

    obs_hist = obs[ep, t - history + 1 : t + 1].astype(np.float32)
    obs_t = torch.from_numpy(obs_hist).unsqueeze(0).to(device)

    pred_norm = policy.sample_action_chunk(obs_t).clamp(-1.0, 1.0)[0].cpu().numpy()
    norm_stats = ActionNormStats.from_dict(ckpt["action_norm_stats"])
    pred_real = ((pred_norm + 1.0) * 0.5) * norm_stats.action_range + norm_stats.action_min

    np.set_printoptions(precision=4, suppress=True)
    print("Predicted normalized action chunk:")
    print(pred_norm)
    print("\nPredicted denormalized action chunk:")
    print(pred_real)
    print("\nFirst action to execute:")
    print(pred_real[0])


if __name__ == "__main__":
    main()
