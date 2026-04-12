import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy import (
    ActionDiffusionTransformer,
    DiffusionPolicy,
    DiffusionScheduler,
    DualArmSequenceDataset,
)
from diffusion_policy.data import ActionNormStats
from diffusion_policy.utils import default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate diffusion policy offline")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--obs-file", type=str, default="liftpot_images.npy")
    parser.add_argument("--action-file", type=str, default="liftpot_actions.npy")
    parser.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means full evaluation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def reconstruct_datasets(
    obs_path: Path,
    action_path: Path,
    checkpoint: Dict,
    split: str,
    val_ratio: float,
    seed: int,
) -> DualArmSequenceDataset:
    cfg = checkpoint["config"]
    history = int(cfg["history"])
    horizon = int(cfg["horizon"])
    stride = int(cfg.get("stride", 1))
    train_episode_limit = int(cfg.get("train_episode_limit", 0) or 0)

    obs_mm = np.load(obs_path, mmap_mode="r")
    num_eps = obs_mm.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_eps)
    n_val = max(1, int(round(num_eps * val_ratio)))
    n_val = min(n_val, num_eps - 1)
    val_eps = perm[:n_val]
    train_eps = perm[n_val:]
    if train_episode_limit > 0:
        train_eps = train_eps[: min(train_episode_limit, len(train_eps))]

    if split == "train":
        eps = train_eps
    elif split == "val":
        eps = val_eps
    else:
        eps = np.arange(num_eps, dtype=np.int32)

    norm_stats = ActionNormStats.from_dict(checkpoint["action_norm_stats"])
    return DualArmSequenceDataset(
        obs_path=obs_path,
        action_path=action_path,
        history=history,
        horizon=horizon,
        stride=stride,
        episode_indices=eps,
        norm_stats=norm_stats,
    )


@torch.no_grad()
def evaluate(policy: DiffusionPolicy, loader: DataLoader, dataset: DualArmSequenceDataset, max_batches: int) -> Dict[str, float]:
    policy.eval()
    mse_norm = []
    mae_norm = []
    mse_real = []
    mae_real = []
    smooth = []

    for i, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False), start=1):
        if max_batches > 0 and i > max_batches:
            break
        obs = batch["obs"].to(next(policy.parameters()).device, non_blocking=True)
        gt = batch["action"].to(next(policy.parameters()).device, non_blocking=True)
        pred = policy.sample_action_chunk(obs).clamp(-1.0, 1.0)

        mse_norm.append(torch.mean((pred - gt) ** 2).item())
        mae_norm.append(torch.mean(torch.abs(pred - gt)).item())

        pred_np = pred.cpu().numpy()
        gt_np = gt.cpu().numpy()
        pred_real = dataset.denormalize_actions(pred_np)
        gt_real = dataset.denormalize_actions(gt_np)

        mse_real.append(float(np.mean((pred_real - gt_real) ** 2)))
        mae_real.append(float(np.mean(np.abs(pred_real - gt_real))))

        # smoothness proxy: average L2 norm of action finite differences
        d = np.diff(pred_real, axis=1)
        smooth.append(float(np.mean(np.linalg.norm(d, axis=-1))))

    return {
        "num_batches": len(mse_norm),
        "mse_norm": float(np.mean(mse_norm)) if mse_norm else float("nan"),
        "mae_norm": float(np.mean(mae_norm)) if mae_norm else float("nan"),
        "mse_real": float(np.mean(mse_real)) if mse_real else float("nan"),
        "mae_real": float(np.mean(mae_real)) if mae_real else float("nan"),
        "smoothness_l2_step": float(np.mean(smooth)) if smooth else float("nan"),
    }


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

    dataset = reconstruct_datasets(
        obs_path=args.data_dir / args.obs_file,
        action_path=args.data_dir / args.action_file,
        checkpoint=ckpt,
        split=args.split,
        val_ratio=float(args.val_ratio if args.val_ratio is not None else ckpt["config"].get("val_ratio", 0.1)),
        seed=int(args.seed if args.seed is not None else ckpt["config"].get("seed", 42)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    metrics = evaluate(policy, loader, dataset, max_batches=args.max_batches)
    print(json.dumps(metrics, indent=2))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
