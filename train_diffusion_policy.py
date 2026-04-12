import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy import (
    ActionDiffusionTransformer,
    DiffusionPolicy,
    DiffusionScheduler,
    create_train_val_datasets,
)
from diffusion_policy.utils import default_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual-arm diffusion policy")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--obs-file", type=str, default="liftpot_images.npy")
    parser.add_argument("--action-file", type=str, default="liftpot_actions.npy")
    parser.add_argument("--stats-file", type=str, default="stats.json")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))

    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--train-episode-limit",
        type=int,
        default=0,
        help="0 means use all train episodes after split",
    )

    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 means full epoch")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 means full validation")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def evaluate(
    policy: DiffusionPolicy,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> float:
    policy.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            if max_batches > 0 and i > max_batches:
                break
            batch = _move_batch(batch, device)
            loss = policy.compute_loss(batch["obs"], batch["action"])["loss"]
            losses.append(loss.item())
    return float(sum(losses) / max(len(losses), 1))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = default_device(args.device)

    obs_path = args.data_dir / args.obs_file
    action_path = args.data_dir / args.action_file
    stats_path = args.data_dir / args.stats_file
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = create_train_val_datasets(
        obs_path=obs_path,
        action_path=action_path,
        history=args.history,
        horizon=args.horizon,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stride=args.stride,
        stats_path=stats_path,
        train_episode_limit=args.train_episode_limit,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = ActionDiffusionTransformer(
        obs_dim=train_ds.obs_dim,
        action_dim=train_ds.act_dim,
        history=args.history,
        horizon=args.horizon,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    scheduler = DiffusionScheduler(
        num_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    policy = DiffusionPolicy(
        model=model,
        scheduler=scheduler,
        history=args.history,
        horizon=args.horizon,
        action_dim=train_ds.act_dim,
        obs_dim=train_ds.obs_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    log_path = args.out_dir / "train_log.jsonl"
    best_ckpt = args.out_dir / "best.pt"
    latest_ckpt = args.out_dir / "latest.pt"
    best_val = float("inf")

    config_dict = {}
    for k, v in vars(args).items():
        config_dict[k] = str(v) if isinstance(v, Path) else v
    config_dict["device"] = str(device)
    config_dict["obs_dim"] = train_ds.obs_dim
    config_dict["action_dim"] = train_ds.act_dim
    config_dict["train_windows"] = len(train_ds)
    config_dict["val_windows"] = len(val_ds)
    (args.out_dir / "train_config.json").write_text(
        json.dumps(config_dict, indent=2, default=str),
        encoding="utf-8",
    )

    for epoch in range(1, args.epochs + 1):
        policy.train()
        running_loss = 0.0
        train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs:03d}", leave=False)
        for i, batch in enumerate(pbar, start=1):
            if args.max_train_batches > 0 and i > args.max_train_batches:
                break
            batch = _move_batch(batch, device)
            loss = policy.compute_loss(batch["obs"], batch["action"])["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            train_batches += 1
            pbar.set_postfix(train_loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(train_batches, 1)
        val_loss = evaluate(policy, val_loader, device, max_batches=args.max_val_batches)

        ckpt = {
            "epoch": epoch,
            "model_state": policy.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": config_dict,
            "model_kwargs": {
                "obs_dim": train_ds.obs_dim,
                "action_dim": train_ds.act_dim,
                "history": args.history,
                "horizon": args.horizon,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            },
            "scheduler_kwargs": {
                "num_steps": args.diffusion_steps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
            },
            "action_norm_stats": train_ds.norm_stats.to_dict(),
        }
        torch.save(ckpt, latest_ckpt)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, best_ckpt)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss},
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    print(f"Training complete. Best val loss: {best_val:.6f}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
