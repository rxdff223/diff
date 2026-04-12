import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _run(cmd: List[str], cwd: Path) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _train_cmd(args: argparse.Namespace, run_dir: Path, overrides: Dict[str, int], seed: int) -> List[str]:
    cmd = [
        sys.executable,
        "train_diffusion_policy.py",
        "--data-dir",
        str(args.data_dir),
        "--obs-file",
        args.obs_file,
        "--action-file",
        args.action_file,
        "--stats-file",
        args.stats_file,
        "--out-dir",
        str(run_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip",
        str(args.grad_clip),
        "--d-model",
        str(args.d_model),
        "--nhead",
        str(args.nhead),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--beta-start",
        str(args.beta_start),
        "--beta-end",
        str(args.beta_end),
        "--val-ratio",
        str(args.val_ratio),
        "--stride",
        str(args.stride),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    if args.max_train_batches > 0:
        cmd += ["--max-train-batches", str(args.max_train_batches)]
    if args.max_val_batches > 0:
        cmd += ["--max-val-batches", str(args.max_val_batches)]

    # default values used when not overridden by ablation
    merged = {
        "history": args.history,
        "horizon": args.horizon,
        "diffusion_steps": args.diffusion_steps,
        "train_episode_limit": args.train_episode_limit,
    }
    merged.update(overrides)
    cmd += [
        "--history",
        str(merged["history"]),
        "--horizon",
        str(merged["horizon"]),
        "--diffusion-steps",
        str(merged["diffusion_steps"]),
        "--train-episode-limit",
        str(merged["train_episode_limit"]),
    ]
    return cmd


def _eval_cmd(args: argparse.Namespace, checkpoint: Path, out_json: Path, seed: int) -> List[str]:
    cmd = [
        sys.executable,
        "eval_diffusion_policy.py",
        "--checkpoint",
        str(checkpoint),
        "--data-dir",
        str(args.data_dir),
        "--obs-file",
        args.obs_file,
        "--action-file",
        args.action_file,
        "--split",
        "val",
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--val-ratio",
        str(args.val_ratio),
        "--out-json",
        str(out_json),
        "--device",
        args.device,
    ]
    if args.max_eval_batches > 0:
        cmd += ["--max-batches", str(args.max_eval_batches)]
    return cmd


def _build_experiments(args: argparse.Namespace) -> List[Dict]:
    studies = []
    if args.study in ("history", "full"):
        for v in _parse_int_list(args.history_values):
            studies.append(
                {
                    "study": "history",
                    "variable": "history",
                    "value": v,
                    "overrides": {"history": v},
                }
            )
    if args.study in ("horizon", "full"):
        for v in _parse_int_list(args.horizon_values):
            studies.append(
                {
                    "study": "horizon",
                    "variable": "horizon",
                    "value": v,
                    "overrides": {"horizon": v},
                }
            )
    if args.study in ("diffusion_steps", "full"):
        for v in _parse_int_list(args.diffusion_values):
            studies.append(
                {
                    "study": "diffusion_steps",
                    "variable": "diffusion_steps",
                    "value": v,
                    "overrides": {"diffusion_steps": v},
                }
            )
    if args.study in ("demo_count", "full"):
        for v in _parse_int_list(args.demo_values):
            studies.append(
                {
                    "study": "demo_count",
                    "variable": "train_episode_limit",
                    "value": v,
                    "overrides": {"train_episode_limit": v},
                }
            )
    if not studies:
        raise ValueError("No experiments generated. Check --study and value lists.")
    return studies


def _write_summary(summary_path_csv: Path, summary_path_jsonl: Path, rows: List[Dict]) -> None:
    summary_path_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with summary_path_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not rows:
        return
    fieldnames = sorted(rows[0].keys())
    with summary_path_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diffusion-policy ablation experiments")
    parser.add_argument("--study", type=str, default="full", choices=["history", "horizon", "diffusion_steps", "demo_count", "full"])
    parser.add_argument("--out-root", type=Path, default=Path("ablations"))

    parser.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    parser.add_argument("--obs-file", type=str, default="liftpot_images.npy")
    parser.add_argument("--action-file", type=str, default="liftpot_actions.npy")
    parser.add_argument("--stats-file", type=str, default="stats.json")

    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--train-episode-limit", type=int, default=0)
    parser.add_argument("--history-values", type=str, default="1,2,4,8")
    parser.add_argument("--horizon-values", type=str, default="1,4,8,16")
    parser.add_argument("--diffusion-values", type=str, default="20,50,100,200")
    parser.add_argument("--demo-values", type=str, default="50,100,200,400")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.out_root
    root.mkdir(parents=True, exist_ok=True)

    seeds = _parse_int_list(args.seeds)
    experiments = _build_experiments(args)
    results = []
    total = len(experiments) * len(seeds)
    done = 0

    for exp in experiments:
        for seed in seeds:
            done += 1
            run_name = f"{exp['study']}_{exp['value']}_seed{seed}"
            run_dir = root / run_name
            eval_json = run_dir / "eval_metrics.json"
            checkpoint = run_dir / "best.pt"

            print(f"[{done}/{total}] {run_name}")
            if args.resume and eval_json.exists() and checkpoint.exists():
                metrics = json.loads(eval_json.read_text(encoding="utf-8"))
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                _run(_train_cmd(args, run_dir, exp["overrides"], seed), cwd=Path.cwd())
                _run(_eval_cmd(args, checkpoint, eval_json, seed), cwd=Path.cwd())
                metrics = json.loads(eval_json.read_text(encoding="utf-8"))

            row = {
                "run_name": run_name,
                "study": exp["study"],
                "variable": exp["variable"],
                "value": exp["value"],
                "seed": seed,
                "checkpoint": str(checkpoint),
                "mse_norm": metrics.get("mse_norm"),
                "mae_norm": metrics.get("mae_norm"),
                "mse_real": metrics.get("mse_real"),
                "mae_real": metrics.get("mae_real"),
                "smoothness_l2_step": metrics.get("smoothness_l2_step"),
                "num_batches": metrics.get("num_batches"),
            }
            results.append(row)
            _write_summary(root / "summary.csv", root / "summary.jsonl", results)

    print(f"Ablations completed. Results saved to: {root}")
    print(f"Summary CSV: {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
