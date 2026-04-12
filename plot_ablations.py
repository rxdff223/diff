import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation summary from summary.csv")
    parser.add_argument("--summary-csv", type=Path, default=Path("ablations/summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("ablations/plots"))
    parser.add_argument("--metric", type=str, default="mse_real")
    parser.add_argument("--secondary-metric", type=str, default="smoothness_l2_step")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def aggregate(rows: List[Dict[str, str]], metric: str) -> Dict[str, List[Tuple[float, float, float]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        study = row["study"]
        value = float(row["value"])
        v = float(row[metric])
        grouped[study][value].append(v)

    out = {}
    for study, by_value in grouped.items():
        points = []
        for value in sorted(by_value.keys()):
            vals = by_value[value]
            mu = mean(vals)
            sd = stdev(vals) if len(vals) > 1 else 0.0
            points.append((value, mu, sd))
        out[study] = points
    return out


def plot_metric(agg: Dict[str, List[Tuple[float, float, float]]], metric: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for study, points in agg.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        es = [p[2] for p in points]

        plt.figure(figsize=(7, 4.2))
        plt.plot(xs, ys, marker="o", linewidth=2)
        if any(e > 0 for e in es):
            low = [y - e for y, e in zip(ys, es)]
            high = [y + e for y, e in zip(ys, es)]
            plt.fill_between(xs, low, high, alpha=0.2)
        plt.title(f"{study} ablation ({metric})")
        plt.xlabel("value")
        plt.ylabel(metric)
        plt.grid(alpha=0.3)
        out_path = out_dir / f"{study}_{metric}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.summary_csv}")

    for metric in [args.metric, args.secondary_metric]:
        agg = aggregate(rows, metric)
        plot_metric(agg, metric, args.out_dir)
    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
