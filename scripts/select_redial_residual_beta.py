#!/usr/bin/env python
"""Select one residual beta using validation loss only."""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_SEEDS = [7, 13, 21, 22, 42, 77, 100, 2024, 3407, 9999]


def beta_tag(beta):
    return str(beta).replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/redial_residual_eval"))
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    summaries = []
    for beta in args.betas:
        losses = []
        recalls = []
        for seed in args.seeds:
            path = args.root / f"valid_beta_{beta_tag(beta)}" / f"seed_{seed}" / "eval_valid.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            losses.append(float(metrics["valid/loss"]))
            recalls.append(float(metrics["valid/recall@10"]))
        summaries.append({
            "beta": beta,
            "mean_valid_loss": float(np.mean(losses)),
            "std_valid_loss": float(np.std(losses, ddof=1)),
            "mean_valid_recall_at_10": float(np.mean(recalls)),
            "std_valid_recall_at_10": float(np.std(recalls, ddof=1)),
        })

    selected = min(summaries, key=lambda row: row["mean_valid_loss"])
    output = {
        "selection_rule": "minimum mean validation loss across paired seeds",
        "seeds": args.seeds,
        "candidates": summaries,
        "selected_beta": selected["beta"],
    }
    args.root.mkdir(parents=True, exist_ok=True)
    output_path = args.root / "selection.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for row in summaries:
        print(
            f"beta={row['beta']:.2f} "
            f"valid_loss={row['mean_valid_loss']:.6f} +/- {row['std_valid_loss']:.6f} "
            f"valid_R10={row['mean_valid_recall_at_10']:.4f} +/- "
            f"{row['std_valid_recall_at_10']:.4f}"
        )
    print(f"SELECTED_BETA={selected['beta']}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
