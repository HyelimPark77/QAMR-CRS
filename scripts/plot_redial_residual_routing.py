#!/usr/bin/env python
"""Plot raw and residual-adjusted ReDial routing coefficients."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from summarize_redial_residual_eval import load_coefficients


COLORS = ["#355C91", "#C65D0E", "#4F7D3A", "#76558F"]
NAMES = ["KG", "Co-occ.", "Text", "Image"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/redial_residual_eval"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/routing_residual_tradeoff"),
        help="Output path without an extension.",
    )
    args = parser.parse_args()

    selection_path = args.root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    beta = float(selection["selected_beta"])
    seeds = selection["seeds"]
    entity_fraction, raw, adjusted = load_coefficients(args.root, beta, seeds)
    values = np.vstack([raw, adjusted])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    fig, ax = plt.subplots(figsize=(3.35, 1.75))

    y = np.array([1, 0])
    left = np.zeros(2)
    for index, (name, color) in enumerate(zip(NAMES, COLORS)):
        widths = values[:, index]
        bars = ax.barh(
            y,
            widths,
            left=left,
            height=0.48,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=name,
        )
        for row, bar in enumerate(bars):
            width = widths[row]
            if width >= 0.055:
                ax.text(
                    left[row] + width / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width * 100:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                    fontweight="bold",
                )
        left += widths

    ax.set_yticks(y, ["Raw routing", f"Residual ($\\beta={beta:g}$)"])
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("Modality coefficient (%)", labelpad=2)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.8,
        fontsize=7,
    )

    fig.tight_layout(pad=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"selected beta: {beta}")
    print(f"entity-bearing fraction: {entity_fraction:.4f}")
    print("raw:", ", ".join(f"{name}={value:.4f}" for name, value in zip(NAMES, raw)))
    print(
        "residual-adjusted:",
        ", ".join(f"{name}={value:.4f}" for name, value in zip(NAMES, adjusted)),
    )
    print(f"saved: {args.output.with_suffix('.pdf')}")
    print(f"saved: {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
