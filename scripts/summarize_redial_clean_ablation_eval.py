#!/usr/bin/env python
"""Summarize clean ablations from collision-safe per-run evaluation JSON."""

import json
from pathlib import Path

import numpy as np
from scipy import stats

from summarize_paired_rec import METRICS, load_group, metric_value, print_group


ROOT_DIR = Path(__file__).resolve().parents[1]
SEEDS = [7, 13, 21, 22, 42, 77, 100, 2024, 3407, 9999]


def load_eval_group(variant):
    runs = {}
    for seed in SEEDS:
        path = (
            ROOT_DIR
            / "runs"
            / "redial_clean_ablation_10seed"
            / variant
            / f"seed_{seed}"
            / "recovered_test"
            / "eval_test.json"
        )
        if not path.is_file():
            raise ValueError(f"missing evaluation result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[seed] = {
            "path": path,
            "seed": seed,
            "output_dir": str(path.parent),
            "epoch": -1,
            "valid_loss": float("nan"),
            "test": payload["metrics"],
        }
    return runs


def print_eval_group(name, runs):
    print(f"\n== {name} ==")
    for seed in SEEDS:
        run = runs[seed]
        print(
            f"seed {seed:>4} "
            f"R@10={metric_value(run, 'recall@10'):.4f} "
            f"N@10={metric_value(run, 'ndcg@10'):.4f} "
            f"M@10={metric_value(run, 'mrr@10'):.4f}"
        )
    print("summary:")
    for metric in METRICS:
        values = np.array([metric_value(runs[seed], metric) for seed in SEEDS])
        print(f"  {metric:>9}: {values.mean():.4f} +/- {values.std(ddof=1):.4f}")


def print_paired(title, baseline, treatment):
    print(f"\n== {title} ==")
    print("metric       mean_delta  std_delta   paired_t_p   wilcoxon_p")
    for metric in METRICS:
        base = np.array([metric_value(baseline[seed], metric) for seed in SEEDS])
        treat = np.array([metric_value(treatment[seed], metric) for seed in SEEDS])
        delta = treat - base
        t_p = stats.ttest_rel(treat, base).pvalue
        try:
            w_p = stats.wilcoxon(treat, base).pvalue
        except ValueError:
            w_p = float("nan")
        print(
            f"{metric:>9}  {delta.mean():+10.6f}  {delta.std(ddof=1):9.6f}  "
            f"{t_p:11.6g}  {w_p:11.6g}"
        )


def main():
    baseline = load_group(
        str(ROOT_DIR / "../MSCRS-main/log/*.log"), "redial_10seed_mscrs"
    )
    static = load_eval_group("static_prior")
    global_query = load_eval_group("global_query")
    entity_query = load_eval_group("entity_query")
    full = load_group(str(ROOT_DIR / "log/*.log"), "redial_h2_matched_10seed/seed_")

    for name, runs in (("MSCRS", baseline), ("QAMR-CRS", full)):
        if set(runs) != set(SEEDS):
            raise ValueError(f"{name} seed mismatch: {sorted(runs)}")

    print_group("MSCRS", baseline, SEEDS)
    print_eval_group("Static prior only", static)
    print_eval_group("Global query-aware routing", global_query)
    print_eval_group("Entity-conditioned query routing", entity_query)
    print_group("QAMR-CRS", full, SEEDS)

    print_paired("Static prior only - MSCRS", baseline, static)
    print_paired("Global routing - static prior", static, global_query)
    print_paired("Entity-conditioned - global routing", global_query, entity_query)
    print_paired("QAMR-CRS - entity-conditioned", entity_query, full)


if __name__ == "__main__":
    main()
