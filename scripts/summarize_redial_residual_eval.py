#!/usr/bin/env python
"""Compare validation-selected residual evaluation with the trained H2 model."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from summarize_paired_rec import METRICS, load_group, metric_value


def beta_tag(beta):
    return str(beta).replace(".", "p")


def load_residual_runs(root, beta, seeds):
    runs = {}
    for seed in seeds:
        path = root / f"test_beta_{beta_tag(beta)}" / f"seed_{seed}" / "eval_test.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[seed] = payload["metrics"]
    return runs


def load_coefficients(root, beta, seeds):
    raw_sum = np.zeros(4)
    count = 0
    for seed in seeds:
        path = root / f"test_beta_{beta_tag(beta)}" / f"seed_{seed}" / "evidence_test.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            batch_size = len(record["labels"])
            raw_sum += np.asarray(record["router_weights"]) * batch_size
            count += batch_size
    raw_unconditional = raw_sum / count
    entity_bearing_fraction = float(raw_unconditional.sum())
    if entity_bearing_fraction <= 0:
        raise ValueError("No entity-bearing examples were found in the evidence logs.")

    # Entity-aware summaries are all-zero for examples without mentioned
    # entities. Condition on entity-bearing examples before comparing modality
    # distributions so the four coefficients sum to one.
    raw_conditional = raw_unconditional / entity_bearing_fraction
    adjusted_conditional = beta / len(raw_conditional) + (1.0 - beta) * raw_conditional
    return entity_bearing_fraction, raw_conditional, adjusted_conditional


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/redial_residual_eval"))
    parser.add_argument("--baseline-logs", default="log/*.log")
    parser.add_argument("--baseline-tag", default="redial_h2_matched_10seed/seed_")
    args = parser.parse_args()

    selection = json.loads((args.root / "selection.json").read_text(encoding="utf-8"))
    beta = float(selection["selected_beta"])
    seeds = selection["seeds"]
    baseline = load_group(args.baseline_logs, args.baseline_tag)
    if set(baseline) != set(seeds):
        raise ValueError(f"seed mismatch: baseline={sorted(baseline)}, selected={sorted(seeds)}")
    residual = load_residual_runs(args.root, beta, seeds)

    print(f"selected_beta: {beta}")
    print(f"paired seeds ({len(seeds)}): {seeds}")
    print("\nmetric       H2_mean  residual_mean  mean_delta   paired_t_p   wilcoxon_p")
    for metric in METRICS:
        base = np.asarray([metric_value(baseline[seed], metric) for seed in seeds])
        treat = np.asarray([residual[seed][f"test/{metric}"] for seed in seeds])
        delta = treat - base
        t_p = stats.ttest_rel(treat, base).pvalue
        try:
            w_p = stats.wilcoxon(treat, base).pvalue
        except ValueError:
            w_p = float("nan")
        print(
            f"{metric:>9}  {base.mean():8.4f}  {treat.mean():13.4f}  "
            f"{delta.mean():+10.6f}  {t_p:11.6g}  {w_p:11.6g}"
        )

    entity_fraction, raw, adjusted = load_coefficients(args.root, beta, seeds)
    names = ["KG", "Co-occurrence", "Text", "Image"]
    print("\n10-seed test routing summary (conditioned on entity-bearing examples):")
    print(f"entity-bearing fraction: {entity_fraction:.4f}")
    for index, name in enumerate(names):
        print(f"{name:>13}: raw={raw[index]:.4f} residual_adjusted={adjusted[index]:.4f}")


if __name__ == "__main__":
    main()
