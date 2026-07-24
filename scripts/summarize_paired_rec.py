#!/usr/bin/env python
"""Summarize paired recommendation runs selected by validation loss."""

import argparse
import ast
import glob
import re
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = [
    "recall@1",
    "recall@10",
    "recall@50",
    "ndcg@10",
    "ndcg@50",
    "mrr@10",
    "mrr@50",
]


def parse_dict(line):
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end < start:
        return None
    payload = re.sub(r"np\.float64\(([^()]*)\)", r"\1", line[start : end + 1])
    try:
        value = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def parse_run(path):
    config = None
    valid_rows = []
    test_rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        record = parse_dict(line)
        if record is None:
            continue
        if config is None and "seed" in record and "output_dir" in record:
            config = record
        if "valid/loss" in record and "epoch" in record:
            valid_rows.append(record)
        if "test/recall@10" in record and "epoch" in record:
            test_rows.append(record)

    if config is None or not valid_rows or not test_rows:
        return None

    selected_valid = min(valid_rows, key=lambda row: row["valid/loss"])
    selected_epoch = selected_valid["epoch"]
    matching_test = [row for row in test_rows if row["epoch"] == selected_epoch]
    if len(matching_test) != 1:
        raise ValueError(
            f"{path}: expected one test row for validation-selected epoch "
            f"{selected_epoch}, found {len(matching_test)}"
        )

    return {
        "path": path,
        "seed": int(config["seed"]),
        "output_dir": str(config["output_dir"]),
        "epoch": selected_epoch,
        "valid_loss": float(selected_valid["valid/loss"]),
        "test": matching_test[0],
    }


def load_group(pattern, run_tag):
    runs = {}
    for name in sorted(glob.glob(pattern)):
        path = Path(name)
        if run_tag not in path.read_text(encoding="utf-8", errors="ignore"):
            continue
        run = parse_run(path)
        if run is None or run_tag not in run["output_dir"]:
            continue
        seed = run["seed"]
        if seed in runs:
            raise ValueError(
                f"duplicate seed {seed} for tag {run_tag!r}: "
                f"{runs[seed]['path']} and {run['path']}"
            )
        runs[seed] = run
    if not runs:
        raise ValueError(f"no complete runs found for tag {run_tag!r} in {pattern!r}")
    return runs


def metric_value(run, metric):
    return float(run["test"][f"test/{metric}"])


def print_group(name, runs, seeds):
    print(f"\n== {name} ==")
    for seed in seeds:
        run = runs[seed]
        print(
            f"seed {seed:>4} epoch {run['epoch']:>2} "
            f"valid_loss={run['valid_loss']:.6f} "
            f"R@10={metric_value(run, 'recall@10'):.4f} "
            f"N@10={metric_value(run, 'ndcg@10'):.4f} "
            f"M@10={metric_value(run, 'mrr@10'):.4f}"
        )
    print("summary:")
    for metric in METRICS:
        values = np.array([metric_value(runs[seed], metric) for seed in seeds])
        print(f"  {metric:>9}: {values.mean():.4f} +/- {values.std(ddof=1):.4f}")


def print_paired(baseline, treatment, seeds):
    print("\n== paired treatment - baseline ==")
    print("metric       mean_delta  std_delta   paired_t_p   wilcoxon_p")
    for metric in METRICS:
        base = np.array([metric_value(baseline[seed], metric) for seed in seeds])
        treat = np.array([metric_value(treatment[seed], metric) for seed in seeds])
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
    parser = argparse.ArgumentParser(
        description="Compare paired recommendation runs using validation-loss selection."
    )
    parser.add_argument("--baseline-logs", required=True, help="Baseline log glob.")
    parser.add_argument("--baseline-tag", required=True, help="Substring in baseline output_dir.")
    parser.add_argument("--treatment-logs", required=True, help="Treatment log glob.")
    parser.add_argument("--treatment-tag", required=True, help="Substring in treatment output_dir.")
    parser.add_argument("--baseline-name", default="Baseline")
    parser.add_argument("--treatment-name", default="Treatment")
    args = parser.parse_args()

    baseline = load_group(args.baseline_logs, args.baseline_tag)
    treatment = load_group(args.treatment_logs, args.treatment_tag)
    baseline_seeds = set(baseline)
    treatment_seeds = set(treatment)
    if baseline_seeds != treatment_seeds:
        raise ValueError(
            f"seed mismatch: baseline={sorted(baseline_seeds)}, "
            f"treatment={sorted(treatment_seeds)}"
        )

    seeds = sorted(baseline_seeds)
    if len(seeds) < 2:
        raise ValueError("paired statistics require at least two seeds")
    print(f"paired seeds ({len(seeds)}): {seeds}")
    print_group(args.baseline_name, baseline, seeds)
    print_group(args.treatment_name, treatment, seeds)
    print_paired(baseline, treatment, seeds)


if __name__ == "__main__":
    main()
