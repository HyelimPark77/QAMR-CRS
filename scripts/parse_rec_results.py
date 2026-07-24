#!/usr/bin/env python
"""Parse recommendation metrics from QAMR-CRS log files.

The training scripts log Python dictionaries such as:

    {'test/recall@10': 0.24, 'test/ndcg@10': 0.15, ...}

This utility extracts those rows and prints a compact table row for paper
updates. Current training scripts evaluate the test split once after restoring
the checkpoint selected by validation loss.
"""

import argparse
import ast
import csv
import glob
import re
import sys
from pathlib import Path


METRIC_KEYS = [
    "recall@1",
    "recall@10",
    "recall@50",
    "ndcg@10",
    "ndcg@50",
    "mrr@10",
    "mrr@50",
]


def _clean_payload(payload: str) -> str:
    return re.sub(r"np\.float64\(([^()]*)\)", r"\1", payload)


def parse_log(path: Path, split: str):
    rows = []
    prefix = f"{split}/"
    marker = f"'{prefix}recall@10'"

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if marker not in line:
            continue
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end < start:
            continue
        payload = _clean_payload(line[start : end + 1])
        try:
            record = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        if f"{prefix}recall@10" in record:
            rows.append(record)
    return rows


def select_row(rows, split: str, best_by: str):
    key = f"{split}/{best_by}"
    if best_by == "last":
        return rows[-1]
    if split == "test" and len(rows) > 1:
        raise ValueError(
            "refusing to select an epoch by a test metric; use a log produced "
            "by the validation-selected evaluation workflow"
        )
    if key not in rows[0]:
        raise KeyError(f"metric {key!r} not found in parsed log rows")
    return max(rows, key=lambda row: row[key])


def compact_values(row, split: str):
    values = {}
    for key in METRIC_KEYS:
        values[key] = row.get(f"{split}/{key}")
    values["loss"] = row.get(f"{split}/loss")
    values["epoch"] = row.get("epoch")
    return values


def print_text(path: Path, row, split: str, best_by: str):
    values = compact_values(row, split)
    print(f"\n== {path} | best_by={best_by} ==")
    print(
        "epoch {epoch} | "
        "R@1 {recall@1:.4f} R@10 {recall@10:.4f} R@50 {recall@50:.4f} | "
        "N@10 {ndcg@10:.4f} N@50 {ndcg@50:.4f} | "
        "M@10 {mrr@10:.4f} M@50 {mrr@50:.4f}".format(**values)
    )
    if values["loss"] is not None:
        print(f"{split}/loss {values['loss']:.6f}")


def write_csv(rows, output_path: Path):
    fieldnames = ["log", "best_by", "epoch"] + METRIC_KEYS + ["loss"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Parse recommendation metrics from QAMR-CRS logs.")
    parser.add_argument("logs", nargs="+", help="Log files or glob patterns, e.g. 'log/*.log'.")
    parser.add_argument("--split", default="test", choices=["valid", "test"], help="Metric split to parse.")
    parser.add_argument(
        "--best-by",
        default="last",
        choices=["recall@1", "recall@10", "recall@50", "ndcg@10", "ndcg@50", "mrr@10", "mrr@50", "last"],
        help="Row selection rule. Test logs should contain one validation-selected report.",
    )
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    paths = []
    for pattern in args.logs:
        matches = [Path(p) for p in glob.glob(pattern)]
        paths.extend(matches if matches else [Path(pattern)])
    paths = sorted(dict.fromkeys(paths))

    csv_rows = []
    missing = []

    for path in paths:
        if not path.exists():
            missing.append(path)
            continue
        rows = parse_log(path, args.split)
        if not rows:
            continue
        row = select_row(rows, args.split, args.best_by)
        print_text(path, row, args.split, args.best_by)

        values = compact_values(row, args.split)
        values["log"] = str(path)
        values["best_by"] = args.best_by
        csv_rows.append(values)

    if args.csv:
        write_csv(csv_rows, Path(args.csv))
        print(f"\nwrote {args.csv}")

    if missing:
        print("\nMissing log files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
