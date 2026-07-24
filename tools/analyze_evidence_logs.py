"""Summarize modality routing and selected evidence from JSONL logs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


MODALITY_NAMES = ["kg", "co", "text", "image"]


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def update_router_stats(router_stats, split, router_weights):
    if router_weights is None:
        return
    for name, weight in zip(MODALITY_NAMES, router_weights):
        router_stats[split][name].append(weight)


def update_evidence_stats(evidence_stats, split, selected_evidence):
    if not selected_evidence:
        return

    for modality, payload in selected_evidence.items():
        entity_ids = payload.get("entity_ids", [])
        scores = payload.get("scores", [])
        for batch_ids, batch_scores in zip(entity_ids, scores):
            for entity_id, score in zip(batch_ids, batch_scores):
                evidence_stats[split][modality]["frequency"][str(entity_id)] += 1
                evidence_stats[split][modality]["score_sum"][str(entity_id)] += float(score)


def build_summary(records, topn):
    router_stats = defaultdict(lambda: defaultdict(list))
    evidence_stats = defaultdict(
        lambda: defaultdict(lambda: {"frequency": Counter(), "score_sum": defaultdict(float)})
    )
    split_counts = Counter()

    for record in records:
        split = record.get("split", "unknown")
        split_counts[split] += 1
        update_router_stats(router_stats, split, record.get("router_weights"))
        update_evidence_stats(evidence_stats, split, record.get("selected_evidence"))

    summary = {
        "num_records": len(records),
        "splits": {},
    }

    for split in sorted(split_counts.keys()):
        split_summary = {
            "num_records": split_counts[split],
            "router_mean": {
                modality: mean(router_stats[split][modality])
                for modality in MODALITY_NAMES
            },
            "top_evidence": {},
        }

        for modality in MODALITY_NAMES:
            freq_counter = evidence_stats[split][modality]["frequency"]
            score_sums = evidence_stats[split][modality]["score_sum"]
            top_items = []
            for entity_id, freq in freq_counter.most_common(topn):
                avg_score = score_sums[entity_id] / max(freq, 1)
                top_items.append(
                    {
                        "entity_id": int(entity_id),
                        "frequency": freq,
                        "avg_score": avg_score,
                    }
                )
            split_summary["top_evidence"][modality] = top_items

        summary["splits"][split] = split_summary

    return summary


def write_markdown(summary, output_path):
    lines = []
    lines.append("# Evidence Log Summary")
    lines.append("")
    lines.append(f"- Records: {summary['num_records']}")
    lines.append("")

    for split, payload in summary["splits"].items():
        lines.append(f"## {split}")
        lines.append("")
        lines.append("### Router Mean")
        lines.append("")
        for modality, weight in payload["router_mean"].items():
            lines.append(f"- {modality}: {weight:.4f}")
        lines.append("")
        lines.append("### Top Evidence")
        lines.append("")
        for modality, top_items in payload["top_evidence"].items():
            lines.append(f"#### {modality}")
            if not top_items:
                lines.append("")
                lines.append("- none")
                lines.append("")
                continue
            lines.append("")
            for item in top_items:
                lines.append(
                    f"- entity {item['entity_id']}: freq={item['frequency']}, avg_score={item['avg_score']:.4f}"
                )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize evidence routing logs.")
    parser.add_argument("input", type=str, help="Path to an evidence jsonl log file.")
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Number of top evidence entities to keep per modality and split.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save summary json. Defaults to <input>.summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional path to save markdown report. Defaults to <input>.summary.md",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_json = Path(args.output_json) if args.output_json else input_path.with_suffix(input_path.suffix + ".summary.json")
    output_md = Path(args.output_md) if args.output_md else input_path.with_suffix(input_path.suffix + ".summary.md")

    records = load_records(input_path)
    summary = build_summary(records, args.topn)

    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, output_md)

    print(f"wrote {output_json}")
    print(f"wrote {output_md}")


if __name__ == "__main__":
    main()
